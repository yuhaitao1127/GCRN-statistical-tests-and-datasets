#!/usr/bin/env python3
"""Reproduce the corrected eight-block Friedman and GCRN-versus-baseline tests."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


VERSION = "1.0.0"
ALPHA = 0.05
SEEDS = [42, 43, 44, 45, 46]
DATASETS = ["Temp", "RTE", "Face", "LabelMe", "MS", "CF*", "Bird", "Possent", "CF"]
COMPOSITE_MEMBERS = ["CF", "CF*"]
COMPOSITE_NAME = "CF/CF* composite"
INFERENTIAL_BLOCKS = ["Temp", "RTE", "Face", "LabelMe", "MS", "Bird", "Possent", COMPOSITE_NAME]
METHODS = ["MV", "PM", "GLAD", "DS", "EBCC", "BWA", "TiReMGE", "GOVERN", "CrowdFM", "GCRN"]
METRICS = ["Accuracy", "Macro-F1", "Weighted-F1"]
METRIC_SLUG = {
    "Accuracy": "accuracy",
    "Macro-F1": "macro_f1",
    "Weighted-F1": "weighted_f1",
}


def load_and_validate(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = ["metric", "dataset", "method", "seed", "value"]
    if list(data.columns) != required:
        raise ValueError(f"Expected columns {required}, found {list(data.columns)}")

    data["seed"] = pd.to_numeric(data["seed"], errors="raise").astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    if not np.isfinite(data["value"].to_numpy(dtype=float)).all():
        raise ValueError("The value column contains non-finite entries.")

    expected_sets = {
        "metric": set(METRICS),
        "dataset": set(DATASETS),
        "method": set(METHODS),
        "seed": set(SEEDS),
    }
    for column, expected in expected_sets.items():
        observed = set(data[column].unique())
        if observed != expected:
            raise ValueError(
                f"Unexpected {column} values. Missing={sorted(expected-observed)}; "
                f"extra={sorted(observed-expected)}"
            )

    key = ["metric", "dataset", "method", "seed"]
    if data.duplicated(key).any():
        raise ValueError("Duplicate metric-dataset-method-seed rows were found.")

    counts = data.groupby(["metric", "dataset", "method"])["seed"].nunique()
    if len(data) != 1350 or len(counts) != 270 or not (counts == 5).all():
        raise ValueError(
            "Incomplete seed-level data: expected 1,350 rows and five seeds in each "
            "of 270 metric-dataset-method cells."
        )
    return data


def build_inferential_data(data: pd.DataFrame) -> pd.DataFrame:
    """Combine CF and CF* within each metric, method, and seed before averaging runs."""
    members = data.loc[data["dataset"].isin(COMPOSITE_MEMBERS)]
    pivot = members.pivot(
        index=["metric", "method", "seed"], columns="dataset", values="value"
    ).reindex(columns=COMPOSITE_MEMBERS)
    if pivot.isna().any().any() or pivot.shape != (150, 2):
        raise ValueError("CF and CF* do not provide a complete paired five-seed composite.")
    composite = pivot.mean(axis=1).rename("value").reset_index()
    composite.insert(1, "dataset", COMPOSITE_NAME)
    composite = composite[["metric", "dataset", "method", "seed", "value"]]
    other = data.loc[~data["dataset"].isin(COMPOSITE_MEMBERS)]
    return pd.concat([other, composite], ignore_index=True)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="stable")
    adjusted_sorted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted_sorted[rank] = min(running_max, 1.0)
    adjusted = np.empty(len(p_values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def exact_wilcoxon(diff: np.ndarray) -> dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    if np.any(np.isclose(diff, 0.0)):
        raise ValueError("Exact Wilcoxon calculation requires non-zero paired differences.")

    ranks = stats.rankdata(np.abs(diff), method="average")
    if not np.array_equal(np.sort(ranks), np.arange(1.0, diff.size + 1.0)):
        raise ValueError("Tied absolute differences are not supported by this exact calculation.")

    w_plus = float(ranks[diff > 0.0].sum())
    w_minus = float(ranks[diff < 0.0].sum())
    total_rank = float(ranks.sum())
    statistic = min(w_plus, w_minus)

    sign_assignments = np.asarray(
        list(itertools.product((0.0, 1.0), repeat=diff.size)), dtype=float
    )
    null_w_plus = sign_assignments @ ranks
    observed_deviation = abs(w_plus - total_rank / 2.0)
    raw_p = float(
        np.mean(
            np.abs(null_w_plus - total_rank / 2.0)
            >= observed_deviation - 1e-12
        )
    )
    rank_biserial = float((w_plus - w_minus) / total_rank)
    return {
        "w_plus": w_plus,
        "w_minus": w_minus,
        "wilcoxon_w": statistic,
        "raw_p": raw_p,
        "rank_biserial": rank_biserial,
        "enumerations": int(sign_assignments.shape[0]),
    }


def hodges_lehmann_interval(diff: np.ndarray, alpha: float) -> dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    n = diff.size
    walsh = np.sort(
        np.asarray(
            [(diff[i] + diff[j]) / 2.0 for i in range(n) for j in range(i, n)],
            dtype=float,
        )
    )

    ranks = np.arange(1, n + 1, dtype=int)
    sign_assignments = np.asarray(
        list(itertools.product((0, 1), repeat=n)), dtype=int
    )
    null_w_plus = sign_assignments @ ranks
    valid = [
        value
        for value in sorted(set(int(x) for x in null_w_plus))
        if np.mean(null_w_plus <= value) <= alpha / 2.0 + 1e-15
    ]
    critical = max(valid) if valid else -1
    lower_index = critical
    upper_index = len(walsh) - critical - 1
    lower_tail = float(np.mean(null_w_plus <= critical))
    return {
        "hodges_lehmann": float(np.median(walsh)),
        "ci95_lower": float(walsh[lower_index]),
        "ci95_upper": float(walsh[upper_index]),
        "ci_coverage": float(1.0 - 2.0 * lower_tail),
    }


def critical_difference(n_methods: int, n_datasets: int, alpha: float) -> float:
    q_alpha = stats.studentized_range.isf(alpha, n_methods, np.inf) / math.sqrt(2.0)
    return float(q_alpha * math.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets)))


def analyze(data: pd.DataFrame, output_dir: Path, alpha: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    inferential_data = build_inferential_data(data)
    friedman_rows = []
    rank_rows = []
    wilcoxon_rows = []

    for metric in METRICS:
        metric_data = inferential_data.loc[inferential_data["metric"] == metric]
        matrix = (
            metric_data.groupby(["dataset", "method"], sort=False)["value"]
            .mean()
            .unstack("method")
            .loc[INFERENTIAL_BLOCKS, METHODS]
        )
        matrix.index.name = "Dataset"
        matrix.to_csv(output_dir / f"{METRIC_SLUG[metric]}_testing_matrix.csv")

        rank_values = np.vstack(
            [stats.rankdata(-row, method="average") for row in matrix.to_numpy(dtype=float)]
        )
        average_ranks = pd.Series(rank_values.mean(axis=0), index=METHODS).sort_values()
        for method, average_rank in average_ranks.items():
            rank_rows.append(
                {"metric": metric, "method": method, "average_rank": float(average_rank)}
            )

        friedman_stat, friedman_p = stats.friedmanchisquare(
            *[matrix[method].to_numpy(dtype=float) for method in METHODS]
        )
        friedman_rows.append(
            {
                "metric": metric,
                "n_datasets": len(matrix),
                "n_methods": len(METHODS),
                "friedman_chi_square": float(friedman_stat),
                "friedman_p": float(friedman_p),
                "critical_difference": critical_difference(
                    len(METHODS), len(matrix), alpha
                ),
            }
        )

        metric_pairwise = []
        for comparator in [method for method in METHODS if method != "GCRN"]:
            diff = (
                matrix["GCRN"].to_numpy(dtype=float)
                - matrix[comparator].to_numpy(dtype=float)
            )
            wilcoxon = exact_wilcoxon(diff)
            interval = hodges_lehmann_interval(diff, alpha)
            metric_pairwise.append(
                {
                    "metric": metric,
                    "control": "GCRN",
                    "comparator": comparator,
                    "n_datasets": len(diff),
                    **wilcoxon,
                    **interval,
                }
            )

        adjusted = holm_adjust(
            np.asarray([row["raw_p"] for row in metric_pairwise], dtype=float)
        )
        for row, adjusted_p in zip(metric_pairwise, adjusted):
            row["holm_adjusted_p"] = float(adjusted_p)
            wilcoxon_rows.append(row)

    pd.DataFrame(friedman_rows).to_csv(
        output_dir / "friedman_summary.csv", index=False
    )
    pd.DataFrame(rank_rows).to_csv(output_dir / "average_ranks.csv", index=False)
    pd.DataFrame(wilcoxon_rows).to_csv(
        output_dir / "wilcoxon_holm.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "seed_level_results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_and_validate(args.input)
    analyze(data, args.output, args.alpha)
    print(f"Reproducible results written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
