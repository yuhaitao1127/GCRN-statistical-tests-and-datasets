#!/usr/bin/env python3
"""Reproduce the corrected eight-block main-experiment statistics and CD figures."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats


VERSION = "1.0.0"
ALPHA = 0.05
SEEDS = [42, 43, 44, 45, 46]
RAW_DATASETS = ["Temp", "RTE", "Face", "LabelMe", "MS", "CF*", "Bird", "Possent", "CF"]
COMPOSITE_MEMBERS = ["CF", "CF*"]
COMPOSITE_NAME = "CF/CF* composite"
INFERENTIAL_BLOCKS = [
    "Temp",
    "RTE",
    "Face",
    "LabelMe",
    "MS",
    "Bird",
    "Possent",
    COMPOSITE_NAME,
]
APPENDIX_ROWS = RAW_DATASETS + [COMPOSITE_NAME]
METHODS = ["MV", "PM", "GLAD", "DS", "EBCC", "BWA", "TiReMGE", "GOVERN", "CrowdFM", "GCRN"]
METRICS = ["Accuracy", "Macro-F1", "Weighted-F1"]
METRIC_SLUG = {
    "Accuracy": "accuracy",
    "Macro-F1": "macro_f1",
    "Weighted-F1": "weighted_f1",
}
EXPECTED_INPUT_SHA256 = "d5d7948701bfbeb1258ccab49fe4a9d3164e0e65da24a4cd1710d22508f6e8a2"
EXPECTED_8BLOCK = {
    "Accuracy": (32.97715156130999, 0.0001347808404210867, 1.875),
    "Macro-F1": (20.46322971948445, 0.015259224930169115, 3.0),
    "Weighted-F1": (30.95754359363154, 0.0003010372854721093, 2.0),
}
EXPECTED_9BLOCK = {
    "Accuracy": (35.193482688391065, 5.507783259659904e-05, 1.8888888888888888),
    "Macro-F1": (19.867924528301906, 0.01874521501243312, 3.2222222222222223),
    "Weighted-F1": (31.69407008086255, 0.00022488231395728468, 2.111111111111111),
}

# Preserve the compact black-and-white CD style used in the submitted manuscript.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 11.5
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "dataset": set(RAW_DATASETS),
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


def derive_composite_seed_level(data: pd.DataFrame) -> pd.DataFrame:
    members = data.loc[data["dataset"].isin(COMPOSITE_MEMBERS)].copy()
    pivot = members.pivot(
        index=["metric", "method", "seed"], columns="dataset", values="value"
    ).reindex(columns=COMPOSITE_MEMBERS)
    if pivot.isna().any().any() or pivot.shape != (150, 2):
        raise ValueError("CF and CF* do not provide a complete paired five-seed composite.")
    composite = pivot.mean(axis=1).rename("value").reset_index()
    composite.insert(1, "dataset", COMPOSITE_NAME)
    composite = composite[["metric", "dataset", "method", "seed", "value"]]
    return composite.sort_values(["metric", "method", "seed"]).reset_index(drop=True)


def build_matrices(
    data: pd.DataFrame, composite_seed: pd.DataFrame, metric: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    descriptive = (
        data.loc[data["metric"].eq(metric)]
        .groupby(["dataset", "method"], sort=False)["value"]
        .mean()
        .unstack("method")
        .loc[RAW_DATASETS, METHODS]
    )
    descriptive.index.name = "Dataset"
    composite = (
        composite_seed.loc[composite_seed["metric"].eq(metric)]
        .groupby("method")["value"]
        .mean()
        .reindex(METHODS)
    )

    appendix = descriptive.copy()
    appendix.loc[COMPOSITE_NAME] = composite
    appendix = appendix.loc[APPENDIX_ROWS]
    appendix.index.name = "Dataset_or_derived_block"

    testing = descriptive.drop(index=COMPOSITE_MEMBERS).copy()
    testing.loc[COMPOSITE_NAME] = composite
    testing = testing.loc[INFERENTIAL_BLOCKS]
    testing.index.name = "Independent_block"
    return descriptive, appendix, testing


def rank_matrix(values: pd.DataFrame) -> pd.DataFrame:
    ranks = np.vstack(
        [stats.rankdata(-row, method="average") for row in values.to_numpy(dtype=float)]
    )
    return pd.DataFrame(ranks, index=values.index, columns=values.columns)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
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


def exact_wilcoxon(diff: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    if np.any(np.isclose(diff, 0.0)):
        raise ValueError("Exact Wilcoxon calculation requires non-zero paired differences.")
    ranks = stats.rankdata(np.abs(diff), method="average")
    if not np.array_equal(np.sort(ranks), np.arange(1.0, diff.size + 1.0)):
        raise ValueError("Tied absolute differences are not supported by this exact calculation.")

    w_plus = float(ranks[diff > 0.0].sum())
    w_minus = float(ranks[diff < 0.0].sum())
    total_rank = float(ranks.sum())
    assignments = np.asarray(
        list(itertools.product((0.0, 1.0), repeat=diff.size)), dtype=float
    )
    null_w_plus = assignments @ ranks
    observed_deviation = abs(w_plus - total_rank / 2.0)
    raw_p = float(
        np.mean(
            np.abs(null_w_plus - total_rank / 2.0)
            >= observed_deviation - 1e-12
        )
    )
    return {
        "w_plus": w_plus,
        "w_minus": w_minus,
        "wilcoxon_w": min(w_plus, w_minus),
        "raw_p": raw_p,
        "rank_biserial": float((w_plus - w_minus) / total_rank),
        "enumerations": int(assignments.shape[0]),
    }


def hodges_lehmann_interval(diff: np.ndarray, alpha: float) -> Dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    n = diff.size
    walsh = np.sort(
        np.asarray(
            [(diff[i] + diff[j]) / 2.0 for i in range(n) for j in range(i, n)],
            dtype=float,
        )
    )
    ranks = np.arange(1, n + 1, dtype=int)
    assignments = np.asarray(list(itertools.product((0, 1), repeat=n)), dtype=int)
    null_w_plus = assignments @ ranks
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
        "hl_critical_rank": int(critical),
    }


def critical_difference(n_methods: int, n_blocks: int, alpha: float) -> float:
    q_alpha = stats.studentized_range.isf(alpha, n_methods, np.inf) / math.sqrt(2.0)
    return float(q_alpha * math.sqrt(n_methods * (n_methods + 1) / (6.0 * n_blocks)))


def analyze_matrix(
    values: pd.DataFrame, metric: str, analysis: str, alpha: float
) -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranks = rank_matrix(values)
    average_ranks = ranks.mean(axis=0).sort_values()
    friedman_stat, friedman_p = stats.friedmanchisquare(
        *[values[method].to_numpy(dtype=float) for method in METHODS]
    )
    cd = critical_difference(len(METHODS), len(values), alpha)
    summary = {
        "analysis": analysis,
        "metric": metric,
        "n_raw_datasets": len(RAW_DATASETS),
        "n_independent_blocks": int(len(values)),
        "n_methods": len(METHODS),
        "friedman_chi_square": float(friedman_stat),
        "friedman_df": len(METHODS) - 1,
        "friedman_p": float(friedman_p),
        "alpha": alpha,
        "critical_difference": cd,
        "best_method": str(average_ranks.index[0]),
        "best_average_rank": float(average_ranks.iloc[0]),
        "block_definition": (
            "CF and CF* equally weighted as one composite block"
            if analysis == "primary_8block"
            else "CF and CF* treated separately (legacy sensitivity analysis only)"
        ),
    }
    rank_rows = pd.DataFrame(
        {
            "analysis": analysis,
            "metric": metric,
            "method": average_ranks.index,
            "average_rank": average_ranks.to_numpy(dtype=float),
            "mean_across_blocks": values[average_ranks.index].mean(axis=0).to_numpy(dtype=float),
            "sd_across_blocks": values[average_ranks.index].std(axis=0, ddof=1).to_numpy(dtype=float),
        }
    )

    pairwise_rows = []
    for comparator in [method for method in METHODS if method != "GCRN"]:
        diff = (values["GCRN"] - values[comparator]).to_numpy(dtype=float)
        wilcoxon = exact_wilcoxon(diff)
        interval = hodges_lehmann_interval(diff, alpha)
        pairwise_rows.append(
            {
                "analysis": analysis,
                "metric": metric,
                "control": "GCRN",
                "comparator": comparator,
                "n_blocks": len(diff),
                "mean_control": float(values["GCRN"].mean()),
                "mean_comparator": float(values[comparator].mean()),
                "mean_difference": float(diff.mean()),
                "median_difference": float(np.median(diff)),
                "wins": int(np.sum(diff > 1e-12)),
                "ties": int(np.sum(np.abs(diff) <= 1e-12)),
                "losses": int(np.sum(diff < -1e-12)),
                **wilcoxon,
                **interval,
            }
        )
    adjusted = holm_adjust([row["raw_p"] for row in pairwise_rows])
    for row, adjusted_p in zip(pairwise_rows, adjusted):
        row["holm_adjusted_p"] = float(adjusted_p)
        row["reject_holm"] = bool(adjusted_p < alpha)
        row["holm_family_size"] = len(pairwise_rows)
        row["correction_scope"] = "within metric, GCRN versus nine baselines"
        row["ci_note"] = "pointwise exact conservative paired Hodges-Lehmann CI"
    pairwise = pd.DataFrame(pairwise_rows)

    n_methods = len(METHODS)
    n_blocks = len(values)
    se = math.sqrt(n_methods * (n_methods + 1) / (6.0 * n_blocks))
    nemenyi_rows = []
    for method_a, method_b in itertools.combinations(average_ranks.index, 2):
        rank_diff = abs(float(average_ranks[method_a] - average_ranks[method_b]))
        q_stat = rank_diff / se
        p_value = float(
            stats.studentized_range.sf(q_stat * math.sqrt(2.0), n_methods, np.inf)
        )
        nemenyi_rows.append(
            {
                "analysis": analysis,
                "metric": metric,
                "method_a": method_a,
                "method_b": method_b,
                "rank_a": float(average_ranks[method_a]),
                "rank_b": float(average_ranks[method_b]),
                "absolute_rank_difference": rank_diff,
                "nemenyi_q": float(q_stat),
                "nemenyi_p": p_value,
                "critical_difference": cd,
                "reject_nemenyi": bool(rank_diff > cd),
            }
        )
    nemenyi = pd.DataFrame(nemenyi_rows).sort_values(
        ["nemenyi_p", "absolute_rank_difference"], ascending=[True, False]
    )
    return summary, rank_rows, pairwise, nemenyi, ranks


def maximal_cd_groups(
    sorted_ranks: pd.Series, cd: float
) -> List[Tuple[float, float, int]]:
    ranks = sorted_ranks.to_numpy(dtype=float)
    groups: List[Tuple[float, float, int]] = []
    for start in range(len(ranks)):
        best_end = None
        for end in range(start + 1, len(ranks)):
            if ranks[end] - ranks[start] <= cd + 1e-12:
                best_end = end
            else:
                break
        if best_end is not None:
            groups.append((ranks[start], ranks[best_end], best_end - start + 1))
    filtered = []
    for group in groups:
        start, end, length = group
        contained = any(
            other != group
            and other[2] > length
            and other[0] <= start + 1e-12
            and other[1] >= end - 1e-12
            for other in groups
        )
        if not contained:
            filtered.append(group)
    filtered.sort(key=lambda item: (-item[2], item[0], item[1]))
    return filtered[:6]


def draw_cd_panel(
    ax: plt.Axes,
    average_ranks: pd.Series,
    cd: float,
    metric: str,
    panel_label: str | None = None,
) -> None:
    average_ranks = average_ranks.sort_values(kind="stable")
    methods = list(average_ranks.index)
    k = len(methods)
    left_count = int(math.ceil(k / 2.0))
    groups = maximal_cd_groups(average_ranks, cd)

    # The submitted main-comparison diagrams use this fixed geometry.
    ax.set_xlim(-0.56, 11.56)
    ax.set_ylim(-0.01, 1.02)
    ax.axis("off")
    axis_y = 0.64
    ax.hlines(axis_y, 1, k, color="black", linewidth=1.0)
    for rank in range(1, k + 1):
        ax.vlines(rank, axis_y - 0.028, axis_y + 0.028, color="black", linewidth=1.0)
        ax.text(rank, axis_y + 0.032, str(rank), ha="center", va="bottom", fontsize=11.5)

    cd_y = 0.924
    ax.hlines(cd_y, 1.0, 1.0 + cd, color="black", linewidth=1.4)
    ax.vlines([1.0, 1.0 + cd], cd_y - 0.03, cd_y + 0.03, color="black", linewidth=1.2)
    ax.text(
        1.0 + cd / 2.0,
        cd_y + 0.025,
        f"CD = {cd:.3f}",
        ha="center",
        va="bottom",
        fontsize=11.5,
    )

    for index, (start_rank, end_rank, _) in enumerate(groups):
        y = 0.8025 - index * 0.0495
        ax.hlines(y, start_rank, end_rank, color="black", linewidth=2.0)
        ax.vlines([start_rank, end_rank], y - 0.016, y + 0.016, color="black", linewidth=1.1)

    label_y = [0.476, 0.381, 0.286, 0.191, 0.096]
    left_line_end = 0.58
    right_line_end = 10.42
    for method, y in zip(methods[:left_count], label_y):
        rank = float(average_ranks[method])
        weight = "bold" if method == "GCRN" else "normal"
        linewidth = 1.25 if method == "GCRN" else 0.9
        ax.hlines(y, left_line_end, rank, color="black", linewidth=linewidth)
        ax.vlines(rank, y, axis_y, color="black", linewidth=linewidth)
        ax.text(
            left_line_end - 0.04,
            y,
            f"{method} ({rank:.2f})",
            ha="right",
            va="center",
            fontsize=11.5,
            color="black",
            fontweight=weight,
        )

    for method, y in zip(methods[left_count:], label_y):
        rank = float(average_ranks[method])
        weight = "bold" if method == "GCRN" else "normal"
        linewidth = 1.25 if method == "GCRN" else 0.9
        ax.hlines(y, rank, right_line_end, color="black", linewidth=linewidth)
        ax.vlines(rank, y, axis_y, color="black", linewidth=linewidth)
        ax.text(
            right_line_end + 0.04,
            y,
            f"{method} ({rank:.2f})",
            ha="left",
            va="center",
            fontsize=11.5,
            color="black",
            fontweight=weight,
        )

    if panel_label:
        ax.text(
            -0.45,
            0.99,
            f"({panel_label}) {metric}",
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
        )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"))
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=600)
    fig.savefig(
        output_base.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def make_figures(
    rank_tables: Dict[str, pd.Series], cd: float, figures_dir: Path
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_names = {
        "Accuracy": "CD_Accuracy_8blocks",
        "Macro-F1": "CD_Macro_F1_8blocks",
        "Weighted-F1": "CD_Weighted_F1_8blocks",
    }
    for metric in METRICS:
        fig = plt.figure(figsize=(633.1165 / 72.0, 188.1141 / 72.0))
        ax = fig.add_axes([0.045, 0.0, 0.91, 1.0])
        draw_cd_panel(ax, rank_tables[metric], cd, metric)
        save_figure(fig, figures_dir / figure_names[metric])

    fig = plt.figure(figsize=(633.1165 / 72.0, 3 * 188.1141 / 72.0))
    for index, (metric, label) in enumerate(zip(METRICS, ["a", "b", "c"])):
        ax = fig.add_axes([0.045, (2 - index) / 3.0, 0.91, 1.0 / 3.0])
        draw_cd_panel(ax, rank_tables[metric], cd, metric, panel_label=label)
    save_figure(fig, figures_dir / "CD_All_Metrics_8blocks")

    readme = """# Critical Difference figures

These figures use the corrected primary analysis with eight statistically independent
blocks. CF and CF* retain separate descriptive results but enter inference only through
their equally weighted `CF/CF* composite` block. Five seeds are averaged within each
method and raw dataset before ranking.

- Methods: 10
- Independent blocks: 8
- Nemenyi alpha: 0.05
- Critical difference: 4.789263851159032
- Primary editable format: SVG
- Additional formats: PDF, 600 dpi TIFF, and 600 dpi PNG

Horizontal bars connect maximal groups whose average-rank separation does not exceed
the critical difference. These diagrams visualize omnibus ranks. Targeted claims about
GCRN versus a baseline must use `results/wilcoxon_holm_8blocks.csv`.
"""
    (figures_dir / "README.md").write_text(readme, encoding="utf-8")


def build_sensitivity_comparison(
    primary_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    primary_ranks: pd.DataFrame,
    sensitivity_ranks: pd.DataFrame,
    primary_wilcoxon: pd.DataFrame,
    sensitivity_wilcoxon: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = primary_summary.merge(
        sensitivity_summary,
        on="metric",
        suffixes=("_8block", "_9block"),
        validate="one_to_one",
    )
    summary_out = pd.DataFrame(
        {
            "metric": summary["metric"],
            "friedman_chi_square_8block": summary["friedman_chi_square_8block"],
            "friedman_p_8block": summary["friedman_p_8block"],
            "critical_difference_8block": summary["critical_difference_8block"],
            "gcrn_average_rank_8block": summary["best_average_rank_8block"],
            "friedman_chi_square_9block": summary["friedman_chi_square_9block"],
            "friedman_p_9block": summary["friedman_p_9block"],
            "critical_difference_9block": summary["critical_difference_9block"],
            "gcrn_average_rank_9block": summary["best_average_rank_9block"],
        }
    )
    summary_out["friedman_chi_square_delta_8_minus_9"] = (
        summary_out["friedman_chi_square_8block"] - summary_out["friedman_chi_square_9block"]
    )
    summary_out["gcrn_rank_delta_8_minus_9"] = (
        summary_out["gcrn_average_rank_8block"] - summary_out["gcrn_average_rank_9block"]
    )

    ranks = primary_ranks[["metric", "method", "average_rank"]].merge(
        sensitivity_ranks[["metric", "method", "average_rank"]],
        on=["metric", "method"],
        suffixes=("_8block", "_9block"),
        validate="one_to_one",
    )
    ranks["rank_delta_8_minus_9"] = ranks["average_rank_8block"] - ranks["average_rank_9block"]

    wilcoxon = primary_wilcoxon[
        ["metric", "comparator", "raw_p", "holm_adjusted_p", "reject_holm"]
    ].merge(
        sensitivity_wilcoxon[
            ["metric", "comparator", "raw_p", "holm_adjusted_p", "reject_holm"]
        ],
        on=["metric", "comparator"],
        suffixes=("_8block", "_9block"),
        validate="one_to_one",
    )
    return summary_out, ranks, wilcoxon


def add_check(checks: List[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def validate_outputs(
    input_path: Path,
    data: pd.DataFrame,
    composite_seed: pd.DataFrame,
    primary_summary: pd.DataFrame,
    primary_ranks: pd.DataFrame,
    primary_wilcoxon: pd.DataFrame,
    primary_nemenyi: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    sensitivity_wilcoxon: pd.DataFrame,
    figures_dir: Path,
    validation_dir: Path,
) -> pd.DataFrame:
    checks: List[dict] = []
    actual_hash = sha256_file(input_path)
    add_check(
        checks,
        "Original seed-level input unchanged",
        actual_hash == EXPECTED_INPUT_SHA256,
        f"SHA256={actual_hash}",
    )
    counts = data.groupby(["metric", "dataset", "method"])["seed"].nunique()
    add_check(
        checks,
        "Raw input dimensions",
        len(data) == 1350 and len(counts) == 270 and bool(counts.eq(5).all()),
        f"rows={len(data)}; complete cells={len(counts)}",
    )
    add_check(
        checks,
        "Derived composite dimensions",
        len(composite_seed) == 150
        and composite_seed.groupby(["metric", "method"])["seed"].nunique().eq(5).all(),
        f"rows={len(composite_seed)}; expected=150",
    )
    source = data.loc[data["dataset"].isin(COMPOSITE_MEMBERS)].pivot(
        index=["metric", "method", "seed"], columns="dataset", values="value"
    )
    expected_composite = source[COMPOSITE_MEMBERS].mean(axis=1).sort_index()
    observed_composite = composite_seed.set_index(["metric", "method", "seed"])["value"].sort_index()
    composite_error = float(np.max(np.abs(expected_composite - observed_composite)))
    add_check(
        checks,
        "Composite is per-seed equal-weight mean",
        composite_error < 1e-12,
        f"maximum absolute error={composite_error:.3g}",
    )

    summary_lookup = primary_summary.set_index("metric")
    rank_lookup = primary_ranks.loc[primary_ranks["method"].eq("GCRN")].set_index("metric")
    max_summary_error = 0.0
    for metric, (expected_chi, expected_p, expected_rank) in EXPECTED_8BLOCK.items():
        max_summary_error = max(
            max_summary_error,
            abs(float(summary_lookup.loc[metric, "friedman_chi_square"]) - expected_chi),
            abs(float(summary_lookup.loc[metric, "friedman_p"]) - expected_p),
            abs(float(rank_lookup.loc[metric, "average_rank"]) - expected_rank),
        )
    add_check(
        checks,
        "Eight-block reference statistics",
        max_summary_error < 1e-12,
        f"maximum absolute error={max_summary_error:.3g}",
    )
    expected_cd = 4.789263851159032
    cd_error = float(np.max(np.abs(primary_summary["critical_difference"] - expected_cd)))
    add_check(
        checks,
        "Eight-block critical difference",
        cd_error < 1e-12,
        f"CD={primary_summary['critical_difference'].iloc[0]:.15g}; error={cd_error:.3g}",
    )
    add_check(
        checks,
        "Eight-block exact Wilcoxon family",
        len(primary_wilcoxon) == 27
        and primary_wilcoxon["n_blocks"].eq(8).all()
        and primary_wilcoxon["enumerations"].eq(256).all()
        and int(primary_wilcoxon["reject_holm"].sum()) == 0,
        (
            f"rows={len(primary_wilcoxon)}; enumerations="
            f"{sorted(primary_wilcoxon['enumerations'].unique())}; "
            f"Holm rejections={int(primary_wilcoxon['reject_holm'].sum())}"
        ),
    )
    add_check(
        checks,
        "Eight-block Nemenyi family",
        len(primary_nemenyi) == 135,
        f"rows={len(primary_nemenyi)}; expected=135",
    )

    sensitivity_lookup = sensitivity_summary.set_index("metric")
    sensitivity_rank_error = 0.0
    for metric, (expected_chi, expected_p, expected_rank) in EXPECTED_9BLOCK.items():
        row = sensitivity_lookup.loc[metric]
        # The GCRN rank is stored in the summary because it remains the best method.
        sensitivity_rank = float(row["best_average_rank"])
        sensitivity_rank_error = max(
            sensitivity_rank_error,
            abs(float(row["friedman_chi_square"]) - expected_chi),
            abs(float(row["friedman_p"]) - expected_p),
            abs(sensitivity_rank - expected_rank),
        )
    add_check(
        checks,
        "Nine-block sensitivity reproduces archived analysis",
        sensitivity_rank_error < 1e-12
        and int(sensitivity_wilcoxon["reject_holm"].sum()) == 4,
        (
            f"maximum absolute error={sensitivity_rank_error:.3g}; "
            f"Holm rejections={int(sensitivity_wilcoxon['reject_holm'].sum())}"
        ),
    )

    expected_bases = [
        "CD_Accuracy_8blocks",
        "CD_Macro_F1_8blocks",
        "CD_Weighted_F1_8blocks",
        "CD_All_Metrics_8blocks",
    ]
    expected_figure_files = [
        figures_dir / f"{base}.{suffix}"
        for base in expected_bases
        for suffix in ("svg", "pdf", "png", "tiff")
    ]
    figures_exist = all(path.exists() and path.stat().st_size > 0 for path in expected_figure_files)
    add_check(
        checks,
        "Figure export bundle",
        figures_exist,
        f"present={sum(path.exists() for path in expected_figure_files)}/16",
    )
    raster_ok = True
    raster_details = []
    for base in expected_bases:
        png_path = figures_dir / f"{base}.png"
        tiff_path = figures_dir / f"{base}.tiff"
        with Image.open(png_path) as image:
            grayscale = np.asarray(image.convert("L"))
            raster_ok &= image.width > 1000 and image.height > 500 and float(grayscale.std()) > 5.0
            raster_details.append(f"{base}:PNG={image.width}x{image.height},sd={grayscale.std():.2f}")
        with Image.open(tiff_path) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
            raster_ok &= min(dpi_x, dpi_y) >= 599.0
            raster_details.append(f"TIFFdpi={dpi_x:.1f}x{dpi_y:.1f}")
    add_check(checks, "Raster figure quality", bool(raster_ok), "; ".join(raster_details))
    svg_text_ok = all(
        "<text" in (figures_dir / f"{base}.svg").read_text(encoding="utf-8")
        for base in expected_bases
    )
    add_check(
        checks,
        "SVG text remains editable",
        svg_text_ok,
        "all four SVG files contain text elements",
    )

    validation = pd.DataFrame(checks)
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(validation_dir / "validation_checks.csv", index=False)
    passed = int(validation["status"].eq("PASS").sum())
    failed = int(validation["status"].eq("FAIL").sum())
    lines = [
        "Main experiment eight-block validation",
        f"Version: {VERSION}",
        f"PASS: {passed}",
        f"FAIL: {failed}",
        "",
    ]
    lines.extend(
        f"[{row.status}] {row.check}: {row.detail}" for row in validation.itertuples()
    )
    (validation_dir / "validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise ValueError(f"Validation failed: {failed} check(s).")
    return validation


def write_manifest(package_root: Path, validation_dir: Path) -> None:
    manifest_path = validation_dir / "sha256_manifest.csv"
    rows = []
    for path in sorted(package_root.rglob("*")):
        if (
            path.is_file()
            and path != manifest_path
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ):
            rows.append(
                {
                    "relative_path": path.relative_to(package_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


def write_results_readme(results_dir: Path) -> None:
    text = """# Result files

Primary corrected inference uses eight independent blocks. The raw CF and CF* rows
are retained for description, but only their per-seed equal-weight composite enters
Friedman, average-rank, Nemenyi/CD, and Wilcoxon-Holm calculations.

- `*_descriptive_matrix_9datasets.csv`: original nine dataset means, unchanged.
- `*_appendix_matrix_9datasets_plus_composite.csv`: the nine descriptive rows plus the derived composite row.
- `*_testing_matrix_8blocks.csv`: exact primary inferential inputs.
- `*_rank_matrix_8blocks.csv`: within-block ranks used for Friedman and CD analysis.
- `derived_cf_cfstar_composite_seed_level.csv`: 150 derived values (3 metrics x 10 methods x 5 seeds).
- `friedman_summary_8blocks.csv`: corrected omnibus tests and CD.
- `average_ranks_8blocks.csv`: corrected average ranks.
- `wilcoxon_holm_8blocks.csv`: 27 targeted GCRN-versus-baseline exact tests.
- `nemenyi_pairwise_8blocks.csv`: all 45 rank comparisons per metric.
- `sensitivity_9block/`: legacy nine-block calculation retained only as a sensitivity comparison.
- `sensitivity_comparison_8block_vs_9block.csv`: side-by-side omnibus results.

The pointwise Hodges-Lehmann confidence intervals are not multiplicity-adjusted;
inferential decisions use the Holm-adjusted p-values.
"""
    (results_dir / "README.md").write_text(text, encoding="utf-8")


def run(input_path: Path, package_root: Path, alpha: float) -> None:
    results_dir = package_root / "results"
    sensitivity_dir = results_dir / "sensitivity_9block"
    figures_dir = package_root / "figures"
    validation_dir = package_root / "validation"
    for directory in (results_dir, sensitivity_dir, figures_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data = load_and_validate(input_path)
    composite_seed = derive_composite_seed_level(data)
    composite_seed.to_csv(results_dir / "derived_cf_cfstar_composite_seed_level.csv", index=False)

    primary_summaries = []
    primary_rank_rows = []
    primary_wilcoxon_rows = []
    primary_nemenyi_rows = []
    sensitivity_summaries = []
    sensitivity_rank_rows = []
    sensitivity_wilcoxon_rows = []
    sensitivity_nemenyi_rows = []
    figure_rank_tables: Dict[str, pd.Series] = {}

    for metric in METRICS:
        slug = METRIC_SLUG[metric]
        descriptive, appendix, testing = build_matrices(data, composite_seed, metric)
        descriptive.to_csv(results_dir / f"{slug}_descriptive_matrix_9datasets.csv")
        appendix.to_csv(results_dir / f"{slug}_appendix_matrix_9datasets_plus_composite.csv")
        testing.to_csv(results_dir / f"{slug}_testing_matrix_8blocks.csv")

        summary, ranks, wilcoxon, nemenyi, rank_values = analyze_matrix(
            testing, metric, "primary_8block", alpha
        )
        rank_values.to_csv(results_dir / f"{slug}_rank_matrix_8blocks.csv")
        primary_summaries.append(summary)
        primary_rank_rows.append(ranks)
        primary_wilcoxon_rows.append(wilcoxon)
        primary_nemenyi_rows.append(nemenyi)
        figure_rank_tables[metric] = ranks.set_index("method")["average_rank"].sort_values()

        sensitivity_summary, sensitivity_ranks, sensitivity_wilcoxon, sensitivity_nemenyi, sensitivity_rank_matrix = analyze_matrix(
            descriptive, metric, "sensitivity_9block", alpha
        )
        descriptive.to_csv(sensitivity_dir / f"{slug}_testing_matrix_9blocks.csv")
        sensitivity_rank_matrix.to_csv(sensitivity_dir / f"{slug}_rank_matrix_9blocks.csv")
        sensitivity_summaries.append(sensitivity_summary)
        sensitivity_rank_rows.append(sensitivity_ranks)
        sensitivity_wilcoxon_rows.append(sensitivity_wilcoxon)
        sensitivity_nemenyi_rows.append(sensitivity_nemenyi)

    primary_summary = pd.DataFrame(primary_summaries)
    primary_ranks = pd.concat(primary_rank_rows, ignore_index=True)
    primary_wilcoxon = pd.concat(primary_wilcoxon_rows, ignore_index=True)
    primary_nemenyi = pd.concat(primary_nemenyi_rows, ignore_index=True)
    sensitivity_summary = pd.DataFrame(sensitivity_summaries)
    sensitivity_ranks = pd.concat(sensitivity_rank_rows, ignore_index=True)
    sensitivity_wilcoxon = pd.concat(sensitivity_wilcoxon_rows, ignore_index=True)
    sensitivity_nemenyi = pd.concat(sensitivity_nemenyi_rows, ignore_index=True)

    primary_summary.to_csv(results_dir / "friedman_summary_8blocks.csv", index=False)
    primary_ranks.to_csv(results_dir / "average_ranks_8blocks.csv", index=False)
    primary_wilcoxon.to_csv(results_dir / "wilcoxon_holm_8blocks.csv", index=False)
    primary_nemenyi.to_csv(results_dir / "nemenyi_pairwise_8blocks.csv", index=False)
    sensitivity_summary.to_csv(sensitivity_dir / "friedman_summary_9blocks.csv", index=False)
    sensitivity_ranks.to_csv(sensitivity_dir / "average_ranks_9blocks.csv", index=False)
    sensitivity_wilcoxon.to_csv(sensitivity_dir / "wilcoxon_holm_9blocks.csv", index=False)
    sensitivity_nemenyi.to_csv(sensitivity_dir / "nemenyi_pairwise_9blocks.csv", index=False)

    comparison, rank_changes, wilcoxon_changes = build_sensitivity_comparison(
        primary_summary,
        sensitivity_summary,
        primary_ranks,
        sensitivity_ranks,
        primary_wilcoxon,
        sensitivity_wilcoxon,
    )
    comparison.to_csv(results_dir / "sensitivity_comparison_8block_vs_9block.csv", index=False)
    rank_changes.to_csv(results_dir / "sensitivity_average_rank_changes.csv", index=False)
    wilcoxon_changes.to_csv(results_dir / "sensitivity_wilcoxon_changes.csv", index=False)
    write_results_readme(results_dir)

    cd = float(primary_summary["critical_difference"].iloc[0])
    if not np.allclose(primary_summary["critical_difference"], cd, rtol=0.0, atol=1e-15):
        raise ValueError("Critical difference unexpectedly differs among metrics.")
    make_figures(figure_rank_tables, cd, figures_dir)

    validate_outputs(
        input_path,
        data,
        composite_seed,
        primary_summary,
        primary_ranks,
        primary_wilcoxon,
        primary_nemenyi,
        sensitivity_summary,
        sensitivity_wilcoxon,
        figures_dir,
        validation_dir,
    )
    write_manifest(package_root, validation_dir)
    print(f"Corrected eight-block package generated at: {package_root.resolve()}")
    for row in primary_summary.itertuples(index=False):
        print(
            f"{row.metric}: Friedman chi2={row.friedman_chi_square:.6f}, "
            f"p={row.friedman_p:.9g}, CD={row.critical_difference:.6f}, "
            f"best rank={row.best_method} ({row.best_average_rank:.4f})"
        )
    print(
        "Holm-adjusted Wilcoxon rejections: "
        f"{int(primary_wilcoxon['reject_holm'].sum())}/27"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "data" / "seed_level_results.csv",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=script_dir,
        help="Directory that receives results, figures, and validation outputs.",
    )
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input.resolve(), args.package_root.resolve(), args.alpha)


if __name__ == "__main__":
    main()
