# GCRN main-experiment statistical tests: corrected eight-block analysis

Version: **1.0.0**

This package resolves the non-independence of CF and CF*. The two datasets share the
same 300 tasks and gold labels but use different worker pools and annotation sets.
Their original results are therefore retained as separate descriptive rows, while
inferential tests use one equally weighted composite block.

For metric `q`, method `m`, and seed `s`, the derived value is

```text
CF/CF* composite(q,m,s) = [CF(q,m,s) + CF*(q,m,s)] / 2.
```

The five derived seed values are then averaged within each method and metric. The
primary Friedman, average-rank, Nemenyi/CD, and exact paired Wilcoxon-Holm analyses
use eight independent blocks:

```text
Temp, RTE, Face, LabelMe, MS, Bird, Possent, CF/CF* composite
```

The source file remains unchanged at 1,350 observations
(`3 metrics x 9 descriptive datasets x 10 methods x 5 seeds`).

## Primary results

| Metric | Friedman chi-square (df=9) | p-value | GCRN average rank |
|---|---:|---:|---:|
| Accuracy | 32.977152 | 0.000134781 | 1.8750 |
| Macro-F1 | 20.463230 | 0.015259225 | 3.0000 |
| Weighted-F1 | 30.957544 | 0.000301037 | 2.0000 |

The Nemenyi critical difference is **4.789264** at alpha=0.05. All 27 targeted
GCRN-versus-baseline exact paired Wilcoxon tests are non-significant after Holm
correction; the smallest adjusted p-value is 0.0703125. Omnibus/rank evidence and
targeted pairwise evidence should therefore be described separately.

## Package layout

- `data/seed_level_results.csv`: unchanged five-seed source data.
- `datasets/`: unchanged raw datasets and metadata from the original archive at commit `3b06e8907860`.
- `run_statistical_tests.py`: complete analysis, figure generation, and validation.
- `results/`: descriptive, appendix, primary 8-block, and legacy 9-block sensitivity outputs.
- `figures/`: three individual CD diagrams and one combined three-metric figure.
- `validation/`: numerical checks, raster/export checks, and SHA-256 manifest.

The appendix matrices contain all nine original descriptive rows plus the derived
composite row. Only the eight rows in `*_testing_matrix_8blocks.csv` enter primary
inference. The 9-block outputs are retained solely as a sensitivity comparison and
must not be presented as the corrected primary analysis.

## Exact environment

- Python 3.9.18
- NumPy 1.25.0
- pandas 1.5.3
- SciPy 1.10.1
- Matplotlib 3.9.4
- Pillow 11.2.1

## Reproduce

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_statistical_tests.py
```

The script fails if the copied seed-level input differs from its archived SHA-256,
if any expected statistical value changes, if the composite is not the exact per-seed
arithmetic mean, or if required figure exports are missing or invalid.
