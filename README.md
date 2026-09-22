# GCRN statistical tests and seed-level results

Version: **1.0.0**

This minimal repository contains the five-seed results and the code used for the cross-dataset statistical tests. Seeds 42, 43, 44, 45, and 46 are repeated runs. CF and CF* share the same tasks and gold labels. For each metric, method, and seed, the script first averages their two values with equal weight to form one CF/CF* composite. It then averages the five seeds within each method--block--metric cell and uses eight independent paired blocks: Temp, RTE, Face, LabelMe, MS, Bird, Possent, and CF/CF* composite. The original nine datasets and all 1,350 seed-level observations are retained unchanged.

## Files

- `data/seed_level_results.csv`: 1,350 seed-level observations (3 metrics x 9 datasets x 10 methods x 5 seeds).
- `datasets/`: the nine raw crowdsourcing datasets (`label.csv` and `truth.csv` only), dataset dimensions and format notes.
- `run_statistical_tests.py`: reconstructs the three exact eight-block testing matrices and computes the Friedman tests, average ranks, Nemenyi critical difference, and GCRN-versus-baseline exact Wilcoxon tests with Holm correction, matched-pairs rank-biserial effects, and exact Hodges--Lehmann confidence intervals.
- `requirements.txt`, `.python-version`, and `VERSION`: fixed runtime, dependency, and release versions.

## Exact environment

- Python 3.9.18
- NumPy 1.25.0
- pandas 1.5.3
- SciPy 1.10.1

## Run

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_statistical_tests.py
```

The command creates a local `results/` directory containing the exact testing matrices and statistical results. The six output CSV filenames and columns are unchanged; the existing `n_datasets` column now reports eight inferential blocks. Generated results and figures are not included in this repository. The corrected analysis has a Nemenyi critical difference of 4.789264 at alpha=0.05; all 27 targeted GCRN-versus-baseline Wilcoxon comparisons are non-significant after Holm correction. For archival citation, upload this directory as release **v1.0.0** and cite either that release tag or the immutable Git commit SHA assigned by GitHub.
