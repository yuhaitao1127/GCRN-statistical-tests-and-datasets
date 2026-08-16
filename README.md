# GCRN statistical tests and seed-level results

Version: **1.0.0**

This minimal repository contains the five-seed results and the code used for the cross-dataset statistical tests. Seeds 42, 43, 44, 45, and 46 are repeated runs. The script first averages these runs within each method--dataset--metric cell and then treats the nine datasets as the independent paired blocks.

## Files

- `data/seed_level_results.csv`: 1,350 seed-level observations (3 metrics x 9 datasets x 10 methods x 5 seeds).
- `datasets/`: the nine raw crowdsourcing datasets (`label.csv` and `truth.csv` only), dataset dimensions, format notes, and SHA-256 checksums.
- `run_statistical_tests.py`: reconstructs the three exact testing matrices and computes the Friedman tests, average ranks, Nemenyi critical difference, and GCRN-versus-baseline exact Wilcoxon tests with Holm correction, matched-pairs rank-biserial effects, and exact Hodges--Lehmann confidence intervals.
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

The command creates a local `results/` directory containing the exact testing matrices and statistical results. For archival citation, upload this directory as release **v1.0.0** and cite either that release tag or the immutable Git commit SHA assigned by GitHub.
