# Datasets

This directory contains the nine crowdsourcing truth-inference datasets used in the reported cross-dataset comparisons. Only the two source CSV files required for each dataset are included; generated graph caches, trained models, experimental outputs, and training code are not included.

CF and CF* share the same 300 tasks and gold labels but contain different worker pools and annotation sets. Both are retained here and in descriptive performance tables. In the corrected primary inferential analysis, they are not treated as independent blocks: their per-seed results are averaged with equal weight to form one `CF/CF* composite` block. See the repository-level README and generated testing matrices for the exact construction.

## Directory layout

```text
datasets/
  raw/<dataset>/label.csv
  raw/<dataset>/truth.csv
  DATASET_METADATA.csv
```

Each `label.csv` has three integer columns:

- `task`: task/item identifier;
- `worker`: annotator identifier;
- `answer`: class label supplied by that annotator.

Each `truth.csv` has two integer columns:

- `task`: task/item identifier;
- `truth`: reference class label.

Identifiers and class labels are stored exactly as obtained; no train/test split, normalization, imputation, sampling, or feature extraction has been applied. The public directory name `CF_star` represents the manuscript name `CF*`, because `*` is not a valid Windows directory character. `PosSent` corresponds to `Possent` in the statistical-results table.

## Integrity verification

Run `python run_statistical_tests.py` from the repository root. The script checks
the archived seed-level input hash and writes SHA-256 hashes for the complete
reproducibility package to `validation/sha256_manifest.csv`. Dataset dimensions
and name mappings are recorded in `datasets/DATASET_METADATA.csv`.
