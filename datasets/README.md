# Datasets

This directory contains the nine crowdsourcing truth-inference datasets used in the reported cross-dataset comparisons. Only the two source CSV files required for each dataset are included; generated graph caches, trained models, experimental outputs, and training code are not included.

## Directory layout

```text
datasets/
  raw/<dataset>/label.csv
  raw/<dataset>/truth.csv
  DATASET_METADATA.csv
  SHA256SUMS
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

Run the following command from the repository root in PowerShell:

```powershell
Get-ChildItem datasets/raw -File -Recurse |
  Sort-Object FullName |
  ForEach-Object { (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
```

Compare the results with `datasets/SHA256SUMS`. Dataset dimensions and name mappings are recorded in `datasets/DATASET_METADATA.csv`.
