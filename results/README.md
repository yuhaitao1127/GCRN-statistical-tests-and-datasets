# Result files

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
