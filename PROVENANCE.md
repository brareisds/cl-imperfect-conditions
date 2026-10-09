# Provenance

## Raw logs (`results/`)

The text logs under `results/` (`args-*.txt`, `stdout-*.txt`, `log_*_output.log`) are copies. They were produced on
Alliance Canada's Fir cluster by the author's working copy of `LifeLonger_default` (private working repository, commit
8c49e56), not by the code in this repository. The folders were renamed on copy (for example `results_label_noise`
became `results/label_noise`). Absolute paths inside the logs were edited to `/home/barbara/project/LifeLonger_default`;
no metric or other content was changed. Checkpoints, Fisher matrices and exemplar sets are not included; they remain on
the cluster.

- Phases 1 to 3 and the ablation: Slurm jobs 63847735 to 63847756 (22 jobs, all completed), run on 2026-10-09.
- Label-noise runs and clean baselines (`results/label_noise`, `results/official_savemodel`): earlier jobs.
- Runs made from this repository are written to `results/verify_*` and are the ones to compare against these copies.

## Analysis outputs (`analysis/`)

Regenerated in this repository from the logs above (CPU only): `per_seed.csv`, `summary.csv`, `paired_stats.csv`,
`recency_by_run.csv`, `recency_stats.csv`, `test_corruption_summary.csv`, `test_corruption_paired.csv` and
`dataset_audit.csv`. Each was compared with the version produced in the working repository and matched, after dropping
the rows that refer to a method not studied here. `paired_stats.csv` needs scipy for the exact t quantile.

New in this repository (no earlier version): `per_seed_phases.csv`, `summary_phases.csv`, `paired_phases.csv`.

Copied from the cluster, because they come from GPU jobs that need checkpoints or saved Fisher matrices (the scripts
that produce them are included): the confusion matrices (`analysis/*_confusion.csv`,
`analysis/label_noise/confusions/`), `phase0_base_runs.csv`, `phase0_base_norms.csv`, `test_corruption_*_ewc.csv`,
`test_corruption_calibration*.csv`, `fisher_by_noise.csv`, `fisher_stability.csv` and `fisher_confidence_*.csv`. Rows and
columns that refer to a method not studied here were removed from the copies.
