# Continual Learning under Imperfect Conditions

Code for a sensitivity study of two class-incremental learning methods, **EWC** and **iCaRL**, on medical image
classification (OrganAMNIST and BloodMNIST, MedMNIST). It was developed during a Mitacs Globalink research internship
at Ontario Tech University.

The study asks how these methods behave when training and test conditions are not ideal:

- **Label noise during training**, within the current task (intra-task) and across tasks (cross-task).
- **Test-time image corruption** (Gaussian noise, blur, contrast, brightness), with severity levels calibrated on the data.
- **Recency bias**: how much of the forgetting is explained by old-task samples being predicted as last-task classes,
  with and without Weight Aligning (WA) applied to the final layer.
- **Why EWC degrades under noise**: how label noise changes the Fisher information, tested by rescaling each task's
  Fisher to a fixed mean (a control, not a fix).
- **iCaRL classifier**: logits versus its native nearest-mean-of-exemplars (NME) classifier, and a class-weight ablation
  of the loss. EWC (run with an exemplar memory of 200 samples) is evaluated with both classifiers as well, so the two
  methods can be compared under the same classifier.
- **Latest-task prediction bias (LTB)**: the mean over earlier tasks of the share of their test samples predicted as
  classes of the last task (Łapacz et al., arXiv:2411.04723), with confusion matrices after the last task.
- **Additional datasets**: PathMNIST and TissueMNIST, clean stream only.

## Origin of the code

This repository is derived from [LifeLonger](https://github.com/mmderakhshani/LifeLonger) ("LifeLonger: A Benchmark for
Continual Disease Classification", MICCAI 2022), which in turn builds on
[FACIL](https://github.com/mmasana/FACIL) (Masana et al., "Class-Incremental Learning: Survey and Performance Evaluation
on Image Classification", TPAMI 2023).

Upstream reference: LifeLonger commit `5a3cb47` (see `patches/`).

### What changed with respect to LifeLonger

`patches/src_modified_vs_upstream.diff` contains the unified diff of every file that also exists upstream, and
`patches/new_files.txt` lists the files that do not exist upstream (experiment scripts and analysis code). Seven
upstream files were modified.

Setup differences that affect the results:

- `src/datasets/memory_dataset.py`: the class-order shuffle is disabled upstream (`if False:`). With
  `SHUFFLE_CLASSES_FIX=1` (exported by the launch scripts) the class order is shuffled, so each seed has its own order.
- `src/datasets/dataset_config.py`: normalization statistics for OrganAMNIST, BloodMNIST and TissueMNIST (upstream uses
  `None`), computed on the training data (`src/compute_normalize_stats.py`), and `extend_channel: 3` for the
  single-channel datasets.
- `src/datasets/data_loader.py`: the grey-to-RGB channel extension now runs before normalization, so that `Normalize`
  receives three channels.

Label noise (opt-in through environment variables, no effect when unset):

- `src/datasets/memory_dataset.py`: with `LABEL_NOISE=p`, each training label is flipped with probability `p` to a
  different class. `LABEL_NOISE_MODE=intra` draws the wrong label among the classes of the same task; `cross` draws it
  among the classes of all tasks seen so far. The noise has its own generator (`LABEL_NOISE_SEED`), so a noisy run and
  its clean counterpart share the class order and the validation split. The validation split now iterates over the
  labels actually present.
- `src/main_incremental.py`: the class frequencies used by iCaRL's loss weights only count labels that belong to the
  current task (needed once cross-task noise puts labels from other tasks in the data). It also prints a class-order
  check line.

Method-specific options:

- `src/approach/ewc.py`: `FISHER_NORM=c` rescales each task's Fisher so that its mean equals `c`;
  `SAVE_FISHER=1` saves the raw, accumulated and (with `FISHER_NORM`) unnormalized Fisher matrices;
  `SAVE_EXEMPLARS=1` writes the exemplar memory (`images`, `labels`) of each task to
  `<exp_path>/exemplars/task<t>.pt` after `collect_exemplars`.
- `src/approach/icarl.py`: `NO_CLASS_WEIGHTS=1` removes the class-frequency weights from the cross-entropy;
  `SAVE_EXEMPLARS=1` saves the exemplar sets and class means.

Bug fixes (not opt-in):

- `src/approach/icarl.py`: `squeeze()` became `squeeze(-1)` in the nearest-mean distances, because a last batch of one
  sample collapsed the batch dimension and raised an error.
- `src/approach/incremental_learning.py` and `src/approach/icarl.py`: training skips batches of a single sample, which
  BatchNorm cannot process.

`src/datasets/dataset_config.py` and `src/datasets/data_loader.py` also keep a few configurations and a loader branch
(`bloodmnist_64`, `organamnist_64`, `organamnist_nonorm`, `tissuemnist_balanced`) that this study does not use.

## Layout

| Path | Content |
|---|---|
| `src/` | Training framework (FACIL/LifeLonger) with the modifications above |
| `scripts/` | LifeLonger's launch scripts, unmodified, plus ours (`script_<dataset>_noise.sh`) |
| `slurm/` | Slurm jobs and submitters |
| `analysis/` | Parsers, statistics, figures and their CSV inputs and outputs |
| `results/` | Text logs of the runs (see below) |
| `patches/` | Diff against upstream and list of files that do not exist upstream |
| `PROVENANCE.md` | Where the logs and CSVs come from |

LifeLonger's scripts are kept as cloned, for reference. Because `dataset_config.py` now defines normalization
statistics (see above), running them here does not reproduce the results published for LifeLonger.

### `results/`

One folder per experiment, with one subfolder per dataset, method and noise level
(`<dataset>_<method>_p<p>/<dataset>_<method>_fixd_<seed>/`), each holding `args-*.txt` and `stdout-*.txt`:

| Folder | Content |
|---|---|
| `label_noise/` | EWC and iCaRL with intra-task label noise |
| `official_savemodel/` | Clean baselines (EWC and iCaRL) |
| `phase1_icarl/` | iCaRL reruns with exemplars saved (logits versus NME) |
| `phase2_cross/` | Cross-task label noise |
| `phase3_fisher_norm/` | EWC with normalized Fisher |
| `ablation_noweight/` | iCaRL without class-frequency loss weights |
| `phase4_ewc_exemplars/` | EWC with exemplars, intra-task label noise (p = 0, 0.1, 0.2, 0.3) |
| `phase5_ewc_cross_exemplars/` | EWC with exemplars, cross-task label noise (p = 0.1, 0.3) |
| `phase6_other_datasets/` | iCaRL and EWC on the clean stream of PathMNIST and TissueMNIST |

Checkpoints, Fisher matrices and exemplar sets are not included. `results/smoke/` and `results/verify_*/` are written
by the smoke test and by reruns from this repository, and are not tracked.

## Running

The jobs target a Slurm cluster (they were run on Alliance Canada's Fir with one MIG slice of an H100). Submit from the
repository root, because the scripts use `$SLURM_SUBMIT_DIR` as the project root. Each script activates a Python 3.11
virtual environment at `~/envs/hello_world`; edit the `source` line if yours is elsewhere. The datasets are expected
under `data/` (see `src/datasets/dataset_config.py`).

The training jobs (`run_smoke_flags.sh`, `run_label_noise.sh`, `run_phase.sh`) prepend `$HOME/cudalibs` to
`LD_LIBRARY_PATH`. This is a workaround for `libcuda` having moved to `/usr/lib64` on the Fir nodes after the October
2026 software update. The directory is not part of the repository; it holds links to the driver libraries:

```bash
mkdir -p ~/cudalibs && cd ~/cudalibs
for f in libcuda.so.1 libnvidia-ml.so.1 libnvidia-nvvm.so.4 libnvidia-ptxjitcompiler.so.1; do ln -sf /usr/lib64/$f $f; done
```

Remove the `LD_LIBRARY_PATH` line from the scripts if the libraries are found without it.

Smoke test (four short runs covering label noise, Fisher normalization and exemplar saving):

```bash
sbatch slurm/run_smoke_flags.sh
```

Experiments (each job runs 5 seeds):

```bash
bash slurm/submit_label_noise.sh                  # label noise within the current task, EWC and iCaRL
bash slurm/submit_phases.sh phase1                # iCaRL with exemplars saved (logits vs NME)
bash slurm/submit_phases.sh phase2                # cross-task label noise
FNORM_ORGANAMNIST=<c> FNORM_BLOODMNIST=<c> \
    bash slurm/submit_phases.sh phase3            # normalized Fisher (constants from analysis/label_noise/fisher_norm_constant.py)
bash slurm/submit_phases.sh ablation              # iCaRL without class-frequency loss weights
bash slurm/submit_phases.sh phase4                # EWC with exemplars, intra-task label noise
bash slurm/submit_phases.sh phase5                # EWC with exemplars, cross-task label noise
bash slurm/submit_phases.sh phase6                # PathMNIST and TissueMNIST, clean stream
```

`DRY=1` makes `submit_phases.sh` print the `sbatch` commands without submitting, and `DEP=<jobid>` makes the jobs wait
for another job (for example the smoke test).

## Analysis

Run from the repository root with the environment active (numpy, scipy, matplotlib). Without scipy,
`paired_stats.py` and `parse_phases.py` fall back to a rounded t table and the intervals change in the fourth digit.

These steps read only the text logs and CSVs in this repository:

```bash
python analysis/label_noise/parse_label_noise.py          # per_seed.csv, summary.csv
python analysis/label_noise/paired_stats.py               # paired_stats.csv
python analysis/label_noise/parse_phases.py               # per_seed_phases.csv, summary_phases.csv, paired_phases.csv
python analysis/label_noise/recency_under_noise.py        # recency per run, from the confusion matrices
python analysis/label_noise/recency_stats.py              # recency_stats.csv
python analysis/label_noise/test_corruption_analysis.py   # test_corruption_summary.csv, test_corruption_paired.csv
python analysis/label_noise/make_report_figures.py        # figures (PDF), written to analysis/label_noise/report_figs
python analysis/audit_dataset_properties.py --data-root data --output-csv analysis/dataset_audit.csv
python analysis/plot_confusion_from_csv.py -h             # confusion-matrix heatmaps (usage in the file header)
python analysis/label_noise/nme_paired.py                 # paired_nme.csv, summary_nme.csv (needs nme_eval.csv)
python analysis/label_noise/test_corruption_nme_analysis.py  # test_corruption_nme_summary.csv, test_corruption_nme_paired.csv
python analysis/label_noise/make_paper_figures.py         # PNG figures, written to analysis/label_noise/paper_figs
python analysis/label_noise/make_confusion_figs.py -h     # averaged confusion-matrix figures (usage in the file header)
```

`parse_phases.py` pairs the runs by seed and reports mean differences with 95% t intervals: iCaRL reruns against the
originals, cross-task against intra-task noise, normalized against standard Fisher, iCaRL without against with class
weights, and iCaRL NME against iCaRL logits (the latter reads `phase0_base_runs.csv`).

These steps need a GPU and the checkpoints or Fisher matrices kept on the cluster; their outputs are included in
`analysis/` (see `PROVENANCE.md`):

| Script | Slurm job | Output |
|---|---|---|
| `phase0_recency_wa.py` | `slurm/run_phase0.sh` | `phase0_base_runs.csv`, `phase0_base_norms.csv` |
| `test_corruption_eval.py` | `slurm/run_test_corruption.sh` | `test_corruption_<dataset>_ewc.csv`, calibration CSVs |
| `confusion_matrix_check.py` | `slurm/run_noise_confusions.sh` | confusion matrices |
| `fisher_by_noise.py`, `fisher_stability.py`, `fisher_confidence.py` | `slurm/run_fisher_*.sh` | `fisher_*.csv` |
| `nme_eval.py` | `slurm/run_nme_eval.sh` | `nme_eval.csv` |
| `test_corruption_nme_eval.py` | `slurm/run_test_corruption_nme.sh` | `test_corruption_nme_<dataset>_<method>.csv` |
| `confusion_eval.py` | `slurm/run_confusion_eval.sh` | `confusions/*_confusion.csv` |

`nme_eval.py` loads the final checkpoint and the saved exemplars of each run, computes the class means from
L2-normalised exemplar features (renormalised, as in iCaRL) and classifies by nearest mean. It also recomputes the
logit-based accuracy and compares it with the training log (tolerance 0.15); the last line of the output reports
`mismatches: X/N`:

```bash
sbatch slurm/run_nme_eval.sh --setup intra cross other --methods icarl ewc \
    --legacy-root <path to the earlier LifeLonger runs>
```

`test_corruption_nme_eval.py` evaluates the phase 1 (iCaRL) and phase 4 (EWC) checkpoints on corrupted test images
with both classifiers; the corruptions and levels are those of `test_corruption_eval.py` and the exemplars stay clean.
The clean evaluation of each model is checked against `nme_eval.csv` (tolerance 0.15). Submit one job per dataset and
method (`DATASET`, `METHOD`, `LEGACY_ROOT`, optionally `EXTRA="--seeds 0 --train-noise 0"` for a smoke test) and
concatenate the four CSVs, keeping one header, into `analysis/label_noise/test_corruption_nme.csv`.

`confusion_eval.py` rebuilds the final model of each run and writes the confusion matrix after the last task
(rows: true class, columns: predicted class) for the logit classifier and for NME, comparing the native classifier
with the existing matrices (`existing files compared: differ X/N`).

## Reproducibility

Runs are seeded. Reruns of the same code, made after a cluster software update, reproduced the per-seed final TAg
accuracies of the original runs exactly (to the one decimal printed) for iCaRL with 5 seeds on OrganAMNIST at label
noise 0, 0.1, 0.2 and 0.3 and on BloodMNIST with clean labels. Other configurations have not been compared.

The NME evaluation recomputes the logit accuracy of every run and compared it with the training log without any
mismatch (0 of 140 runs). In the test-time corruption jobs, the clean accuracy was reproduced within 0.07 points; the
differences of up to two test images that appear in some runs are attributed to numerical differences on the GPU.
