# `analysis/README.md`

# `analysis/` — Analysis notebooks and shared helpers

This folder contains the notebooks that turn the trained-model pickles
into every figure and table reported in the paper, plus a small helper
module that the notebooks share.

Nothing here trains a model. Every notebook loads pickles produced by
the `training/` notebooks, converts them into trial-level arrays,
computes statistics at the trial or participant level, and writes
figures (PDF) and tables (CSV/LaTeX) into a per-experiment
`Analysis-Figures/` and `Analysis-Tables/` folder.

---

## Table of contents

- [Folder layout](#folder-layout)
- [Notebook reference](#notebook-reference)
- [`helpers.py`](#helperspy)
- [Statistical conventions](#statistical-conventions)
- [Output layout](#output-layout)
- [Recommended execution order](#recommended-execution-order)

---

## Folder layout

```
analysis/
├── __init__.py
├── helpers.py
├── 01_loss_function.ipynb
├── 02_segment_lengths.ipynb
├── 03_null_models.ipynb
└── 04_eog_eeg.ipynb
```

Each notebook consumes the pickles of the corresponding experiment
under `DATA_ROOT/Experiments/Cross-Validation/<Experiment>/Results/`
and writes its outputs into `Results/Analysis-Figures/` and
`Results/Analysis-Tables/`.

---

## Notebook reference

### `01_loss_function.ipynb` — MSE vs Neg-Pearson

**Input.** `Loss-Functions/Results/sub02_chFz_{MSE,NegPearson}.pkl`.

**Scope.** Subject 2, channel Fz, T = 1 s, 40 held-out trials.

**What it does.**

1. Loads the two pickles via `load_result(..., variant=...)`.
2. Extracts trial-level arrays via `ExperimentResult.get_all_trials()`.
3. Verifies trial alignment and identical EEG targets across conditions
   with `validate_trial_alignment`.
4. Computes trial-level summaries with `summarize_trials`.
5. Runs a paired trial-level comparison with
   `compare_methods_trial_level` and converts it to tidy DataFrames
   with `comparison_to_dataframes`.
6. Plots band-wise RBP and coherence with `plot_band_comparison`,
   and mean coherence and PSD curves with `plot_mean_curve_with_ci`.

**Outputs.**

- `Loss-Functions/Results/Analysis-Tables/exp1_overall.{csv,tex}`
- `Loss-Functions/Results/Analysis-Tables/exp1_coherence.{csv,tex}`
- `Loss-Functions/Results/Analysis-Tables/exp1_rbp_error.{csv,tex}`
- `Loss-Functions/Results/Analysis-Tables/exp1_summary.csv`
- `Loss-Functions/Results/Analysis-Figures/exp1_band_comparison.pdf`
- `.../exp1_coh_mean_curve_with_ci.pdf`
- `.../exp1_psd_pred_mean_curve_with_ci.pdf`

**Paper.** Section 4.3 (Tables 2–4, Figures 7–9).

---

### `02_segment_lengths.ipynb` — Segment-length scan

**Input.** `Segment-Lengths/Results/sub02_chFz_tw{0.5,1,2,3}.pkl`.

**Scope.** Subject 2, channel Fz, T ∈ {0.5, 1, 2, 3} s, 40 held-out
trials each.

**What it does.**

1. Loads all four variants via `load_result(..., variant='twX')`.
2. Verifies that trial indices and EEG targets are identical across
   segment lengths.
3. Computes trial-level summaries with `summarize_trials`.
4. Builds a descriptive table (Fisher-z aggregated r and RMSE with
   bootstrap CIs) for the manuscript.
5. Runs the multi-condition paired analyzer
   `analyze_segment_lengths_trial_level`, which returns omnibus
   (Friedman + RM-ANOVA) and pairwise tests for RMSE, r, and per-band
   coherence and RBP error, with BH or Holm correction.
6. Renders the segment-length curves with `plot_segment_metric_curves`
   (RMSE + r panel, per-band coherence panel).

**Outputs.**

- `Segment-Lengths/Results/Analysis-Tables/exp2_descriptive_summary.{csv,tex}`
- `.../exp2_omnibus.{csv,tex}`
- `.../exp2_pairwise.{csv,tex}`
- `Segment-Lengths/Results/Analysis-Figures/exp2_band_comparison.pdf`
- `.../exp2_coh_mean_curve_with_ci.pdf`
- `.../exp2_psd_pred_mean_curve_with_ci.pdf`
- `.../exp2_RMSE_and_R.pdf`
- `.../exp2_Coherence_per_band.pdf`

**Paper.** Section 4.4 (Tables 5–6, Figures 10–11).

---

### `03_null_models.ipynb` — Trivial and shuffled null models

**Input.**

- `Between-Participant-Variability/Results/sub{XX}_chFz.pkl` (main)
- `Null-Models/Shuffled/Results/sub{XX}_chFz.pkl`
- `Null-Models/Trivial/Results/sub{XX}_chFz.pkl` (three predictions per fold)

**Scope.** 18 subjects, channel Fz, T = 2 s, 40 held-out trials each.

**What it does.**

1. Loads the main, shuffled, and trivial results.
2. Checks that all conditions share the same trial indices;
   verifies fold structure (5 folds × 8 held-out trials).
3. **Critical convention:** the shuffled prediction is always scored
   against the **original** EEG target of the corresponding trial,
   not against the shuffled target that was used during training.
4. Computes trial-level summaries for all five conditions with
   `summarize_trials`:
   Proposed model, Trial-shuffled model, Training-set mean waveform,
   Training-mean constant, Zero-output baseline.
5. Produces a participant-level table (`df_subject`) and a
   participant-level table (`df_group`).
6. Runs within-subject paired trial-level comparisons against the
   Proposed model (`compare_methods_trial_level`), and aggregates the
   participant-level effects to the population.
7. Corrects p-values with Holm within each metric family.
8. Renders the summary figure with `plot_null_models_summary`.
9. Runs a **native shuffled-target** analysis side by side with the
   common-target analysis, plus an audit that the shuffling actually
   modified the targets, and a paired comparison of the two
   evaluations of the same shuffled predictions.

**Outputs.**

- `Null-Models/Analysis-Tables/exp3_subject_level.csv`
- `.../exp3_population_summary.csv`
- `.../exp3_population_effects.csv`
- `.../exp3_within_subject_effects.csv`
- `.../exp3_within_subject_spectral_effects.csv`
- `.../exp3_shuffled_native_vs_common.csv`
- `.../exp3_shuffle_audit.csv`
- `Null-Models/Analysis-Figures/exp3_null_models.pdf`

**Paper.** Section 4.5, Reviewer 2, Comment 2.

---

### `04_eog_eeg.ipynb` — Peripheral EOG controls

**Input.**

- `Between-Participant-Variability/Results/sub{XX}_chFz.pkl`
- `Channel-Specificity/Results/sub{XX}_ch{Pz,Oz}.pkl`
- `EOG-Regression/Results/sub{XX}_ch{Fz,Pz,Oz}.pkl`

**Scope.** 18 subjects, channels Fz/Pz/Oz, references hEOG + vEOG,
T = 2 s.

**What it does.** Two complementary analyses:

1. **Signal-level coupling.** Runs
   `analyze_eeg_eog_emg_coupling` for every (subject, channel) pair,
   storing broadband Pearson *r* and per-band coherence separately.
   Builds a wide-format table (`df_coupling_wide`) with one row per
   (Subject, Target, Reference) and columns
   `r_broadband_4_45, coh_theta, coh_alpha, coh_beta`, where the
   broadband *r* is Fisher-z aggregated across the 40 trials and
   coherence is the linear mean.
2. **EOG→EEG regression baseline.** Loads the EOG-regression pickles
   and compares them, under the same held-out trials, against the
   video-model predictions.

Then:

3. Verifies that the video model and the EOG regression use identical
   fold assignments and identical held-out trial partitions.
4. Builds complete trial-level arrays via `get_all_trials()`.
5. Aggregates the coupling table to the population level, with
   Fisher-z aggregation for *r*, linear aggregation for coherence,
   percentile-bootstrap CIs over the 18 participants, sign-flip
   permutation p-values, Wilcoxon p-values, Cohen's *d_z*, and Holm
   correction within each metric family.
6. Runs a paired Video − EOG analysis, both participant-level and
   population-level, for correlation, per-band coherence, and
   per-band RBP error. All contrasts use the sign convention
   **A − B = Video − EOG**.
7. Renders the figures:
   - `plot_coupling_heatmap_grid` (subject × channel × reference,
     broadband r and theta coherence),
   - `plot_supplementary_coupling_alpha_beta` (alpha and beta
     coherence),
   - `plot_population_level_coupling` (main coupling figure),
   - `plot_supplementary_population_level_coupling` (per-band
     coherence, three panels),
   - `plot_main_figure_video_vs_eog` (4 panels: r, Δz(r), Δcoh,
     ΔRBP error),
   - `plot_supplementary_spectral_heatmaps` (subject × band × channel
     heatmaps of coherence, RBP error, and their Δ).

**Outputs.**

- `EOG-Regression/Results/Analysis-Tables/Table_S2_coupling_population.csv`
- `.../Table_S3_subject_prediction.csv`
- `.../Table_S4_subject_spectral.csv`
- `.../Table_S5_subject_paired_prediction.csv`
- `.../Table_S6_subject_paired_spectral.csv`
- `.../Table_S7_population_spectral.csv`
- `.../Table_S8_channel_specificity.csv`
- `.../Table_1_main_video_vs_eog.csv`
- `.../Table_2_main_theta_coherence.csv`
- `EOG-Regression/Results/Analysis-Figures/fig_coupling_heatmap_all_channels_and_subjects.pdf`
- `.../fig_S_EEG_EOG_alpha_beta_coupling.pdf`
- `.../fig_coupling_main.pdf`
- `.../fig_S_coupling_bands.pdf`
- `.../fig_main_video_vs_eog.pdf`
- `.../fig_S_spectral_heatmaps.pdf`

**Paper.** Section 4.5, Reviewer 1, Comment 1.

---

## `helpers.py`

Small utility module imported by every notebook. It has no statistics
of its own — only file and DataFrame plumbing.

| Function                  | Purpose                                                              |
|---------------------------|----------------------------------------------------------------------|
| `setup(base_dir, ...)`    | Normalizes `base_dir` to an absolute path and bundles common config into a dict |
| `experiment_paths(base, name)` | Builds `<base>/<name>/Results/` and `<base>/<name>/Results/Analysis-Figures/`, creating the figure folder on demand |
| `load_result(results_dir, subject, channel, variant, pred_key)` | Loads `<results_dir>/sub{XX}_ch{CH}[_{variant}].pkl` as an `ExperimentResult` |
| `get_arrays(result, signal_length=7680, pred_key)` | Concatenates `y_test`, `y_pred`, and `trials` across folds and reshapes to `(n_trials, signal_length)` |
| `get_trials_dataframe(summaries_dict)` | Builds the tidy per-trial DataFrame expected by `plot_metrics_grid` (rmse, r, rbp_pred_<band>, coh_<band>) |
| `validate_trial_alignment(ids_a, ids_b, ...)` | Raises unless two trial-id arrays are equal, duplicate-free, and same length |

All three loaders in the `load_*` family in `face2eeg.core.results` are
also available; `helpers.load_result` is the notebook-friendly wrapper
that builds the filename from `(subject, channel, variant)`.

---

## Statistical conventions

The analysis notebooks follow a strict separation between three
hierarchical levels:

| Level               | Unit           | n            | Where used                                            |
|---------------------|----------------|--------------|-------------------------------------------------------|
| Within-subject      | Trial          | 40           | Per-subject summaries, paired tests within a subject   |
| Participant         | Subject        | 18           | Population-level effect sizes and bootstrap CIs        |
| Population          | Subject        | 18           | Sign-flip permutation p-values, Wilcoxon, Holm         |

Rules that apply across all notebooks:

- **Pearson *r* is always aggregated via Fisher z**, both within a
  subject (across trials) and across subjects. The transform and its
  inverse live in `face2eeg.core.stats`.
- **Coherence is never Fisher-z transformed.** It is bounded in
  [0, 1] and is not a correlation; it is aggregated linearly.
- **RMSE and RBP error are aggregated linearly.**
- **Permutation p-values use the Phipson & Smyth (2010) Monte Carlo
  correction** `(extreme + 1) / (n_perm + 1)`.
- **Multiple-comparison correction is Holm-Bonferroni within each
  metric family** (all three channels, or all three bands, depending
  on the notebook). BH FDR is available in `stats.py` when needed.
- **Sign convention is `A − B`.** In every comparison the reference
  (usually the Proposed model) is A. This is verified explicitly in
  `04_eog_eeg.ipynb` before the population-level analysis starts.
- **Effect size at the population level is Cohen's *d_z* on the 18
  participant-level effects**, not on the 40 correlated trial-level
  differences (see `population_cohens_dz` in `stats.py`).

---

## Output layout

```
DATA_ROOT/Experiments/Cross-Validation/
├── Loss-Functions/
│   └── Results/
│       ├── Analysis-Figures/
│       └── Analysis-Tables/
├── Segment-Lengths/
│   └── Results/
│       ├── Analysis-Figures/
│       └── Analysis-Tables/
├── Between-Participant-Variability/
│   └── Results/
│       ├── Analysis-Figures/
│       └── Analysis-Tables/
├── Channel-Specificity/
│   └── Results/
│       ├── Analysis-Figures/
│       └── Analysis-Tables/
├── EOG-Regression/
│   └── Results/
│       ├── Analysis-Figures/
│       └── Analysis-Tables/
└── Null-Models/
    ├── Analysis-Figures/
    └── Analysis-Tables/
```

Every figure is written as a PDF at 400 dpi with vector text
(`pdf.fonttype = 42`), which is the format required by most journals.
Tables are written both as CSV (for downstream tooling) and as LaTeX
(for direct `\input{}` in the manuscript).

---

## Recommended execution order

1. `01_loss_function` — single subject, fast
2. `02_segment_lengths` — single subject, four conditions
3. `03_null_models` — depends on `Between-Participant-Variability` and the two null-model folders
4. `04_eog_eeg` — depends on `Between-Participant-Variability`, `Channel-Specificity`, and `EOG-Regression`

Notebooks 3 and 4 both require the outputs of the training notebooks
listed in `training/README.md`.

The notebooks are idempotent in the sense that re-running them
regenerates their figures and tables from the same pickles. They do
not mutate the pickles.

---

## Related documentation

- [`../README.md`](../README.md) — project overview
- [`../training/README.md`](../training/README.md) — notebooks that produce the pickles
- [`../face2eeg/core/stats.py`](../face2eeg/core/stats.py) — trial-level statistics used everywhere
- [`../face2eeg/core/plotting.py`](../face2eeg/core/plotting.py) — every figure function called here
- [`../face2eeg/core/results.py`](../face2eeg/core/results.py) — `ExperimentResult` container and loaders
- [`../face2eeg/core/coupling.py`](../face2eeg/core/coupling.py) — signal-level EEG-EOG coupling used in notebook 04
