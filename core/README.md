# `face2eeg/core/README.md`

# `face2eeg.core` — Core library

The `core` package contains every reusable piece of logic in the project.
Nothing here is notebook-specific: everything is importable, testable, and
free of hard-coded data paths.

It is organized into four subpackages and eight top-level modules.

---

## Table of contents

- [Package layout](#package-layout)
- [Top-level modules](#top-level-modules)
- [Subpackages](#subpackages)
- [Typical usage](#typical-usage)
- [Design conventions](#design-conventions)

---

## Package layout

```
core/
├── __init__.py
├── config.py            global constants (seed, channels, bands, subjects)
├── signals.py           Butterworth band-pass filters
├── stats.py             trial-level statistics + effect sizes + CI
├── coupling.py          signal-level EEG ↔ EOG/EMG coupling
├── results.py           ExperimentResult container + loaders
├── evaluation.py        batched prediction + trial-level evaluation
├── training.py          single training loop with resume support
├── plotting.py          all figures used in the paper
│
├── data/                data loading and preprocessing (see data/README.md)
├── experiments/         end-to-end pipelines (see experiments/README.md)
└── models/              network definitions and losses (see models/README.md)
```

---

## Top-level modules

### `config.py`

Single source of truth for every fixed value used anywhere in the project:

| Group            | Contents                                                    |
|------------------|-------------------------------------------------------------|
| Reproducibility  | `SEED = 1338`                                               |
| Sampling         | `FS_EEG = 128`, `FS_VIDEO = 50`, `TRIAL_LEN = 60`, `SIGNAL_LEN = 7680` |
| Channels         | `DEAP_EEG_CHANNELS`, `FRONTAL_EEG_CHANNELS`, `PERIPHERAL_CHANNELS` |
| Bands            | `FREQ_BANDS`, `EEG_ANALYSIS_BANDS` (theta, alpha, beta)      |
| DEAP constants   | `N_TRIALS_PER_SUBJECT`, `N_SUBJECTS`, `N_EEG_CHANNELS`      |
| Subject lists    | `EXCLUDED_SUBJECTS`, `VALID_SUBJECTS`, `N_SUBJECTS_VALID`, `GLASSES_SUBJECTS` |

No data path is defined here; paths are passed explicitly at runtime.

---

### `signals.py`

Two zero-phase Butterworth band-pass filters.

```python
bandpass_filter(data, freq_band, fs, order=4, freq_bands=FREQ_BANDS)
bandpass_freq_range(data, lo, hi, fs=128.0, order=4)
```

- `bandpass_filter` uses a named band from `config.FREQ_BANDS` (`'theta'`, `'alpha'`, …).
- `bandpass_freq_range` takes an explicit `(lo, hi)` range in Hz.

Both validate the normalized cutoffs against `(0, 1)` before filtering.

---

### `stats.py`

The statistical backbone of the paper. Every inference is performed at
the **trial level** (n = 40 per subject) or at the **fold level**
(n = 5 or 18), never at the sample level.

Primary API:

| Function                              | Purpose                                                  |
|---------------------------------------|----------------------------------------------------------|
| `summarize_trials`                    | Trial-level metrics + pooled statistics + CIs for one method |
| `compare_methods_trial_level`         | Paired trial-level test between two methods              |
| `comparison_to_dataframes`            | Tidy DataFrames from a paired comparison                 |
| `analyze_segment_lengths_trial_level` | Multi-condition paired analysis (Experiment 2)           |
| `segment_lengths_to_dataframes`       | Tidy DataFrames from the multi-condition analysis        |
| `summarize_fold_trials_unit`          | Fold-level trial-as-unit summary (legacy helper)         |
| `aggregate_across_folds`              | Fold-level aggregation across CV folds                   |
| `compare_methods_across_folds`        | Fold-level paired comparison between two methods         |

Supporting helpers:

| Group                  | Functions                                             |
|------------------------|--------------------------------------------------------|
| Frequency              | `eeg_band_range`, `eeg_analysis_bands`, `band_mask`, `relative_bandpower` |
| Correlation transforms | `fisher_z`, `inv_fisher_z`, `safe_pearsonr`           |
| Spectral               | `psd_welch`, `mscohere`, `eeg_frequency_analysis`     |
| Inference              | `sign_flip_pvalue`, `bootstrap_ci_trial`              |
| Multiple comparisons   | `_fdr_bh`, `_holm_bonferroni`                         |
| Omnibus                | `_friedman_test_matrix`, `_rm_anova_oneway_rm`        |
| Effect sizes           | `population_cohens_dz`, plus Cohen's *d_z* and Hedges' *g* inside `_paired_summary` |

**Key conventions**

- **Fisher z-transform** is applied before averaging or comparing Pearson *r*.
- **Coherence** is averaged linearly (no transform); the standard deviation is computed with `ddof=1`.
- **Phipson & Smyth (2010)** Monte Carlo correction `(extreme + 1) / (n_perm + 1)` is used for all permutation p-values.
- All confidence intervals are 95% and are reported either as t-based (`ci_t`) or percentile-bootstrap (`ci_boot`).

---

### `coupling.py`

Signal-level coupling between one EEG channel and the peripheral
channels (hEOG, vEOG, zEMG). No model is trained.

```python
results = analyze_eeg_eog_emg_coupling(
    eeg_channel_name='Fz',
    participant_id=2,
    experiment_ids=list(range(40)),
    fs=128.0,
    nperseg=1024,
    bands=('theta', 'alpha', 'beta'),
    references=('heog', 'veog', 'zemg'),
    root_dir=DATA_ROOT,
)
```

Returns per-trial arrays:

| Key                        | Shape         | Meaning                              |
|----------------------------|---------------|--------------------------------------|
| `eeg_<ref>_r`              | `(n_trials,)` | Broadband (4–45 Hz) Pearson *r*      |
| `eeg_<ref>_<band>_coh`     | `(n_trials,)` | Band-averaged coherence              |
| `coh_<ref>_spectra`        | `(n_trials, n_freq)` | Full coherence spectra          |
| `f_coh`                    | `(n_freq,)`   | Frequency axis                       |
| `_bands_used`              | tuple         | Metadata                             |

The broadband *r* is stored once per reference and is **not** tied to
any specific band; only the coherence values vary by band.

---

### `results.py`

Universal container for training results, plus three loaders.

```python
@dataclass
class ExperimentResult:
    file_path, subject, channel, time_window, variant,
    n_splits, folds, config, pred_key = "y_pred"
```

Key methods:

| Method                    | Returns                                       |
|---------------------------|-----------------------------------------------|
| `fold_ids()`              | Sorted fold indices                           |
| `y_test(fold_idx)`        | Ground-truth segments for one fold            |
| `y_pred(fold_idx)`        | Predictions for one fold (`pred_key`)         |
| `get_fold_trials(k)`      | Segment→trial reshaped *(y_test, y_pred)*     |
| `get_all_trials()`        | Concatenated trials across all folds          |
| `get_trial_indices()`     | Concatenated trial indices across all folds   |
| `history(k)`              | Training history for one fold (or `None`)     |

Loaders:

```python
load_experiment_result(path, variant=None, pred_key="y_pred")
load_results_from_folder(folder, pattern="sub*_ch*.pkl", pred_key="y_pred")
load_results_by_subject(folder, subject_ids, channel_name, variant=None, pred_key="y_pred")
```

`pred_key` selects which prediction to read when a pickle contains
multiple baselines (`const_zero`, `const_train_mean`, `mean_waveform`).

---

### `evaluation.py`

Batched prediction and trial-level evaluation.

```python
evaluate_model(y_true, y_pred, fs=128.0, seed=42, n_boot=10000)
```

Returns a dict with:

- Scalar metrics (`r_mean`, `r_pooled`, `rmse_mean`, `r_pooled_ci`, `rmse_ci`, `n_trials`)
- Raw per-trial arrays (`r_trials`, `rmse_trials`)
- Plot-ready containers (`overall`, `curves`) accepted by `plot_band_comparison` and `plot_mean_curve_with_ci`

```python
run_model_and_predict(model, X_test, device, batch_size=16)
```

Runs a PyTorch model in batches, moves input to device only once per
batch, frees CUDA memory once at the end.

---

### `training.py`

Single training loop with validation, LR scheduling, checkpointing,
resume support, and full reproducibility.

```python
history = train_model(
    model,
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    device=device,
    epochs=10,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=1e-5,
    loss_fn=nn.MSELoss(),
    scheduler_step_size=5,
    scheduler_gamma=0.1,
    model_save_path=save_path,
    resume_from=save_path,   # optional
    verbose=True,
)
```

Returns a `history` dict:

| Key           | Meaning                                          |
|---------------|--------------------------------------------------|
| `loss`        | Mean train loss per epoch                        |
| `val_loss`    | Mean val loss per epoch (empty if no validation) |
| `lr`          | LR after `scheduler.step()` each epoch           |
| `best_val`    | Best validation loss seen so far                 |
| `best_epoch`  | 1-based epoch of `best_val`                      |
| `time`        | Wall-clock seconds spent in this call            |

When `resume_from` points to an existing checkpoint, the model,
optimizer, scheduler, and history are restored, and training continues
from `ckpt['epoch']`. The checkpoint is overwritten every epoch.

---

### `plotting.py`

Every figure used in the paper. All plots share a Nature/IEEE-style
`rcParams` block defined at module level, and a common color palette.

| Function                                    | Purpose                                                       |
|---------------------------------------------|---------------------------------------------------------------|
| `plot_band_comparison`                      | Bar chart of RBP and coherence per model                      |
| `plot_paired_metrics`                       | RMSE + Pearson r paired across folds                          |
| `plot_mean_curve_with_ci`                   | Mean PSD or coherence with CI vs frequency                    |
| `plot_segment_metric_curves`                | Segment-length curves from precomputed mean + CI DataFrames   |
| `plot_null_models_summary`                  | Null-model comparison (Proposed / Shuffled / Mean waveform / …) |
| `plot_metrics_grid`                         | Box + swarm for trial-level RMSE, r, RBP, coherence           |
| `plot_band_comparison_grid`                 | Grid of RBP and coherence panels per subject                  |
| `plot_coupling_heatmap_grid`                | Subject × channel × reference heatmap of broadband r and theta coherence |
| `plot_supplementary_coupling_alpha_beta`    | Supplementary heatmaps of alpha and beta coherence            |
| `plot_population_level_coupling`            | Population-level coupling across channels (main figure)       |
| `plot_supplementary_population_level_coupling` | Coherence per band (theta, alpha, beta) as three side-by-side panels |
| `plot_main_figure_video_vs_eog`             | Video vs EOG comparison (4 panels: r, Δz(r), Δcoh, ΔRBP error) |
| `plot_supplementary_spectral_heatmaps`      | Subject × (band × channel) heatmaps of coherence, RBP error, and their Δ |

Every function returns the Matplotlib figure(s) and accepts
`save_fig=True` and `file_name=...`. Most also accept an `error` switch
(`"ci95"` vs `"ci_boot"`) for the error bars.

---

## Subpackages

### `data/`

Reading, preprocessing, and segmentation of DEAP data.
See [`data/README.md`](data/README.md) for full detail.

### `experiments/`

End-to-end pipelines that consume the loaded data:
the main cross-validation pipeline, the EOG baseline, and the trivial
baselines. See [`experiments/README.md`](experiments/README.md).

### `models/`

Network definitions (`PhysNet_3DCNN_ED`), loss functions (MSE, negative
Pearson), and checkpoint helpers. See [`models/README.md`](models/README.md).

---

## Typical usage

### Train one fold

```python
from face2eeg.core.data     import fit_scaler, load_participant_segments, get_cv_splits
from face2eeg.core.models   import PhysNet_3DCNN_ED
from face2eeg.core.training import train_model
from face2eeg.core.evaluation import run_model_and_predict, evaluate_model

tr, te, _ = get_cv_splits(n_points=40, n_splits=5)
fold_idx = 0
train_ids, test_ids = tr[fold_idx], te[fold_idx]

scaler = fit_scaler(
    participants=[2], experiments=train_ids, channels=[18],
    method='z-score', root_dir=DATA_ROOT,
)
X_tr, y_tr = load_participant_segments(2, train_ids, scaler, 'Fz', time_window=2, root_dir=DATA_ROOT)
X_te, y_te = load_participant_segments(2, test_ids,  scaler, 'Fz', time_window=2, root_dir=DATA_ROOT)

model = PhysNet_3DCNN_ED(X_tr.shape[2], y_tr.shape[1]).to(device)
history = train_model(model, X_tr, y_tr, X_te, y_te,
                      epochs=10, model_save_path='fold0.pt')

y_pred = run_model_and_predict(model, X_te, device)
metrics = evaluate_model(y_te, y_pred)
```

### Load a saved result

```python
from face2eeg.core.results import load_experiment_result
r = load_experiment_result('sub02_chFz_MSE.pkl')
y_true, y_pred = r.get_all_trials()
```

### Statistical comparison of two methods

```python
from face2eeg.core.stats import summarize_trials, compare_methods_trial_level, comparison_to_dataframes

s_A = summarize_trials(y_true, y_pred_A)
s_B = summarize_trials(y_true, y_pred_B)

comp = compare_methods_trial_level(s_A, s_B, method_names=("MSE", "NegPearson"))
dfs  = comparison_to_dataframes(comp)
# dfs["overall"], dfs["coherence"], dfs["rbp_error"]
```

### EEG ↔ EOG coupling

```python
from face2eeg.core.coupling import analyze_eeg_eog_emg_coupling

res = analyze_eeg_eog_emg_coupling('Fz', 2, list(range(40)), root_dir=DATA_ROOT)
r_heog = res['eeg_heog_r']            # (40,)
coh_theta_heog = res['eeg_heog_theta_coh']
```

---

## Design conventions

| Rule                                | Why                                                                 |
|-------------------------------------|---------------------------------------------------------------------|
| No hard-coded data paths in `core`  | Paths are passed as arguments; notebook code decides them           |
| All randomness is seeded            | Every bootstrap, permutation, and shuffle accepts a `seed`          |
| Trial is the unit of inference      | 40 trials per subject; fold-level aggregation where appropriate     |
| Pearson r is aggregated via Fisher z| Corrects for the skewness of the r-sampling distribution            |
| Coherence is aggregated linearly    | Matches EEG convention; bounded in [0, 1]                           |
| Effect sizes reported alongside p   | Cohen's *d_z*, Hedges' *g*, Kendall's *W*, partial η²               |
| Multiple-comparison correction      | BH FDR is default; Holm-Bonferroni available                        |
| `pred_key` selects the prediction   | Allows trivial baselines to coexist in one pickle with the main model |

---

## Related documentation

- [`data/README.md`](data/README.md) — loaders, scaling, segmentation
- [`experiments/README.md`](experiments/README.md) — end-to-end pipelines
- [`models/README.md`](models/README.md) — network, losses, checkpointing
- [`../README.md`](../README.md) — project overview