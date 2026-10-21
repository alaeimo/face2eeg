# `face2eeg/core/experiments/README.md`

# `face2eeg.core.experiments` — End-to-end experiment runners

This subpackage contains every training and reference-point experiment
in the paper. Each runner takes a subject id and a channel name and
returns a pickle-ready payload. No plotting, no figures — those live in
`face2eeg.core.plotting` and the analysis notebooks.

Three runners are exposed:

| Runner                     | Trains? | Needs GPU? | Purpose                                                  |
|----------------------------|---------|------------|----------------------------------------------------------|
| `run_kfold_pipeline`       | Yes     | Yes (or CPU) | Main video → EEG model (PhysNet-3DCNN-ED)             |
| `run_eog_regression`       | Yes     | No         | Linear-regression baseline from EOG → EEG                |
| `run_trivial_baselines`    | No      | No         | Floor references: zero, train-mean, train-mean waveform  |

All three use **exactly the same GroupKFold splits**, produced by
`face2eeg.core.data.get_cv_splits`. This guarantees that:

- the same trial appears in the test fold of every runner;
- trial *i* in method A and trial *i* in method B refer to the same
  physical trial;
- paired statistical comparisons at the trial level (see
  `face2eeg.core.stats`) are valid without any realignment.

---

## Table of contents

- [Design principles](#design-principles)
- [Common protocol](#common-protocol)
- [Module reference](#module-reference)
- [Pickle schemas](#pickle-schemas)
- [Typical usage](#typical-usage)
- [Related documentation](#related-documentation)

---

## Design principles

1. **One fold, one model.** No state is carried between folds; the
   optimizer, scheduler, and scaler are created fresh inside each fold.
2. **No data leakage.** The scaler is always fit on the training
   trials of the current fold only, and the same fitted scaler is then
   applied to the test trials.
3. **Schema parity.** All three runners produce pickles that load
   through `face2eeg.core.results.load_experiment_result`. Differences
   are confined to extra keys inside each fold dict.
4. **Reproducibility.** Every stochastic operation takes an explicit
   seed. `run_kfold_pipeline` enforces this for null-model shuffling.
5. **Separation from plotting and statistics.** Runners return raw
   arrays and metadata; downstream modules do everything else.

---

## Common protocol

For every runner:

1. Compute the GroupKFold splits once (5 folds over 40 trials).
2. For each fold, fit a scaler on the **training** trials.
3. Apply that scaler to both training and test EEG.
4. Produce per-fold arrays of shape `(n_test_segments, s_window)` or
   `(n_test_trials, 7680)`, depending on whether segmentation is used.
5. Write a single `.pkl` per `(subject, channel)` to `save_dir`, with
   the name `sub{XX}_ch{CH}.pkl`.

The **only** thing that varies between runners is what happens inside
the fold loop: train a network, fit a linear regression, or just
compute trivial predictors.

---

## Module reference

### `pipeline.py` — `run_kfold_pipeline`

The main cross-validation runner.

```python
results = run_kfold_pipeline(
    subject_id=2,
    channel_name='Fz',
    time_window=2,
    n_splits=5,
    epochs=10,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=1e-5,
    train_overlap=True,  train_overlap_ratio=0.5,
    test_overlap=False,  test_overlap_ratio=0.0,
    save_dir=MODELS_DIR,
    root_dir=DATA_ROOT,
)
```

**Per fold:**

1. Fit `StandardScaler` on training trials.
2. Load and segment **training** data (`load_participant_segments`)
   with `signal_normalize=True`.
3. Load and segment **test** data with the same scaler.
4. Instantiate a fresh `PhysNet_3DCNN_ED`.
5. Train for `epochs` with `train_model`. The test set is **not**
   passed as validation; only the training loss is tracked.
6. Predict on the test fold with `run_model_and_predict`.
7. Store `y_test`, `y_pred`, `trials`, and (optionally) the training
   `history`.
8. Delete the model and free the CUDA cache.

**Null-model mode (`shuffle_pairing=True`):** before concatenation,
the EEG list is permuted inside `load_participant_segments` so that
each video segment is paired with the EEG of a different trial of the
same subject. The video keeps its order; the signals are shuffled. A
fixed-point-free permutation is enforced (no trial is paired with
itself). The RNG uses `shuffle_seed + fold_idx` for training and
`shuffle_seed + 1000 + fold_idx` for the test fold, so train and test
use independent permutation streams. When `shuffle_pairing=True`, a
fixed `shuffle_seed` is **required**.

**Not exposed:**

- `resume_from` — every fold starts from scratch.
- External optimizer / scheduler — creating them outside the fold loop
  would leak state across folds.

**Key parameters:**

| Parameter           | Meaning                                                       |
|---------------------|---------------------------------------------------------------|
| `time_window`       | Segment length in seconds (0.5, 1, 2, 3)                      |
| `train_overlap`     | Whether training segments overlap (50% in the paper)          |
| `test_overlap`      | Kept off to avoid temporal dependence between test segments   |
| `shuffle_pairing`   | Enables the shuffled null model (Reviewer 2, Comment 2)       |
| `shuffle_seed`      | Required for the shuffled null; ignored otherwise             |
| `model_exp_start`   | Prefix number for checkpoint filenames                        |
| `return_history`    | Whether to store per-epoch train loss in the pickle           |

**Checkpointing:** one `.pt` file per fold is written to `save_dir`
with the name `model-E{exp}_sub{id}_ch{ch}_tw{tw}_fold{k}.pt`. The file
is overwritten every epoch.

---

### `eog_baseline.py` — `run_eog_regression` and `eog_regression_fold`

The EOG→EEG linear-regression baseline for Reviewer 2, Comment 1.
Torch-free.

#### `eog_regression_fold` (one fold)

```python
fold_result = eog_regression_fold(
    participant_id=2,
    eeg_channel_name='Fz',
    train_trials=tr,
    test_trials=te,
    experiment_ids=list(range(40)),
    references=('heog', 'veog'),
    fs=128.0,
    root_dir=DATA_ROOT,
)
```

**Model:** per-sample linear regression

```
Fz(t) ≈ β0 + β1 · hEOG(t) + β2 · vEOG(t)
```

**Preprocessing:**

1. Load EEG + EOG via `load_eeg_eog_emg`, which band-passes the
   peripheral channels to 4–45 Hz. The EEG channel is already
   band-passed by DEAP.
2. Fit `StandardScaler` on the training samples for **X** (EOG) and a
   second `StandardScaler` for **y** (EEG). This is mathematically
   identical to fitting a single scaler on the concatenation of all
   training trials, but keeps memory bounded.
3. Fit `sklearn.linear_model.LinearRegression` on the z-scored data.
4. For each test trial, z-score the EOG with the training statistics,
   predict the z-scored EEG, and z-score the ground-truth EEG with the
   same training statistics.

**Returned keys (per fold):**

| Key                 | Shape         | Meaning                                        |
|---------------------|---------------|------------------------------------------------|
| `y_test`            | `(n_test, 7680)` | Ground-truth EEG, z-scored with train stats  |
| `y_pred`            | `(n_test, 7680)` | Predicted EEG, same z-scale                  |
| `trials`            | `(n_test,)`     | Trial indices of the test fold                |
| `model_coef`        | `(n_refs,)`     | Linear-regression coefficients                 |
| `model_intercept`   | scalar          | Linear-regression intercept                    |
| `scaler_X_mean`     | `(n_refs,)`     | Mean of the training EOG, per reference        |
| `scaler_X_scale`    | `(n_refs,)`     | Std of the training EOG, per reference         |
| `scaler_y_mean`     | scalar          | Mean of the training EEG                       |
| `scaler_y_scale`    | scalar          | Std of the training EEG                        |
| `references`        | tuple of str    | Order of the EOG channels fed to the model     |

This is sufficient to reconstruct the model on any new input:

```python
X_z    = (X_new - scaler_X_mean) / scaler_X_scale
y_z    = X_z @ model_coef + model_intercept
y_raw  = y_z * scaler_y_scale + scaler_y_mean
```

#### `run_eog_regression` (full CV)

Runs `eog_regression_fold` over all folds and writes the pickle in the
**same schema as `run_kfold_pipeline`** so it can be loaded through the
same loader:

```python
{
    'subject':     int,
    'channel':     str,
    'time_window': 60,          # whole trial, no segmentation
    'n_splits':    int,
    'folds': { fold_idx: {...} },
    'config': {
        'method':       'LinearRegression',
        'estimator':    'sklearn.linear_model.LinearRegression',
        'segmentation': False,
        'zscore':       'per-fold, train statistics only',
        'references':   list(references),
        'fs':           float(fs),
    },
}
```

**IMPORTANT:** the filename is `sub{XX}_ch{CH}.pkl` — identical to the
video model. **Save to a dedicated folder** (e.g. `EOG-Regression/Results/`)
to avoid overwriting the video-model pickles.

A JSON summary (`EOG2EEG_summary.json`) is written next to the pickle
and updated incrementally, listing every `(subject, channel)` that has
been processed.

---

### `trivial_baselines.py` — `run_trivial_baselines`

The floor references for Reviewer 2, Comment 2. Torch-free. No model is
trained. Only the EEG side is loaded (`load_participant_signal_segments`),
so the runtime is dominated by I/O.

Three predictors are computed per test fold:

| Predictor           | Definition                                           | Interpretation                                    |
|---------------------|------------------------------------------------------|---------------------------------------------------|
| `const_zero`        | Zero everywhere                                      | Equivalent to predicting the mean of z-scored targets. RMSE = 1 by construction. |
| `const_train_mean`  | Training-set scalar mean, broadcast to test shape    | One-parameter baseline; does not use temporal structure |
| `mean_waveform`     | Training-set mean waveform (per-position average)    | Uses temporal structure but no video information  |

```python
results = run_trivial_baselines(
    subject_id=2,
    channel_name='Fz',
    time_window=2,
    save_dir=None,        # this runner does not write files
    root_dir=DATA_ROOT,
)
```

**Returned keys per fold:**

| Key                  | Shape              | Meaning                                    |
|----------------------|--------------------|--------------------------------------------|
| `y_test`             | `(n_test_seg, s_win)` | Ground-truth EEG segments (z-scored)    |
| `trials`             | `(n_test,)`         | Trial indices                              |
| `const_zero`         | `(n_test_seg, s_win)` | Zeros                                    |
| `const_train_mean`   | `(n_test_seg, s_win)` | Scalar train mean broadcast              |
| `mean_waveform`      | `(n_test_seg, s_win)` | Training-set mean waveform tiled         |
| `y_train_mean`       | scalar              | The scalar used by `const_train_mean`      |

The runner does **not** write a pickle by default; the caller decides
where to save it. To store the results with `load_experiment_result`
compatibility, wrap the returned dict with the standard top-level keys
(`subject`, `channel`, `time_window`, `n_splits`, `config`) and dump it.

**Why `const_zero` has RMSE = 1:** because the targets are z-scored
before prediction, the residual `y - 0` has the same variance as
`y`, i.e. 1. Any RMSE that is not meaningfully below 1 indicates that
the model has not learned anything beyond a constant.

---

## Pickle schemas

All runners produce a pickle with the same top-level structure:

```
{
    'subject':     int,
    'channel':     str,
    'time_window': int | float,
    'n_splits':    int,
    'folds':       { fold_idx: { ... } },
    'config':      { ... },
}
```

They differ only in what each fold dict contains:

| Key                     | `run_kfold_pipeline` | `run_eog_regression` | `run_trivial_baselines` |
|-------------------------|:--------------------:|:--------------------:|:-----------------------:|
| `y_test`                | ✅                   | ✅                   | ✅                      |
| `y_pred`                | ✅                   | ✅                   | —                       |
| `trials`                | ✅                   | ✅                   | ✅                      |
| `history`               | ✅ (if enabled)      | `None` (placeholder) | —                       |
| `model_coef`            | —                    | ✅                   | —                       |
| `model_intercept`       | —                    | ✅                   | —                       |
| `scaler_X_mean`         | —                    | ✅                   | —                       |
| `scaler_X_scale`        | —                    | ✅                   | —                       |
| `scaler_y_mean`         | —                    | ✅                   | —                       |
| `scaler_y_scale`        | —                    | ✅                   | —                       |
| `references`            | —                    | ✅                   | —                       |
| `const_zero`            | —                    | —                    | ✅                      |
| `const_train_mean`      | —                    | —                    | ✅                      |
| `mean_waveform`         | —                    | —                    | ✅                      |
| `y_train_mean`          | —                    | —                    | ✅                      |

Because the top-level keys are identical, all three can be loaded with:

```python
from face2eeg.core.results import load_experiment_result

r = load_experiment_result('sub02_chFz.pkl')
y_true, y_pred = r.get_all_trials()
```

For trivial baselines, pass `pred_key='const_zero'`, `'const_train_mean'`,
or `'mean_waveform'` to select which prediction to read.

---

## Typical usage

### Main model across CV folds

```python
from face2eeg.core.experiments import run_kfold_pipeline

results = run_kfold_pipeline(
    subject_id=2,
    channel_name='Fz',
    time_window=2,
    epochs=10,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=1e-5,
    train_overlap=True,  train_overlap_ratio=0.5,
    test_overlap=False,  test_overlap_ratio=0.0,
    save_dir=NN_RESULTS_DIR,
    root_dir=DATA_ROOT,
    verbose=True,
)
```

### EOG regression baseline

```python
from face2eeg.core.experiments import run_eog_regression

run_eog_regression(
    subject_id=2,
    channel_name='Fz',
    references=('heog', 'veog'),
    save_dir=EOG_RESULTS_DIR,
    root_dir=DATA_ROOT,
)
```

### Trivial baselines

```python
from face2eeg.core.experiments import run_trivial_baselines

results = run_trivial_baselines(
    subject_id=2,
    channel_name='Fz',
    time_window=2,
    save_dir=None,          # return dict; caller decides where to save
    root_dir=DATA_ROOT,
)
```

### Shuffled null model

```python
from face2eeg.core.experiments import run_kfold_pipeline

run_kfold_pipeline(
    subject_id=2,
    channel_name='Fz',
    time_window=2,
    shuffle_pairing=True,
    shuffle_seed=1338,
    save_dir=SHUFFLED_RESULTS_DIR,
    root_dir=DATA_ROOT,
)
```

### Comparing all three under identical splits

```python
from face2eeg.core.results import load_experiment_result
from face2eeg.core.stats   import summarize_trials, compare_methods_trial_level

r_main     = load_experiment_result(f'{NN_DIR}/sub02_chFz.pkl')
r_eog      = load_experiment_result(f'{EOG_DIR}/sub02_chFz.pkl')
r_zero     = load_experiment_result(f'{TRIVIAL_DIR}/sub02_chFz.pkl', pred_key='const_zero')
r_wave     = load_experiment_result(f'{TRIVIAL_DIR}/sub02_chFz.pkl', pred_key='mean_waveform')

yt, yp_main = r_main.get_all_trials()
_,  yp_eog  = r_eog.get_all_trials()
_,  yp_zero = r_zero.get_all_trials()
_,  yp_wave = r_wave.get_all_trials()

s_main = summarize_trials(yt, yp_main)
s_eog  = summarize_trials(yt, yp_eog)
s_zero = summarize_trials(yt, yp_zero)
s_wave = summarize_trials(yt, yp_wave)

comp = compare_methods_trial_level(s_main, s_eog, method_names=('Video', 'EOG'))
```

Every `summarize_trials` and `compare_methods_trial_level` result is
directly comparable because the trial orderings are identical.

---

## Related documentation

- [`../../README.md`](../../README.md) — project overview
- [`../README.md`](../README.md) — core library
- [`../data/README.md`](../data/README.md) — loaders, scaling, segmentation
- [`../models/README.md`](../models/README.md) — network and losses
- [`../stats.py`](../stats.py) — trial-level statistics used by the analysis notebooks
- [`../results.py`](../results.py) — `ExperimentResult` container and loaders