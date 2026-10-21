# `training/README.md`

# `training/` — Model training notebooks

This folder contains the six Jupyter notebooks that produce every
trained model used in the paper. They are thin drivers: all real logic
lives in `face2eeg.core`. A notebook's job is to

1. set up `sys.path` and `DATA_ROOT`;
2. pick the correct experiment directory and hyperparameters;
3. call the appropriate runner from `face2eeg.core.experiments`;
4. save the returned dict as a `.pkl` and skip work that is already done.

Because of point 4, all notebooks are **idempotent**: re-running them
after an interruption continues from where they stopped.

---

## Table of contents

- [Common setup](#common-setup)
- [Notebook reference](#notebook-reference)
- [Output layout](#output-layout)
- [Relationship to the paper](#relationship-to-the-paper)
- [Recommended execution order](#recommended-execution-order)
- [Design conventions](#design-conventions)

---

## Common setup

Every notebook starts with the same four blocks:

**1. Environment detection**

```python
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
```

If running on Colab, Drive is mounted and `pims` is installed. Local
runs skip both steps.

**2. Project paths**

```python
SOURCE_ROOT = r'D:\Thesis\Codes\Source'
DATA_ROOT   = r'D:\Thesis\Codes\Data'
```

A `FileNotFoundError` is raised if `SOURCE_ROOT/face2eeg` is missing,
which catches almost every deployment mistake before training starts.

**3. Reproducibility**

```python
np.random.seed(SEED)
torch.manual_seed(SEED)
```

`SEED = 1338` is imported from `face2eeg.core.config`.

**4. Directory creation**

Each notebook defines its own `RESULTS_DIR` and (when applicable)
`MODELS_DIR` under `DATA_ROOT/Experiments/Cross-Validation/<Experiment>/`
and creates them with `os.makedirs(..., exist_ok=True)`.

Every notebook ends with a small status printout that lists, per
subject (and per channel where relevant), whether the corresponding
`.pkl` file exists.

---

## Notebook reference

### `01_loss_function.ipynb` — MSE vs Neg-Pearson

**Question.** Does the loss function change prediction quality?

**Scope.**

| Item          | Value                                  |
|---------------|----------------------------------------|
| Subject       | 2                                       |
| Channel       | Fz                                      |
| Time window   | 1 s                                     |
| CV            | 5-fold GroupKFold over 40 trials        |
| Overlap       | Train 0 %, Test 0 %                     |
| Epochs        | 20                                      |
| Batch size    | 16                                      |
| Optimizer     | Adam (lr = 1e-4, wd = 1e-5)             |
| LR schedule   | StepLR, γ = 0.1 every 10 epochs         |

**Runs.** Two identical PhysNet-3DCNN-ED instances, differing only in
the loss:

- MSE            → `exp_start = 131`
- Neg-Pearson    → `exp_start = 136`

**Output.**
`Loss-Functions/Results/sub02_chFz_MSE.pkl` and
`Loss-Functions/Results/sub02_chFz_NegPearson.pkl`.

**Paper.** Section 4.3, Tables 2–4, Figures 7–9.

---

### `02_segment_lengths.ipynb` — Segment-length scan

**Question.** How does the temporal input length *T* affect waveform
estimation and spectral fidelity?

**Scope.**

| Item          | Value                                  |
|---------------|----------------------------------------|
| Subject       | 2                                       |
| Channel       | Fz                                      |
| Time windows  | T ∈ {0.5, 1, 2, 3} s                    |
| Loss          | MSE (fixed)                             |
| CV            | 5-fold GroupKFold over 40 trials        |
| Overlap       | Train 0 %, Test 0 %                     |
| Epochs        | 20                                      |
| Optimizer     | Adam (lr = 1e-4, wd = 1e-5)             |
| LR schedule   | StepLR, γ = 0.1 every 10 epochs         |

**Runs.**

| Variant | T (s) | exp_start | Note                                          |
|---------|-------|-----------|-----------------------------------------------|
| `tw0.5` | 0.5   | 141       |                                                |
| `tw1`   | 1.0   | (skipped) | Reuses the MSE run from `01_loss_function`     |
| `tw2`   | 2.0   | 151       |                                                |
| `tw3`   | 3.0   | 156       |                                                |

**Output.**
`Segment-Lengths/Results/sub02_chFz_tw{0.5,1,2,3}.pkl`.

**Paper.** Section 4.4, Tables 5–6, Figures 10–11.

---

### `03_between_participant_variability.ipynb` — One model per subject

**Question.** How does prediction quality vary across participants
under identical intra-subject conditions?

**Scope.**

| Item          | Value                                     |
|---------------|-------------------------------------------|
| Subjects      | `VALID_SUBJECTS` (n = 18)                 |
| Channel       | Fz                                        |
| Time window   | 2 s (selected from `02_segment_lengths`)  |
| Loss          | MSE                                       |
| CV            | 5-fold GroupKFold over 40 trials          |
| Overlap       | Train 50 %, Test 0 %                      |
| Epochs        | 10                                        |
| Optimizer     | Adam (lr = 1e-4, wd = 1e-5)               |
| LR schedule   | StepLR, γ = 0.1 every 5 epochs            |

`exp_start` is assigned per subject as `200 + 10·index` to avoid
checkpoint collisions.

**Output.** `Between-Participant-Variability/Results/sub{XX}_chFz.pkl`.

**Notes.** The notebook calls `clear_video_cache()` and
`clear_signals_cache()` between subjects to keep RAM bounded, and frees
the CUDA cache after each subject.

**Paper.** Section 4.5, Table 7, Figure 12.

---

### `04_channel_specificity.ipynb` — Posterior channels (Pz, Oz)

**Question.** Is the model's performance spatially specific to the
frontal channel (Fz) or does it generalize to posterior sites?

**Scope.**

| Item          | Value                                     |
|---------------|-------------------------------------------|
| Subjects      | `VALID_SUBJECTS` (n = 18)                 |
| Channels      | Pz, Oz (Fz is the reference from exp. 3)  |
| Time window   | 2 s                                       |
| Loss          | MSE                                       |
| CV            | 5-fold GroupKFold over 40 trials          |
| Overlap       | Train 50 %, Test 0 %                      |
| Epochs        | 10                                        |
| Optimizer     | Adam (lr = 1e-4, wd = 1e-5)               |
| LR schedule   | StepLR, γ = 0.1 every 5 epochs            |

`exp_start` is set per channel:

- Pz → 300
- Oz → 400

**Output.** `Channel-Specificity/Results/sub{XX}_ch{Pz,Oz}.pkl`.

**Notes.** Directly addresses Reviewer 1, Comment 1. When combined
with the Fz results of experiment 3 and the EOG regression baseline
of experiment 5, it allows the topography of the coupling to be
compared against the topography of the residual ocular signal.

**Paper.** Section 4.5 (participant-level variability) and the discussion
of ocular contamination in the Introduction and 3.4.2.

---

### `05_eog_eeg.ipynb` — EOG → EEG linear regression baseline

**Question.** How much of the reported performance can a plain linear
regression from the horizontal and vertical EOG explain?

**Scope.**

| Item          | Value                                     |
|---------------|-------------------------------------------|
| Subjects      | `VALID_SUBJECTS` (n = 18)                 |
| Channels      | Fz, Pz, Oz                                |
| References    | hEOG, vEOG                                |
| CV            | 5-fold GroupKFold over 40 trials          |
| Preprocessing | 4–45 Hz band-pass on EOG, per-fold z-scoring on train |

**Model.** Per-sample linear regression:

```
EEG(t) ≈ β0 + β1 · hEOG(t) + β2 · vEOG(t)
```

**Output.** `EOG-Regression/Results/sub{XX}_ch{CH}.pkl` and
`EOG-Regression/Results/EOG2EEG_summary.json`.

**Notes.** No GPU is required. The pickle schema matches
`run_kfold_pipeline`, so the same loader
(`load_experiment_result`) can read it. The filename pattern is
identical to the video model; the runner must be pointed at its own
`RESULTS_DIR`.

**Paper.** Section 4.5 and the discussion of ocular contamination in
the Introduction and 3.4.2.

---

### `06_null_models.ipynb` — Trivial and shuffled null models

**Question.** What does the same pipeline give when video and EEG are
unrelated? What is the floor for RMSE and correlation?

**Scope.**

| Item          | Value                                     |
|---------------|-------------------------------------------|
| Subjects      | `VALID_SUBJECTS` (n = 18)                 |
| Channel       | Fz                                        |
| Time window   | 2 s                                       |
| CV            | 5-fold GroupKFold over 40 trials          |
| Overlap       | Train 50 %, Test 0 % (shuffled model)     |
| Epochs        | 10 (shuffled model only)                  |
| Optimizer     | Adam (lr = 1e-4, wd = 1e-5)               |
| LR schedule   | StepLR, γ = 0.1 every 5 epochs            |
| Shuffle seed  | `SHUFFLE_SEED = SEED` (= 1338)            |

**Part 1 — Trivial baselines (no training).** Under the same CV
splits, three predictors are computed on the test folds:

| Predictor           | Definition                                    |
|---------------------|-----------------------------------------------|
| `const_zero`        | Zero everywhere (RMSE = 1 by construction)    |
| `const_train_mean`  | Training-set scalar mean                       |
| `mean_waveform`     | Training-set mean waveform                     |

Only the EEG signal is loaded (no video decoding).

**Part 2 — Shuffled pairing.** Trains the full pipeline with video and
EEG paired across *different* trials of the same subject. Everything
else — network, loss, optimizer, epochs, folds, overlap, seeds — is
identical to the real model. The permutation is fixed-point-free so
that no trial is paired with itself.

**Output.**

- `Null-Models/Trivial/Results/sub{XX}_chFz.pkl`
- `Null-Models/Shuffled/Results/sub{XX}_chFz.pkl`

**Paper.** Section 4.5 and the discussion of the null-model reference
points in response to Reviewer 2, Comment 2.

---

## Output layout

```
DATA_ROOT/
└── Experiments/
    └── Cross-Validation/
        ├── Loss-Functions/
        │   ├── Results/    sub02_chFz_MSE.pkl, sub02_chFz_NegPearson.pkl
        │   └── Models/     model-E131_sub2_chFz_tw1_fold{k}.pt …
        ├── Segment-Lengths/
        │   ├── Results/    sub02_chFz_tw0.5.pkl, sub02_chFz_tw2.pkl, sub02_chFz_tw3.pkl
        │   └── Models/     model-E{141,151,156}_sub2_chFz_tw{T}_fold{k}.pt …
        ├── Between-Participant-Variability/
        │   ├── Results/    sub{01..22}_chFz.pkl
        │   └── Models/     model-E{200,210,..}_sub{XX}_chFz_tw2_fold{k}.pt …
        ├── Channel-Specificity/
        │   ├── Results/    sub{XX}_chPz.pkl, sub{XX}_chOz.pkl
        │   └── Models/     model-E{300,400}_sub{XX}_ch{CH}_tw2_fold{k}.pt …
        ├── EOG-Regression/
        │   └── Results/    sub{XX}_ch{CH}.pkl, EOG2EEG_summary.json
        └── Null-Models/
            ├── Shuffled/
            │   ├── Results/ sub{XX}_chFz.pkl
            │   └── Models/  model-E5{XX}_sub{XX}_chFz_tw2_fold{k}.pt …
            └── Trivial/
                └── Results/ sub{XX}_chFz.pkl
```

Every `.pkl` under `Results/` follows the same schema and loads through
`face2eeg.core.results.load_experiment_result`. The trivial baselines
store three predictions in each fold and are read via `pred_key=`.

---

## Relationship to the paper

| Notebook                        | Paper section               |
|---------------------------------|-----------------------------|
| `01_loss_function`              | 4.3 (Tables 2–4, Figs. 7–9) |
| `02_segment_lengths`            | 4.4 (Tables 5–6, Figs. 10–11) |
| `03_between_participant_variability` | 4.5 (Table 7, Fig. 12) |
| `04_channel_specificity`        | 4.5 + Reviewer 1, Comment 1 |
| `05_eog_eeg`                    | 4.5 + Reviewer 1, Comment 1 |
| `06_null_models`                | 4.5 + Reviewer 2, Comment 2 |

The corresponding figures are generated by the analysis notebooks in
`face2eeg/analysis/`, not here.

---

## Recommended execution order

If you are running the whole project from scratch:

1. `05_eog_eeg` — no GPU; produces the EOG baseline first
2. `06_null_models` — Part 1 (trivial) is GPU-free, Part 2 (shuffled) needs GPU
3. `01_loss_function` — single subject, fast sanity check
4. `02_segment_lengths` — single subject, four runs
5. `03_between_participant_variability` — main experiment, 18 subjects
6. `04_channel_specificity` — 18 subjects × 2 channels

Notebooks 1–4 are cheap; running them first confirms that the
environment, data paths, and checkpoints are wired correctly before
launching the long runs.

If a run is interrupted, just re-execute the cell — the file-existence
guard skips everything already done.

---

## Design conventions

| Rule                                   | Why                                                        |
|----------------------------------------|------------------------------------------------------------|
| Notebooks are drivers, not logic       | All behavior lives in `face2eeg.core.experiments`          |
| Every notebook is idempotent           | Re-runs resume; no wasted GPU time                         |
| Hyperparameters match across notebooks | Segment length 2 s, MSE, 10 epochs, Train 50 % / Test 0 %  |
| Exp starts are unique per run          | Prevents checkpoint collisions across experiments          |
| RAM is cleared between subjects        | `clear_video_cache()` + `clear_signals_cache()`            |
| CUDA cache is cleared after each run   | `torch.cuda.empty_cache()`                                  |
| `pred_key` distinguishes baselines     | Trivial predictions coexist in one pickle with the main run |
| The model file is overwritten per epoch | Only the last epoch is kept on disk; the loss history is in the pickle |

---

## Related documentation

- [`../README.md`](../README.md) — project overview
- [`../face2eeg/core/experiments/README.md`](../face2eeg/core/experiments/README.md) — the runners these notebooks call
- [`../face2eeg/core/models/README.md`](../face2eeg/core/models/README.md) — PhysNet-3DCNN-ED and losses
- [`../face2eeg/core/training.py`](../face2eeg/core/training.py) — training loop and checkpoint format
- [`../face2eeg/core/results.py`](../face2eeg/core/results.py) — `ExperimentResult` container and loaders
- [`../analysis/README.md`](../analysis/README.md) — notebooks that consume these pickles