# Face2EEG

> **Official implementation of the paper.** *Paper title, authors, and venue are being finalized and will be filled in once the manuscript is accepted.*

**Face2EEG** is a research codebase for predicting frontal-channel EEG from facial video on the DEAP dataset. It reproduces every experiment, figure, and table reported in the paper under a single, reproducible pipeline, with strict attention to group-aware cross-validation, trial-level statistics, and reviewer-requested control analyses.

---

## Table of contents

- [Overview](#overview)
- [Key features](#key-features)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Expected data layout](#expected-data-layout)
- [Quickstart](#quickstart)
- [End-to-end workflow](#end-to-end-workflow)
- [Experiments](#experiments)
- [Design principles](#design-principles)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contributors](#contributors)

---

## Overview

Facial video carries a wealth of physiological information: pulse, movement, ocular activity, and small expressive changes. **Face2EEG** asks how much of a frontal EEG waveform can be recovered from a 60 s facial video recorded *simultaneously* with EEG, and how much of that recovery is genuinely cortical rather than driven by residual ocular or movement-related coupling.

The pipeline:

- ingests **DEAP** raw `.dat` files and cropped facial videos;
- trains a 3D convolutional encoder–decoder (`PhysNet_3DCNN_ED`, adapted from [Yu et al., 2019](https://github.com/ZitongYu/PhysNet/tree/master)) to regress EEG from video;
- evaluates under **5-fold intra-subject GroupKFold** with strict control of data leakage;
- reports **trial-level** inference (40 held-out trials per subject) and **population-level** inference (18 independent subjects);
- includes all reviewer-requested controls: loss-function comparison, segment-length scan, between-participant variability, channel specificity (Fz/Pz/Oz), EOG-only linear-regression baseline, and two families of null models (shuffled pairing and trivial predictors).

The paper's results are produced end-to-end by the notebooks in [`training/`](training/) and [`analysis/`](analysis/).

---

## Key features

- **Reproducible by design.** Fixed seed (`SEED = 1338`), idempotent notebooks, JSON manifests of every run, and a strict separation between training and analysis.
- **Group-aware CV.** `GroupKFold` over 40 trials per subject; the same trial partition is reused by every experiment so that trial *i* in method A and trial *i* in method B refer to the same physical trial.
- **Trial-level statistics.** Fisher-z aggregation of Pearson correlations, percentile-bootstrap CIs, sign-flip permutation tests (Phipson & Smyth correction), Wilcoxon and t-test fallbacks, Cohen's *d_z*, Hedges' *g*, Kendall's *W*, and partial η².
- **Reviewer-response baselines.** Shuffled-pairing null model, three trivial baselines (zero output, train-mean constant, train-mean waveform), EOG-only linear regression, and posterior-channel controls (Pz, Oz) — all evaluated under identical folds.
- **Nature/IEEE-quality figures.** Publication-ready PDFs at 400 dpi with vector text and consistent styling across all panels.
- **Full separation of concerns.** `core/` contains reusable logic, `training/` runs the models, `analysis/` produces figures and tables. Notebooks are drivers, not implementations.
- **Zero hidden state.** Caches are explicit and clearable; scalers are always fit on training folds only.

---

## Repository structure

```
face2eeg/
├── README.md                    ← this file
├── requirements.txt
├── .gitignore
├── __init__.py
│
├── core/                        reusable library — see core/README.md
│   ├── config.py                global constants
│   ├── signals.py               band-pass filters
│   ├── stats.py                 trial-level statistics + effect sizes
│   ├── coupling.py              EEG ↔ EOG/EMG signal coupling
│   ├── results.py               ExperimentResult container + loaders
│   ├── evaluation.py            batched prediction + evaluation
│   ├── training.py              training loop with resume support
│   ├── plotting.py              every figure in the paper
│   │
│   ├── data/                    loading and preprocessing — see data/README.md
│   ├── experiments/             end-to-end runners — see experiments/README.md
│   └── models/                  PhysNet-3DCNN-ED and losses — see models/README.md
│
├── training/                    six training notebooks — see training/README.md
│   ├── 01_loss_function.ipynb
│   ├── 02_segment_lengths.ipynb
│   ├── 03_between_participant_variability.ipynb
│   ├── 04_channel_specificity.ipynb
│   ├── 05_eog_eeg.ipynb
│   └── 06_null_models.ipynb
│
└── analysis/                    four analysis notebooks — see analysis/README.md
    ├── 01_loss_function.ipynb
    ├── 02_segment_lengths.ipynb
    ├── 03_null_models.ipynb
    └── 04_eog_eeg.ipynb
```

Every subfolder has its own README with module-level documentation:

- [`core/README.md`](core/README.md)
- [`core/data/README.md`](core/data/README.md)
- [`core/experiments/README.md`](core/experiments/README.md)
- [`core/models/README.md`](core/models/README.md)
- [`training/README.md`](training/README.md)
- [`analysis/README.md`](analysis/README.md)

---

## Installation

### Requirements

Python 3.10+ is recommended. Install the project dependencies with:

```bash
pip install -r requirements.txt
```

The main third-party dependencies are:

| Package            | Purpose                                              |
|--------------------|------------------------------------------------------|
| `numpy`, `pandas`  | numerical and tabular operations                     |
| `scipy`            | signal processing, Welch PSD, coherence, statistics  |
| `scikit-learn`     | scalers, GroupKFold, linear regression               |
| `statsmodels`      | FDR / Holm correction helpers                        |
| `torch`            | model training and inference                         |
| `opencv-python`    | video frame resizing and encoding                    |
| `pims`             | lazy video frame reading                             |
| `dlib`             | face detection (used once during preprocessing)      |
| `matplotlib`, `seaborn` | publication-quality figures                     |

A CUDA-capable GPU is strongly recommended for the training notebooks (notebooks `01`, `02`, `03`, `04`, `06`). Notebook `05` (EOG regression) and the trivial-baseline part of `06` run on CPU.

### Face-detection backend

`dlib` requires a C++ compiler and CMake. On Ubuntu: `sudo apt install cmake build-essential`. On macOS: `brew install cmake`. On Windows: install Visual Studio Build Tools with the C++ workload.

---

## Expected data layout

The pipeline expects the following directory structure under a single
`DATA_ROOT`:

```
DATA_ROOT/
├── experiment2trial.csv                # DEAP trial mapping
├── Physiological_data/
│   ├── s01.dat
│   ├── s02.dat
│   └── ...
├── Face_videos/
│   ├── s01_trial01_cropped.avi
│   ├── s01_trial02_cropped.avi
│   └── ...
└── Experiments/
    └── Cross-Validation/
        ├── Loss-Functions/
        ├── Segment-Lengths/
        ├── Between-Participant-Variability/
        ├── Channel-Specificity/
        ├── EOG-Regression/
        └── Null-Models/
```

Raw facial videos are preprocessed into `Face_videos/` by a one-off
batch script:

```bash
python -m face2eeg.core.data.preprocess_faces
```

This performs face detection, cropping, and resizing exactly once per
trial. After this step, every downstream notebook runs on the cropped
videos and never touches the raw data again.

Experiment outputs (results `.pkl`, model checkpoints, figures, and
tables) are written into the corresponding subfolders of
`Experiments/Cross-Validation/`.

---

## Quickstart

Clone the repository, install dependencies, and run a single-subject
sanity check:

```bash
git clone <repo-url>
cd face2eeg
pip install -r requirements.txt
```

Then open `training/01_loss_function.ipynb` and execute all cells.
This trains two models (MSE and Neg-Pearson) on Subject 2 for one
fold configuration, and writes:

- `Loss-Functions/Results/sub02_chFz_MSE.pkl`
- `Loss-Functions/Results/sub02_chFz_NegPearson.pkl`

Once those files exist, `analysis/01_loss_function.ipynb` regenerates
every figure and table for Section 4.3 of the paper.

---

## End-to-end workflow

The pipeline has four stages. They should be run in this order:

**1. Preprocess facial videos (once per machine).**

```bash
python -m face2eeg.core.data.preprocess_faces
```

Runs face detection on raw `.avi` files and writes cropped 256×256
videos to `Face_videos/`.

**2. Train the models.** Each notebook in `training/` produces one
family of pickles under `Experiments/Cross-Validation/<Experiment>/`:

| Notebook                              | Produces                                            |
|---------------------------------------|-----------------------------------------------------|
| `01_loss_function.ipynb`              | MSE and Neg-Pearson models on Subject 2 (T = 1 s)   |
| `02_segment_lengths.ipynb`            | Subject 2 across T ∈ {0.5, 1, 2, 3} s (MSE)         |
| `03_between_participant_variability`  | 18-subject Fz model (T = 2 s)                       |
| `04_channel_specificity.ipynb`        | 18-subject Pz and Oz models (T = 2 s)               |
| `05_eog_eeg.ipynb`                    | EOG→EEG linear-regression baseline for Fz/Pz/Oz     |
| `06_null_models.ipynb`                | Shuffled null model and three trivial baselines     |

Every notebook is idempotent: it skips work whose output already
exists. Re-running after an interruption picks up where it left off.

**3. Analyze the results.** The notebooks in `analysis/` read the
pickles produced in step 2 and write figures and tables into
`Results/Analysis-Figures/` and `Results/Analysis-Tables/` under each
experiment.

**4. Reproduce the paper.** The figures referenced in the manuscript
correspond one-to-one with files produced by the analysis notebooks:

| Paper element          | Generated by                                  |
|------------------------|-----------------------------------------------|
| Section 4.3 (Tables 2–4, Figs. 7–9) | `analysis/01_loss_function.ipynb`   |
| Section 4.4 (Tables 5–6, Figs. 10–11) | `analysis/02_segment_lengths.ipynb` |
| Section 4.5 (Table 7, Fig. 12) | `analysis/03_null_models.ipynb`        |
| Section 4.5 and Reviewer 1, Comment 1 (Tables 1–2, main figure, and supplementary tables) | `analysis/04_eog_eeg.ipynb` |
| Reviewer 2, Comment 2 (null models) | `analysis/03_null_models.ipynb` |

---

## Experiments

The paper reports six experiments, all implemented in this repository:

| # | Experiment                          | Purpose                                                                 |
|---|-------------------------------------|-------------------------------------------------------------------------|
| 1 | **Loss-function comparison**        | Compare MSE and negative Pearson correlation under identical folds (Subject 2, T = 1 s). |
| 2 | **Segment-length scan**             | Sweep T ∈ {0.5, 1, 2, 3} s (Subject 2, MSE) and select the best T for downstream experiments. |
| 3 | **Between-participant variability** | Train one model per subject (n = 18) at T = 2 s on Fz and characterise inter-individual differences. |
| 4 | **Channel specificity**             | Train the same model on Pz and Oz to test whether performance is frontal-specific or spatially general. |
| 5 | **EOG→EEG linear baseline**         | Fit a per-sample linear regression from hEOG and vEOG to the EEG target under the same folds. Provides a reference for how much of the reported performance can be explained by ocular activity. |
| 6 | **Null models**                     | Two references: (i) a shuffled-pairing model that trains with deliberately broken video–EEG correspondence; (ii) three trivial baselines (zero output, training-set mean, training-set mean waveform). Establishes the floor for RMSE and correlation. |

For a detailed description of each experiment, see
[`training/README.md`](training/README.md) and
[`analysis/README.md`](analysis/README.md).

---

## Design principles

The codebase is organized around the following rules. They are
enforced throughout `core/`, `training/`, and `analysis/`.

1. **No data leakage.** Scalers are always fit on training trials only
   and applied to test trials with those same statistics. Every model
   is trained under `GroupKFold` over 40 trials so that trial *i*
   appears in exactly one test fold.
2. **Trial is the unit of inference.** Metrics are computed
   independently per 60 s trial and aggregated across trials within a
   subject. Population-level inference is performed on the 18
   independent participants, not on the 720 correlated trials.
3. **Fisher z for correlation.** Pearson *r* is always Fisher-z
   transformed before averaging or testing. Coherence and RMSE are
   aggregated on their native scales.
4. **Phipson & Smyth correction for permutations.** All sign-flip
   permutation p-values use `(extreme + 1) / (n_perm + 1)`, which is
   exact under the null and never returns zero.
5. **Multiple-comparison correction is explicit.** Holm-Bonferroni is
   used within each metric family; Benjamini-Hochberg is available when
   a family-level FDR is preferred.
6. **Notebooks are drivers, not implementations.** All logic lives in
   `face2eeg.core`. Notebooks configure paths, call runners, and save
   outputs. They are idempotent and can be re-run safely.
7. **Schema parity across experiments.** All experiment runners write a
   pickle with the same top-level structure (`subject`, `channel`,
   `time_window`, `n_splits`, `folds`, `config`), so a single loader
   reads every result.
8. **Publication-quality outputs by default.** All figures are PDF at
   400 dpi with vector text; all tables are written as CSV and LaTeX.

---

## Reproducibility

- **Fixed seed.** `SEED = 1338` is defined in `face2eeg/core/config.py`
  and propagated to NumPy, PyTorch, and every stochastic procedure.
- **Deterministic CV.** `get_cv_splits(n_points=40, n_splits=5)`
  produces the same fold assignment on every machine.
- **Deterministic null-model shuffling.** The shuffled-pairing null
  model uses `shuffle_seed = SEED` and applies a fixed-point-free
  permutation so that no trial is paired with itself.
- **Explicit checkpoints.** Every model checkpoint stores its epoch,
  optimizer, scheduler, and full history; the analysis notebooks read
  them without re-running training.
- **JSON manifests.** Each experiment writes a JSON manifest of every
  completed run, so that resuming a partially finished sweep is a
  one-liner.
- **Environment.** Tested with Python 3.10, PyTorch 2.x, NumPy 1.24+,
  SciPy 1.10+, scikit-learn 1.2+, pandas 1.5+.

A single-command reproduction of every paper figure is not provided,
because training 18 subjects on GPU takes several hours. Instead, each
analysis notebook regenerates its figures and tables directly from the
saved pickles and is safe to re-run at any time.

---

## Citation

*Citation information will be added once the paper is accepted. In the
meantime, please refer to this repository as the official
implementation.*

```bibtex
@misc{face2eeg,
  title  = {Face2EEG: Predicting EEG from Facial Video on the DEAP Dataset},
  author = {Alaei, Mohammad},
  year   = {2025},
  note   = {Official implementation. Paper details pending.},
  url    = {https://github.com/alaeimo/face2eeg}
}
```

---

## License

This project is released under the **MIT License**. See `LICENSE` for
details.

---

## Acknowledgments

- The DEAP dataset: Koelstra et al., *DEAP: A Database for Emotion
  Analysis Using Physiological Signals*, IEEE Transactions on Affective
  Computing, 2012.
- The PhysNet architecture: Yu, Li, & Zhao, *Remote
  Photoplethysmograph Signal Measurement from Facial Videos Using
  Spatio-Temporal Networks*, IJCAI 2019.
  Reference implementation: https://github.com/ZitongYu/PhysNet/tree/master
- The ShuffleNet / GroupKFold ecosystem provided by scikit-learn.
- The statistical community for the Phipson & Smyth (2010) correction
  to permutation p-values.

---

## Contributors

**Mohammad Alaei** — [alaeimo.ir](https://alaeimo.ir)

*For questions, bug reports, or collaboration, please open an issue or
reach out via the website above.*