# `face2eeg/core/data/README.md`


# `face2eeg.core.data` — Data loading & preparation

This subpackage contains all code that reads, preprocesses and segments
data from the DEAP dataset before it enters the training pipeline.
Nothing here trains a model — it only produces well-defined NumPy
arrays and scikit-learn scalers.

---

## Table of contents

- [Design principles](#design-principles)
- [Folder layout](#folder-layout)
- [Module reference](#module-reference)
- [Typical usage](#typical-usage)
- [Caching](#caching)
- [Dependencies](#dependencies)

---

## Design principles

1. **Separation of concerns** — every module does one thing:
   - `mapper.py` → experiment_id ↔ trial_id mapping
   - `physio.py` → raw `.dat` reading
   - `video.py` → face detection, cropping, frame loading
   - `scaler.py` → per-fold z-score / minmax fitting
   - `segmentation.py` → fixed-length windowing of video and EEG
   - `participant.py` → joint (video, EEG, label) triple
   - `cv.py` → GroupKFold split generation
2. **Caching at the I/O boundary** — raw files are read from disk once
   per process and stored in module-level caches (`cache.py`).
3. **No data leakage** — scalers are always fit on training folds only,
   the caller is responsible for passing the right `experiments` list.
4. **Reproducibility** — random operations (null-model shuffling) accept
   an explicit seed argument.

---

## Folder layout

```
core/data/
├── __init__.py            public API re-exports
├── cache.py               module-level caches
├── mapper.py              experiment2trial.csv loader
├── physio.py              raw .dat loader
├── peripheral.py          EOG / EMG channels loader
├── video.py               face detection, cropping, frame loading
├── preprocess_faces.py    one-off batch preprocessing CLI
├── scaler.py              per-fold scaler fitting
├── participant.py         joint video + EEG + label loader
├── segmentation.py        windowing of video and EEG
└── cv.py                  GroupKFold splits
```

---

## Module reference

### `cache.py`

Three plain Python dictionaries that memoize expensive I/O:

| Cache                    | Keyed by                                         |
|--------------------------|--------------------------------------------------|
| `_EXP2TRIAL_MAPPER_CACHE`| path to `experiment2trial.csv`                   |
| `_PHYSIO_CACHE`          | `(participant_id, root_dir, data_subdir)`        |
| `_VIDEO_CACHE`           | `(participant_id, experiment_id, resize, …)`     |

Public helpers:

```python
from face2eeg.core.data.cache import clear_video_cache, clear_signals_cache

clear_video_cache()      # empties _VIDEO_CACHE
clear_signals_cache()    # empties _PHYSIO_CACHE
```

Use these between subjects in long-running notebooks to keep RAM bounded.

---

### `mapper.py`

```python
trials, labels = get_exp2trial_mapper(path_to_csv)
```

Reads the DEAP `experiment2trial.csv` file and returns two NumPy arrays:

| Array    | Shape         | Meaning                                                |
|----------|---------------|--------------------------------------------------------|
| `trials` | `(32, 40)`    | trial index (video number) for each participant/experiment |
| `labels` | `(32, 40, 4)` | `(valence, arousal, dominance, liking)` per trial      |

The mapping is 1-based on `participant_id` and 0-based on
`experiment_id`.

---

### `physio.py`

```python
signals, labels = load_participant_physiological_data(
    participant_id, experiments=None, channels=None,
    remove_pretrial=True, resample=False, new_sample_rate=128,
    root_dir='.', data_subdir='Physiological_data',
)
```

Reads `sXX.dat` for one participant and returns:

| Output   | Shape                       | Notes                                        |
|----------|-----------------------------|----------------------------------------------|
| signals  | `(n_trials, n_ch, 7680)`    | 60 s @ 128 Hz, 3 s baseline removed          |
| labels   | `(n_trials, 4)`             | valence, arousal, dominance, liking          |

Arguments:

- `experiments` — list of experiment ids in `[0, 40)` (or `None` for all)
- `channels`    — list of channel indices (or `None` for all 40)
- `remove_pretrial` — drop the first 3 s of each trial (default `True`)
- `resample`    — optional resampling to `new_sample_rate`

---

### `peripheral.py`

```python
data = load_eeg_eog_emg(
    eeg_channel_name, participant_id, experiment_ids,
    fs=128.0, filter_band=(4.0, 45.0), root_dir=None,
)
```

Loads **one EEG channel** plus the three peripheral channels needed for
EOG/EMG analyses:

| Key      | Shape            | Content                                     |
|----------|------------------|---------------------------------------------|
| `'eeg'`  | `(n_trials, 7680)` | selected EEG channel (already filtered by DEAP) |
| `'heog'` | `(n_trials, 7680)` | horizontal EOG, band-passed `(4–45 Hz)`     |
| `'veog'` | `(n_trials, 7680)` | vertical EOG, band-passed `(4–45 Hz)`       |
| `'zemg'` | `(n_trials, 7680)` | zygomaticus EMG, band-passed `(4–45 Hz)`    |

`eeg_channel_name` is a string key in `config.DEAP_EEG_CHANNELS`
(e.g. `'Fz'`, `'Pz'`, `'Oz'`).

---

### `video.py`

Three responsibilities:

#### 1. Face detection & cropping

```python
preprocess_video(
    video_path, out_path,
    frame_size=(256, 256), fps=50.0,
    detect_step=25, detect_max_frames=3000,
    skip_if_exists=True, verbose=False,
)
```

Runs `dlib.get_frontal_face_detector()` on a subsampled grid of frames,
averages the detected boxes into a single bounding box, crops every
frame with that box, resizes to `frame_size`, and writes a new `.avi`
with the DIVX codec.

This is a direct port of the original notebook pipeline — the
BGR↔RGB conversion in `crop_faces` is intentional and self-consistent
with `load_participant_facial_video`, which reads via `pims`.

Returns a small metadata dict (`n_frames`, `bbox`, `skipped`).

#### 2. Frame loading (used during training)

```python
video, label = load_participant_facial_video(
    participant_id, experiment_id,
    resize=True, frame_size=(32, 32),
    normalize=True, centering=True,
    root_dir='.', mapper_filename='experiment2trial.csv',
    video_subdir='Face_videos',
)
```

Returns:

| Output  | Shape                    | Notes                                     |
|---------|--------------------------|-------------------------------------------|
| `video` | `(3000, H, W, 3)`        | `H, W` from `frame_size`; float32          |
| `label` | `(4,)`                   | valence, arousal, dominance, liking        |

If the video file does not exist, returns `(None, label)`.

The per-frame processing pipeline is:

1. `cv2.resize` (bicubic) if `resize=True`
2. divide by 255 if `normalize=True`
3. subtract per-frame channel mean if `centering=True`

#### 3. Batch preprocessing CLI

`preprocess_faces.py` runs `preprocess_video` over every trial of every
subject listed in `config.VALID_SUBJECTS`. It edits two module-level
constants (`RAW_ROOT`, `OUT_ROOT`) that must point to the local layout
of the raw and processed video folders.

Run once per machine:

```bash
python -m face2eeg.core.data.preprocess_faces
```

---

### `scaler.py`

```python
scaler = fit_scaler(
    participants, experiments, channels,
    method='z-score', feature_range=(0, 1),
    remove_pretrial=True, resample=False, new_sample_rate=128,
    root_dir='.',
)
```

Returns a **fitted** `StandardScaler` (`method='z-score'`) or
`MinMaxScaler` (`method='minmax'`).

The scaler is fit with `partial_fit` on each trial of each participant,
which is mathematically identical to `fit` on the concatenated data but
keeps memory usage bounded. **Only pass training experiments** — never
fit on test data.

---

### `participant.py`

```python
video, signals, labels = load_participant_data(
    participant_id, experiment_id, channel, scaler,
    remove_pretrial=True, resample=False, new_sample_rate=128,
    resize=True, frame_size=(32, 32),
    video_normalize=True, video_centering=True, signal_normalize=True,
    root_dir='.',
)
```

Convenience wrapper that combines `load_participant_physiological_data`
and `load_participant_facial_video` for one experiment, applying the
scaler to the EEG signal if `signal_normalize=True`.

Returns:

- `video`   — `(3000, H, W, 3)` float32, or `None`
- `signals` — `(1, 7680)` z-scored, or `None`
- `labels`  — `(4,)` emotional ratings

---

### `segmentation.py`

Four public entry points:

#### `segment_video_and_signal(...)`
Core low-level windowing function. Slices a video and EEG into
fixed-length windows, with optional overlap, and supports three target
types (`RawEEG`, `PSD`, `FBP`).

| Output | Shape                              |
|--------|------------------------------------|
| `X`    | `(n_segments, 3, v_window, H, W)`  |
| `y`    | `(n_segments, s_window)` for RawEEG, or `(n_segments, n_freqs)` for PSD/FBP |

#### `load_participant_segments(...)`
High-level wrapper that:
1. Loads video + EEG for a list of experiments
2. Applies the scaler to the EEG
3. Optionally permutes the EEG list to break video↔EEG pairing
   (null-model mode: `shuffle_pairing=True`, `shuffle_seed=<int>`)
4. Concatenates and segments

#### `segment_signal_only(...)` and `load_participant_signal_segments(...)`
Same as above but only for the EEG side. Useful for trivial baselines
(e.g. constant predictor) that do not need the video.

---

### `cv.py`

```python
training_sets, testing_sets, cv = get_cv_splits(n_points=40, n_splits=5)
```

Returns `GroupKFold` splits over 40 trial indices. Each trial appears in
exactly one test fold, and no trial is split between train and test.

`plot_cv_indices` renders a small diagram of the splits, useful for
sanity-checking the CV assignment.

---

## Typical usage

### End-to-end single-subject training loop

```python
from face2eeg.core.data import (
    fit_scaler, load_participant_segments,
    get_cv_splits, clear_video_cache, clear_signals_cache,
)

tr, te, _ = get_cv_splits(n_points=40, n_splits=5)

for fold_idx, (train_ids, test_ids) in enumerate(zip(tr, te)):
    scaler = fit_scaler(
        participants=[subject_id],
        experiments=train_ids,
        channels=[18],                    # Fz
        method='z-score',
        root_dir=DATA_ROOT,
    )
    X_tr, y_tr = load_participant_segments(
        subject_id, train_ids, scaler, channel_name='Fz',
        time_window=2, frame_size=(32, 32),
        video_fr=50, signal_fs=128,
        root_dir=DATA_ROOT,
    )
    X_te, y_te = load_participant_segments(
        subject_id, test_ids, scaler, channel_name='Fz',
        time_window=2, frame_size=(32, 32),
        video_fr=50, signal_fs=128,
        root_dir=DATA_ROOT,
    )
    # ... train model on (X_tr, y_tr), evaluate on (X_te, y_te)

    clear_video_cache()
    clear_signals_cache()
```

### EOG–EEG coupling analyses

```python
from face2eeg.core.data import load_eeg_eog_emg

data = load_eeg_eog_emg('Fz', subject_id, experiment_ids=list(range(40)))
# data['eeg'], data['heog'], data['veog'], data['zemg']
```

### Null-model shuffle

```python
X, y = load_participant_segments(
    subject_id, train_ids, scaler, channel_name='Fz',
    time_window=2, shuffle_pairing=True, shuffle_seed=1338,
    root_dir=DATA_ROOT,
)
```

---

## Caching

Three independent caches prevent re-reading the same file twice within
one Python process:

- **Mapper CSV** → cached by file path
- **Physiological `.dat`** → cached by `(participant, root_dir, subdir)`
- **Video `.avi`** → cached by `(participant, experiment, frame params)`

They grow unbounded, so long-running notebooks should call the
`clear_*` helpers between subjects to release memory.

The raw `.dat` cache holds the full 40×40×8064 array (~52 MB per
subject), so keep an eye on RAM when processing many subjects.

---

## Dependencies

| Package        | Used for                                      |
|----------------|-----------------------------------------------|
| `numpy`        | array operations throughout                   |
| `scipy.signal` | resampling, Welch PSD                          |
| `scikit-learn` | `StandardScaler`, `MinMaxScaler`, `GroupKFold` |
| `pandas`       | reading `experiment2trial.csv`                 |
| `pims`         | video frame reading                            |
| `opencv-python`| frame resizing, video writing                  |
| `dlib`         | face detection                                 |

All are listed in `requirements.txt` at the repository root.

---

## Related modules

- `face2eeg.core.config` — paths, channel indices, frequency bands
- `face2eeg.core.experiments` — the training / evaluation pipelines that consume these arrays
- `face2eeg.core.data.video.preprocess_faces` — batch preprocessing CLI