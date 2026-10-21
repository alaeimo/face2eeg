# face2eeg/core/data/__init__.py
"""
Data loading and preparation for DEAP.

Submodules
----------
cache         — module-level caches for CSV, .dat, and video files
mapper        — experiment_id → trial_id mapping (experiment2trial.csv)
video         — facial video loading and preprocessing
physio        — raw physiological (.dat) loading
scaler        — per-fold z-score / minmax scaler
participant   — joint loader (video + EEG + labels)
segmentation  — window segmentation of video and EEG
cv            — GroupKFold splits and visualization
"""

# --- caches (no internal deps) ---
from .cache import (
    clear_video_cache,
    clear_signals_cache,
)

# --- mapper (depends on cache) ---
from .mapper import get_exp2trial_mapper

# --- video (depends on mapper) ---
from .video import load_participant_facial_video

# --- physio (depends on cache) ---
from .physio import load_participant_physiological_data

# --- scaler (depends on physio) ---
from .scaler import fit_scaler

# --- participant (depends on video + physio) ---
from .participant import load_participant_data

# --- segmentation (depends on participant) ---
from .segmentation import (
    segment_video_and_signal,
    segment_signal_only,
    load_participant_segments,
    load_participant_signal_segments,
)

# --- CV splits (independent) ---
from .cv import (
    get_cv_splits,
    plot_cv_indices,
)

from .peripheral import load_eeg_eog_emg

__all__ = [
    # cache
    'clear_video_cache',
    'clear_signals_cache',
    # mapper
    'get_exp2trial_mapper',
    # peripheral
    'load_eeg_eog_emg',
    # video
    'load_participant_facial_video',
    # physio
    'load_participant_physiological_data',
    # scaler
    'fit_scaler',
    # participant
    'load_participant_data',
    # segmentation
    'segment_video_and_signal',
    'segment_signal_only',
    'load_participant_segments',
    'load_participant_signal_segments',
    # cv
    'get_cv_splits',
    'plot_cv_indices',

]