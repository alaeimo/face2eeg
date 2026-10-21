# face2eeg/core/data/peripheral.py
"""
Load peripheral channels (EOG, EMG) from DEAP .dat files.
"""
import numpy as np

from ..config  import PERIPHERAL_CHANNELS
from ..signals import bandpass_freq_range
from .physio   import load_participant_physiological_data


# EEG channel name -> index in DEAP
_EEG_INDEX = None  # populate lazily to avoid circular import


def _get_eeg_index(name):
    global _EEG_INDEX
    if _EEG_INDEX is None:
        from ..config import DEAP_EEG_CHANNELS
        _EEG_INDEX = DEAP_EEG_CHANNELS
    if name not in _EEG_INDEX:
        raise KeyError(f"Unknown EEG channel '{name}'.")
    return _EEG_INDEX[name]


def load_eeg_eog_emg(eeg_channel_name, participant_id, experiment_ids,
                     fs=128.0, filter_band=(4.0, 45.0),
                     root_dir=None):
    """
    Load one EEG channel + hEOG + vEOG + zEMG for the given trials.

    Returns
    -------
    dict : {'eeg', 'heog', 'veog', 'zemg'}
        each (n_trials, 7680), all band-passed to `filter_band`.
    """

    eeg_idx  = _get_eeg_index(eeg_channel_name)
    heog_idx = PERIPHERAL_CHANNELS['hEOG']
    veog_idx = PERIPHERAL_CHANNELS['vEOG']
    zemg_idx = PERIPHERAL_CHANNELS['zEMG']

    raw, _ = load_participant_physiological_data(
        participant_id,
        experiments=list(experiment_ids),
        channels=[eeg_idx, heog_idx, veog_idx, zemg_idx],
        root_dir=root_dir,
    )
    n_trials, _, n_samples = raw.shape

    eeg = raw[:, 0, :].astype(np.float64, copy=True)   # already filtered by DEAP

    heog = np.empty_like(eeg); veog = np.empty_like(eeg); zemg = np.empty_like(eeg)
    for i in range(n_trials):
        heog[i] = bandpass_freq_range(raw[i, 1], *filter_band, fs=fs)
        veog[i] = bandpass_freq_range(raw[i, 2], *filter_band, fs=fs)
        zemg[i] = bandpass_freq_range(raw[i, 3], *filter_band, fs=fs)

    return {'eeg': eeg, 'heog': heog, 'veog': veog, 'zemg': zemg}