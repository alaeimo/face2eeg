# face2eeg/core/config.py
"""
Global constants for the face2eeg package.

Contains only values that are fixed across all experiments:
sampling rates, channel indices, frequency bands, random seed,
and the local code root (for convenience, not for data access).

Data paths are NOT defined here — they are resolved at runtime
via `face2eeg.analysis.setup(data_root)` in the analysis notebooks,
and via explicit arguments in the training notebooks.
"""

# ============================================================
# 1. Reproducibility
# ============================================================
SEED = 1338


# ============================================================
# 2. Sampling rates 
# ============================================================
FS_EEG      = 128            # Hz — DEAP preprocessed sampling rate
FS_VIDEO    = 50             # fps — facial video frame rate
TRIAL_LEN   = 60             # seconds per trial (after removing 3s baseline)
SIGNAL_LEN  = FS_EEG * TRIAL_LEN    # 7680 — samples per trial


# ============================================================
# 3. DEAP EEG channels (0-based, Geneva order)
# ============================================================
# The order follows the DEAP dataset documentation. Channel `n`
# in the docs (1-indexed) has Python index `n - 1`.
DEAP_EEG_CHANNELS = {
    # Frontal
    'Fp1':  0,  'AF3':  1,  'F3':   2,  'F7':   3,
    'FC5':  4,  'FC1':  5,
    'Fp2': 16,  'AF4': 17,  'Fz':  18,  'F4':  19,
    'F8':  20,  'FC6': 21,  'FC2': 22,
    # Central
    'C3':   6,  'Cz':  23,  'C4':  24,
    # Temporal
    'T7':   7,  'T8':  25,
    # Centro-parietal
    'CP5':  8,  'CP1':  9,  'CP6': 26, 'CP2': 27,
    # Parietal
    'P3':  10,  'P7':  11,  'Pz':  15,  'P4': 28, 'P8': 29,
    # Parieto-occipital
    'PO3': 12,  'PO4': 30,
    # Occipital
    'O1':  13,  'Oz':  14,  'O2':  31,
}

# Frontal subset (used for Fz-only analyses)
FRONTAL_EEG_CHANNELS = {
    'Fp1': 0,  'AF3': 1,  'F3':  2,  'F7':  3,
    'FC5': 4,  'FC1': 5,
    'Fp2': 16, 'AF4': 17, 'Fz': 18,  'F4': 19,
    'F8':  20, 'FC6': 21, 'FC2': 22,
}


# ============================================================
# 4. Frequency bands (Hz) for spectral analysis
# ============================================================
FREQ_BANDS = {
    # Not present in DEAP (band-pass 4-45 Hz) — kept for completeness
    "delta":      (0.5, 4.0),
    
    # Fully representable from 50 fps video
    "theta":      (4.0, 8.0),
    "alpha":      (8.0, 13.0),
    
    # Representable portion of beta (capped at video Nyquist)
    "beta":       (13.0, 25.0),
    
    # Full beta, for supplementary descriptive only
    "beta_full":  (13.0, 30.0),
    
    # Above video Nyquist — supplementary only, NOT for inference
    "beta_high":  (25.0, 30.0),
    "gamma":      (30.0, 45.0),
}

# Bands used for inferential statistics (must be representable
# from 50-fps facial video, i.e. below 25 Hz Nyquist).
EEG_ANALYSIS_BANDS = ['theta', 'alpha', 'beta']


# ============================================================
# 5. DEAP-specific constants
# ============================================================
N_TRIALS_PER_SUBJECT = 40
N_SUBJECTS           = 22
N_EEG_CHANNELS       = 32
N_PERIPHERAL_CHANNELS = 8      # EOG, EMG, GSR, PPG, Resp, Plethysmo, Temp, Status

# --- Peripheral channel definitions (0-based) ---
# After the 32 EEG channels, DEAP stores 8 peripheral channels:
#   - 2 EOG (horizontal, vertical)
#   - 2 EMG (zygomaticus, trapezius)
#   - 4 other (GSR, Respiration, Plethysmograph, Temperature)
PERIPHERAL_CHANNELS = {
    'hEOG': 32,   # horizontal EOG
    'vEOG': 33,   # vertical EOG
    'zEMG': 34,   # zygomaticus major EMG
    'tEMG': 35,   # trapezius EMG
    'GSR':  36,   # galvanic skin response
    'Resp': 37,   # respiration
    'PPG':  38,   # plethysmograph
    'Temp': 39,   # skin temperature
}


# ============================================================
# 6. Subject exclusions / missing trials
# ============================================================
# For participants 3, 5, 11 and 14, one or several of the last trials
# are missing due to technical issues (i.e. the tape ran out).
# These subjects are excluded from the main analyses.

EXCLUDED_SUBJECTS = [3, 5, 11, 14]
VALID_SUBJECTS    = [1, 2, 4, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22]
N_SUBJECTS_VALID  = len(VALID_SUBJECTS)   # 18

# ============================================================
# 7. Participant metadata: glasses
# ============================================================
# Subjects who are wearing glasses during recording.
GLASSES_SUBJECTS = [3, 4, 5, 12, 14, 16, 20, 21, 22]

__all__ = [
    # reproducibility
    'SEED',
    # sampling
    'FS_EEG', 'FS_VIDEO', 'TRIAL_LEN', 'SIGNAL_LEN',
    # channels
    'DEAP_EEG_CHANNELS', 'FRONTAL_EEG_CHANNELS',
    # bands
    'FREQ_BANDS', 'EEG_ANALYSIS_BANDS',
    # DEAP
    'N_TRIALS_PER_SUBJECT', 'N_SUBJECTS', 'VALID_SUBJECTS',
    'N_SUBJECTS_VALID','N_EEG_CHANNELS', 'N_PERIPHERAL_CHANNELS', 
    'PERIPHERAL_CHANNELS', 'EXCLUDED_SUBJECTS', 'GLASSES_SUBJECTS'
]