# face2eeg/core/experiments/trivial_baselines.py
"""
Trivial null-model baselines for trial-level reference points.

Under the same GroupKFold splits used by the trained model, this
module computes three trivial predictors on each test fold without
any training:

  - const_zero       : predicts 0 everywhere (equivalent to predicting
                       the mean of z-scored targets; RMSE = 1 by
                       construction)
  - const_train_mean : predicts the scalar mean of the training set
  - mean_waveform    : predicts the training-set mean waveform
                       (per-position average)

These baselines establish the floor against which the trained
model's performance is compared (Reviewer 2, Comment 2).

Notes
-----
Only the EEG signal is loaded (no video decoding), using
`load_participant_signal_segments`. This keeps the runtime low
while producing y segments identical to the trained-model pipeline.
"""
import numpy as np

from ..config import DEAP_EEG_CHANNELS
from ..data   import (
    fit_scaler,
    load_participant_signal_segments,
    get_cv_splits,
)


def run_trivial_baselines(
    subject_id,
    channel_name='Fz',
    time_window=2,
    # ---------------- CV splits ----------------
    n_points=40,
    n_splits=5,
    # ---------------- scaler (fit_scaler) ----------------
    scaler_method='z-score',
    scaler_feature_range=(0, 1),
    scaler_remove_pretrial=True,
    scaler_resample=False,
    scaler_new_sample_rate=128,
    # ---------------- data generation (shared) ----------------
    signal_fs=128,
    y_type='RawEEG',
    signal_normalize=True,
    # ---------------- data generation (train-specific) ----------------
    train_overlap=False,
    train_overlap_ratio=0.0,
    # ---------------- data generation (test-specific) ----------------
    test_overlap=False,
    test_overlap_ratio=0.0,
    # ---------------- output ----------------
    channel_dict=None,
    verbose=True,
    root_dir='.'
):
    """
    Compute trivial predictors on the test folds of a subject.

    Parameters
    ----------
    subject_id : int
    channel_name : str
    time_window : int or float
    n_points : int
    n_splits : int
    scaler_method : str
    scaler_feature_range : tuple
    scaler_remove_pretrial : bool
    scaler_resample : bool
    scaler_new_sample_rate : int
    signal_fs : int
    y_type : str
    signal_normalize : bool
    train_overlap : bool Whether to use overlapping segments when computing the train-set statistics. Should match the pipeline's training setting.
    train_overlap_ratio : float Overlap ratio when train_overlap=True.
    test_overlap : bool
    test_overlap_ratio : float
    channel_dict : dict or None
    verbose : bool
    root_dir: (str) root directory containing physiological data and facial videos
    
    Returns
    -------
    results : dict
        {
            'subject':     int,
            'channel':     str,
            'time_window': int or float,
            'n_splits':    int,
            'folds': {
                fold_idx: {
                    'y_test':           np.ndarray,
                    'trials':           np.ndarray,
                    'const_zero':       np.ndarray,
                    'const_train_mean': np.ndarray,
                    'mean_waveform':    np.ndarray,
                    'y_train_mean':     float,
                },
                ...
            },
            'config': {...},
        }
    """
    # ============================================================
    # Defaults & validation
    # ============================================================
    if channel_dict is None:
        channel_dict = DEAP_EEG_CHANNELS

    if channel_name not in channel_dict:
        raise KeyError(
            f"Channel '{channel_name}' not found. "
            f"Available: {list(channel_dict.keys())}"
        )

    # ============================================================
    # CV splits
    # ============================================================
    training_sets, testing_sets, _ = get_cv_splits(
        n_points=n_points, n_splits=n_splits
    )

    results = {
        'subject':      subject_id,
        'channel':      channel_name,
        'time_window':  time_window,
        'n_splits':     n_splits,
        'folds':        {},
        'config': {
            'scaler_method':         scaler_method,
            'scaler_feature_range':  scaler_feature_range,
            'signal_fs':             signal_fs,
            'y_type':                y_type,
            'signal_normalize':      signal_normalize,
            'train_overlap':         train_overlap,           
            'train_overlap_ratio':   train_overlap_ratio,    
            'test_overlap':          test_overlap,
            'test_overlap_ratio':    test_overlap_ratio,
        },
    }

    # ============================================================
    # Fold loop
    # ============================================================
    for fold_idx, (tr, te) in enumerate(zip(training_sets, testing_sets)):
        if verbose:
            print(f"\n=== Fold {fold_idx + 1}/{n_splits} "
                  f"(train={len(tr)}, test={len(te)}) ===")

        # --------------------------------------------------------
        # 1. Fit scaler ONLY on training trials (no leakage)
        # --------------------------------------------------------
        scaler = fit_scaler(
            participants=[subject_id],
            experiments=tr,
            channels=[channel_dict[channel_name]],
            method=scaler_method,
            feature_range=scaler_feature_range,
            remove_pretrial=scaler_remove_pretrial,
            resample=scaler_resample,
            new_sample_rate=scaler_new_sample_rate,
            root_dir=root_dir,
        )

        # --------------------------------------------------------
        # 2. Train EEG segments (signal-only; no video)
        # --------------------------------------------------------
        y_train = load_participant_signal_segments(
            participant_id=subject_id,
            experiments=tr,
            scaler=scaler,
            channel_name=channel_name,
            time_window=time_window,
            overlap=train_overlap,
            overlap_ratio=train_overlap_ratio,
            shuffle_output=False,
            signal_normalize=signal_normalize,
            signal_fs=signal_fs,
            y_type=y_type,
            deap_eeg_channels=channel_dict,
            root_dir=root_dir,
        )

        # --------------------------------------------------------
        # 3. Test EEG segments (signal-only; no video)
        # --------------------------------------------------------
        y_test = load_participant_signal_segments(
            participant_id=subject_id,
            experiments=te,
            scaler=scaler,
            channel_name=channel_name,
            time_window=time_window,
            overlap=test_overlap,
            overlap_ratio=test_overlap_ratio,
            shuffle_output=False,
            signal_normalize=signal_normalize,
            signal_fs=signal_fs,
            y_type=y_type,
            deap_eeg_channels=channel_dict,
            root_dir=root_dir,
        )

        if verbose:
            print(f"  y_train {y_train.shape}   y_test {y_test.shape}")

        # --------------------------------------------------------
        # 4. Build trivial predictors
        # --------------------------------------------------------
        n_test = y_test.shape[0]

        train_mean_scalar   = float(np.mean(y_train))
        train_mean_waveform = y_train.mean(axis=0)

        const_zero       = np.zeros_like(y_test)
        const_train_mean = np.full_like(y_test, train_mean_scalar)
        mean_waveform    = np.tile(train_mean_waveform, (n_test, 1))

        if verbose:
            print(f"  train_mean = {train_mean_scalar:+.4f}")

        # --------------------------------------------------------
        # 5. Store
        # --------------------------------------------------------
        results['folds'][fold_idx] = {
            'y_test':            y_test,
            'trials':            np.asarray(te),
            'const_zero':        const_zero,
            'const_train_mean':  const_train_mean,
            'mean_waveform':     mean_waveform,
            'y_train_mean':      train_mean_scalar,
        }

    return results