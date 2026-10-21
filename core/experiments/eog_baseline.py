# face2eeg/core/experiments/eog_baseline.py
"""
EOG→EEG linear-regression baseline (Reviewer 2, Comment 1).

Per-fold, per-sample linear regression from peripheral channels
(hEOG, vEOG) to the target EEG channel. Uses the same GroupKFold
splits as the video model.
"""
import os
import json
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ..data.peripheral import load_eeg_eog_emg
from ..data import get_cv_splits

# ============================================================
# 1. Per-fold regression
# ============================================================

def eog_regression_fold(
    participant_id,
    eeg_channel_name,
    train_trials,
    test_trials,
    experiment_ids,
    references=('heog', 'veog'),
    fs=128.0,
    root_dir=None,
):
    """
    Fit EOG→EEG linear regression for ONE fold (no segmentation).

    Preprocessing mirrors `fit_scaler` from revised_utils:
      - StandardScaler (method='z-score')
      - fitted on TRAINING trials only
      - one global (mean, std) per channel

    Built inline to avoid re-reading the .dat file from disk.

    Model is per-sample:
        Fz(t) ≈ β0 + β1·hEOG(t) + β2·vEOG(t)

    Returns
    -------
    dict with:
        y_test          : (n_test, 7680)  z-scored Fz
        y_pred          : (n_test, 7680)  predicted Fz (same z-scale)
        trials          : (n_test,)
        model_coef      : (n_refs,)
        model_intercept : scalar
        scaler_X_mean   : (n_refs,)
        scaler_X_scale  : (n_refs,)
        scaler_y_mean   : scalar
        scaler_y_scale  : scalar
    """


    # 1. Load channels (4–45 Hz filtered) ----------
    data = load_eeg_eog_emg(eeg_channel_name, participant_id,
                             experiment_ids, fs=fs, root_dir=root_dir)
    y_all = data['eeg'].astype(np.float64)              # (n_trials, 7680)
    X_all = np.stack(
        [data[ref] for ref in references], axis=-1
    ).astype(np.float64)                                # (n_trials, 7680, n_refs)
    n_refs = X_all.shape[-1]

    # 2. Build training matrices ----------
    X_tr_flat = X_all[train_trials].reshape(-1, n_refs)   # (n_train*7680, n_refs)
    y_tr_flat = y_all[train_trials].reshape(-1)           # (n_train*7680,)

    # 3. Fit scalers (equivalent to fit_scaler, but inline) ----------
    # fit_scaler creates StandardScaler and does partial_fit on each trial.
    # This is mathematically identical to StandardScaler().fit() on the
    # concatenation of all training samples — no file I/O needed here.
    scaler_X = StandardScaler().fit(X_tr_flat)                     # (n_refs,), (n_refs,)
    scaler_y = StandardScaler().fit(y_tr_flat.reshape(-1, 1))      # scalar, scalar

    X_tr_flat_z = scaler_X.transform(X_tr_flat)
    y_tr_flat_z = scaler_y.transform(y_tr_flat.reshape(-1, 1)).reshape(-1)

    # ---------- 4. Fit plain linear regression ----------
    reg = LinearRegression().fit(X_tr_flat_z, y_tr_flat_z)

    # ---------- 5. Predict each test trial (full 60-s waveform) ----------
    y_test_out = np.empty((len(test_trials), y_all.shape[1]), dtype=np.float64)
    y_pred_out = np.empty_like(y_test_out)

    for i, te in enumerate(test_trials):
        x_te = X_all[te]                                  # (7680, n_refs)
        y_te = y_all[te]                                  # (7680,)

        x_te_z = scaler_X.transform(x_te)
        y_pred_out[i] = reg.predict(x_te_z)

        y_te_z = scaler_y.transform(y_te.reshape(-1, 1)).reshape(-1)
        y_test_out[i] = y_te_z

    return {
        'y_test':          y_test_out,                    # (n_test, 7680)
        'y_pred':          y_pred_out,                    # (n_test, 7680)
        'trials':          np.asarray(test_trials),
        'model_coef':      reg.coef_.copy(),              # (n_refs,)
        'model_intercept': float(reg.intercept_),
        'scaler_X_mean':   scaler_X.mean_.copy(),         # (n_refs,)
        'scaler_X_scale':  scaler_X.scale_.copy(),        # (n_refs,)
        'scaler_y_mean':   float(scaler_y.mean_[0]),
        'scaler_y_scale':  float(scaler_y.scale_[0]),
        'references':references,                          # ('heog', 'veog')
    }

# ============================================================
# 2. Full CV runner for one (subject, channel)
# ============================================================
def run_eog_regression(
    subject_id,
    channel_name='Fz',
    n_points=40,
    n_splits=5,
    references=('heog', 'veog'),
    fs=128.0,
    save_dir=None,
    json_filename='EOG2EEG_summary.json',
    root_dir=None,
    verbose=True,
):
    """
    Train EOG→EEG regression for one (subject, channel) across `n_splits`
    GroupKFold folds.

    The output pickle has the SAME structure as `run_kfold_pipeline`:

        {
            'subject':     int,
            'channel':     str,
            'time_window': 60,          # whole trial, no segmentation
            'n_splits':    int,
            'folds': {
                fold_idx: {
                    'y_test':          (n_test, 7680),
                    'y_pred':          (n_test, 7680),
                    'trials':          (n_test,),
                    'history':         None,          # placeholder for schema parity
                    'model_coef':      (n_refs,),
                    'model_intercept': float,
                    'scaler_X_mean':   (n_refs,),
                    'scaler_X_scale':  (n_refs,),
                    'scaler_y_mean':   float,
                    'scaler_y_scale':  float,
                },
                ...
            },
            'config': {...},
        }

    and can be loaded with `face2eeg.core.results.load_experiment_result`
    or `face2eeg.analysis.load_result`.

    IMPORTANT
    ---------
    The filename is `sub{XX}_ch{CH}.pkl`, identical to `run_kfold_pipeline`.
    You MUST save to a dedicated folder (e.g. `EOG-Regression/Results/`)
    to avoid overwriting the video-model results.

    Parameters
    ----------
    subject_id : int
    channel_name : str
    n_points : int
    n_splits : int
    references : tuple of str
    fs : float
    save_dir : str
        Directory to save the .pkl. Required.
    json_filename : str
        Name of the shared JSON summary inside `save_dir`.
    root_dir : str or None
        Root of the DEAP data folder.
    verbose : bool

    Returns
    -------
    results : dict
        The pickle payload described above.
    """

    if save_dir is None:
        raise ValueError("save_dir must be provided.")
    os.makedirs(save_dir, exist_ok=True)

    training_sets, testing_sets, _ = get_cv_splits(n_points, n_splits)

    results = {
        'subject':     subject_id,
        'channel':     channel_name,
        'time_window': 60,          # whole trial, no segmentation
        'n_splits':    n_splits,
        'folds':       {},
        'config': {
            'method':       'LinearRegression',
            'estimator':    'sklearn.linear_model.LinearRegression',
            'segmentation': False,
            'zscore':       'per-fold, train statistics only',
            'references':   list(references),
            'fs':           float(fs),
            # No 'variant' key — the folder itself distinguishes EOG from NN.
        },
    }

    if verbose:
        print(f"\n=== EOG regression: S{subject_id:02d} | {channel_name} ===")

    for fold_idx, (tr, te) in enumerate(zip(training_sets, testing_sets)):
        fold_result = eog_regression_fold(
            participant_id=subject_id,
            eeg_channel_name=channel_name,
            train_trials=tr,
            test_trials=te,
            experiment_ids=list(range(n_points)),
            references=references,
            fs=fs,
            root_dir=root_dir,
        )
        # Add 'history' placeholder for schema parity with run_kfold_pipeline
        fold_result['history'] = None
        results['folds'][fold_idx] = fold_result

        if verbose:
            print(f"  fold {fold_idx}: train={len(tr)} | test={len(te)}")

    # --------------------------------------------------------
    # Save binary (same filename as run_kfold_pipeline)
    # --------------------------------------------------------
    base     = f"sub{subject_id:02d}_ch{channel_name}.pkl"
    pkl_path = os.path.join(save_dir, base)
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    if verbose:
        print(f"  saved: {pkl_path}")

    # --------------------------------------------------------
    # Update shared JSON summary
    # --------------------------------------------------------
    json_path = os.path.join(save_dir, json_filename)
    if os.path.isfile(json_path):
        with open(json_path, 'r') as f:
            summary = json.load(f)
    else:
        summary = {
            'estimator':    'sklearn.linear_model.LinearRegression',
            'segmentation': False,
            'zscore':       'per-fold, train statistics only',
            'references':   list(references),
            'fs':           float(fs),
            'experiments':  {},
        }

    exp_name = f'EOG2EEG_LinReg_{channel_name}'
    summary['experiments'].setdefault(exp_name, {'models': {}})
    summary['experiments'][exp_name]['models'][
        f'sub{subject_id:02d}_ch{channel_name}'
    ] = {
        'model_details': {
            'subject': subject_id,
            'channel': channel_name,
            'n_folds': n_splits,
        },
        'pkl_file':  base,
        'n_folds':   n_splits,
    }

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return results