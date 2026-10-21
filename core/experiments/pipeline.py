# face2eeg/core/experiments/pipeline.py
"""
Main cross-validation pipeline.

Runs n_splits-fold GroupKFold over the 40 trials of one subject
and one EEG channel, training a fresh model per fold.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from ..config     import DEAP_EEG_CHANNELS
from ..data       import fit_scaler, load_participant_segments, get_cv_splits
from ..models     import PhysNet_3DCNN_ED
from ..training   import train_model
from ..evaluation import run_model_and_predict


def run_kfold_pipeline(
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
    frame_size=(32, 32),
    video_fr=50,
    signal_fs=128,
    y_type='RawEEG',
    # ---------------- data generation (train-specific) ----------------
    train_overlap=False,
    train_overlap_ratio=0.0,
    train_shuffle_output=True,
    train_signal_normalize=True,
    # ---------------- data generation (test-specific) ----------------
    test_overlap=False,
    test_overlap_ratio=0.0,
    test_shuffle_output=False,
    test_signal_normalize=True,
    # ---------------- null model: shuffled pairing ----------------
    shuffle_pairing=False,        
    shuffle_seed=None,            
    # ---------------- model ----------------
    video_length=None,   # None -> X_train.shape[2]
    signal_length=None,  # None -> y_train.shape[1]
    # ---------------- train_model ----------------
    epochs=10,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=1e-5,
    loss_fn=None,                # None -> nn.MSELoss()
    scheduler_step_size=5,
    scheduler_gamma=0.1,
    # ---------------- prediction (run_model_and_predict) ----------------
    predict_batch_size=16,
    # ---------------- output ----------------
    model_exp_start=200,
    save_dir=None,
    device=None,
    channel_dict=None,
    return_history=True,
    verbose=True,
    root_dir='.'
):
    """
    Run n_splits-fold GroupKFold cross-validation for one subject, one channel.

    Notes
    -----
    - `load_participant_segments` in this project does NOT expose
      `remove_pretrial` / `resample` / `new_sample_rate` — they are
      hardcoded inside it (`remove_pretrial=True, resample=False`).
      Only `fit_scaler` accepts them, exposed here as `scaler_*`.
    - optimizer / scheduler are created fresh inside `train_model` at
      each fold (passing them from outside would leak state across folds).
    - `resume_from` is intentionally not exposed: each fold starts from
      scratch.
    - `X_test` is used as the validation set inside `train_model`,
      matching the original notebook pattern.

    - `shuffle_pairing`: (bool) if True, both training and test EEG
        lists are permuted before concatenation, breaking the video-EEG
        trial correspondence in every phase of the pipeline.
        Videos keep their original order; signals are shuffled.
        Used for null models (Reviewer 2, Comment 2).
    - `shuffle_seed`: (int/None) RNG seed for the permutation. When
        `shuffle_pairing=True`, this must be an int for reproducibility;
        each fold uses `shuffle_seed + fold_idx` for train and
        `shuffle_seed + 1000 + fold_idx` for test.

    - `root_dir`: (str) root directory containing physiological data and facial videos
        
    """
    # ============================================================
    # Defaults & validation
    # ============================================================
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if channel_dict is None:
        channel_dict = DEAP_EEG_CHANNELS

    if channel_name not in channel_dict:
        raise KeyError(
            f"Channel '{channel_name}' not found. "
            f"Available: {list(channel_dict.keys())}"
        )

    if shuffle_pairing and shuffle_seed is None:
        raise ValueError(
            "shuffle_pairing=True requires a fixed shuffle_seed "
            "for reproducible null-model results."
        )

    if save_dir is None:
        raise ValueError("save_dir must be provided.")
    os.makedirs(save_dir, exist_ok=True)

    if loss_fn is None:
        loss_fn = nn.MSELoss()

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
            'scaler_method':        scaler_method,
            'scaler_feature_range': scaler_feature_range,
            'frame_size':           frame_size,
            'video_fr':             video_fr,
            'signal_fs':            signal_fs,
            'y_type':               y_type,
            'train_overlap':        train_overlap,
            'train_overlap_ratio':  train_overlap_ratio,
            'train_shuffle_output': train_shuffle_output,
            'test_overlap':         test_overlap,
            'test_overlap_ratio':   test_overlap_ratio,
            'test_shuffle_output':  test_shuffle_output,
            'shuffle_pairing':      shuffle_pairing,        
            'shuffle_seed':         shuffle_seed,    
            'epochs':               epochs,
            'batch_size':           batch_size,
            'learning_rate':        learning_rate,
            'weight_decay':         weight_decay,
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
        # 2. Train data generation
        # --------------------------------------------------------
        X_train, y_train = load_participant_segments(
            participant_id=subject_id,
            experiments=tr,
            scaler=scaler,
            channel_name=channel_name,
            time_window=time_window,
            overlap=train_overlap,
            overlap_ratio=train_overlap_ratio,
            shuffle_output=train_shuffle_output,
            signal_normalize=train_signal_normalize,
            frame_size=frame_size,
            video_fr=video_fr,
            signal_fs=signal_fs,
            y_type=y_type,
            deap_eeg_channels=channel_dict,
            shuffle_pairing=shuffle_pairing,
            shuffle_seed=(shuffle_seed + fold_idx) if shuffle_seed is not None else None,
            root_dir=root_dir,
        )

        # --------------------------------------------------------
        # 3. Test data generation
        # --------------------------------------------------------
        X_test, y_test = load_participant_segments(
            participant_id=subject_id,
            experiments=te,
            scaler=scaler,
            channel_name=channel_name,
            time_window=time_window,
            overlap=test_overlap,
            overlap_ratio=test_overlap_ratio,
            shuffle_output=test_shuffle_output,
            signal_normalize=test_signal_normalize,
            frame_size=frame_size,
            video_fr=video_fr,
            signal_fs=signal_fs,
            y_type=y_type,
            deap_eeg_channels=channel_dict,
            shuffle_pairing=shuffle_pairing,
            shuffle_seed=(shuffle_seed + 1000 + fold_idx) if shuffle_seed is not None else None,
            root_dir=root_dir,
        )

        if verbose:
            print(f"  X_train {X_train.shape}  y_train {y_train.shape}")
            print(f"  X_test  {X_test.shape}   y_test  {y_test.shape}")

        # --------------------------------------------------------
        # 4. Fresh model
        # --------------------------------------------------------
        _video_length  = video_length  if video_length  is not None else X_train.shape[2]
        _signal_length = signal_length if signal_length is not None else y_train.shape[1]

        model = PhysNet_3DCNN_ED(_video_length, _signal_length).to(device)

        # --------------------------------------------------------
        # 5. Train (checkpoint overwritten every epoch)
        # --------------------------------------------------------
        parts = [
            f"E{model_exp_start + fold_idx}",
            f"sub{subject_id}",
            f"ch{channel_name}",
            f"tw{time_window}",
            f"fold{fold_idx}",
        ]

        file_name = f"model-{'_'.join(parts)}.pt"
        save_path = os.path.join(save_dir, file_name)
        
        history = train_model(
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=None, y_val=None,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_fn=loss_fn,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            model_save_path=save_path,
            verbose=verbose,
        )

        # --------------------------------------------------------
        # 6. Predict on test fold
        # --------------------------------------------------------
        y_pred = run_model_and_predict(
            model, X_test, device, batch_size=predict_batch_size
        )

        # --------------------------------------------------------
        # 7. Store
        # --------------------------------------------------------
        fold_result = {
            'y_test': y_test,
            'y_pred': y_pred,
            'trials': np.asarray(te),
        }
        if return_history:
            fold_result['history'] = history
        results['folds'][fold_idx] = fold_result

        # --------------------------------------------------------
        # 8. Cleanup
        # --------------------------------------------------------
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return results