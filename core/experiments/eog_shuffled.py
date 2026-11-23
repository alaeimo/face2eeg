# face2eeg/core/experiments/eog_shuffled.py
"""
Shuffled-trial EOG→EEG linear-regression control.

This experiment breaks the trial-level correspondence between hEOG/vEOG
and the target EEG signal within each participant and CV partition.

For each fold:
    Training: EEG target trial i <- EOG from a different training trial.
    Testing:  EEG target trial j <- EOG from a different held-out trial.

The original EEG target trials are retained for prediction and evaluation
in the separate analysis workflow.

Preprocessing:
    - DEAP preprocessed Python (.dat) arrays
    - hEOG and vEOG
    - 4–45 Hz, fourth-order Butterworth filtering
    - Forward-backward zero-phase filtering
    - Fold-specific StandardScaler fitted on training trials only

No neural network or evaluation metrics are calculated.
"""

import os
import json
import pickle

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ..data.peripheral import load_eeg_eog_emg
from ..data import get_cv_splits


def make_derangement(n, seed):
    """
    Generate a reproducible random derangement using the same
    fixed-point repair procedure as the video shuffled-pairing control.

    The returned permutation contains local positions within a CV
    partition, not global experiment IDs.
    """
    if n < 2:
        raise ValueError(
            f"A derangement requires at least two trials. Got n={n}."
        )
    if seed is None:
        raise ValueError("A fixed random seed is required.")

    perm = np.random.default_rng(seed).permutation(n)

    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]

    if not np.array_equal(np.sort(perm), np.arange(n)):
        raise RuntimeError("Generated mapping is not a valid permutation.")
    if np.any(perm == np.arange(n)):
        raise RuntimeError("Generated permutation contains a fixed point.")

    return perm


def eog_shuffled_regression_fold(
    y_all,
    X_all,
    train_trials,
    test_trials,
    train_seed,
    test_seed,
):
    """
    Fit shuffled EOG→EEG linear regression for one CV fold.

    Parameters
    ----------
    y_all : np.ndarray
        EEG targets, shape (n_trials, n_samples).

    X_all : np.ndarray
        Joint hEOG/vEOG signals, shape (n_trials, n_samples, 2).

    train_trials : array-like
        Global trial indices in the training partition.

    test_trials : array-like
        Global trial indices in the held-out partition.

    train_seed : int
        Random seed for the training EOG derangement.

    test_seed : int
        Random seed for the held-out EOG derangement.

    Returns
    -------
    dict
        Held-out targets and predictions, trial mappings, regression
        parameters, and scaler parameters.

    Notes
    -----
    EEG target ordering is never shuffled. Only the EOG source-trial
    assignment is changed.
    """
    y_all = np.asarray(y_all, dtype=np.float64)
    X_all = np.asarray(X_all, dtype=np.float64)
    train_trials = np.asarray(train_trials, dtype=int)
    test_trials = np.asarray(test_trials, dtype=int)

    if y_all.ndim != 2:
        raise ValueError(
            f"y_all must have shape (trials, samples); got {y_all.shape}."
        )
    if X_all.ndim != 3:
        raise ValueError(
            "X_all must have shape (trials, samples, references)."
        )
    if X_all.shape[:2] != y_all.shape:
        raise ValueError(
            f"EEG/EOG shape mismatch: {y_all.shape} vs {X_all.shape}."
        )
    if X_all.shape[-1] != 2:
        raise ValueError(
            f"Exactly two EOG predictors are required; got {X_all.shape[-1]}."
        )
    if np.unique(train_trials).size != len(train_trials):
        raise ValueError("Training trial indices contain duplicates.")
    if np.unique(test_trials).size != len(test_trials):
        raise ValueError("Testing trial indices contain duplicates.")
    if np.intersect1d(train_trials, test_trials).size:
        raise ValueError("Training and testing trials must be disjoint.")

    n_trials, n_samples = y_all.shape
    n_refs = X_all.shape[-1]

    if np.any(train_trials < 0) or np.any(train_trials >= n_trials):
        raise IndexError("Invalid training trial index.")
    if np.any(test_trials < 0) or np.any(test_trials >= n_trials):
        raise IndexError("Invalid testing trial index.")

    # Generate separate training and testing derangements.
    train_perm = make_derangement(len(train_trials), train_seed)
    test_perm = make_derangement(len(test_trials), test_seed)

    # Map local permutation positions to global trial indices.
    train_eog_source_trials = train_trials[train_perm]
    test_eog_source_trials = test_trials[test_perm]

    if np.any(train_trials == train_eog_source_trials):
        raise RuntimeError("Training pairing contains a fixed point.")
    if np.any(test_trials == test_eog_source_trials):
        raise RuntimeError("Testing pairing contains a fixed point.")

    # Fit scalers using only the original training trial sets.
    # The EOG permutation uses every training EOG trial exactly once,
    # so the original and permuted training EOG sets are identical.
    scaler_X = StandardScaler().fit(
        X_all[train_trials].reshape(-1, n_refs)
    )
    scaler_y = StandardScaler().fit(
        y_all[train_trials].reshape(-1, 1)
    )

    # Pair shuffled training EOG with the original EEG targets.
    X_train = X_all[train_eog_source_trials].reshape(-1, n_refs)
    y_train = y_all[train_trials].reshape(-1, 1)

    X_train_z = scaler_X.transform(X_train)
    y_train_z = scaler_y.transform(y_train).ravel()

    # Fit the same regression model as the matched EOG baseline.
    reg = LinearRegression()
    reg.fit(X_train_z, y_train_z)

    # Predict the original held-out EEG targets using shuffled EOG.
    X_test = X_all[test_eog_source_trials].reshape(-1, n_refs)
    X_test_z = scaler_X.transform(X_test)

    y_pred = reg.predict(X_test_z).reshape(
        len(test_trials), n_samples
    )

    y_test = scaler_y.transform(
        y_all[test_trials].reshape(-1, 1)
    ).reshape(len(test_trials), n_samples)

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "trials": test_trials.copy(),
        "train_trials": train_trials.copy(),
        "test_trials": test_trials.copy(),
        "train_eog_source_trials": train_eog_source_trials.copy(),
        "test_eog_source_trials": test_eog_source_trials.copy(),
        "train_permutation": train_perm.copy(),
        "test_permutation": test_perm.copy(),
        "model_coef": reg.coef_.copy(),
        "model_intercept": float(reg.intercept_),
        "scaler_X_mean": scaler_X.mean_.copy(),
        "scaler_X_scale": scaler_X.scale_.copy(),
        "scaler_y_mean": float(scaler_y.mean_[0]),
        "scaler_y_scale": float(scaler_y.scale_[0]),
        "train_seed": int(train_seed),
        "test_seed": int(test_seed),
        "references": ("heog", "veog"),
    }


def run_shuffled_eog_regression(
    subject_id,
    channel_name="Fz",
    n_points=40,
    n_splits=5,
    references=("heog", "veog"),
    fs=128.0,
    shuffle_seed=1338,
    filter_band=(4.0, 45.0),
    save_dir=None,
    json_filename="EOG_Shuffled_summary.json",
    root_dir=".",
    verbose=True,
):
    """
    Run within-participant shuffled-trial EOG regression.

    The experiment uses the same GroupKFold splits and derangement
    seed scheme as the shuffled-pairing video control.

    For each fold:
        1. Generate training and testing EOG derangements.
        2. Pair the original EEG targets with shuffled EOG trials.
        3. Fit the linear regression using the training partition.
        4. Predict the original held-out EEG targets.
        5. Store predictions and trial mappings.

    No evaluation metrics are calculated in this function.
    """
    if save_dir is None:
        raise ValueError("save_dir must be provided.")
    if shuffle_seed is None:
        raise ValueError(
            "shuffle_seed must be an integer for reproducible pairing."
        )
    if not isinstance(shuffle_seed, (int, np.integer)) or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be a non-negative integer.")
    if tuple(references) != ("heog", "veog"):
        raise ValueError(
            "references must be ('heog', 'veog') in this implementation."
        )
    if n_points < 2:
        raise ValueError("At least two trials are required.")
    if n_splits < 2 or n_splits > n_points:
        raise ValueError("Invalid number of CV folds.")

    os.makedirs(save_dir, exist_ok=True)

    # Load the original DEAP preprocessed EEG and EOG arrays.
    experiment_ids = np.arange(n_points)

    data = load_eeg_eog_emg(
        eeg_channel_name=channel_name,
        participant_id=subject_id,
        experiment_ids=experiment_ids,
        fs=fs,
        filter_band=filter_band,
        root_dir=root_dir,
    )

    y_all = np.asarray(data["eeg"], dtype=np.float64)
    X_all = np.stack(
        [data[ref] for ref in references],
        axis=-1,
    ).astype(np.float64)

    if y_all.shape[0] != n_points:
        raise ValueError(
            f"Expected {n_points} EEG trials; received {y_all.shape[0]}."
        )
    if X_all.shape[:2] != y_all.shape:
        raise ValueError(
            f"EEG/EOG dimensions are incompatible: "
            f"{y_all.shape} vs {X_all.shape}."
        )
    if X_all.shape[-1] != 2:
        raise ValueError(
            f"Expected two EOG predictors; received {X_all.shape[-1]}."
        )
    if not np.all(np.isfinite(y_all)):
        raise ValueError("Non-finite values detected in EEG data.")
    if not np.all(np.isfinite(X_all)):
        raise ValueError("Non-finite values detected in EOG data.")

    n_samples = y_all.shape[1]

    expected_samples = int(round(60.0 * fs))
    if n_samples != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples per 60-second trial "
            f"at {fs} Hz, but received {n_samples}."
        )
    # Generate the same GroupKFold partitions as the other models.
    training_sets, testing_sets, _ = get_cv_splits(
        n_points=n_points,
        n_splits=n_splits,
    )

    all_test_trials = np.concatenate(testing_sets)
    if not np.array_equal(np.sort(all_test_trials), np.arange(n_points)):
        raise RuntimeError(
            "GroupKFold does not hold out every trial exactly once."
        )

    for tr, te in zip(training_sets, testing_sets):
        if np.intersect1d(tr, te).size:
            raise RuntimeError(
                "Training and testing partitions overlap."
            )
        
    if not np.array_equal(
        np.sort(np.concatenate([tr, te])),
        np.arange(n_points),
    ):
        raise RuntimeError(
            "Training and testing trials do not form a complete partition."
        )
    
    results = {
        "subject": subject_id,
        "channel": channel_name,
        "time_window": 60,
        "n_splits": n_splits,
        "folds": {},
        "config": {
            "method": "LinearRegression",
            "estimator": "sklearn.linear_model.LinearRegression",
            "control": "shuffled_pairing",
            "pairing_scheme": "Within-participant, within-fold EOG shuffled pairing",
            "references": list(references),
            "source_data": "DEAP preprocessed Python .dat arrays",
            "eog_filter_band_hz": list(filter_band),
            "eog_filter": (
                "Fourth-order Butterworth, "
                "forward-backward zero-phase filtfilt"
            ),
            "eog_scaling": "StandardScaler fitted on training EOG trials",
            "eeg_scaling": "StandardScaler fitted on training EEG trials",
            "cv": "GroupKFold",
            "n_points": n_points,
            "n_splits": n_splits,
            "shuffle_pairing": True,
            "shuffle_seed": int(shuffle_seed),
            "seed_scheme": (
                "Training: shuffle_seed + fold_idx; "
                "testing: shuffle_seed + 1000 + fold_idx"
            ),
            "fs": float(fs),
            "n_samples_per_trial": int(n_samples),
            "no_fixed_points": True,
        },
    }

    if verbose:
        print(
            f"\n=== Shuffled EOG regression: "
            f"S{subject_id:02d} | {channel_name} ==="
        )
        print(
            f"  Trials: {n_points} | Folds: {n_splits} | "
            f"Seed: {shuffle_seed}"
        )
        print(f"  EEG shape: {y_all.shape} | EOG shape: {X_all.shape}")

    # Fit and predict once per fold.
    for fold_idx, (tr, te) in enumerate(
        zip(training_sets, testing_sets)
    ):
        if verbose:
            print(
                f"\n=== Fold {fold_idx + 1}/{n_splits} "
                f"(train={len(tr)}, test={len(te)}) ==="
            )

        # Use exactly the seed scheme in pipeline.py.
        train_seed = int(shuffle_seed + fold_idx)
        test_seed = int(shuffle_seed + 1000 + fold_idx)

        fold = eog_shuffled_regression_fold(
            y_all=y_all,
            X_all=X_all,
            train_trials=tr,
            test_trials=te,
            train_seed=train_seed,
            test_seed=test_seed,
        )

        if verbose:
            print(
                f"  y_test {fold['y_test'].shape}  "
                f"y_pred {fold['y_pred'].shape}"
            )

        # Store predictions and pairing information.
        results["folds"][fold_idx] = {
            "y_test": fold["y_test"],
            "y_pred": fold["y_pred"],
            "trials": fold["trials"],
            "train_trials": fold["train_trials"],
            "test_trials": fold["test_trials"],
            "train_eog_source_trials": fold["train_eog_source_trials"],
            "test_eog_source_trials": fold["test_eog_source_trials"],
            "train_permutation": fold["train_permutation"],
            "test_permutation": fold["test_permutation"],
            "train_seed": fold["train_seed"],
            "test_seed": fold["test_seed"],
            "model_coef": fold["model_coef"],
            "model_intercept": fold["model_intercept"],
            "scaler_X_mean": fold["scaler_X_mean"],
            "scaler_X_scale": fold["scaler_X_scale"],
            "scaler_y_mean": fold["scaler_y_mean"],
            "scaler_y_scale": fold["scaler_y_scale"],
        }

    # Save the result.
    pkl_filename = f"sub{subject_id:02d}_ch{channel_name}.pkl"
    pkl_path = os.path.join(save_dir, pkl_filename)

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print(f"\nSaved: {pkl_path}")

    # Update the JSON summary.
    json_path = os.path.join(save_dir, json_filename)

    if os.path.isfile(json_path):
        with open(json_path, "r") as f:
            summary = json.load(f)
    else:
        summary = {
            "estimator": "sklearn.linear_model.LinearRegression",
            "control": "shuffled_pairing",
            "experiments": {},
        }

    exp_name = f"EOG2EEG_Shuffled_{channel_name}"
    summary["experiments"].setdefault(exp_name, {"models": {}})
    summary["experiments"][exp_name]["models"][
        f"sub{subject_id:02d}_ch{channel_name}"
    ] = {
        "model_details": {
            "subject": int(subject_id),
            "channel": channel_name,
            "n_folds": int(n_splits),
            "shuffle_seed": int(shuffle_seed),
        },
        "pkl_file": pkl_filename,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return results