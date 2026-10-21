# face2eeg/core/evaluation.py

"""
Model evaluation and batched prediction.
"""
import numpy as np
import torch

from .stats import summarize_trials

def evaluate_model(y_true, y_pred, fs=128.0, seed=42, n_boot=10000):
    """
    Evaluate a model's predictions at the trial level.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (n_trials, n_samples)
        Ground-truth and predicted signals.
    fs : float
        Sampling frequency (Hz).
    seed : int
        Random seed for bootstrap resampling.
    n_boot : int
        Number of bootstrap resamples for confidence intervals.

    Returns
    -------
    dict with:
        Simple metrics
            r_mean, r_std, r_pooled, r_pooled_ci
            rmse_mean, rmse_sd, rmse_ci
            n_trials
        Raw trial arrays (for paired testing)
            r_trials, rmse_trials
        Ready-to-plot containers
            overall : dict       -> pass to plot_band_comparison
            curves  : dict       -> pass to plot_mean_curve_with_ci
    """
    s = summarize_trials(y_true, y_pred, fs=fs, n_boot=n_boot, seed=seed)

    # Container expected by plot_band_comparison
    overall = {
        "band_summary": s["band_summary"],
    }

    # Container expected by plot_mean_curve_with_ci
    curves = {
        "curves_summary": s["curves_summary"],
    }

    return {
        # Numeric summary
        "n_trials":     s["trials"],
        "r_mean":       s["r_mean"],
        "r_std":        s["r_std"],
        "r_pooled":     s["r_pooled"],
        "r_pooled_ci":  s["r_pooled_ci_boot"],
        "rmse_mean":    s["rmse_mean"],
        "rmse_sd":      s["rmse_sd"],
        "rmse_ci":      s["rmse_trial_ci_boot"],

        # Raw trial arrays
        "r_trials":     s["r_trials"],
        "rmse_trials":  s["rmse_trials"],

        # Plot-ready
        "overall": overall,
        "curves":  curves,
    }


def run_model_and_predict(model, X_test, device, batch_size=16):
    """
    Run model prediction in batches to avoid CUDA out of memory.

    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model
    X_test : numpy array
        Input array, shape (N_seg, channels, T, W, H) or similar
    device : torch.device
        CPU or GPU device
    batch_size : int
        Number of samples per batch

    Returns:
    --------
    y_pred : numpy array
        Predicted output, shape (N_seg, signal_length)
    """
    model.eval()
    N = X_test.shape[0]
    y_pred_list = []

    # Pre-allocate contiguous array once
    X_test_np = np.ascontiguousarray(X_test)

    with torch.inference_mode():
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            X_batch = X_test_np[start_idx:end_idx]
            X_batch_tensor = torch.from_numpy(X_batch).float().to(device)

            y_batch = model(X_batch_tensor).cpu().numpy()
            y_pred_list.append(y_batch)

    # Free GPU memory once at the end (not every batch)
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    y_pred = np.concatenate(y_pred_list, axis=0)
    return y_pred