# face2eeg\analysis\helpers.py
"""
Notebook-facing helpers for the analysis notebooks.
"""
import os
import numpy as np
import pandas as pd

from ..core.results import load_experiment_result
from ..core.stats   import eeg_analysis_bands


# ============================================================
# 1. Base setup
# ============================================================
def setup(base_dir,
          figures_subdir='Analysis-Figures',
          subjects=[],
          make_fig_dir=True):
    """
    Configure the base directory for all experiments.

    Parameters
    ----------
    base_dir : str
        Directory that contains all experiment folders.
        E.g. r'D:\\Face2EEG\\Codes\\Data\\Experiments\\Cross-Validation'.
    figures_subdir : str
        Subfolder under each experiment's Results/ where figures are saved.
        Default: 'Analysis-Figures'.
    subjects : iterable of int
        Subjects used in the paper.
    make_fig_dir : bool
        If True, create each experiment's figures folder on demand.

    Returns
    -------
    base : dict
        {
            'base_dir':       str,   # abspath of base_dir
            'figures_subdir': str,   # e.g. 'Analysis-Figures'
            'subjects':       list[int],
            'make_fig_dir':   bool,
        }
    """
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    return {
        'base_dir':       base_dir,
        'figures_subdir': figures_subdir,
        'subjects':       list(subjects),
        'make_fig_dir':   make_fig_dir,
    }


# ============================================================
# 2. Per-experiment paths
# ============================================================
def experiment_paths(base, experiment_name):
    """
    Build the Results/ and Figures paths for one experiment.

    Parameters
    ----------
    base : dict
        Output of setup().
    experiment_name : str
        e.g. 'Loss-Functions', 'Segment-Lengths', 'Between-Participant-Variability',
             'Channel-Specificity', 'EOG-EEG'.

    Returns
    -------
    paths : dict
        {
            'name':    experiment_name,
            'root':    <base_dir>/<experiment_name>,
            'results': <base_dir>/<experiment_name>/Results,
            'figures': <base_dir>/<experiment_name>/Results/Analysis-Figures,
        }
    """
    root    = os.path.join(base['base_dir'], experiment_name)
    results = os.path.join(root, 'Results')
    figures = os.path.join(results, base['figures_subdir'])

    if not os.path.isdir(results):
        raise FileNotFoundError(
            f"Results folder not found: {results}"
        )
    if base['make_fig_dir']:
        os.makedirs(figures, exist_ok=True)

    return {
        'name':    experiment_name,
        'root':    root,
        'results': results,
        'figures': figures,
    }


# ============================================================
# 3. Load a single result
# ============================================================
def load_result(results_dir, subject, channel='Fz', variant=None, pred_key="y_pred"):
    """
    Load from <results_dir>/sub{XX}_ch{CH}[_{variant}].pkl.
    """
    base  = f"sub{subject:02d}_ch{channel}"
    fname = f"{base}_{variant}.pkl" if variant else f"{base}.pkl"
    path  = os.path.join(results_dir, fname)
    return load_experiment_result(path, pred_key=pred_key)


# ============================================================
# 4. Trial-level arrays
# ============================================================
def get_arrays(result, signal_length=7680, pred_key="y_pred"):
    """Return test targets, predictions, and trial IDs."""
    y_test, y_pred, trial_ids = [], [], []

    for fold in sorted(result["folds"], key=str):
        data = result["folds"][fold]

        if "y_test" not in data:
            raise KeyError(f"{fold}: missing 'y_test'")
        if pred_key not in data:
            raise KeyError(f"{fold}: missing '{pred_key}'")
        if "trials" not in data:
            raise KeyError(f"{fold}: missing 'trials'")

        yt = np.asarray(data["y_test"]).reshape(-1, signal_length)
        yp = np.asarray(data[pred_key]).reshape(-1, signal_length)
        tr = np.asarray(data["trials"])

        if not (len(yt) == len(yp) == len(tr)):
            raise ValueError(
                f"{fold}: y_test={len(yt)}, {pred_key}={len(yp)}, "
                f"trials={len(tr)}"
            )

        y_test.append(yt)
        y_pred.append(yp)
        trial_ids.extend(tr.tolist())

    return (
        np.concatenate(y_test),
        np.concatenate(y_pred),
        np.asarray(trial_ids),
    )


# ============================================================
# 5. Tidy DataFrame for plot_metrics_grid
# ============================================================
def get_trials_dataframe(summaries_dict):
    """
    Build the tidy per-trial DataFrame expected by plot_metrics_grid.
    """
    bands = eeg_analysis_bands()
    rows = []
    for label, summ in summaries_dict.items():
        n = summ['trials']
        for t in range(n):
            tr = summ['frequency_analysis_trials'][t]
            row = {
                'model': label,
                'trial': t + 1,
                'rmse':  summ['rmse_trials'][t],
                'r':     summ['r_trials'][t],
            }
            for b in bands:
                row[f'rbp_pred_{b}'] = tr['rbp_pred'][b]
                row[f'coh_{b}']      = tr['coh_bands'][b]
            rows.append(row)
    return pd.DataFrame(rows)

def validate_trial_alignment(trial_ids_a, trial_ids_b, name_a="A", name_b="B"):
    """Ensure two conditions contain exactly the same trials in the same order."""
    a = np.asarray(trial_ids_a)
    b = np.asarray(trial_ids_b)

    if len(a) != len(b):
        raise ValueError(
            f"{name_a} and {name_b} have different numbers of trials: "
            f"{len(a)} vs {len(b)}"
        )

    if len(np.unique(a)) != len(a):
        raise ValueError(f"{name_a} contains duplicate trial IDs.")

    if len(np.unique(b)) != len(b):
        raise ValueError(f"{name_b} contains duplicate trial IDs.")

    if not np.array_equal(a, b):
        raise ValueError(
            f"{name_a} and {name_b} have different trial IDs or ordering."
        )

    return True