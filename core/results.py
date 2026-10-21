# face2eeg/core/results.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, List, Tuple
import os
import pickle
import numpy as np
import glob

@dataclass
class ExperimentResult:
    """
    Container for one result pickle in new format.

    Attributes
    ----------
    file_path : str
        Absolute path of the file it was loaded from.
    subject : int
    channel : str
    time_window : int or float or None
    variant : str or None
    n_splits : int
    folds : dict
        {fold_idx: {'y_test', 'y_pred', 'trials', ['history']}}
        y_test / y_pred shape: (n_segments, s_window)
    config : dict
    pred_key: str (y_pred)
    """
    file_path:   str
    subject:     int
    channel:     str
    time_window: Optional[Union[int, float]]
    variant:     Optional[str]
    n_splits:    int
    folds:       Dict[int, Dict[str, Any]]
    config:      Dict[str, Any]
    pred_key:    str = "y_pred"
    # ----------------------------------------------------------------
    # Basic accessors
    # ----------------------------------------------------------------
    def fold_ids(self) -> List[int]:
        return sorted(self.folds.keys())

    def y_test(self, fold_idx: int) -> np.ndarray:
        return self.folds[fold_idx]['y_test']

    def y_pred(self, fold_idx: int) -> np.ndarray:
        return self.folds[fold_idx][self.pred_key]

    def trials(self, fold_idx: int) -> np.ndarray:
        return self.folds[fold_idx]['trials']

    def history(self, fold_idx: int) -> Optional[Dict[str, Any]]:
        return self.folds[fold_idx].get('history')

    # ----------------------------------------------------------------
    # Segment -> trial reshaping
    # ----------------------------------------------------------------
    def _segments_to_trials(self, y: np.ndarray,
                            signal_fs: int = 128,
                            trial_len_sec: int = 60) -> np.ndarray:
        if self.time_window is None:
            raise ValueError("time_window is None; cannot reshape to trials.")
        s_window = int(self.time_window * signal_fs)
        trial_len = int(trial_len_sec * signal_fs)
        segs_per_trial = trial_len // s_window
        if segs_per_trial * s_window != trial_len:
            raise ValueError(
                f"time_window={self.time_window} does not divide "
                f"trial_len={trial_len} evenly."
            )
        n_trials = y.shape[0] // segs_per_trial
        return y.reshape(n_trials, segs_per_trial, s_window) \
                .reshape(n_trials, trial_len)

    def get_fold_trials(self, fold_idx: int,
                        signal_fs: int = 128,
                        trial_len_sec: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Return (y_true, y_pred) for one fold, at trial level."""
        yt = self._segments_to_trials(self.y_test(fold_idx),
                                      signal_fs, trial_len_sec)
        yp = self._segments_to_trials(self.y_pred(fold_idx),
                                      signal_fs, trial_len_sec)
        return yt, yp

    def get_all_trials(self,
                       signal_fs: int = 128,
                       trial_len_sec: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate all folds -> (n_trials_total, trial_len)."""
        yts, yps = [], []
        for k in self.fold_ids():
            yt, yp = self.get_fold_trials(k, signal_fs, trial_len_sec)
            yts.append(yt)
            yps.append(yp)
        return np.concatenate(yts, axis=0), np.concatenate(yps, axis=0)

    def get_trial_indices(self) -> np.ndarray:
        """Concatenated trial indices across all folds."""
        return np.concatenate([self.trials(k) for k in self.fold_ids()], axis=0)

    # ----------------------------------------------------------------
    # Display
    # ----------------------------------------------------------------
    def summary(self) -> str:
        base = os.path.basename(self.file_path)
        return (f"ExperimentResult(file={base!r}, "
                f"subject={self.subject}, channel={self.channel!r}, "
                f"tw={self.time_window}, variant={self.variant!r}, "
                f"n_splits={self.n_splits})")

    def __repr__(self):
        return self.summary()


# ============================================================
# Loader — takes a file path
# ============================================================
def load_experiment_result(
    file_path: str,
    variant: Optional[str] = None,
    pred_key: str="y_pred", 
    verbose: bool = False,
) -> ExperimentResult:
    """
    Load a new-format result pickle into an ExperimentResult.

    Parameters
    ----------
    file_path : str
        Full path to the .pkl file.
        e.g. r'D:\\...\\Loss-Functions\\Results\\sub02_chFz_MSE.pkl'
    variant : str or None
        If given, overrides the variant stored in config.
        If None, reads from `config['variant']` (migration metadata).

    pred_key: str, 
        Default is `y_pred` for NN models and 
        (`const_zero`, `const_train_mean`, `mean_waveform` for trivial models 
    verbose : bool

    Returns
    -------
    ExperimentResult
    """
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Not found: {file_path}")

    with open(file_path, 'rb') as f:
        d = pickle.load(f)

    needed = {'subject', 'channel', 'time_window', 'n_splits',
              'folds', 'config'}
    if not isinstance(d, dict) or not needed.issubset(d.keys()):
        raise ValueError(
            f"File is not in new format: {file_path}\n"
            f"  keys = {list(d.keys()) if isinstance(d, dict) else type(d)}"
        )

    # resolve variant
    if variant is None:
        variant = (d.get('config') or {}).get('variant', None)

    if verbose:
        print(f"[load] {file_path}")

    return ExperimentResult(
        file_path   = file_path,
        subject     = d['subject'],
        channel     = d['channel'],
        time_window = d['time_window'],
        variant     = variant,
        n_splits    = d['n_splits'],
        folds       = d['folds'],
        config      = d['config'],
        pred_key    = pred_key
    )


def load_results_from_folder(
    folder: str,
    pattern: str = 'sub*_ch*.pkl',
    pred_key: str="y_pred", 
    verbose: bool = False,
) -> List[ExperimentResult]:
    """
    Load all matching .pkl files in a folder as ExperimentResult objects.
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    out = []
    for p in paths:
        if os.path.basename(p).startswith('_backup'):
            continue
        try:
            out.append(load_experiment_result(p, pred_key=pred_key, verbose=verbose))
        except Exception as e:
            if verbose:
                print(f"  [fail] {p}: {e}")
    return out


def load_results_by_subject(
    folder: str,
    subject_ids: List[int],
    channel_name: str,
    variant: Optional[str] = None,
    pred_key: str="y_pred", 
    verbose: bool = False,
) -> Dict[int, ExperimentResult]:
    """
    Load multiple subjects from one folder.
    filename pattern: sub{XX}_ch{CH}[_{variant}].pkl
    """
    out = {}
    for sid in subject_ids:
        base = f"sub{sid:02d}_ch{channel_name}"
        fname = f"{base}_{variant}.pkl" if variant else f"{base}.pkl"
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            if verbose:
                print(f"  [miss] {path}")
            continue
        try:
            out[sid] = load_experiment_result(path, pred_key=pred_key, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  [fail] {path}: {e}")
    return out


if __name__ == "__main__":
    CROSS_VAL = r'D:\Thesis\Codes\Data\Experiments\Cross-Validation'

    r = load_experiment_result(
        os.path.join(CROSS_VAL, 'Loss-Functions', 'Results', 'sub02_chFz_MSE.pkl')
    )
    print(r.summary())
    # ExperimentResult(file='sub02_chFz_MSE.pkl', subject=2, channel='Fz', tw=1, variant='MSE', n_splits=5)

    # trial arrays
    y_true, y_pred = r.get_all_trials()
    print(y_true.shape, y_pred.shape)   # (40, 7680) (40, 7680)