# face2eeg/core/experiments/__init__.py
"""
Experiment runners.

`pipeline` and `trivial_baselines` depend on torch.
`eog_baseline` is torch-free.
"""
from .eog_baseline import run_eog_regression, eog_regression_fold


def __getattr__(name):
    """Lazily import torch-dependent runners."""
    if name == 'run_kfold_pipeline':
        from .pipeline import run_kfold_pipeline
        return run_kfold_pipeline
    if name == 'run_trivial_baselines':
        from .trivial_baselines import run_trivial_baselines
        return run_trivial_baselines
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'run_eog_regression',
    'eog_regression_fold',
    'run_kfold_pipeline',
    'run_trivial_baselines',
]