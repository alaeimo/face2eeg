# face2eeg\analysis\__init__.py
"""
Notebook-facing analysis helpers.
"""
from .helpers import (
    setup,
    experiment_paths,
    load_result,
    get_arrays,
    get_trials_dataframe,
    validate_trial_alignment
)

__all__ = [
    'setup',
    'experiment_paths',
    'load_result',
    'get_arrays',
    'get_trials_dataframe',
    'validate_trial_alignment'
]