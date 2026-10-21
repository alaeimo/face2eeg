# face2eeg/__init__.py
"""
face2eeg — EEG prediction from facial video on DEAP.

Light submodules (config, results, stats, plotting, signals) are
imported eagerly — no torch required.

Heavy submodules (data, models, training, evaluation, experiments)
are imported lazily — torch is pulled in only on first access.
"""
__version__ = '0.1.0'

import importlib
import sys


# ============================================================
# 1. Eager imports — light modules
#    Register each as `face2eeg.<name>` in sys.modules so that
#    `from face2eeg.stats import X` works even though the file
#    lives at face2eeg/core/stats.py.
# ============================================================
_EAGER = ['config', 'results', 'stats', 'plotting', 'signals']

for _name in _EAGER:
    try:
        _mod = importlib.import_module(f'.core.{_name}', __name__)
    except Exception as e:
        raise ImportError(
            f"face2eeg: cannot import face2eeg.core.{_name} — "
            f"{type(e).__name__}: {e}"
        ) from e

    # Attribute on the package
    globals()[_name] = _mod

    # Also register under the short name so `from face2eeg.stats import X` works
    sys.modules[f'{__name__}.{_name}'] = _mod

del _name, _mod


# ============================================================
# 2. Lazy imports — heavy modules (torch)
# ============================================================
_HEAVY = {
    'data':        'face2eeg.core.data',
    'models':      'face2eeg.core.models',
    'training':    'face2eeg.core.training',
    'evaluation':  'face2eeg.core.evaluation',
    'experiments': 'face2eeg.core.experiments',
}


def __getattr__(name):
    """Lazily import torch-dependent submodules on first access."""
    if name in _HEAVY:
        mod = importlib.import_module(_HEAVY[name])
        globals()[name] = mod
        sys.modules[f'{__name__}.{name}'] = mod
        return mod
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__():
    return sorted(_EAGER + list(_HEAVY.keys()))