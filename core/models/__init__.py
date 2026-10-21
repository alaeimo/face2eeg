# face2eeg/core/models/__init__.py
"""
Model architectures, losses, and checkpoint utilities.

Submodules
----------
losses      — Neg_Pearson loss
physnet     — PhysNet 3D-CNN encoder-decoder
checkpoint  — device selection and checkpoint loading
"""

# --- losses ---
from .losses import Neg_Pearson

# --- architecture ---
from .physnet import PhysNet_3DCNN_ED

# --- checkpoint (depends on physnet) ---
from .checkpoint import (
    try_gpu,
    load_model,
)


__all__ = [
    'Neg_Pearson',
    'PhysNet_3DCNN_ED',
    'try_gpu',
    'load_model',
]# face2eeg/core/models/__init__.py
"""
Model architectures, losses, and checkpoint utilities.

Submodules
----------
losses      — Neg_Pearson loss
physnet     — PhysNet 3D-CNN encoder-decoder
checkpoint  — device selection and checkpoint loading
"""

# --- losses ---
from .losses import Neg_Pearson

# --- architecture ---
from .physnet import PhysNet_3DCNN_ED

# --- checkpoint (depends on physnet) ---
from .checkpoint import (
    try_gpu,
    load_model,
)


__all__ = [
    'Neg_Pearson',
    'PhysNet_3DCNN_ED',
    'try_gpu',
    'load_model',
]