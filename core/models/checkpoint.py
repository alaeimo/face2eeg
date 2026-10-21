# face2eeg/core/models/checkpoint.py

"""
Device selection and checkpoint loading.
"""
import os
import torch
from .physnet import PhysNet_3DCNN_ED

# Selecting Device (cpu or gpu)
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def load_model(video_length, signal_length, model_path, device):
    torch.cuda.empty_cache()
    model = PhysNet_3DCNN_ED(video_length, signal_length)
    model = model.to(device=device)
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    raise ModuleNotFoundError()


