# face2eeg/core/data/cache.py

"""
Caches for repeated I/O (mapper CSV, physiological .dat, video .avi).
"""

_EXP2TRIAL_MAPPER_CACHE = {}
_PHYSIO_CACHE = {}
_VIDEO_CACHE = {}

def clear_video_cache():
    _VIDEO_CACHE.clear()
    print("[video cache] cleared")

def clear_signals_cache():
    _PHYSIO_CACHE.clear()
    print("[signal cache] cleared")