
# face2eeg/core/signals.py

"""
Signal-processing helpers (band-pass filters).
"""
from scipy.signal import butter, filtfilt
from .config import FREQ_BANDS

def bandpass_filter(data, freq_band, fs, order=4, freq_bands={}):
    """
    description:
    ------------
    Apply a zero-phase Butterworth bandpass filter to the signal.

    parameters:
    ------------
    -param data: (numpy ndarray) input signal
    -param freq_band: (str) band name, key in `freq_bands` dict
    -param fs: (int/float) sampling frequency (Hz)
    -param order: (int) Butterworth filter order
    -param freq_bands: (dict/None) frequency-band definitions;
                       if None, uses global FREQ_BANDS

    return:
    ------------
    y: (numpy ndarray) filtered signal
    """
    if freq_band not in freq_bands:
        raise ValueError(
            f"Unknown band '{freq_band}'. Available: {list(freq_bands.keys())}"
        )

    nyquist = 0.5 * fs
    lowcut, highcut = freq_bands[freq_band]
    low  = lowcut  / nyquist
    high = highcut / nyquist

    if not (0 < low < high < 1):
        raise ValueError(
            f"Invalid band {freq_band}={freq_bands[freq_band]} for fs={fs}. "
            f"Normalized: low={low:.3f}, high={high:.3f}"
        )

    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)  # zero-phase filter
    return y


def bandpass_freq_range(data, lo, hi, fs=128.0, order=4):
    """
    description:
    ------------
    Apply a zero-phase Butterworth bandpass filter using an explicit
    frequency range (in Hz) instead of a named band from `FREQ_BANDS`.

    parameters:
    ------------
    -param data: (numpy ndarray) input signal
    -param lo: (int/float) lower cutoff frequency in Hz
    -param hi: (int/float) upper cutoff frequency in Hz
    -param fs: (int/float) sampling frequency (Hz)
    -param order: (int) Butterworth filter order

    return:
    ------------
    y: (numpy ndarray) filtered signal
    """
    nyq = 0.5 * fs
    low, high = lo / nyq, hi / nyq

    if not (0 < low < high < 1):
        raise ValueError(
            f"Invalid range ({lo}, {hi}) for fs={fs}. "
            f"Normalized: low={low:.3f}, high={high:.3f}"
        )

    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)  # zero-phase filter
    return y