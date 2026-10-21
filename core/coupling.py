# face2eeg/core/coupling.py
"""
Signal-level coupling analysis between one EEG channel and peripheral
references (hEOG, vEOG, zEMG). No model is trained.
"""
import numpy as np

from .data.peripheral import load_eeg_eog_emg
from .stats           import safe_pearsonr, mscohere, eeg_band_range, band_mask


def analyze_eeg_eog_emg_coupling(eeg_channel_name, participant_id,
                                  experiment_ids, fs=128.0,
                                  nperseg=1024,
                                  bands=('theta', 'alpha', 'beta'),
                                  references=('heog', 'veog', 'zemg'),
                                  root_dir=None):
    """
    Compute trial-level Pearson r and band-averaged coherence between
    one EEG channel and each reference channel.

        Parameters
    ----------
    eeg_channel_name : str
        Name of the EEG channel ('Fz', 'Pz', 'Oz').
    participant_id : int
        DEAP participant id (1-32).
    experiment_ids : array-like of int
        DEAP `experiment_id` indices (0-39) to include.
    fs : float, default=128.0
        Sampling frequency in Hz.
    nperseg : int, default=1024
        Segment length (samples) for Welch coherence estimation.
        Overlap is set to `nperseg // 2`.
    bands : tuple of str, default=('theta', 'alpha', 'beta')
        Frequency bands for the band-averaged coherence summaries.
        Each entry must be a valid key in `eeg_band_range`.
    references : tuple, list of references, default=('heog', 'veog', 'zemg')
        Peripheral reference channels.
    root_dir : str, root directory containing physiological data and facial videos

    Returns
    -------
    dict
        Scalar summaries, each of length n_trials:
          'eeg_heog_r', 'eeg_veog_r', 'eeg_zemg_r'
              Broadband (4-45 Hz) Pearson r, band-independent.
          'eeg_heog_<band>_coh', 'eeg_veog_<band>_coh', 'eeg_zemg_<band>_coh'
              Band-averaged coherence for each band in `bands`.

        Full coherence spectra, for plotting:
          'f_coh'              : (n_freq_bins,) frequency axis
          'coh_heog_spectra'   : (n_trials, n_freq_bins)
          'coh_veog_spectra'   : (n_trials, n_freq_bins)
          'coh_zemg_spectra'   : (n_trials, n_freq_bins)

        Metadata:
          '_bands_used'        : tuple of str, the bands computed in this call.
                                 Use this to verify which frequency bands the
                                 band-averaged coherence keys correspond to.
                                 Note: the broadband `_r` values are NOT tied
                                 to any band in `_bands_used`.
    """
    # Resolve band ranges for each requested band
    band_ranges = {b: eeg_band_range(b) for b in bands}

    data = load_eeg_eog_emg(eeg_channel_name, participant_id,
                             experiment_ids, fs=fs, root_dir=root_dir)
    eeg = data['eeg']
    n_trials = eeg.shape[0]

    # Pre-allocate scalar arrays
    scalar_r   = {ref: np.empty(n_trials) for ref in references}
    scalar_coh = {
        ref: {b: np.empty(n_trials) for b in bands}
        for ref in references
    }

    # Pre-allocate coherence spectra storage
    n_bins = nperseg // 2 + 1
    f_coh = np.empty(n_bins)
    spectra = {ref: np.empty((n_trials, n_bins)) for ref in references}

    for i in range(n_trials):
        for ref in references:
            sig = data[ref][i]
            # --- Pearson correlations (stats.safe_pearsonr) ---
            scalar_r[ref][i] = safe_pearsonr(eeg[i], sig)

            # --- Magnitude-squared coherence (stats.mscohere) ---
            f, C = mscohere(eeg[i], sig, fs=fs,
                            nperseg=nperseg, noverlap=nperseg // 2)
            
            # Store frequency axis once (identical across trials)
            if i == 0:
                f_coh[:] = f

            spectra[ref][i] = C

            # --- Band-averaged coherence per band (stats.band_mask) ---
            for b in bands:
                band_lo, band_hi = band_ranges[b]
                m = band_mask(f, band_lo, band_hi)
                scalar_coh[ref][b][i] = (
                    float(np.mean(C[m])) if np.any(m) else np.nan
                )

    out = {}
    for ref in references:
        # Broadband r is stored ONCE, independent of bands.
        # This r corresponds to the full 4-45 Hz range and is
        # not tied to any specific frequency band.
        out[f'eeg_{ref}_r'] = scalar_r[ref]
        for b in bands:
            out[f'eeg_{ref}_{b}_coh'] = scalar_coh[ref][b]
        out[f'coh_{ref}_spectra'] = spectra[ref]
    out['f_coh'] = f_coh
    out['_bands_used'] = tuple(bands)   # metadata: which bands were computed
    return out