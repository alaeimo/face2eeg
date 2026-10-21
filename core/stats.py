# face2eeg/core/stats.py
"""
Trial-level statistical analysis for the paper.

This module implements the trial-as-unit inference used throughout
the paper. Every metric is computed independently per 60-second trial
and then aggregated with bootstrap or t-based confidence intervals.

Main API
--------
summarize_trials                    — primary trial-level summary
compare_methods_trial_level         — paired comparison of two methods
comparison_to_dataframes            — tidy DataFrame output
analyze_segment_lengths_trial_level — paired comparison of k conditions
segment_lengths_to_dataframes       — tidy DataFrame output

Supporting functions
--------------------
eeg_band_range, eeg_analysis_bands, band_mask, relative_bandpower
fisher_z, inv_fisher_z, safe_pearsonr
psd_welch, mscohere, sign_flip_pvalue
_*  — internal helpers
"""

from typing import Dict, Any, List, Tuple, Union, Sequence, Optional, Callable
import math
import itertools
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- config-driven band definitions ---
from .config import FREQ_BANDS, EEG_ANALYSIS_BANDS


# optional SciPy
try:
    from scipy import stats
    from scipy.signal import welch, coherence
    _HAS_SCIPY = True
except Exception:
    stats = None
    welch = None
    coherence = None
    _HAS_SCIPY = False

ArrayLike = Union[float, np.ndarray, list, tuple]

# ============================================================
# 1. Frequency / EEG helpers (config-driven)
# ============================================================
def eeg_band_range(name: str) -> Tuple[float, float]:
    """
    Return the (low, high) frequency range in Hz for a given band name.

    Bands are defined in `face2eeg.core.config.FREQ_BANDS`.

    Notes
    -----
    - delta (0.5-4 Hz) is not present in DEAP (band-pass 4-45 Hz).
    - gamma (30-45 Hz) lies entirely above the 25 Hz video Nyquist.
    - Only theta, alpha, and beta are used for inferential statistics.
    """
    name = name.lower()
    if name not in FREQ_BANDS:
        raise ValueError(
            f"Unknown band name: {name!r}. "
            f"Available: {list(FREQ_BANDS.keys())}"
        )
    return FREQ_BANDS[name]


def eeg_analysis_bands() -> List[str]:
    """
    Bands used for inferential statistics.

    Returns a copy of `config.EEG_ANALYSIS_BANDS` (typically
    ['theta', 'alpha', 'beta'] — the bands fully representable from
    50-fps facial video).
    """
    return list(EEG_ANALYSIS_BANDS)

def band_mask(freqs: ArrayLike, lo: float, hi: float) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    return (freqs >= lo) & (freqs <= hi)

def relative_bandpower(Pxx: ArrayLike, f: ArrayLike, blo: float, bhi: float,
                       ref_lo: float = 4.0, ref_hi: float = 25.0) -> float:
    """
    Relative band power with reference band matching video Nyquist.
    
    ref_hi defaults to 25.0 Hz because facial video at 50 fps cannot
    represent spectral content above 25 Hz, and including 25-45 Hz in
    the denominator would create an inconsistency between the true and
    predicted spectra.
    """
    Pxx = np.asarray(Pxx, dtype=float)
    f = np.asarray(f, dtype=float)
    if Pxx.shape != f.shape:
        return float(np.nan)
    bmask = band_mask(f, blo, bhi)
    rmask = band_mask(f, ref_lo, ref_hi)
    if not np.any(bmask) or not np.any(rmask):
        return float(np.nan)
    num = np.trapezoid(Pxx[bmask], f[bmask])
    den = np.trapezoid(Pxx[rmask], f[rmask])
    return float(num / den) if den > 0 else float(np.nan)

# -------------------------
# Fisher transform
# -------------------------

def fisher_z(r: ArrayLike, return_scalar: bool = False) -> Union[np.ndarray, float]:
    r_arr = np.asarray(r, dtype=float)
    # clip to nearest representable <1 and >-1
    upper = np.nextafter(1.0, 0.0)
    lower = np.nextafter(-1.0, 0.0)
    r_clipped = np.clip(r_arr, lower, upper)
    z = np.arctanh(r_clipped)
    if return_scalar and r_arr.ndim == 0:
        return float(z)
    return z

def inv_fisher_z(z: ArrayLike, return_scalar: bool = False) -> Union[np.ndarray, float]:
    z_arr = np.asarray(z, dtype=float)
    r = np.tanh(z_arr)
    if return_scalar and z_arr.ndim == 0:
        return float(r)
    return r

# -------------------------
# Robust correlation
# -------------------------

def safe_pearsonr(x: ArrayLike, y: ArrayLike) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float(np.nan)
    # zero-variance check
    if np.allclose(x, x.mean()) or np.allclose(y, y.mean()):
        return float(np.nan)
    # use numpy corrcoef (fast) but guard
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else float(np.nan)

# -------------------------
# PSD / coherence (Welch) with SciPy fallback
# -------------------------

def psd_welch(x: ArrayLike, fs: float = 128.0, nperseg: int = 1024, noverlap: int = 512):
    x = np.asarray(x, dtype=float)
    if x.size < 8:
        return np.array([np.nan]), np.array([np.nan])
    if _HAS_SCIPY and welch is not None:
        seg = min(nperseg, x.size)
        nover = min(noverlap, max(0, seg - 1))
        f, Pxx = welch(x, fs=fs, nperseg=seg, noverlap=nover)
        return f, Pxx
    # fallback single-window periodogram (Hann)
    seg = x.size
    win = np.hanning(seg)
    X = np.fft.rfft(x * win)
    Pxx = (np.abs(X)**2) / (fs * np.sum(win**2))
    f = np.fft.rfftfreq(seg, 1.0 / fs)
    return f, Pxx

def mscohere(x: ArrayLike, y: ArrayLike, fs: float = 128.0, nperseg: int = 1024, noverlap: int = 512):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 8:
        return np.array([np.nan]), np.array([np.nan])
    if _HAS_SCIPY and coherence is not None:
        seg = min(nperseg, x.size)
        nover = min(noverlap, max(0, seg - 1))
        f, Cxy = coherence(x, y, fs=fs, nperseg=seg, noverlap=nover)
        return f, Cxy
    # fallback: single-window estimate (higher variance)
    seg = x.size
    win = np.hanning(seg); norm = fs * np.sum(win**2)
    X = np.fft.rfft(x * win); Y = np.fft.rfft(y * win)
    Sxx = (np.abs(X)**2) / norm
    Syy = (np.abs(Y)**2) / norm
    Sxy = (X * np.conj(Y)) / norm
    Cxy = (np.abs(Sxy)**2) / (Sxx * Syy + 1e-12)
    f = np.fft.rfftfreq(seg, 1.0 / fs)
    return f, np.clip(Cxy, 0.0, 1.0)

# -------------------------
# Permutation test
# -------------------------

def sign_flip_pvalue(d, n_perm=10000, seed=42):
    """
    Two-sided sign-flip permutation test.

    Uses the exact Monte Carlo correction of Phipson & Smyth (2010):
    p = (extreme + 1) / (n_perm + 1), which avoids returning p = 0
    and is an exact test under the null hypothesis.

    Parameters
    ----------
    d : array-like
        Paired differences (or sample under null of symmetric
        distribution around zero).
    n_perm : int, default=10000
        Number of permutation samples.
    seed : int, default=42
        RNG seed.

    Returns
    -------
    float
        Two-sided p-value in [1/(n_perm+1), 1].
        NaN if no finite values in d.

    References
    ----------
    Phipson, B., & Smyth, G. K. (2010). Permutation P-values should
    never be zero: calculating exact P-values when permutations are
    randomly drawn. Statistical Applications in Genetics and Molecular
    Biology, 9(1), Article 39.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]

    if d.size == 0:
        return float(np.nan)

    obs = float(np.mean(d))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm_means = np.mean(signs * d[None, :], axis=1)

    extreme = int(np.sum(np.abs(perm_means) >= abs(obs)))
    return float((extreme + 1) / (n_perm + 1))

# -------------------------
# Small helpers
# -------------------------

def _percentile_bounds(ci_level: float) -> Tuple[float, float]:
    lo = (1.0 - ci_level) / 2.0 * 100.0
    hi = (1.0 + ci_level) / 2.0 * 100.0
    return lo, hi

def _mean_sd_ci(arr: ArrayLike, ci_level: float = 0.95) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return {"n": 0, "mean": float(np.nan), "std": float(np.nan), "ci95": (float(np.nan), float(np.nan))}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else float(np.nan)
    if n > 1:
        tcrit = float(stats.t.ppf(0.5 + ci_level / 2, df=n - 1)) if _HAS_SCIPY else 1.96
        se = std / math.sqrt(n)
        ci = (float(mean - tcrit * se), float(mean + tcrit * se))
    else:
        ci = (mean, mean)
    return {"n": n, "mean": mean, "std": std, "ci95": ci}

def _curve_summary(arr: np.ndarray, ci_level: float = 0.95) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim < 2 or arr.shape[0] == 0:
        # expect shape (n_samples, n_points)
        return {"mean": np.full(arr.shape[1] if arr.ndim > 1 else 1, np.nan),
                "std": np.full(arr.shape[1] if arr.ndim > 1 else 1, np.nan),
                "ci95": (np.full(arr.shape[1] if arr.ndim > 1 else 1, np.nan),
                         np.full(arr.shape[1] if arr.ndim > 1 else 1, np.nan))}
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
    if arr.shape[0] > 1:
        se = std / math.sqrt(arr.shape[0])
        tcrit = float(stats.t.ppf(0.5 + ci_level / 2, df=arr.shape[0] - 1)) if _HAS_SCIPY else 1.96
        ci_lo = mean - tcrit * se
        ci_hi = mean + tcrit * se
        ci95 = (ci_lo, ci_hi)
    else:
        ci95 = (mean, mean)
    return {"mean": mean, "std": std, "ci95": ci95}

def bootstrap_ci_trial(arr, n_boot=10000, ci_level=0.95, seed=42):
    """Percentile bootstrap CI on the mean of a trial-level array."""
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        m = float(np.mean(x)) if x.size else float(np.nan)
        return (m, m)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo = (1 - ci_level) / 2 * 100
    hi = (1 + ci_level) / 2 * 100
    return (float(np.percentile(boots, lo)),
            float(np.percentile(boots, hi)))

# ============================================================
# Population-level effect size helper
# ============================================================
# Cohen's dz is computed on the 18 independent participant-level
# effects, NOT on the 40 correlated trial-level differences.
# The trial-level dz (produced internally by _paired_summary) is
# retained for descriptive use only and is NOT reported in the
# main manuscript table.
def population_cohens_dz(participant_effects):
    """
    Cohen's dz for a paired within-subject design at the
    population level.

    Parameters
    ----------
    participant_effects : array-like
        One paired effect per participant (length = N_subjects).
        Each value should be a participant-level mean of the
        within-subject paired differences.

    Returns
    -------
    float
        mean(participant_effects) / std(participant_effects, ddof=1).
        NaN if fewer than two finite values.
    """
    x = np.asarray(participant_effects, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float(np.nan)
    sd = float(np.std(x, ddof=1))
    if sd == 0.0:
        return float(np.nan)
    return float(np.mean(x) / sd)

# -------------------------
# Paired summary: tests + CI + effect sizes
# -------------------------

def _paired_summary(
    d: ArrayLike,
    ci_level: float = 0.95,
    n_perm: int = 10000,
    n_boot: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    out = {
        "n": int(d.size),
        "mean": float(np.nan),
        "std": float(np.nan),
        "se": float(np.nan),
        "ci_t": (float(np.nan), float(np.nan)),
        "ci_boot": (float(np.nan), float(np.nan)),
        "tests": {"sign_perm_p": float(np.nan), "wilcoxon_p": float(np.nan), "ttest_p": float(np.nan)},
        "effect": {"cohen_dz": float(np.nan), "hedges_g": float(np.nan)}
    }
    if d.size == 0:
        return out
    mean = float(np.mean(d))
    std = float(np.std(d, ddof=1)) if d.size > 1 else float(np.nan)
    se = float(std / math.sqrt(d.size)) if d.size > 1 else float(np.nan)
    out.update({"mean": mean, "std": std, "se": se})

    # t-based CI
    if d.size > 1:
        tcrit = float(stats.t.ppf(0.5 + ci_level / 2.0, df=d.size - 1)) if _HAS_SCIPY else 1.96
        ci_t = (mean - tcrit * se, mean + tcrit * se)
    else:
        ci_t = (mean, mean)
    out["ci_t"] = (float(ci_t[0]), float(ci_t[1]))

    # bootstrap CI (percentile)
    if d.size >= 1:
        rng = np.random.default_rng(seed)
        boots = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
        lo, hi = _percentile_bounds(ci_level)
        out["ci_boot"] = (float(np.percentile(boots, lo)), float(np.percentile(boots, hi)))

    # sign-flip permutation (two-sided)
    out["tests"]["sign_perm_p"] = float(sign_flip_pvalue(d, n_perm=n_perm, seed=seed))

    # Wilcoxon and t-test (if SciPy present)
    if _HAS_SCIPY and d.size > 0:
        try:
            W = stats.wilcoxon(d, alternative='two-sided', zero_method='wilcox', correction=False)
            out["tests"]["wilcoxon_p"] = float(W.pvalue)
        except Exception:
            out["tests"]["wilcoxon_p"] = float(np.nan)
        try:
            T = stats.ttest_1samp(d, 0.0)
            out["tests"]["ttest_p"] = float(T.pvalue)
        except Exception:
            out["tests"]["ttest_p"] = float(np.nan)

    # effect sizes for paired design
    if d.size > 1 and np.isfinite(std) and std > 0:
        dz = mean / std  # Cohen's dz
        df = d.size - 1
        # small-sample correction J
        J = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else 1.0
        g = dz * J
        out["effect"]["cohen_dz"] = float(dz)
        out["effect"]["hedges_g"] = float(g)

    return out

# -------------------------
# Multiple comparison corrections
# -------------------------

def _fdr_bh(pvals: Sequence[Optional[float]]) -> List[float]:
    p = np.asarray([np.nan if x is None else float(x) for x in pvals], dtype=float)
    n = p.size
    padj = np.full(n, np.nan, dtype=float)
    # treat nan as infinite so they go last when sorting
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranks = np.empty(n, dtype=int); ranks[order] = np.arange(1, n + 1)
    finite_mask = np.isfinite(p)
    if not np.any(finite_mask):
        return padj.tolist()
    p_f = p[finite_mask]
    r_f = ranks[finite_mask]
    padj_f = p_f * n / r_f
    # ensure monotonicity (nonincreasing after sorting ascending)
    idx_sort = np.argsort(p_f)
    padj_sorted = padj_f[idx_sort]
    padj_monotone = np.minimum.accumulate(padj_sorted[::-1])[::-1]
    padj_f_final = np.minimum(1.0, padj_monotone)
    padj_f_final_reordered = np.empty_like(padj_f_final)
    padj_f_final_reordered[idx_sort] = padj_f_final
    padj[finite_mask] = padj_f_final_reordered
    return padj.tolist()

def _holm_bonferroni(pvals: Sequence[Optional[float]]) -> List[float]:
    p = np.asarray([np.nan if x is None else float(x) for x in pvals], dtype=float)
    n = p.size
    padj = np.full(n, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    if not np.any(finite_mask):
        return padj.tolist()
    p_f = p[finite_mask]
    # sort ascending
    order = np.argsort(p_f)
    padj_f = np.empty_like(p_f)
    # step-up multiplicative factor: m = n - i
    for rank_idx, idx in enumerate(order):
        m = n - rank_idx
        padj_f[idx] = min(1.0, p_f[idx] * m)
    # enforce monotonicity (non-decreasing with sorted order)
    padj_f_sorted = np.maximum.accumulate(padj_f[order])
    padj_f_monotone = np.empty_like(padj_f)
    padj_f_monotone[order] = padj_f_sorted
    padj[finite_mask] = padj_f_monotone
    return padj.tolist()

# -------------------------
# Friedman + RM-ANOVA
# -------------------------

def _rm_anova_oneway_rm(mat: np.ndarray) -> Dict[str, Any]:
    """
    One-way repeated measures ANOVA (subjects x conditions).
    Returns dict with F, p, eta2 (partial), n, k.
    Note: This is the classic univariate RM-ANOVA (assumes sphericity).
    For k>2 users should consider testing/adjusting for sphericity (GG/HF) externally.
    """
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2:
        raise ValueError("mat must be 2D (n_subjects, k_conditions)")
    n, k = mat.shape
    if n < 2 or k < 2:
        return {"F": np.nan, "p": np.nan, "eta2": np.nan, "n": n, "k": k}
    GM = np.nanmean(mat)
    mean_j = np.nanmean(mat, axis=0)
    mean_i = np.nanmean(mat, axis=1)
    SS_total = np.nansum((mat - GM) ** 2)
    SS_cond = n * np.nansum((mean_j - GM) ** 2)
    SS_subj = k * np.nansum((mean_i - GM) ** 2)
    SS_error = SS_total - SS_cond - SS_subj
    df_cond = k - 1
    df_error = (k - 1) * (n - 1)
    if df_error <= 0:
        return {"F": np.nan, "p": np.nan, "eta2": np.nan, "n": n, "k": k}
    MS_cond = SS_cond / df_cond
    MS_error = SS_error / df_error
    if MS_error <= 0:
        F = np.nan; p = np.nan
    else:
        F = MS_cond / MS_error
        p = float(stats.f.sf(F, df_cond, df_error)) if _HAS_SCIPY else np.nan
    denom = SS_cond + SS_error
    eta2 = float(SS_cond / denom) if denom > 0 else np.nan
    return {"F": float(F) if np.isfinite(F) else np.nan,
            "p": float(p) if np.isfinite(p) else np.nan,
            "eta2": float(eta2) if np.isfinite(eta2) else np.nan,
            "n": n, "k": k}

def _friedman_test_matrix(mat: np.ndarray) -> Dict[str, Any]:
    """
    Friedman test for mat shape (n_subjects, k_conditions).
    Returns chi2, p, kendall_w, n, k. Uses SciPy for chi2/p when available,
    but always computes kendall_w from ranks (S-statistic) to avoid mismatch
    between different χ² definitions or tie-corrections.
    """
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2:
        raise ValueError("mat must be 2D (n_subjects, k_conditions)")
    n, k = mat.shape
    if n < 2 or k < 2:
        return {"chi2": np.nan, "p": np.nan, "kendall_w": np.nan, "n": n, "k": k}

    # complete rows only
    ok_rows = np.all(np.isfinite(mat), axis=1)
    mat_ok = mat[ok_rows, :]
    n_ok = mat_ok.shape[0]
    if n_ok < 2:
        return {"chi2": np.nan, "p": np.nan, "kendall_w": np.nan, "n": n_ok, "k": k}

    # Compute ranks per subject (average ranks for ties)
    try:
        # prefer scipy.stats.rankdata if available
        from scipy import stats as _sci_stats
        ranks = np.apply_along_axis(lambda row: _sci_stats.rankdata(row, method='average'), axis=1, arr=mat_ok)
    except Exception:
        # fallback to simple average-rank implementation
        def rankdata_avg(a):
            a = np.asarray(a)
            order = np.argsort(a)
            ranks = np.empty_like(a, dtype=float)
            sorted_a = a[order]
            i = 0
            nrow = len(a)
            while i < nrow:
                j = i + 1
                while j < nrow and sorted_a[j] == sorted_a[i]:
                    j += 1
                avg_rank = (i + 1 + j) / 2.0
                ranks[order[i:j]] = avg_rank
                i = j
            return ranks
        ranks = np.apply_along_axis(rankdata_avg, 1, mat_ok)

    # compute S and Kendall's W from ranks (rows = subjects)
    Rj = np.nansum(ranks, axis=0)                # sum of ranks per condition
    S = np.sum((Rj - n_ok * (k + 1) / 2.0) ** 2) # sum of squared deviations
    denom = (n_ok ** 2) * (k ** 3 - k)
    kendall_w = 12.0 * S / denom if denom != 0 else np.nan

    # numeric safety: clamp tiny numerical overshoot to 1.0
    if not np.isnan(kendall_w):
        if kendall_w > 1.0 and kendall_w <= 1.0 + 1e-12:
            kendall_w = 1.0
        elif kendall_w < 0 and kendall_w >= -1e-12:
            kendall_w = 0.0

    # Compute chi2/p using SciPy if available (for reporting), but don't derive W from it
    chi2 = np.nan
    p = np.nan
    if _HAS_SCIPY:
        try:
            args = [mat_ok[:, j] for j in range(k)]
            res = stats.friedmanchisquare(*args)
            chi2 = float(res.statistic)
            p = float(res.pvalue)
        except Exception:
            chi2 = np.nan
            p = np.nan
    else:
        # manual chi2 (no tie-correction)
        chi2 = (12.0 / (n_ok * k * (k + 1))) * np.sum(Rj ** 2) - 3.0 * n_ok * (k + 1)
        p = float(stats.chi2.sf(chi2, df=k - 1)) if _HAS_SCIPY else np.nan

    return {
        "chi2": float(chi2) if np.isfinite(chi2) else np.nan,
        "p": float(p) if np.isfinite(p) else np.nan,
        "kendall_w": float(kendall_w) if np.isfinite(kendall_w) else np.nan,
        "n": int(n_ok),
        "k": int(k)
    }


# -------------------------
# Robust matrix builder (subjects x conditions)
# -------------------------

def build_matrix_from_extractor(
    fold_summaries_dict: Dict[str, List[Dict[str, Any]]],
    extractor,
) -> np.ndarray:
    """
    Build numeric matrix (subjects x conditions).
    fold_summaries_dict: mapping condition_label -> list_of_fold_summaries (length K)
    extractor: function(fs) -> scalar (or raises) used to extract metric per fold
    Returns M shape (K, num_conditions) filled with np.nan where missing/invalid.
    """
    conditions = list(fold_summaries_dict.keys())
    if len(conditions) == 0:
        return np.empty((0, 0), dtype=float)
    K = len(next(iter(fold_summaries_dict.values())))
    # sanity: ensure all lists same length
    for lbl, lst in fold_summaries_dict.items():
        if len(lst) != K:
            raise ValueError("All conditions must have same number of fold summaries")
    M = np.full((K, len(conditions)), np.nan, dtype=float)
    for j, lbl in enumerate(conditions):
        for i, fs in enumerate(fold_summaries_dict[lbl]):
            try:
                v = extractor(fs)
            except Exception:
                v = np.nan
            try:
                v = float(v)
            except Exception:
                v = np.nan
            M[i, j] = v if np.isfinite(v) else np.nan
    return M

# -------------------------
# Helpers for fold-level metrics extraction
# -------------------------

def _extract_fold_coh_mean(fold_summary: Dict[str, Any], band: str) -> float:
    try:
        return float(fold_summary["bands"]["coh_bands"][band]["mean"])
    except Exception:
        return float(np.nan)

def _compute_fold_rbp_error(fold_summary: Dict[str, Any], band: str) -> float:
    trials = fold_summary.get("frequency_analysis_trials", [])
    if len(trials) == 0:
        return float(np.nan)
    diffs = []
    for tr in trials:
        try:
            e = abs(float(tr["rbp_pred"][band]) - float(tr["rbp_true"][band]))
            diffs.append(e)
        except Exception:
            continue
    if len(diffs) == 0:
        return float(np.nan)
    return float(np.mean(diffs))

def _collect_common_bands(fold_summaries: List[Dict[str, Any]]) -> List[str]:
    try:
        return list(fold_summaries[0]["bands"]["coh_bands"].keys())
    except Exception:
        return ["theta", "alpha", "beta", "gamma"]

# -------------------------
# Fold-summary (trial-as-unit) summarizer
# -------------------------

def summarize_fold_trials_unit(
    y_true_fold: np.ndarray,
    y_pred_fold: np.ndarray,
    fs: float = 128.0,
    nperseg: int = 1024,
    noverlap: int = 512,
    n_boot: int = 5000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Summarize all trials inside a fold. Inputs:
      - y_true_fold, y_pred_fold: shape (trials, samples_per_trial)
    Returns dict containing trial-level r and rmse arrays, pooled r (z), PSD/coh summaries, band-level summaries.
    """
    y_true_fold = np.asarray(y_true_fold)
    y_pred_fold = np.asarray(y_pred_fold)
    if y_true_fold.ndim != 2 or y_pred_fold.ndim != 2:
        raise ValueError("y_true_fold and y_pred_fold must be 2D arrays (trials, samples)")
    if y_true_fold.shape != y_pred_fold.shape:
        raise ValueError("y_true_fold and y_pred_fold must have identical shapes")
    trials, m = y_true_fold.shape

    # per-trial correlation and RMSE
    r_trials = np.empty(trials, dtype=float)
    rmse_trials = np.empty(trials, dtype=float)
    for i in range(trials):
        x = y_true_fold[i]
        y = y_pred_fold[i]
        r_trials[i] = safe_pearsonr(x, y)
        rmse_trials[i] = float(np.sqrt(np.mean((x - y) ** 2)))

    # stats for r and rmse
    r_valid_mask = np.isfinite(r_trials)
    r_valid = r_trials[r_valid_mask]
    r_mean = float(np.nanmean(r_trials)) if trials > 0 else float(np.nan)
    r_std = float(np.nanstd(r_trials, ddof=1)) if np.sum(np.isfinite(r_trials)) > 1 else float(np.nan)
    rmse_mean = float(np.nanmean(rmse_trials)) if trials > 0 else float(np.nan)
    rmse_sd = float(np.nanstd(rmse_trials, ddof=1)) if np.sum(np.isfinite(rmse_trials)) > 1 else float(np.nan)

    # fisher pooling across trials (z)
    if r_valid.size == 0:
        z_fold = float(np.nan)
        r_pooled = float(np.nan)
        S_fold = 0.0
        se_theory = se_emp = se_boot = float(np.nan)
        r_ci_boot = (float(np.nan), float(np.nan))
    else:
        z_valid = fisher_z(r_valid)
        # weights: approximate effective sample size per trial ~ m - 3, but at least 1
        w = np.full(z_valid.shape, max(int(m - 3), 1), dtype=float)
        S_fold = float(np.sum(w))
        z_fold = float(np.sum(w * z_valid) / S_fold)
        r_pooled = float(inv_fisher_z(z_fold))
        se_theory = float(1.0 / math.sqrt(S_fold)) if S_fold > 0 else float(np.nan)
        se_emp = float(np.std(z_valid, ddof=1) / math.sqrt(z_valid.size)) if z_valid.size > 1 else float(np.nan)
        # bootstrap on z
        rng = np.random.default_rng(seed)
        n = z_valid.size
        boot = rng.choice(z_valid, size=(n_boot, n), replace=True).mean(axis=1)
        boot = boot[np.isfinite(boot)]
        if boot.size > 1:
            se_boot = float(np.std(boot, ddof=1))
            z_ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
            r_ci_boot = (float(inv_fisher_z(z_ci[0])), float(inv_fisher_z(z_ci[1])))
        else:
            se_boot = float(np.nan)
            r_ci_boot = (r_pooled, r_pooled)

    # frequency analysis per trial
    freq_bands = ["theta", "alpha", "beta", "gamma"]
    freq_grid = None
    coh_grid = None
    psd_true_all, psd_pred_all, coh_all = [], [], []
    frequency_analysis_trials = []
    rbp_true_by_band = {b: [] for b in freq_bands}
    rbp_pred_by_band = {b: [] for b in freq_bands}
    coh_band_by_band = {b: [] for b in freq_bands}

    for i in range(trials):
        res = eeg_frequency_analysis(y_true_fold[i], y_pred_fold[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        frequency_analysis_trials.append(res)
        if freq_grid is None:
            freq_grid = res["f_psd_true"]
        if coh_grid is None:
            coh_grid = res["f_coh"]
        # align PSDs to common grid
        if not np.array_equal(res["f_psd_true"], freq_grid):
            psd_true_all.append(np.interp(freq_grid, res["f_psd_true"], res["psd_true"]))
            psd_pred_all.append(np.interp(freq_grid, res["f_psd_pred"], res["psd_pred"]))
        else:
            psd_true_all.append(res["psd_true"])
            psd_pred_all.append(res["psd_pred"])
        if not np.array_equal(res["f_coh"], coh_grid):
            coh_all.append(np.interp(coh_grid, res["f_coh"], res["coh_vals"]))
        else:
            coh_all.append(res["coh_vals"])
        for b in freq_bands:
            rbp_true_by_band[b].append(res["rbp_true"][b])
            rbp_pred_by_band[b].append(res["rbp_pred"][b])
            coh_band_by_band[b].append(res["coh_bands"][b])

    psd_true_all = np.asarray(psd_true_all)
    psd_pred_all = np.asarray(psd_pred_all)
    coh_all = np.asarray(coh_all)

    freq_summary = {
        "f_psd": freq_grid,
        "psd_true": _curve_summary(psd_true_all),
        "psd_pred": _curve_summary(psd_pred_all),
        "f_coh": coh_grid,
        "coh": _curve_summary(coh_all)
    }

    # band-level summaries & tests (true vs pred within fold)
    bands_summary = {"rbp_true": {}, "rbp_pred": {}, "coh_bands": {}, "tests": {}}
    pvals = []
    for b in freq_bands:
        arr_true = np.array(rbp_true_by_band[b], dtype=float)
        arr_pred = np.array(rbp_pred_by_band[b], dtype=float)
        arr_coh = np.array(coh_band_by_band[b], dtype=float)
        bands_summary["rbp_true"][b] = _mean_sd_ci(arr_true)
        bands_summary["rbp_pred"][b] = _mean_sd_ci(arr_pred)
        bands_summary["coh_bands"][b] = _mean_sd_ci(arr_coh)
        if np.all(np.isfinite(arr_true)) and np.all(np.isfinite(arr_pred)):
            d = arr_pred - arr_true
            p_val = sign_flip_pvalue(d, n_perm=10000, seed=seed)
        else:
            p_val = float(np.nan)
        bands_summary["tests"][b] = {"rbp_true_vs_pred_p": p_val}
        pvals.append(p_val)
    # FDR on rbp tests (within fold)
    pvals_arr = np.asarray(pvals, dtype=float)
    if np.any(np.isfinite(pvals_arr)):
        # fallback to statsmodels only if available; else use _fdr_bh implementation
        try:
            from statsmodels.stats.multitest import multipletests
            finite_mask = np.isfinite(pvals_arr)
            reject, pvals_corr, _, _ = multipletests(pvals_arr[finite_mask], alpha=0.05, method='fdr_bh')
            # map back
            k = len(pvals)
            idx = 0
            for i, b in enumerate(freq_bands):
                if finite_mask[i]:
                    bands_summary["tests"][b]["rbp_true_vs_pred_p_fdr"] = float(pvals_corr[idx])
                    bands_summary["tests"][b]["reject_fdr"] = bool(reject[idx])
                    idx += 1
                else:
                    bands_summary["tests"][b]["rbp_true_vs_pred_p_fdr"] = float(np.nan)
                    bands_summary["tests"][b]["reject_fdr"] = False
        except Exception:
            padj = _fdr_bh(pvals_arr)
            for i, b in enumerate(freq_bands):
                bands_summary["tests"][b]["rbp_true_vs_pred_p_fdr"] = float(padj[i]) if np.isfinite(pvals_arr[i]) else float(np.nan)
                bands_summary["tests"][b]["reject_fdr"] = bool(np.isfinite(padj[i]) and padj[i] < 0.05)
    return {
        "trials": trials,
        "r_trials": r_trials,
        "rmse_trials": rmse_trials,
        "r_mean": r_mean, "r_std": r_std,
        "rmse_mean": rmse_mean, "rmse_sd": rmse_sd,
        "z_fold": z_fold, "S_fold": S_fold,
        "r_pooled": r_pooled,
        "se_theory": se_theory, "se_emp": se_emp, "se_boot": se_boot,
        "r_pooled_ci_boot": r_ci_boot,
        "frequency_analysis_trials": frequency_analysis_trials,
        "freq_summary": freq_summary,
        "bands": bands_summary,
    }

# -------------------------
# Aggregate across folds (fold-as-unit)
# -------------------------

def aggregate_across_folds(
    fold_summaries: List[Dict[str, Any]],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95
) -> Dict[str, Any]:
    K = len(fold_summaries)
    if K == 0:
        raise ValueError("No folds provided to aggregate_across_folds")
    # RMSE
    rmse_means = np.array([float(f['rmse_mean']) for f in fold_summaries], dtype=float)
    rmse_mean = float(np.nanmean(rmse_means))
    rmse_std = float(np.nanstd(rmse_means, ddof=1)) if K > 1 else 0.0
    SE_rmse_mean = rmse_std / math.sqrt(K) if K > 1 else 0.0
    if K > 1 and _HAS_SCIPY:
        tcrit = float(stats.t.ppf(0.5 + ci_level / 2, df=K - 1))
    elif K > 1:
        tcrit = 1.96
    else:
        tcrit = 0.0
    rmse_ci = (rmse_mean - tcrit * SE_rmse_mean, rmse_mean + tcrit * SE_rmse_mean) if K > 1 else (rmse_mean, rmse_mean)
    # bootstrap CI
    if K > 1:
        rng = np.random.default_rng(42)
        boot_means = rng.choice(rmse_means, size=(n_bootstrap, K), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot_means, [(1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100])
        rmse_ci_boot = (float(lo), float(hi))
    else:
        rmse_ci_boot = (rmse_mean, rmse_mean)

    # correlation z across folds
    z_folds = np.array([f.get('z_fold', np.nan) for f in fold_summaries], dtype=float)
    valid_mask = np.isfinite(z_folds)
    if np.sum(valid_mask) == 0:
        r_fold_mean_from_z = np.nan; r_fold_ci = (np.nan, np.nan); r_fold_ci_boot = (np.nan, np.nan)
    else:
        z_valid = z_folds[valid_mask]
        z_mean = float(np.mean(z_valid))
        z_std = float(np.std(z_valid, ddof=1)) if z_valid.size > 1 else 0.0
        SE_z_mean = z_std / math.sqrt(z_valid.size) if z_valid.size > 1 else 0.0
        if z_valid.size > 1 and _HAS_SCIPY:
            tcrit_z = float(stats.t.ppf(0.5 + ci_level / 2, df=z_valid.size - 1))
        elif z_valid.size > 1:
            tcrit_z = 1.96
        else:
            tcrit_z = 0.0
        z_ci = (z_mean - tcrit_z * SE_z_mean, z_mean + tcrit_z * SE_z_mean) if z_valid.size > 1 else (z_mean, z_mean)
        r_fold_mean_from_z = float(inv_fisher_z(z_mean))
        r_fold_ci = (float(inv_fisher_z(z_ci[0])), float(inv_fisher_z(z_ci[1])))
        # bootstrap
        if z_valid.size > 1:
            rng = np.random.default_rng(123)
            boot_means = rng.choice(z_valid, size=(n_bootstrap, z_valid.size), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot_means, [(1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100])
            r_fold_ci_boot = (float(inv_fisher_z(lo)), float(inv_fisher_z(hi)))
        else:
            r_fold_ci_boot = (r_fold_mean_from_z, r_fold_mean_from_z)

    # pooled correlation across folds (weighted by S_fold)
    S_total = float(sum(float(f.get('S_fold', 0.0)) for f in fold_summaries))
    if S_total > 0:
        z_weighted = float(sum((float(f.get('z_fold', 0.0)) * float(f.get('S_fold', 0.0))) for f in fold_summaries) / S_total)
        r_pooled_overall = float(inv_fisher_z(z_weighted))
        SE_z_total = 1.0 / math.sqrt(S_total)
        z_ci_tot = (z_weighted - 1.96 * SE_z_total, z_weighted + 1.96 * SE_z_total)
        r_pooled_overall_ci = (float(inv_fisher_z(z_ci_tot[0])), float(inv_fisher_z(z_ci_tot[1])))
    else:
        r_pooled_overall = np.nan; r_pooled_overall_ci = (np.nan, np.nan)

    # frequency band summaries if available
    bands = ["theta", "alpha", "beta", "gamma"]
    band_summary = {}
    if all("bands" in f for f in fold_summaries):
        rng = np.random.default_rng(999)
        for b in bands:
            true_vals = np.array([f["bands"]["rbp_true"][b]["mean"] for f in fold_summaries], dtype=float)
            pred_vals = np.array([f["bands"]["rbp_pred"][b]["mean"] for f in fold_summaries], dtype=float)
            coh_vals = np.array([f["bands"]["coh_bands"][b]["mean"] for f in fold_summaries], dtype=float)
            # bootstrap helper
            def _bootstrap_ci(vals):
                if K > 1:
                    boot_means = rng.choice(vals, size=(n_bootstrap, K), replace=True).mean(axis=1)
                    lo, hi = np.percentile(boot_means, [(1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100])
                    return (float(lo), float(hi))
                else:
                    return (float(vals[0]), float(vals[0]))
            band_summary[b] = {
                "rbp_true": {"mean_sd_ci": _mean_sd_ci(true_vals, ci_level), "ci_boot": _bootstrap_ci(true_vals)},
                "rbp_pred": {"mean_sd_ci": _mean_sd_ci(pred_vals, ci_level), "ci_boot": _bootstrap_ci(pred_vals)},
                "coh": {"mean_sd_ci": _mean_sd_ci(coh_vals, ci_level), "ci_boot": _bootstrap_ci(coh_vals)},
            }
    # curves summary
    curves_summary = {}
    if all("freq_summary" in f for f in fold_summaries):
        def summarize_curves(curve_key):
            Fs, mats = [], []
            for fs in fold_summaries:
                fr = fs["freq_summary"]
                if curve_key == "coh":
                    F = fr["f_coh"]; C = fr["coh"]["mean"]
                else:
                    F = fr["f_psd"]; C = fr[curve_key]["mean"]
                Fs.append(F); mats.append(C)
            f_common = Fs[0]
            cur_mat = []
            for F, C in zip(Fs, mats):
                if np.array_equal(F, f_common):
                    cur_mat.append(C)
                else:
                    cur_mat.append(np.interp(f_common, F, C))
            cur_mat = np.vstack(cur_mat)
            m = np.nanmean(cur_mat, axis=0)
            s = np.nanstd(cur_mat, axis=0, ddof=1) if cur_mat.shape[0] > 1 else np.zeros_like(m)
            if _HAS_SCIPY and cur_mat.shape[0] > 1:
                tcrit = float(stats.t.ppf(0.5 + ci_level / 2, df=cur_mat.shape[0] - 1))
            else:
                tcrit = 1.96 if cur_mat.shape[0] > 1 else 0.0
            se = s / math.sqrt(max(1, cur_mat.shape[0]))
            ci_lo, ci_hi = m - tcrit * se, m + tcrit * se
            return {"f": f_common, "mean": m, "ci_t": (ci_lo, ci_hi)}
        curves_summary["psd_true"] = summarize_curves("psd_true")
        curves_summary["psd_pred"] = summarize_curves("psd_pred")
        curves_summary["coh"] = summarize_curves("coh")

    return {
        "rmse_fold_mean": rmse_mean,
        "rmse_fold_std": rmse_std,
        "rmse_fold_ci": rmse_ci,
        "rmse_fold_ci_boot": rmse_ci_boot,
        "r_fold_mean_from_z": r_fold_mean_from_z,
        "r_fold_ci_from_z": r_fold_ci,
        "r_fold_ci_boot": r_fold_ci_boot,
        "r_pooled_overall": r_pooled_overall,
        "r_pooled_overall_ci": r_pooled_overall_ci,
        "band_summary": band_summary,
        "curves_summary": curves_summary,
    }

# -------------------------
# Pairwise compare two methods (fold-level)
# -------------------------

def compare_methods_across_folds(
    fold_summaries_A: List[Dict[str, Any]],
    fold_summaries_B: List[Dict[str, Any]],
    method_names: Tuple[str, str] = ("A", "B"),
    ci_level: float = 0.95,
    n_perm: int = 10000,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    if len(fold_summaries_A) != len(fold_summaries_B):
        raise ValueError("A and B must have the same number of folds")
    K = len(fold_summaries_A)
    # RMSE
    rmse_A = np.array([float(fs.get("rmse_mean", np.nan)) for fs in fold_summaries_A], dtype=float)
    rmse_B = np.array([float(fs.get("rmse_mean", np.nan)) for fs in fold_summaries_B], dtype=float)
    mask_rmse = np.isfinite(rmse_A) & np.isfinite(rmse_B)
    d_rmse = (rmse_A - rmse_B)[mask_rmse]
    rmse_summary = _paired_summary(d_rmse, ci_level=ci_level, n_perm=n_perm, n_boot=n_boot, seed=seed)

    # r (z-space)
    zA = np.array([float(fs.get("z_fold", np.nan)) for fs in fold_summaries_A], dtype=float)
    zB = np.array([float(fs.get("z_fold", np.nan)) for fs in fold_summaries_B], dtype=float)
    mask_z = np.isfinite(zA) & np.isfinite(zB)
    d_z = (zA - zB)[mask_z]
    r_summary_z = _paired_summary(d_z, ci_level=ci_level, n_perm=n_perm, n_boot=n_boot, seed=seed)
    mean_r_A = float(inv_fisher_z(np.nanmean(zA[mask_z]))) if np.any(mask_z) else float(np.nan)
    mean_r_B = float(inv_fisher_z(np.nanmean(zB[mask_z]))) if np.any(mask_z) else float(np.nan)
    delta_mean_r = float(mean_r_A - mean_r_B)

    # coherence per-band & rbp-error per-band
    bands = _collect_common_bands(fold_summaries_A)
    coh_band = {"per_band": {}, "p_adj": {"bh": {}, "holm": {}}}
    pvals_coh = []
    for b in bands:
        A_vals = np.array([_extract_fold_coh_mean(fs, b) for fs in fold_summaries_A], dtype=float)
        B_vals = np.array([_extract_fold_coh_mean(fs, b) for fs in fold_summaries_B], dtype=float)
        m = np.isfinite(A_vals) & np.isfinite(B_vals)
        d = (A_vals - B_vals)[m]
        summ = _paired_summary(d, ci_level=ci_level, n_perm=n_perm, n_boot=n_boot, seed=seed)
        coh_band["per_band"][b] = {"A": A_vals[m], "B": B_vals[m], "delta": d, "summary": summ}
        pvals_coh.append(summ["tests"]["sign_perm_p"])
    padj_bh = _fdr_bh(pvals_coh); padj_holm = _holm_bonferroni(pvals_coh)
    for b, p_bh, p_holm in zip(bands, padj_bh, padj_holm):
        coh_band["p_adj"]["bh"][b] = float(p_bh) if np.isfinite(p_bh) else float(np.nan)
        coh_band["p_adj"]["holm"][b] = float(p_holm) if np.isfinite(p_holm) else float(np.nan)

    rbp_err = {"per_band": {}, "p_adj": {"bh": {}, "holm": {}}}
    pvals_err = []
    for b in bands:
        A_vals = np.array([_compute_fold_rbp_error(fs, b) for fs in fold_summaries_A], dtype=float)
        B_vals = np.array([_compute_fold_rbp_error(fs, b) for fs in fold_summaries_B], dtype=float)
        m = np.isfinite(A_vals) & np.isfinite(B_vals)
        d = (A_vals - B_vals)[m]
        summ = _paired_summary(d, ci_level=ci_level, n_perm=n_perm, n_boot=n_boot, seed=seed)
        rbp_err["per_band"][b] = {"A": A_vals[m], "B": B_vals[m], "delta": d, "summary": summ}
        pvals_err.append(summ["tests"]["sign_perm_p"])
    padj_bh = _fdr_bh(pvals_err); padj_holm = _holm_bonferroni(pvals_err)
    for b, p_bh, p_holm in zip(bands, padj_bh, padj_holm):
        rbp_err["p_adj"]["bh"][b] = float(p_bh) if np.isfinite(p_bh) else float(np.nan)
        rbp_err["p_adj"]["holm"][b] = float(p_holm) if np.isfinite(p_holm) else float(np.nan)

    return {
        "methods": {"A": method_names[0], "B": method_names[1]},
        "K": K,
        "rmse": {"A": rmse_A[mask_rmse], "B": rmse_B[mask_rmse], "delta": d_rmse, "summary": rmse_summary},
        "r": {"A_z": zA[mask_z], "B_z": zB[mask_z], "delta_z": d_z, "summary_z": r_summary_z,
              "mean_r_A": mean_r_A, "mean_r_B": mean_r_B, "delta_mean_r": delta_mean_r},
        "coherence": coh_band,
        "rbp_error": rbp_err,
        "bands": bands
    }



# -------------------------
# Participant-level trial summary (merged from fold + aggregate)
# -------------------------

def summarize_trials(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fs: float = 128.0,
    nperseg: int = 1024,
    noverlap: int = 512,
    n_boot: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute trial-level metrics and pooled statistics for one subject
    and one method.

    The unit of analysis is the 60-second trial. Metrics are first
    computed independently for each trial, then aggregated across
    trials to produce pooled point estimates and bootstrap confidence
    intervals. 

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth reference signal of shape
        (n_trials, samples_per_trial), e.g. (40, 7680) for 40 trials
        of 60 seconds at 128 Hz.
    y_pred : np.ndarray
        Predicted signal with the same shape as `y_true`.
    fs : float, default=128.0
        Sampling frequency of the signals in Hz.
    nperseg : int, default=1024
        Segment length for Welch spectral estimation.
    noverlap : int, default=512
        Overlap between Welch segments.
    n_boot : int, default=10000
        Number of bootstrap resamples for confidence intervals.
    ci_level : float, default=0.95
        Confidence level for the intervals (0 < ci_level < 1).
    seed : int, default=42
        Random seed for reproducible bootstrap resampling.

    Returns
    -------
    dict
        A dictionary containing trial-level data, pooled statistics,
        and descriptive summaries:

        Primary trial-level data
        ........................
        trials : int
            Number of trials processed.
        r_trials : np.ndarray, shape (trials,)
            Pearson correlation between reference and predicted signal
            for each trial.
        rmse_trials : np.ndarray, shape (trials,)
            Root-mean-square error between reference and predicted
            signal for each trial.
        frequency_analysis_trials : list of dict
            Per-trial spectral results, each containing PSDs, coherence
            spectra, relative band power, and band-averaged coherence.

        Pooled correlation statistics
        .............................
        r_pooled : float
            Fisher-z pooled correlation across all trials, back-
            transformed to the correlation scale.
        r_pooled_ci_boot : tuple of float
            Bootstrap confidence interval for `r_pooled`, expressed on
            the correlation scale.
        r_mean : float
            Arithmetic mean of per-trial correlations.
        r_std : float
            Standard deviation of per-trial correlations.

        Pooled RMSE statistics
        ......................
        rmse_mean : float
            Mean RMSE across trials.
        rmse_sd : float
            Standard deviation of RMSE across trials.
        rmse_trial_ci_boot : tuple of float
            Bootstrap confidence interval for `rmse_mean`.

        Band-level summaries
        ....................
        band_summary : dict
            Per-band aggregated statistics. Keys are band names
            returned by `eeg_analysis_bands` (theta, alpha, beta).
            Each band entry contains:
              - "rbp_true" : dict with "mean_sd_ci" and "ci_boot"
                for relative band power of the reference signal
              - "rbp_pred" : dict with "mean_sd_ci" and "ci_boot"
                for relative band power of the predicted signal
              - "coh"      : dict with "mean_sd_ci" and "ci_boot"
                for band-averaged coherence
              - "_raw"     : dict with per-trial arrays for downstream
                paired testing ("coh_trials", "rbp_true_trials",
                "rbp_pred_trials")

        Spectrum-level summaries
        ........................
        curves_summary : dict
            Mean spectral curves with confidence intervals, one entry
            per curve type:
              - "psd_true" : reference PSD
              - "psd_pred" : predicted PSD
              - "coh"      : magnitude-squared coherence
            Each entry is a dict with keys "f" (frequency axis),
            "mean" (mean curve), and "ci_t" (t-based confidence
            interval on the mean curve across trials).

    Notes
    -----
    - Correlation pooling uses Fisher's z-transform with equal weight
      per trial. Unequal weighting based on samples per trial is not
      appropriate here, because samples within a 60-second EEG trial
      are temporally correlated and do not represent independent
      observations.
    - Bootstrap resampling is performed over the trial dimension, so
      the resulting intervals quantify trial-to-trial variability
      within a subject.
    - Spectral analysis is restricted to theta (4-8 Hz), alpha
      (8-13 Hz), and beta (13-25 Hz), consistent with the 25 Hz
      Nyquist limit of the facial video.
    - Per-trial RMSE and correlation are mathematically coupled for
      standardized signals. They are reported together as complementary
      views of the same prediction, not as independent evidence.

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("Inputs must be 2D arrays (n_trials, samples)")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")

    n_trials, m = y_true.shape
    rng = np.random.default_rng(seed)

        # ----------------------------------------------------------------
    # 1. Per-trial r and RMSE (PRIMARY INFERENTIAL DATA)
    # ----------------------------------------------------------------
    r_trials = np.empty(n_trials, dtype=float)
    rmse_trials = np.empty(n_trials, dtype=float)

    for i in range(n_trials):
        r_trials[i] = safe_pearsonr(y_true[i], y_pred[i])
        rmse_trials[i] = float(np.sqrt(np.mean((y_true[i] - y_pred[i]) ** 2)))

    # IMPORTANT: r and RMSE have independent validity.
    r_valid = r_trials[np.isfinite(r_trials)]
    rmse_valid = rmse_trials[np.isfinite(rmse_trials)]

    n_r_valid = int(r_valid.size)
    n_rmse_valid = int(rmse_valid.size)

    # ----------------------------------------------------------------
    # 2. Fisher-z pooling at trial level (weight = 1)
    # ----------------------------------------------------------------
    if n_r_valid > 0:
        z_all = fisher_z(r_valid)
        z_mean = float(np.mean(z_all))
        z_std = float(np.std(z_all, ddof=1)) if n_r_valid > 1 else 0.0
        r_pooled = float(inv_fisher_z(z_mean))
    else:
        z_all, z_mean, z_std, r_pooled = np.array([]), float(np.nan), float(np.nan), float(np.nan)

    # ----------------------------------------------------------------
    # 3. Bootstrap CIs
    # ----------------------------------------------------------------
    lo, hi = (1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100

    # ---- Pearson r CI ----
    if n_r_valid > 1:
        boot_z = rng.choice(z_all, size=(n_boot, n_r_valid), replace=True).mean(axis=1)
        r_ci_boot = (float(inv_fisher_z(np.percentile(boot_z, lo))),
                     float(inv_fisher_z(np.percentile(boot_z, hi))))
    elif n_r_valid == 1:
        r_ci_boot = (float(r_pooled), float(r_pooled))
    else:
        r_ci_boot = (np.nan, np.nan)

    # ---- RMSE CI ----
    if n_rmse_valid > 1:
        boot_rmse = rng.choice(rmse_valid, size=(n_boot, n_rmse_valid), replace=True).mean(axis=1)
        rmse_ci_boot = (float(np.percentile(boot_rmse, lo)),
                        float(np.percentile(boot_rmse, hi)))
    elif n_rmse_valid == 1:
        rmse_ci_boot = (float(rmse_valid[0]), float(rmse_valid[0]))
    else:
        rmse_ci_boot = (np.nan, np.nan)

    # ----------------------------------------------------------------
    # 4. Per-trial frequency analysis
    # ----------------------------------------------------------------
    freq_bands = eeg_analysis_bands()   # ["theta", "alpha", "beta"]
    frequency_analysis_trials = []
    for i in range(n_trials):
        frequency_analysis_trials.append(
            eeg_frequency_analysis(y_true[i], y_pred[i],
                                   fs=fs, nperseg=nperseg, noverlap=noverlap)
        )

    # ----------------------------------------------------------------
    # 5. Band-level trial-level summaries (interface-compatible)
    # ----------------------------------------------------------------
    def _boot_mean_ci(vals: np.ndarray) -> Tuple[float, float]:
        """
        Bootstrap CI on the mean. Accepts arrays that may contain NaN;
        the NaNs are dropped internally for the computation. The caller
        is responsible for storing the unfiltered array in `_raw`.
        """
        vals_f = vals[np.isfinite(vals)]
        if vals_f.size < 2:
            m0 = float(np.mean(vals_f)) if vals_f.size else float(np.nan)
            return (m0, m0)
        boot = rng.choice(vals_f, size=(n_boot, vals_f.size), replace=True).mean(axis=1)
        return (
            float(np.percentile(boot, (1 - ci_level) / 2 * 100)),
            float(np.percentile(boot, (1 + ci_level) / 2 * 100)),
        )

    band_summary: Dict[str, Any] = {}
    for b in freq_bands:
        # --- Raw per-trial arrays (length = n_trials, NaN preserved) ---
        coh_trials      = np.array([tr["coh_bands"][b] for tr in frequency_analysis_trials], dtype=float)
        rbp_true_trials = np.array([tr["rbp_true"][b]  for tr in frequency_analysis_trials], dtype=float)
        rbp_pred_trials = np.array([tr["rbp_pred"][b]  for tr in frequency_analysis_trials], dtype=float)

        # --- Filtered views for descriptive point estimates only ---
        coh_v   = coh_trials[np.isfinite(coh_trials)]
        rbp_t_v = rbp_true_trials[np.isfinite(rbp_true_trials)]
        rbp_p_v = rbp_pred_trials[np.isfinite(rbp_pred_trials)]

        band_summary[b] = {
            "rbp_true": {
                "mean_sd_ci": _mean_sd_ci(rbp_t_v, ci_level),
                "ci_boot":    _boot_mean_ci(rbp_true_trials),
            },
            "rbp_pred": {
                "mean_sd_ci": _mean_sd_ci(rbp_p_v, ci_level),
                "ci_boot":    _boot_mean_ci(rbp_pred_trials),
            },
            "coh": {
                "mean_sd_ci": _mean_sd_ci(coh_v, ci_level),
                "ci_boot":    _boot_mean_ci(coh_trials),
            },
            "_raw": {
                # Unfiltered, length = n_trials. NaN preserved so that
                # trial index i refers to the SAME physical trial
                # across methods and across conditions.
                "coh_trials":      coh_trials,
                "rbp_true_trials": rbp_true_trials,
                "rbp_pred_trials": rbp_pred_trials,
            },
        }

    # ----------------------------------------------------------------
    # 6. Curves summary (mean across 40 trials, CI over 40 trials)
    # ----------------------------------------------------------------
    curves_summary: Dict[str, Any] = {}
    f_psd_ref = frequency_analysis_trials[0]["f_psd_true"]
    f_coh_ref = frequency_analysis_trials[0]["f_coh"]

    psd_true_mat, psd_pred_mat, coh_mat = [], [], []
    for tr in frequency_analysis_trials:
        if not np.array_equal(tr["f_psd_true"], f_psd_ref):
            psd_true_mat.append(np.interp(f_psd_ref, tr["f_psd_true"], tr["psd_true"]))
            psd_pred_mat.append(np.interp(f_psd_ref, tr["f_psd_pred"], tr["psd_pred"]))
        else:
            psd_true_mat.append(tr["psd_true"])
            psd_pred_mat.append(tr["psd_pred"])
        if not np.array_equal(tr["f_coh"], f_coh_ref):
            coh_mat.append(np.interp(f_coh_ref, tr["f_coh"], tr["coh_vals"]))
        else:
            coh_mat.append(tr["coh_vals"])

    psd_true_mat = np.vstack(psd_true_mat)
    psd_pred_mat = np.vstack(psd_pred_mat)
    coh_mat = np.vstack(coh_mat)

    def _summarize_curve(mat: np.ndarray, f: np.ndarray) -> Dict[str, Any]:
        m_ = np.nanmean(mat, axis=0)
        s_ = np.nanstd(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(m_)
        tc = (float(stats.t.ppf(0.5 + ci_level / 2, df=mat.shape[0] - 1))
              if _HAS_SCIPY and mat.shape[0] > 1 else 1.96)
        se_ = s_ / math.sqrt(max(1, mat.shape[0]))
        return {"f": f, "mean": m_, "ci_t": (m_ - tc * se_, m_ + tc * se_)}

    curves_summary["psd_true"] = _summarize_curve(psd_true_mat, f_psd_ref)
    curves_summary["psd_pred"] = _summarize_curve(psd_pred_mat, f_psd_ref)
    curves_summary["coh"]      = _summarize_curve(coh_mat, f_coh_ref)

    # ----------------------------------------------------------------
    # 7. Return — interface-compatible with old pipeline
    # ----------------------------------------------------------------
    return {
        # Primary inferential data
        "trials": int(n_trials),                       
        "r_trials": r_trials,
        "rmse_trials": rmse_trials,
        "frequency_analysis_trials": frequency_analysis_trials,

        # Pooled point estimates (same as old r_pooled_overall etc.)
        "r_pooled": r_pooled,
        "r_pooled_ci_boot": r_ci_boot,                
        "r_mean":    float(np.mean(r_valid))            if n_r_valid          else float(np.nan),
        "r_std":     float(np.std(r_valid, ddof=1))     if n_r_valid    > 1   else float(np.nan),
        "rmse_mean": float(np.mean(rmse_valid))         if n_rmse_valid       else float(np.nan),
        "rmse_sd":   float(np.std(rmse_valid, ddof=1))  if n_rmse_valid > 1   else float(np.nan),
        "rmse_trial_ci_boot": rmse_ci_boot,             

        # Descriptive (interface-compatible)
        "band_summary": band_summary,
        "curves_summary": curves_summary,
    }

def compare_methods_trial_level(
    summary_A: Dict[str, Any],
    summary_B: Dict[str, Any],
    method_names: Tuple[str, str] = ("A", "B"),
    use_fisher_z: bool = True,
    ci_level: float = 0.95,
    n_perm: int = 10000,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Paired trial-level comparison between two methods.
    Inputs are direct outputs of summarize_trials.

    Assumes both summaries came from the SAME fold assignment
    (same GroupKFold split), so trial i in A corresponds to trial i in B.
    """
    rA = np.asarray(summary_A["r_trials"], dtype=float)
    rB = np.asarray(summary_B["r_trials"], dtype=float)
    rmseA = np.asarray(summary_A["rmse_trials"], dtype=float)
    rmseB = np.asarray(summary_B["rmse_trials"], dtype=float)

    if rA.shape != rB.shape:
        raise ValueError("A and B must come from identical trial orderings")

    rng = np.random.default_rng(seed)

    # ---------- r (paired in z-space) ----------
    mask_r = np.isfinite(rA) & np.isfinite(rB)
    zA = fisher_z(rA[mask_r]) if use_fisher_z else rA[mask_r]
    zB = fisher_z(rB[mask_r]) if use_fisher_z else rB[mask_r]
    d_z = zA - zB
    r_summary = _paired_summary(d_z, ci_level=ci_level, n_perm=n_perm,
                                 n_boot=n_boot, seed=seed)

    # ---------- RMSE ----------
    mask_e = np.isfinite(rmseA) & np.isfinite(rmseB)
    d_rmse = rmseA[mask_e] - rmseB[mask_e]
    rmse_summary = _paired_summary(d_rmse, ci_level=ci_level, n_perm=n_perm,
                                    n_boot=n_boot, seed=seed)

    # ---------- Per-band coherence and RBP error ----------
    bands = eeg_analysis_bands()
    coh_band = {"per_band": {}, "p_adj": {"bh": {}, "holm": {}}}
    rbp_err  = {"per_band": {}, "p_adj": {"bh": {}, "holm": {}}}
    p_coh, p_err = [], []

    for b in bands:
        cohA = np.asarray(summary_A["band_summary"][b]["_raw"]["coh_trials"], dtype=float)
        cohB = np.asarray(summary_B["band_summary"][b]["_raw"]["coh_trials"], dtype=float)
        errA = np.abs(
            summary_A["band_summary"][b]["_raw"]["rbp_pred_trials"]
            - summary_A["band_summary"][b]["_raw"]["rbp_true_trials"]
        )
        errB = np.abs(
            summary_B["band_summary"][b]["_raw"]["rbp_pred_trials"]
            - summary_B["band_summary"][b]["_raw"]["rbp_true_trials"]
        )

        n_c = min(len(cohA), len(cohB))
        mA = np.isfinite(cohA[:n_c]) & np.isfinite(cohB[:n_c])
        d_coh = cohA[:n_c][mA] - cohB[:n_c][mA]

        n_e = min(len(errA), len(errB))
        mE = np.isfinite(errA[:n_e]) & np.isfinite(errB[:n_e])
        d_err = errA[:n_e][mE] - errB[:n_e][mE]

        s_c = _paired_summary(d_coh, ci_level=ci_level, n_perm=n_perm,
                               n_boot=n_boot, seed=seed)
        s_e = _paired_summary(d_err, ci_level=ci_level, n_perm=n_perm,
                               n_boot=n_boot, seed=seed)
        coh_band["per_band"][b] = {"delta": d_coh, "summary": s_c}
        rbp_err["per_band"][b]  = {"delta": d_err, "summary": s_e}
        p_coh.append(s_c["tests"]["sign_perm_p"])
        p_err.append(s_e["tests"]["sign_perm_p"])

    padj_coh_bh = _fdr_bh(p_coh)
    padj_coh_holm = _holm_bonferroni(p_coh)
    padj_err_bh = _fdr_bh(p_err)
    padj_err_holm = _holm_bonferroni(p_err)
    for i, b in enumerate(bands):
        coh_band["p_adj"]["bh"][b]   = float(padj_coh_bh[i])
        coh_band["p_adj"]["holm"][b] = float(padj_coh_holm[i])
        rbp_err["p_adj"]["bh"][b]    = float(padj_err_bh[i])
        rbp_err["p_adj"]["holm"][b]  = float(padj_err_holm[i])

    return {
        "methods": {"A": method_names[0], "B": method_names[1]},
        "n_trials": int(d_z.size),
        "r": {
            "delta_z": d_z,
            "summary": r_summary,
            "mean_r_A": float(inv_fisher_z(np.mean(fisher_z(rA[mask_r])))),
            "mean_r_B": float(inv_fisher_z(np.mean(fisher_z(rB[mask_r])))),
            "delta_mean_r": float(
                inv_fisher_z(np.mean(fisher_z(rA[mask_r])))
                - inv_fisher_z(np.mean(fisher_z(rB[mask_r])))
            ),
        },
        "rmse": {"delta": d_rmse, "summary": rmse_summary},
        "coherence": coh_band,
        "rbp_error": rbp_err,
        "bands": bands,
    }
# -------------------------
# helpers: DataFrame output, load, rounding nested
# -------------------------
def comparison_to_dataframes(comp: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Convert the output of `compare_methods_trial_level` into tidy
    DataFrames for reporting and plotting.

    Produces three separate tables:

      - ``overall``   : paired tests for the pooled correlation
                        (expressed on Fisher's z scale) and for the
                        pooled RMSE.
      - ``coherence`` : paired tests per frequency band for
                        band-averaged magnitude-squared coherence.
      - ``rbp_error`` : paired tests per frequency band for the
                        absolute relative-band-power error,
                        |RBP_pred - RBP_true|.

    Parameters
    ----------
    comp : dict
        Output of ``compare_methods_trial_level``. Must contain the
        top-level keys ``"r"``, ``"rmse"``, ``"coherence"``,
        ``"rbp_error"``, and ``"bands"``.

    Returns
    -------
    dict of pandas.DataFrame
        Keys are ``"overall"``, ``"coherence"``, and ``"rbp_error"``.

        Common columns across all three tables
        ......................................
        metric       : str, name of the quantity compared.
        n            : int, number of paired trials used in the test.
        delta_mean   : float, mean paired difference (A - B).
        ci_t_low,
        ci_t_high    : float, t-based 95% confidence interval.
        ci_boot_low,
        ci_boot_high : float, percentile-bootstrap 95% confidence
                       interval.
        sign_perm_p  : float, two-sided sign-flip permutation p-value.
        wilcoxon_p   : float, Wilcoxon signed-rank p-value.
        ttest_p      : float, paired t-test p-value.
        cohen_dz     : float, Cohen's dz effect size (paired design).
        hedges_g     : float, Hedges' g small-sample-corrected effect
                       size.

        Additional columns
        ..................
        ``overall`` adds:
            mean_r_A, mean_r_B  : pooled correlations per method,
                                  back-transformed to the r scale.
            delta_mean_r        : difference in pooled correlation on
                                  the r scale.
        ``coherence`` and ``rbp_error`` add:
            p_bh, p_holm        : adjusted p-values for multiple
                                  comparison correction across bands
                                  (Benjamini-Hochberg and
                                  Holm-Bonferroni respectively).

    Notes
    -----
    - All inference is at the trial level (n typically 40 per
      subject). No fold-level aggregation is used.
    - The ``overall`` table reports correlation on Fisher's z scale
      because the paired test is performed after the z-transform.
      Mean correlations per method are additionally reported on the
      original r scale for interpretability.
    - Coherence and RBP-error tables are indexed by band name. Bands
      are restricted to theta, alpha, and beta, consistent with the
      video Nyquist limit.
    """
    import pandas as pd

    # ---------------------------------------------------------------
    # 1. Overall (r and RMSE)
    # ---------------------------------------------------------------
    r_sum = comp["r"]["summary"]
    r_row = {
        "metric":       "r (Fisher z)",
        "n":            r_sum["n"],
        "delta_mean":   r_sum["mean"],
        "ci_t_low":     r_sum["ci_t"][0],
        "ci_t_high":    r_sum["ci_t"][1],
        "ci_boot_low":  r_sum["ci_boot"][0],
        "ci_boot_high": r_sum["ci_boot"][1],
        "sign_perm_p":  r_sum["tests"]["sign_perm_p"],
        "wilcoxon_p":   r_sum["tests"]["wilcoxon_p"],
        "ttest_p":      r_sum["tests"]["ttest_p"],
        "cohen_dz":     r_sum["effect"]["cohen_dz"],
        "hedges_g":     r_sum["effect"]["hedges_g"],
        "mean_r_A":     comp["r"]["mean_r_A"],
        "mean_r_B":     comp["r"]["mean_r_B"],
        "delta_mean_r": comp["r"]["delta_mean_r"],
    }

    rmse_sum = comp["rmse"]["summary"]
    rmse_row = {
        "metric":       "RMSE",
        "n":            rmse_sum["n"],
        "delta_mean":   rmse_sum["mean"],
        "ci_t_low":     rmse_sum["ci_t"][0],
        "ci_t_high":    rmse_sum["ci_t"][1],
        "ci_boot_low":  rmse_sum["ci_boot"][0],
        "ci_boot_high": rmse_sum["ci_boot"][1],
        "sign_perm_p":  rmse_sum["tests"]["sign_perm_p"],
        "wilcoxon_p":   rmse_sum["tests"]["wilcoxon_p"],
        "ttest_p":      rmse_sum["tests"]["ttest_p"],
        "cohen_dz":     rmse_sum["effect"]["cohen_dz"],
        "hedges_g":     rmse_sum["effect"]["hedges_g"],
        "mean_r_A":     np.nan,
        "mean_r_B":     np.nan,
        "delta_mean_r": np.nan,
    }

    df_overall = pd.DataFrame([rmse_row, r_row]).set_index("metric")

    # ---------------------------------------------------------------
    # 2. Coherence per band
    # ---------------------------------------------------------------
    rows = []
    for b in comp["bands"]:
        s = comp["coherence"]["per_band"][b]["summary"]
        rows.append({
            "band":         b,
            "metric":       "coherence",
            "n":            s["n"],
            "delta_mean":   s["mean"],
            "ci_t_low":     s["ci_t"][0],
            "ci_t_high":    s["ci_t"][1],
            "ci_boot_low":  s["ci_boot"][0],
            "ci_boot_high": s["ci_boot"][1],
            "sign_perm_p":  s["tests"]["sign_perm_p"],
            "wilcoxon_p":   s["tests"]["wilcoxon_p"],
            "ttest_p":      s["tests"]["ttest_p"],
            "cohen_dz":     s["effect"]["cohen_dz"],
            "hedges_g":     s["effect"]["hedges_g"],
            "p_bh":         comp["coherence"]["p_adj"]["bh"][b],
            "p_holm":       comp["coherence"]["p_adj"]["holm"][b],
        })
    df_coh = pd.DataFrame(rows).set_index("band").sort_index()

    # ---------------------------------------------------------------
    # 3. RBP error per band
    # ---------------------------------------------------------------
    rows = []
    for b in comp["bands"]:
        s = comp["rbp_error"]["per_band"][b]["summary"]
        rows.append({
            "band":         b,
            "metric":       "rbp_error(|pred-true|)",
            "n":            s["n"],
            "delta_mean":   s["mean"],
            "ci_t_low":     s["ci_t"][0],
            "ci_t_high":    s["ci_t"][1],
            "ci_boot_low":  s["ci_boot"][0],
            "ci_boot_high": s["ci_boot"][1],
            "sign_perm_p":  s["tests"]["sign_perm_p"],
            "wilcoxon_p":   s["tests"]["wilcoxon_p"],
            "ttest_p":      s["tests"]["ttest_p"],
            "cohen_dz":     s["effect"]["cohen_dz"],
            "hedges_g":     s["effect"]["hedges_g"],
            "p_bh":         comp["rbp_error"]["p_adj"]["bh"][b],
            "p_holm":       comp["rbp_error"]["p_adj"]["holm"][b],
        })
    df_rbp_err = pd.DataFrame(rows).set_index("band").sort_index()

    return {
        "overall":   df_overall,
        "coherence": df_coh,
        "rbp_error": df_rbp_err,
    }

def round_nested(x: Any, n: int = 2) -> Any:
    if isinstance(x, dict):
        return {k: round_nested(v, n) for k, v in x.items()}
    elif isinstance(x, (float, int)) and not isinstance(x, bool):
        if np.isnan(x):
            return x
        return round(float(x), n)
    elif isinstance(x, (list, tuple)):
        return type(x)(round_nested(v, n) for v in x)
    else:
        return x

# -------------------------
# Small wrapper: build consistent eeg_frequency_analysis result
# (kept small & reused by summarize_fold_trials_unit)
# -------------------------

def eeg_frequency_analysis(y_true, y_pred, fs: float = 128.0,
                           nperseg: int = 1024, noverlap: int = 512,
                           ref_band: Tuple[float, float] = (4.0, 25.0)):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    band_names = eeg_analysis_bands()
    # empty case
    if y_true.size == 0 or y_pred.size == 0:
        empty = {
            "f_psd_true": np.array([np.nan]), "psd_true": np.array([np.nan]),
            "f_psd_pred": np.array([np.nan]), "psd_pred": np.array([np.nan]),
            "rbp_true": {b: np.nan for b in band_names},
            "rbp_pred": {b: np.nan for b in band_names},
            "f_coh": np.array([np.nan]), "coh_vals": np.array([np.nan]),
            "coh_bands": {b: np.nan for b in band_names},
        }
        return empty
    if y_true.size != y_pred.size:
        n = min(y_true.size, y_pred.size)
        y_true = y_true[:n]; y_pred = y_pred[:n]
    f_psd_true, P_true = psd_welch(y_true, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_psd_pred, P_pred = psd_welch(y_pred, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_coh, C = mscohere(y_true, y_pred, fs=fs, nperseg=nperseg, noverlap=noverlap)
    rbp_true = {}; rbp_pred = {}; coh_bands = {}
    for name in band_names:
        lo, hi = eeg_band_range(name)
        rbp_true[name] = float(relative_bandpower(P_true, f_psd_true, lo, hi, ref_lo=ref_band[0], ref_hi=ref_band[1]))
        rbp_pred[name] = float(relative_bandpower(P_pred, f_psd_pred, lo, hi, ref_lo=ref_band[0], ref_hi=ref_band[1]))
        mask = band_mask(f_coh, lo, hi)
        coh_bands[name] = float(np.nanmean(C[mask])) if np.any(mask) else float(np.nan)
    return {
        "f_psd_true": f_psd_true, "psd_true": P_true,
        "f_psd_pred": f_psd_pred, "psd_pred": P_pred,
        "rbp_true": rbp_true, "rbp_pred": rbp_pred,
        "f_coh": f_coh, "coh_vals": C,
        "coh_bands": coh_bands,
    }


def analyze_segment_lengths_across_folds(
    fold_summaries_dict: Dict[str, List[Dict[str, Any]]],
    method_names: Dict[str, str] | None = None,
    ci_level: float = 0.95,
    n_perm: int = 10000,
    n_boot: int = 10000,
    correction: str = "bh",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Optimized analyzer for Experiment 2 (segment lengths).
    Compares multiple segment lengths (conditions) across folds (subjects).
    Returns omnibus (Friedman + RM-ANOVA) and pairwise comparisons.
    """

    rng = np.random.default_rng(seed)
    conditions = list(fold_summaries_dict.keys())
    k = len(conditions)
    if k < 2:
        raise ValueError("Need ≥ 2 conditions to compare.")

    Kfolds = len(next(iter(fold_summaries_dict.values())))
    for lbl, lst in fold_summaries_dict.items():
        if len(lst) != Kfolds:
            raise ValueError(f"Inconsistent folds for {lbl}")

    # ------------------------------------------------------------------
    # Generic helper to compute omnibus + pairwise for any extractor
    # ------------------------------------------------------------------
    def analyze_metric(extractor: Callable, label: str, transform: Optional[Callable] = None):
        """Run omnibus + pairwise tests for given metric extractor."""
        # Build subject × condition matrix
        M = np.full((Kfolds, k), np.nan, dtype=float)
        for j, lbl in enumerate(conditions):
            for i, fs in enumerate(fold_summaries_dict[lbl]):
                try:
                    val = extractor(fs)
                    if transform:  # e.g. Fisher z → r
                        val = transform(val)
                except Exception:
                    val = np.nan
                M[i, j] = val

        # Keep complete rows for omnibus tests
        complete_mask = np.all(np.isfinite(M), axis=1)
        M_complete = M[complete_mask]

        fried = _friedman_test_matrix(M_complete) if M_complete.size else {"chi2": np.nan, "p": np.nan, "kendall_w": np.nan, "n": 0, "k": k}
        anova = _rm_anova_oneway_rm(M_complete) if M_complete.size else {"F": np.nan, "p": np.nan, "eta2": np.nan, "n": 0, "k": k}

        # Pairwise
        pairs = list(itertools.combinations(range(k), 2))
        rows, pvals = [], []
        for (i, j) in pairs:
            mask = np.isfinite(M[:, i]) & np.isfinite(M[:, j])
            if not np.any(mask):
                continue
            d = M[mask, i] - M[mask, j]
            summ = _paired_summary(d, ci_level=ci_level, n_perm=n_perm, n_boot=n_boot, seed=seed)

            row = {
                "cond_i": conditions[i], "cond_j": conditions[j],
                "n": summ["n"], "delta_mean": summ["mean"],
                "ci_t_low": summ["ci_t"][0], "ci_t_high": summ["ci_t"][1],
                "ci_boot_low": summ["ci_boot"][0], "ci_boot_high": summ["ci_boot"][1],
                **summ["tests"], **summ["effect"]
            }
            rows.append(row)
            pvals.append(summ["tests"]["sign_perm_p"])

        # Adjusted p-values
        padj = _fdr_bh(pvals) if correction == "bh" else _holm_bonferroni(pvals)
        for r, pa in zip(rows, padj):
            r["p_adj"] = float(pa) if np.isfinite(pa) else np.nan

        return {
            "omnibus": {"friedman": fried, "rm_anova": anova, "n_subjects_used": int(np.sum(complete_mask)), "k_conditions": k},
            "pairwise": pd.DataFrame(rows),
            "matrix": M
        }

    results: Dict[str, Any] = {}

    # --- RMSE
    results["rmse"] = analyze_metric(lambda fs: fs.get("rmse_mean", np.nan), "rmse")

    # --- r (Fisher z)
    results["r_z"] = analyze_metric(lambda fs: fs.get("z_fold", np.nan), "r_z")

    # --- Coherence per-band
    bands = _collect_common_bands(next(iter(fold_summaries_dict.values())))
    coh_results = {}
    for band in bands:
        coh_results[band] = analyze_metric(lambda fs, b=band: _extract_fold_coh_mean(fs, b), f"coh_{band}")
    results["coherence"] = {"per_band": coh_results}

    # --- RBP error per-band
    rbp_results = {}
    for band in bands:
        rbp_results[band] = analyze_metric(lambda fs, b=band: _compute_fold_rbp_error(fs, b), f"rbp_{band}")
    results["rbp_error"] = {"per_band": rbp_results}

    # Meta info
    results["_meta"] = {
        "conditions": conditions,
        "n_folds": Kfolds,
        "ci_level": ci_level,
        "n_perm": n_perm,
        "n_boot": n_boot,
        "correction": correction
    }
    return results


def segment_lengths_to_dataframes_for_folds(results: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert analyze_segment_lengths_across_folds results into:
      - df_omnibus: omnibus test results
      - df_pairwise: pairwise comparisons
    """
    rows_omni, rows_pairs = [], []

    def add_metric_rows(metric_name, res, band: str | None = None):
        fried, anova = res["omnibus"]["friedman"], res["omnibus"]["rm_anova"]
        label = f"{metric_name} ({band})" if band else metric_name
        rows_omni.append({
            "metric": label,
            "n_subjects": int(res["omnibus"]["n_subjects_used"]),
            "k_conditions": int(res["omnibus"]["k_conditions"]),
            "friedman_chi2": fried.get("chi2", np.nan),
            "friedman_p": fried.get("p", np.nan),
            "kendall_W": fried.get("kendall_w", np.nan),
            "rm_anova_F": anova.get("F", np.nan),
            "rm_anova_p": anova.get("p", np.nan),
            "eta2_partial": anova.get("eta2", np.nan),
        })
        if not res["pairwise"].empty:
            tmp = res["pairwise"].copy()
            tmp["metric"] = label
            rows_pairs.append(tmp)

    # RMSE + r
    add_metric_rows("RMSE", results["rmse"])
    add_metric_rows("r (z)", results["r_z"])

    # Coherence
    for band, bandres in results["coherence"]["per_band"].items():
        add_metric_rows("coherence", bandres, band)

    # RBP error
    for band, bandres in results["rbp_error"]["per_band"].items():
        add_metric_rows("rbp_error", bandres, band)

    df_omnibus = pd.DataFrame(rows_omni).set_index("metric")
    df_pairwise = pd.concat(rows_pairs, ignore_index=True, sort=False) if rows_pairs else pd.DataFrame()
    return df_omnibus, df_pairwise


# -------------------------
# Multi-condition trial-level analyzer (Experiment 2)
# -------------------------

def analyze_segment_lengths_trial_level(
    summaries_dict: Dict[str, Dict[str, Any]],
    method_names: Dict[str, str] | None = None,
    use_fisher_z: bool = True,
    ci_level: float = 0.95,
    n_perm: int = 10000,
    n_boot: int = 10000,
    correction: str = "holm",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trial-level comparison of multiple conditions (e.g., segment lengths
    T = 0.5, 1, 2, 3 s) for a single subject.

    Inputs are the direct outputs of `summarize_trials`, one per
    condition. All inference is performed at the trial level
    (n_trials per condition, typically 40), using paired test statistics
    and non-parametric omnibus tests.

    
    Pairwise tests use listwise deletion within each pair, so n may differ slightly 
    across pairwise contrasts if trial-level NaN patterns differ between conditions
    
    Parameters
    ----------
    summaries_dict : dict of {condition_label: summary}
        Mapping from a human-readable condition label (e.g. "T=0.5")
        to the corresponding dict produced by `summarize_trials`.
        All summaries must come from the same fold assignment, so that
        trial i in every condition refers to the same physical trial.
        The number of conditions must be ≥ 2.
    method_names : dict, optional
        Optional mapping from condition label to display name. If None,
        the labels in `summaries_dict` are used.
    use_fisher_z : bool, default=True
        If True, Fisher-z transform is applied to trial-level Pearson
        correlations before paired testing (recommended).
    ci_level : float, default=0.95
        Confidence level for the intervals.
    n_perm : int, default=10000
        Number of sign-flip permutations for the paired tests.
    n_boot : int, default=10000
        Number of bootstrap resamples for the CIs.
    correction : {"bh", "holm"}, default="holm"
        Multiple-comparison correction applied across the pairwise
        contrasts.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing, for each metric, an entry with the
        following structure:

        - ``"omnibus"`` : dict with ``"friedman"``, ``"rm_anova"``,
          ``"n_trials_used"``, and ``"k_conditions"``.
        - ``"pairwise"`` : pandas.DataFrame with one row per unordered
          pair of conditions, containing the paired mean difference,
          t-based and bootstrap confidence intervals, sign-flip,
          Wilcoxon, and t-test p-values, adjusted p-values, and
          Cohen's dz / Hedges' g effect sizes.
        - ``"matrix"`` : np.ndarray of shape
        ``(n_trials_ref, k_conditions)`` with one row per trial and one
        column per condition.

        Metrics reported:

        - ``"rmse"``       : trial-level RMSE.
        - ``"r_z"``        : trial-level Pearson r (Fisher z if
          ``use_fisher_z``).
        - ``"coherence"``  : per-band coherence (theta, alpha, beta),
          accessed via ``results["coherence"]["per_band"][band]``.
        - ``"rbp_error"``  : per-band |RBP_pred - RBP_true|,
          accessed via ``results["rbp_error"]["per_band"][band]``.

        A ``"_meta"`` entry stores the configuration used.

    Notes
    -----
    - Omnibus tests (Friedman, repeated-measures ANOVA) are computed
      only on trials with complete data across all conditions.
    - Pairwise tests use the exact Phipson & Smyth (2010) Monte Carlo
      correction for sign-flip p-values.
    - Fisher-z transformed differences are used for correlation
      comparisons; RMSE and RBP-error are compared on the original
      linear scale.
    """
    rng = np.random.default_rng(seed)
    conditions = list(summaries_dict.keys())
    k = len(conditions)
    if k < 2:
        raise ValueError("Need at least 2 conditions to compare.")

    # Sanity: all summaries must have the same n_trials
    n_trials_ref = summaries_dict[conditions[0]]["trials"]
    for lbl in conditions:
        if summaries_dict[lbl]["trials"] != n_trials_ref:
            raise ValueError(
                f"Inconsistent trial counts: {lbl} has "
                f"{summaries_dict[lbl]['trials']} vs "
                f"{n_trials_ref} in {conditions[0]}."
            )

    # ------------------------------------------------------------------
    # Generic helper: build trial × condition matrix from a per-trial
    # extractor
    # ------------------------------------------------------------------
    def _build_matrix(extractor_trial_level: Callable[[Dict[str, Any]], np.ndarray]) -> np.ndarray:
        M = np.full((n_trials_ref, k), np.nan, dtype=float)
        for j, lbl in enumerate(conditions):
            vals = np.asarray(extractor_trial_level(summaries_dict[lbl]), dtype=float)
            if vals.size != n_trials_ref:
                raise ValueError(
                    f"Trial-level array for condition '{lbl}' has length "
                    f"{vals.size}; expected {n_trials_ref}. "
                    f"Trial alignment cannot be guaranteed. "
                    f"Check that `_raw` arrays are stored without filtering."
                )
            M[:, j] = vals
        return M

    def analyze_metric(
        matrix: np.ndarray,
        label: str,
    ) -> Dict[str, Any]:
        """Omnibus + pairwise tests given a trial × condition matrix."""
        complete_mask = np.all(np.isfinite(matrix), axis=1)
        M_complete = matrix[complete_mask]

        if M_complete.size > 0:
            fried = _friedman_test_matrix(M_complete)
            anova = _rm_anova_oneway_rm(M_complete)
        else:
            fried = {"chi2": np.nan, "p": np.nan, "kendall_w": np.nan,
                     "n": 0, "k": k}
            anova = {"F": np.nan, "p": np.nan, "eta2": np.nan,
                     "n": 0, "k": k}

        # ---------------- Pairwise ----------------
        pairs = list(itertools.combinations(range(k), 2))
        rows, pvals = [], []
        for (i, j) in pairs:
            mask = np.isfinite(matrix[:, i]) & np.isfinite(matrix[:, j])
            if not np.any(mask):
                continue
            d = matrix[mask, i] - matrix[mask, j]
            summ = _paired_summary(d, ci_level=ci_level, n_perm=n_perm,
                                   n_boot=n_boot, seed=seed)
            row = {
                "cond_i":        conditions[i],
                "cond_j":        conditions[j],
                "n":             summ["n"],
                "delta_mean":    summ["mean"],
                "ci_t_low":      summ["ci_t"][0],
                "ci_t_high":     summ["ci_t"][1],
                "ci_boot_low":   summ["ci_boot"][0],
                "ci_boot_high":  summ["ci_boot"][1],
                "sign_perm_p":   summ["tests"]["sign_perm_p"],
                "wilcoxon_p":    summ["tests"]["wilcoxon_p"],
                "ttest_p":       summ["tests"]["ttest_p"],
                "cohen_dz":      summ["effect"]["cohen_dz"],
                "hedges_g":      summ["effect"]["hedges_g"],
            }
            rows.append(row)
            pvals.append(summ["tests"]["sign_perm_p"])

        padj = _fdr_bh(pvals) if correction == "bh" else _holm_bonferroni(pvals)
        for r, pa in zip(rows, padj):
            r["p_adj"] = float(pa) if np.isfinite(pa) else np.nan

        return {
            "omnibus": {
                "friedman":         fried,
                "rm_anova":         anova,
                "n_trials_used":    int(np.sum(complete_mask)),
                "k_conditions":     int(k),
                "label":            label,
            },
            "pairwise": pd.DataFrame(rows),
            "matrix":   matrix,
        }

    # ------------------------------------------------------------------
    # 1. RMSE (trial-level)
    # ------------------------------------------------------------------
    M_rmse = _build_matrix(lambda s: s["rmse_trials"])
    results: Dict[str, Any] = {}
    results["rmse"] = analyze_metric(M_rmse, "RMSE")

    # ------------------------------------------------------------------
    # 2. Pearson r (trial-level, Fisher z if requested)
    # ------------------------------------------------------------------
    if use_fisher_z:
        M_r = _build_matrix(lambda s: fisher_z(s["r_trials"]))
        r_label = "r (Fisher z)"
    else:
        M_r = _build_matrix(lambda s: s["r_trials"])
        r_label = "r"
    results["r_z"] = analyze_metric(M_r, r_label)

    # ------------------------------------------------------------------
    # 3. Coherence per band (trial-level)
    # ------------------------------------------------------------------
    bands = eeg_analysis_bands()   # ["theta", "alpha", "beta"]
    coh_results: Dict[str, Any] = {}
    for band in bands:
        M_coh = _build_matrix(
            lambda s, b=band: s["band_summary"][b]["_raw"]["coh_trials"]
        )
        coh_results[band] = analyze_metric(M_coh, f"coherence ({band})")
    results["coherence"] = {"per_band": coh_results}

    # ------------------------------------------------------------------
    # 4. RBP error per band (trial-level, |pred − true|)
    # ------------------------------------------------------------------
    rbp_results: Dict[str, Any] = {}
    for band in bands:
        def _rbp_err(s, b=band):
            pred = np.asarray(s["band_summary"][b]["_raw"]["rbp_pred_trials"], dtype=float)
            true = np.asarray(s["band_summary"][b]["_raw"]["rbp_true_trials"], dtype=float)
            if pred.size != true.size:
                raise ValueError(
                    f"rbp_true and rbp_pred length mismatch in band '{b}': "
                    f"{true.size} vs {pred.size}"
                )
            return np.abs(pred - true)
        M_rbp = _build_matrix(_rbp_err)
        rbp_results[band] = analyze_metric(M_rbp, f"RBP error ({band})")
    results["rbp_error"] = {"per_band": rbp_results}

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------
    results["_meta"] = {
        "conditions":       conditions,
        "method_names":     method_names or {c: c for c in conditions},
        "n_trials":         int(n_trials_ref),
        "k_conditions":     int(k),
        "use_fisher_z":     bool(use_fisher_z),
        "ci_level":         float(ci_level),
        "n_perm":           int(n_perm),
        "n_boot":           int(n_boot),
        "correction":       correction,
    }
    return results


# -------------------------
# DataFrame conversion for the multi-condition analyzer
# -------------------------

def segment_lengths_to_dataframes(
    results: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert the output of `analyze_segment_lengths_trial_level` into
    two tidy DataFrames suitable for reporting.

    Parameters
    ----------
    results : dict
        Output of `analyze_segment_lengths_trial_level`.

    Returns
    -------
    df_omnibus : pandas.DataFrame
        One row per metric, indexed by a human-readable metric label.
        Columns:

        - ``n_trials``      : trials with complete data across all
                              conditions.
        - ``k_conditions``  : number of conditions compared.
        - ``friedman_chi2`` : Friedman chi-squared statistic.
        - ``friedman_p``    : associated p-value.
        - ``kendall_W``     : Kendall's W effect size.
        - ``rm_anova_F``    : repeated-measures ANOVA F statistic.
        - ``rm_anova_p``    : associated p-value.
        - ``eta2_partial``  : partial eta-squared effect size.

    df_pairwise : pandas.DataFrame
        One row per metric × condition pair. Columns include the two
        condition labels, paired mean difference, t-based and
        bootstrap confidence intervals, sign-flip / Wilcoxon / t-test
        p-values, Benjamini-Hochberg adjusted p-values, and Cohen's dz
        and Hedges' g effect sizes. Empty DataFrame if no pairwise
        comparisons were produced.

    Notes
    -----
    - Bands are restricted to theta, alpha, and beta (video Nyquist
      limit).
    - No fold-level inference is performed; every reported statistic
      is derived from paired trial-level comparisons.
    """
    rows_omni: List[Dict[str, Any]] = []
    rows_pairs: List[pd.DataFrame] = []

    def add_metric_rows(metric_name: str, res: Dict[str, Any]) -> None:
        fried = res["omnibus"]["friedman"]
        anova = res["omnibus"]["rm_anova"]
        rows_omni.append({
            "metric":         metric_name,
            "n_trials":       int(res["omnibus"]["n_trials_used"]),
            "k_conditions":   int(res["omnibus"]["k_conditions"]),
            "friedman_chi2":  fried.get("chi2", np.nan),
            "friedman_p":     fried.get("p", np.nan),
            "kendall_W":      fried.get("kendall_w", np.nan),
            "rm_anova_F":     anova.get("F", np.nan),
            "rm_anova_p":     anova.get("p", np.nan),
            "eta2_partial":   anova.get("eta2", np.nan),
        })
        if not res["pairwise"].empty:
            tmp = res["pairwise"].copy()
            tmp["metric"] = metric_name
            rows_pairs.append(tmp)

    # --- RMSE and r ---
    add_metric_rows("RMSE", results["rmse"])
    add_metric_rows("r (Fisher z)", results["r_z"])

    # --- Coherence per band ---
    for band, bandres in results["coherence"]["per_band"].items():
        add_metric_rows(f"coherence ({band})", bandres)

    # --- RBP error per band ---
    for band, bandres in results["rbp_error"]["per_band"].items():
        add_metric_rows(f"RBP error ({band})", bandres)

    df_omnibus = pd.DataFrame(rows_omni).set_index("metric")

    if rows_pairs:
        df_pairwise = pd.concat(rows_pairs, ignore_index=True, sort=False)
        # Move "metric" first for readability
        cols = ["metric"] + [c for c in df_pairwise.columns if c != "metric"]
        df_pairwise = df_pairwise[cols]
    else:
        df_pairwise = pd.DataFrame()

    return df_omnibus, df_pairwise
# -------------------------
# Exported names
# -------------------------
__all__ = [
    "eeg_band_range", "eeg_analysis_bands", "band_mask", "relative_bandpower",
    "fisher_z", "inv_fisher_z", "safe_pearsonr",
    "psd_welch", "mscohere", "sign_flip_pvalue",
    "_mean_sd_ci", "_curve_summary", "_paired_summary",
    "_fdr_bh", "_holm_bonferroni", "_friedman_test_matrix",
    "_rm_anova_oneway_rm", "build_matrix_from_extractor",
    "_extract_fold_coh_mean", "_compute_fold_rbp_error", "_collect_common_bands",
    "summarize_fold_trials_unit", "aggregate_across_folds", "compare_methods_across_folds",
    "summarize_trials", "compare_methods_trial_level",
    "comparison_to_dataframes", "round_nested", "eeg_frequency_analysis", 
    "analyze_segment_lengths_across_folds", "analyze_segment_lengths_trial_level", 
    "segment_lengths_to_dataframes",
    "_HAS_SCIPY", "stats"
]
