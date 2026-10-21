# face2eeg/core/plotting.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import itertools
import os
from matplotlib.colors import TwoSlopeNorm      
from matplotlib.lines import Line2D

plt.rcParams.update({
    # Fonts
    "font.family": "sans-serif",       # IEEE prefers sans-serif (Arial, Helvetica)
    "font.sans-serif": ["Arial"],
    "font.size": 10,                   # base font size (tick labels)
    "axes.titlesize": 12,              # subplot titles
    "axes.labelsize": 12,              # x/y labels
    "xtick.labelsize": 12,              # x-axis tick labels
    "ytick.labelsize": 12,              # y-axis tick labels
    "legend.fontsize": 9,              # legend

    # Axes and spines
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,             # axis thickness

    # Grid
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "grid.linewidth": 0.8,

    # Figure
    "figure.facecolor": "white",
    "figure.dpi": 300,                 # high-res output by default
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


colors = {
    "true": "#4C72B0",   # reference blue
    "a": "#DD8452",       # muted orange
    "b": "#55A868",       # soft green
    "c": "#C44E52",       # brick red
    "d": "#8172B3",       # muted purple
    "e": "#937860",       # warm brown
    "f": "#DA8BC3",       # pink/magenta
    "g": "#8C8C8C",       # gray
    "h": "#CCB974",       # gold/yellow ochre
    "i": "#64B5CD",       # light teal
    "j": "#52854C"        # dark green
}


edge_colors = {
    'true': '#416196', 
    'a': '#bc7046', 'b': '#488f58', 'c': '#a74246', 'd': '#6e6198', 
    'e': '#7d6652', 'f': '#b976a6', 'g': '#777777', 'h': '#ad9d63', 
    'i': '#559aae', 'j': '#467141'
}

patterns = {
    "true": "",
    "a": "", "b": "", "c": "", "d": "", "e": "",
    "f": "", "g": "", "h": "", "i": "", "j": ""
}


def plot_band_comparison(overall_dict, labels_dict, layout="side", error="ci_boot", save_fig=False, file_name="band_comparison.pdf"):
    """
    Plot RBP and Coherence per model, IEEE/Nature style.

    Parameters
    ----------
    overall_dict : dict
        {"mse": overall_mse, "neg": overall_neg_pearson}
    labels_dict : dict
        {"mse": "MSE", "neg": "Neg-Pearson"}
    layout : str
        "side" (1 row 2 cols) or "stacked" (2 rows 1 col)
    error : str
        "ci95" or "ci_boot" for error bars
    """

    bands_names = list(next(iter(overall_dict.values()))["band_summary"].keys())

    # Layout
    if layout == "side":
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=False)
    ax_rbp, ax_coh = axes

    # --- RBP ---
    n_groups = 1 + len(overall_dict)  # True + predicted
    width = 0.7 / n_groups
    x = np.arange(len(bands_names))

    # True
    true_model = next(iter(overall_dict.values()))
    true_means, true_errs = [], [[], []]
    for b in bands_names:
        mean_val = true_model["band_summary"][b]["rbp_true"]["mean_sd_ci"]["mean"]
        if error == "ci95":
            lo, hi = true_model["band_summary"][b]["rbp_true"]["mean_sd_ci"]["ci95"]
        else:
            lo, hi = true_model["band_summary"][b]["rbp_true"]["ci_boot"]
        true_means.append(mean_val)
        true_errs[0].append(mean_val - lo)
        true_errs[1].append(hi - mean_val)
    ax_rbp.bar(x, true_means, width, yerr=true_errs, capsize=3, label="Ground Truth RBP",
               color=colors["true"], alpha=0.9, 
               edgecolor=edge_colors["true"], hatch=patterns["true"])

    # Predicted models
    offset = width
    for key, overall in overall_dict.items():
        pred_means, pred_errs = [], [[], []]
        for b in bands_names:
            mean_val = overall["band_summary"][b]["rbp_pred"]["mean_sd_ci"]["mean"]
            if error == "ci95":
                lo, hi = overall["band_summary"][b]["rbp_pred"]["mean_sd_ci"]["ci95"]
            else:
                lo, hi = overall["band_summary"][b]["rbp_pred"]["ci_boot"]
            pred_means.append(mean_val)
            pred_errs[0].append(mean_val - lo)
            pred_errs[1].append(hi - mean_val)
        ax_rbp.bar(x + offset, pred_means, width, yerr=pred_errs, capsize=3,
                   label=f'Pred ({labels_dict[key]})', color=colors[key], alpha=0.9, 
                   edgecolor=edge_colors[key], hatch=patterns[key])
        offset += width

    ax_rbp.set_xticks(x + (n_groups - 1) * width / 2)
    ax_rbp.set_xticklabels(bands_names, rotation=30, ha="right", fontsize=14)
    ax_rbp.set_ylabel("Mean Power", fontsize=12)
    ax_rbp.set_title("Relative Band Power", fontsize=12, fontweight="bold")
    ax_rbp.grid(True, axis="y", linestyle="--", linewidth=0.8)
    ax_rbp.legend(frameon=False, fontsize=12)

    # --- Coherence ---
    n_groups = len(overall_dict)
    width = 0.7 / n_groups
    x = np.arange(len(bands_names))
    offset = 0
    for key, overall in overall_dict.items():
        coh_means, coh_errs = [], [[], []]
        for b in bands_names:
            mean_val = overall["band_summary"][b]["coh"]["mean_sd_ci"]["mean"]
            if error == "ci95":
                lo, hi = overall["band_summary"][b]["coh"]["mean_sd_ci"]["ci95"]
            else:
                lo, hi = overall["band_summary"][b]["coh"]["ci_boot"]
            coh_means.append(mean_val)
            coh_errs[0].append(mean_val - lo)
            coh_errs[1].append(hi - mean_val)
        ax_coh.bar(x + offset, coh_means, width, yerr=coh_errs, capsize=3,
                   label=f'{labels_dict[key]}', color=colors[key], alpha=0.9, 
                   edgecolor=edge_colors[key], hatch=patterns[key])
        offset += width

    ax_coh.set_xticks(x + (n_groups - 1) * width / 2)
    ax_coh.set_xticklabels(bands_names, rotation=30, ha="right", fontsize=14)
    ax_coh.set_ylabel("Mean Coherence", fontsize=12)
    ax_coh.set_title("Coherence", fontsize=12, fontweight="bold")
    ax_coh.grid(True, axis="y", linestyle="--", linewidth=0.8)
    ax_coh.legend(frameon=False, fontsize=12)

    plt.tight_layout()
    if save_fig: 
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_paired_metrics(fold_summaries_dict, save_fig=False, file_name="paired_metrics.pdf"):
    labels = list(fold_summaries_dict.keys())
    n_models = len(labels)
    n_folds = len(next(iter(fold_summaries_dict.values())))

    labels = list(fold_summaries_dict.keys())
    model_keys = list("abcdefghij")  
    model_colors = {label: colors.get(k, "#000000") for label, k in zip(labels, model_keys)}

    # Extract RMSE and r_pooled
    rmse_dict = {label: np.array([fs["rmse_mean"] for fs in fold_summaries_dict[label]], float)
                 for label in labels}
    r_dict = {label: np.array([fs["r_pooled"] for fs in fold_summaries_dict[label]], float)
              for label in labels}

    if n_models == 2:
        # Two models: RMSE left, r scatter right
        fig, axes = plt.subplots(1, 2, figsize=(12,5))

        # RMSE subplot
        ax = axes[0]
        x = np.arange(n_models)
        for fold_idx in range(n_folds):
            y = [rmse_dict[label][fold_idx] for label in labels]
            ax.plot(x, y, color="gray", alpha=0.6, linewidth=1)
            for xi, yi, label in zip(x, y, labels):
                ax.scatter(xi, yi, color=model_colors[label], edgecolor="black", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels , fontsize=12)
        ax.set_title("RMSE per fold (paired)", fontsize=14)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        # r scatter subplot
        ax = axes[1]
        x_vals = r_dict[labels[0]]
        y_vals = r_dict[labels[1]]
        ax.scatter(x_vals, y_vals, color="gray", edgecolor="black", alpha=0.7)
        lim = [min(np.min(x_vals), np.min(y_vals)), max(np.max(x_vals), np.max(y_vals))]
        ax.plot(lim, lim, linestyle="--", color="black", alpha=0.7)
        ax.set_xlabel(f"r ({labels[0]})", fontsize=12)
        ax.set_ylabel(f"r ({labels[1]})", fontsize=12)
        ax.set_title(f"{labels[0]} vs {labels[1]}", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        if save_fig: 
            plt.savefig(file_name, bbox_inches='tight')
        plt.show()
        plt.close()

    else:
        # More than 2 models: RMSE centered top, pairwise r below
        pairs = list(itertools.combinations(labels, 2))
        n_pairs = len(pairs)

        n_cols = min(3, n_pairs)
        n_rows = 1 + math.ceil(n_pairs / n_cols)
        fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.4)

        # Center RMSE in top row, occupying half width
        mid_col = n_cols // 2
        if n_cols % 2 == 0:
            start_col = mid_col - 1
            end_col = mid_col + 1
        else:
            start_col = mid_col
            end_col = mid_col + 1
        ax_rmse = fig.add_subplot(gs[0, start_col:end_col])

        x = np.arange(n_models)
        for fold_idx in range(n_folds):
            y = [rmse_dict[label][fold_idx] for label in labels]
            ax_rmse.plot(x, y, color="gray", alpha=0.6, linewidth=1)
            for xi, yi, label in zip(x, y, labels):
                ax_rmse.scatter(xi, yi, color=model_colors[label], edgecolor="black", zorder=3)
        ax_rmse.set_xticks(x)
        ax_rmse.set_xticklabels(labels, fontsize=12)
        ax_rmse.set_title("RMSE per fold (paired)", fontsize=14)
        ax_rmse.set_ylabel("RMSE", fontsize=12)
        ax_rmse.grid(True, axis="y", linestyle="--", alpha=0.3)

        # Pairwise r scatter plots below
        for idx, (label_i, label_j) in enumerate(pairs):
            row = 1 + idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            x_vals = r_dict[label_i]
            y_vals = r_dict[label_j]
            ax.scatter(x_vals, y_vals, color="gray", edgecolor="black", alpha=0.7)
            lim = [min(np.min(x_vals), np.min(y_vals)), max(np.max(x_vals), np.max(y_vals))]
            ax.plot(lim, lim, linestyle="--", color="black", alpha=0.7)
            ax.set_xlabel(f"r ({label_i})", fontsize=12)
            ax.set_ylabel(f"r ({label_j})", fontsize=12)
            ax.set_title(f"{label_i} vs {label_j}", fontsize=14)
            ax.grid(True, linestyle="--", alpha=0.3)

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
        if save_fig: 
            plt.savefig(file_name, bbox_inches='tight')
        plt.show()
        plt.close()
        
def plot_mean_curve_with_ci(
    agg_dict, what="coh", freq_range=(4, 25),
    show_true_if_pred=True, save_fig=False,
    file_name="mean_curve_with_ci.pdf"
):
    if what not in ("coh", "psd_pred"):
        raise ValueError("what must be 'coh' or 'psd_pred'")

    PALETTE = {
        "true": "#4C72B0",
        "a":    "#DD8452",
        "b":    "#55A868",
        "c":    "#C44E52",
        "d":    "#8172B3",
        "e":    "#937860",
        "f":    "#DA8BC3",
        "g":    "#8C8C8C",
        "h":    "#CCB974",
        "i":    "#64B5CD",
        "j":    "#52854C",
    }

    label_to_key = {
        "MSE":          "a",
        "NegPearson":   "b",
        "Neg-Pearson":  "b",
        "Ground Truth": "true",
    }

    def get_color(label):
        key = label_to_key.get(label)
        return PALETTE.get(key, "#333333")

    fmin, fmax = freq_range
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, agg in agg_dict.items():
        d = agg["curves_summary"][what]
        f = np.asarray(d["f"])
        m, (lo, hi) = np.asarray(d["mean"]), d["ci_t"]
        mask = (f >= fmin) & (f <= fmax)

        color = get_color(label)
        ax.plot(f[mask], m[mask], lw=2, color=color, label=label)
        ax.fill_between(
            f[mask], np.asarray(lo)[mask], np.asarray(hi)[mask],
            color=color, alpha=.15
        )

    if what == "psd_pred" and show_true_if_pred:
        d = next(iter(agg_dict.values()))["curves_summary"]["psd_true"]
        f = np.asarray(d["f"])
        m, (lo, hi) = np.asarray(d["mean"]), d["ci_t"]
        mask = (f >= fmin) & (f <= fmax)

        gt_color = PALETTE["true"]
        ax.plot(f[mask], m[mask], "--", lw=1.8,
                color=gt_color, label="Ground Truth")
        ax.fill_between(
            f[mask], np.asarray(lo)[mask], np.asarray(hi)[mask],
            color=gt_color, alpha=.08
        )

    ax.set_xlim(fmin, fmax)
    ax.set_xticks([4, 8, 12, 16, 20, 25])
    ax.set_xticklabels(["4", "8", "12", "16", "20", "25"])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence" if what == "coh" else "Power Spectral Density")

    if what == "coh":
        ax.set_ylim(0, 1)

    ax.grid(True, ls="--", alpha=.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    if save_fig:
        fig.savefig(file_name, bbox_inches="tight")

    plt.show()
    plt.close(fig)      

def plot_segment_metric_curves(
    df_rmse=None,
    df_r=None,
    df_per_band=None,
    labels=None,
    x=None,
    ylabel=None,
    ylim=None,
    band_order=("theta", "alpha", "beta"),
    save_fig=False,
    file_name="segment_metrics.pdf",
    close_figs=False,
):
    """
    Render segment-length curves from pre-computed mean + CI DataFrames.

    Pure rendering: no statistics. Only the panels whose DataFrames
    are provided will be built.

    Parameters
    ----------
    df_rmse, df_r : DataFrame, optional
        Output of `extract_per_condition_curves`["rmse"] / ["r"].
        If BOTH are given, the two-panel RMSE + Pearson r figure is built.
    df_per_band : dict, optional
        {band_name: DataFrame}, output of `extract_per_condition_curves`
        ["coh"]. If given, the per-band figure is built.
    labels : list of str, optional
        X tick labels. Default ["T=0.5 s", "T=1 s", "T=2 s", "T=3 s"].
    x : array-like, optional
        X positions. Default [0.5, 1, 2, 3].
    ylabel : str
        Y-label for the per-band figure.
    ylim : tuple, optional
        Y-limits for the per-band figure.
    band_order : tuple of str
        Order of bands in the per-band legend.
    save_fig : bool
        If True, save figure(s) to disk.
    file_name : str
        Base path for the main figure. The per-band figure (when
        df_per_band is given) is saved as `<base>_per_band<ext>` when
        both figures exist, or under `file_name` directly when only
        the per-band figure is built.
    close_figs : bool
        If True, close the figures before returning.

    Returns
    -------
    dict with keys:
        "main"     : Figure or None
        "per_band" : Figure or None
    """
    if labels is None:
        labels = ["T=0.5 s", "T=1 s", "T=2 s", "T=3 s"]
    if x is None:
        x = np.array([0.5, 1, 2, 3], dtype=float)
    x = np.asarray(x, dtype=float)

    def _unpack(df):
        df = df.sort_values("condition_idx").reset_index(drop=True)
        return (df["mean"].to_numpy(float),
                df["ci_low"].to_numpy(float),
                df["ci_high"].to_numpy(float))

    fig_main = None
    fig_band = None

    # ------------------------------------------------------------------
    # Figure 1: RMSE + Pearson r  (only if BOTH DataFrames given)
    # ------------------------------------------------------------------
    if df_rmse is not None and df_r is not None:
        fig_main, axes = plt.subplots(1, 2, figsize=(8, 3.8))

        for ax, (title, df) in zip(
            axes,
            [("RMSE", df_rmse), ("Pearson r", df_r)],
        ):
            mean, lo, hi = _unpack(df)
            ax.plot(x, mean, "o-", lw=1.8)
            ax.fill_between(x, lo, hi, alpha=0.15)
            ax.set_xlabel("Segment length (s)")
            ax.set_ylabel(title)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.grid(True, ls="--", alpha=0.25)

        fig_main.tight_layout()

    # ------------------------------------------------------------------
    # Figure 2: per-band curves  (only if df_per_band given)
    # ------------------------------------------------------------------
    if df_per_band is not None:
        fig_band, ax = plt.subplots(figsize=(7, 4))

        for band in band_order:
            if band not in df_per_band:
                continue
            mean, lo, hi = _unpack(df_per_band[band])
            ax.plot(x, mean, "o-", lw=1.8, label=band)
            ax.fill_between(x, lo, hi, alpha=0.12)

        ax.set_xlabel("Segment length (s)")
        ax.set_ylabel(ylabel or "Coherence")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, ls="--", alpha=0.25)
        ax.legend(frameon=False, ncol=len(band_order))
        fig_band.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if save_fig:
        if fig_main is not None:
            fig_main.savefig(file_name, bbox_inches="tight")
        if fig_band is not None:
            if fig_main is None:
                # only per-band figure: save under file_name directly
                fig_band.savefig(file_name, bbox_inches="tight")
            else:
                base, ext = os.path.splitext(file_name)
                fig_band.savefig(base + "_per_band" + ext,
                                 bbox_inches="tight")

    if close_figs:
        if fig_main is not None:
            plt.close(fig_main)
        if fig_band is not None:
            plt.close(fig_band)

    return {"main": fig_main, "per_band": fig_band}

def plot_null_models_summary(
    df_subject,
    df_group,
    save_fig=False,
    file_name="exp3_null_models.pdf",
    conditions_r=(
        "Video model",
        "Shuffled-pairing model, original-target scoring",
        "Shuffled-pairing model, permuted-target scoring",
        "Training-set mean waveform",
    ),
    conditions_rmse=(
        "Video model",
        "Shuffled-pairing model, original-target scoring",
        "Shuffled-pairing model, permuted-target scoring",
        "Training-set mean waveform",
        "Training-mean constant",
        "Zero-output baseline",
    ),
    labels_r=(
        "Video model",
        "Shuffled (orig. target)",
        "Shuffled (perm. target)",
        "Mean waveform",
    ),
    labels_rmse=(
        "Video model",
        "Shuffled (orig. target)",
        "Shuffled (perm. target)",
        "Mean waveform",
        "Constant mean",
        "Zero output",
    ),
    seed=42,
    general_title="Null-model validation (N = 18 participants)",
):
    from matplotlib.lines import Line2D
    rng = np.random.default_rng(seed)

    panels = [
        dict(
            ax_idx=0,
            panel_label="a",
            conditions=conditions_r,
            labels=labels_r,
            subj_col="Fisher-z pooled r",
            group_col="Participant-aggregated r",
            group_lo="r CI Low",
            group_hi="r CI High",
            ylabel="Fisher-$z$ aggregated Pearson $r$",
            ylim=(-0.2, 1.0),
            zero_line=True,
            reference_line=None,
            title="Correlation",
        ),
        dict(
            ax_idx=1,
            panel_label="b",
            conditions=conditions_rmse,
            labels=labels_rmse,
            subj_col="Trial-level RMSE",
            group_col="Mean participant-level RMSE",
            group_lo="RMSE CI Low",
            group_hi="RMSE CI High",
            ylabel="Mean trial-level RMSE",
            ylim=(0.5, 1.05),
            zero_line=False,
            reference_line=1.0,
            title="RMSE",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for spec in panels:
        ax = axes[spec["ax_idx"]]
        conds = spec["conditions"]

        pivot = (
            df_subject.loc[df_subject["Condition"].isin(conds)]
                      .pivot(index="Subject", columns="Condition",
                             values=spec["subj_col"])
                      .reindex(columns=conds)
        )

        subj_jitter = {s: rng.uniform(-0.05, 0.05) for s in pivot.index}

        # paired lines
        for subj in pivot.index:
            row = pivot.loc[subj].to_numpy(dtype=float)
            xs  = np.arange(len(conds), dtype=float) + subj_jitter[subj]
            mask = np.isfinite(row)
            if mask.sum() >= 2:
                ax.plot(xs[mask], row[mask], color="gray",
                        alpha=0.30, lw=0.7, zorder=1)

        # individual dots
        for i, c in enumerate(conds):
            vals = pivot[c].to_numpy(dtype=float)
            valid_idx = np.isfinite(vals)
            if not valid_idx.any():
                continue
            xs = np.array([i + subj_jitter[s]
                           for s, v in zip(pivot.index, valid_idx) if v])
            ax.scatter(xs, vals[valid_idx], s=16, alpha=0.55,
                       color="C0", zorder=2)

        # population estimate
        for i, c in enumerate(conds):
            row = df_group.loc[df_group["Condition"] == c]
            if row.empty:
                continue
            point = float(row[spec["group_col"]].iloc[0])
            lo    = float(row[spec["group_lo"]].iloc[0])
            hi    = float(row[spec["group_hi"]].iloc[0])
            if not all(np.isfinite([point, lo, hi])):
                continue
            ax.errorbar(i, point, yerr=[[point-lo], [hi-point]],
                        fmt="o", ms=9, capsize=5, lw=1.8,
                        color="black", zorder=3)

        if spec["zero_line"]:
            ax.axhline(0.0, color="red", lw=0.8, ls="--",
                       alpha=0.6, zorder=0)
        if spec["reference_line"] is not None:
            ax.axhline(spec["reference_line"], color="gray",
                       lw=0.8, ls=":", alpha=0.5, zorder=0)

        ax.set_xticks(range(len(spec["labels"])))
        ax.set_xticklabels(spec["labels"], rotation=20, ha="right")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(axis="y", ls="--", alpha=0.25)
        ax.set_ylim(*spec["ylim"])

        ax.text(-0.08, 1.05, spec["panel_label"],
                transform=ax.transAxes, fontweight="bold",
                fontsize=13, va="bottom", ha="left")
        ax.set_title(spec["title"], loc="left",
                     fontweight="normal", fontsize=10, pad=8)

        ax.text(0.98, 0.02, "n = 18", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="0.3")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0",
               markersize=5, alpha=0.55, label="Individual participant"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=8, label="Population estimate (95% bootstrap CI)"),
        Line2D([0], [0], color="gray", alpha=0.4, lw=0.9,
               label="Paired observations"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right",
                   fontsize=8, frameon=False)

    fig.suptitle(general_title, fontsize=11, y=1.02)

    fig.tight_layout()
    if save_fig:
        fig.savefig(file_name, bbox_inches="tight")
    return fig, axes

def plot_band_comparison_grid(overall_dict, labels_dict, error="ci_boot",
                              save_fig=False, file_name="band_comparison_grid.pdf"):
    """
    Plot grid of RBP and Coherence per subject, IEEE/Nature style with striped predicted bars.
    Legend placed on top between the two columns; predicted bars solid color.
    """
    colors = {
        "true": "#4C72B0",   # solid
        "true_edge": "#3D64A1",   # solid
        "pred": "#DD8452",   # solid for predicted
        "pred_edge": "#C57648",   # solid for predicted
        "coherence": "#55A868",
        "coherence_edge": "#498F59",
    }
    patterns = {"true": "", "pred": "//"}  # hatching patterns
    alpha_values = {"true": 0.9, "pred": 0.9}  # reset alpha for predicted bars

    subjects = list(overall_dict.keys())
    n_rows = len(subjects)
    n_cols = 2
    bands_names = list(next(iter(overall_dict.values()))["band_summary"].keys())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3*n_rows), sharey=False)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Compute global y-limits
    rbp_vals_all, coh_vals_all = [], []
    for subj in subjects:
        overall = overall_dict[subj]
        for b in bands_names:
            rbp_vals_all += [
                overall["band_summary"][b]["rbp_true"]["mean_sd_ci"]["mean"],
                overall["band_summary"][b]["rbp_pred"]["mean_sd_ci"]["mean"]
            ]
            coh_vals_all.append(overall["band_summary"][b]["coh"]["mean_sd_ci"]["mean"])

    rbp_min, rbp_max = min(rbp_vals_all)*0.95, max(rbp_vals_all)*1.05
    coh_min, coh_max = min(coh_vals_all)*0.95, max(coh_vals_all)*1.05

    for i, subj in enumerate(subjects):
        overall = overall_dict[subj]
        label = labels_dict[subj]
        x = np.arange(len(bands_names))
        width = 0.35

        # --- RBP ---
        ax_rbp = axes[i, 0]
        true_means, true_errs = [], [[], []]
        pred_means, pred_errs = [], [[], []]

        for b in bands_names:
            t_mean = overall["band_summary"][b]["rbp_true"]["mean_sd_ci"]["mean"]
            p_mean = overall["band_summary"][b]["rbp_pred"]["mean_sd_ci"]["mean"]
            if error=="ci95":
                t_lo, t_hi = overall["band_summary"][b]["rbp_true"]["mean_sd_ci"]["ci95"]
                p_lo, p_hi = overall["band_summary"][b]["rbp_pred"]["mean_sd_ci"]["ci95"]
            else:
                t_lo, t_hi = overall["band_summary"][b]["rbp_true"]["ci_boot"]
                p_lo, p_hi = overall["band_summary"][b]["rbp_pred"]["ci_boot"]
            true_means.append(t_mean)
            true_errs[0].append(t_mean - t_lo)
            true_errs[1].append(t_hi - t_mean)
            pred_means.append(p_mean)
            pred_errs[0].append(p_mean - p_lo)
            pred_errs[1].append(p_hi - p_mean)

        ax_rbp.bar(x, true_means, width, yerr=true_errs, capsize=3,
                   color=colors["true"], alpha=alpha_values["true"],
                   label="Ground Truth RBP", edgecolor=colors["true_edge"], hatch=patterns["true"])
        ax_rbp.bar(x + width, pred_means, width, yerr=pred_errs, capsize=3,
                   color=colors["pred"], alpha=alpha_values["pred"],
                   label="Predicted RBP", edgecolor=colors["pred_edge"], hatch=patterns["pred"])

        ax_rbp.set_xticks(x + width/2)
        ax_rbp.set_xticklabels(bands_names, rotation=0, ha="right", fontsize=12)
        ax_rbp.set_ylabel("Mean RBP", fontsize=12)
        ax_rbp.set_title(f"{label} - Relative Band Power", fontsize=12, fontweight="bold")
        ax_rbp.set_ylim(rbp_min, rbp_max)
        ax_rbp.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_rbp.spines['top'].set_visible(False)
        ax_rbp.spines['right'].set_visible(False)

        # --- Coherence ---
        ax_coh = axes[i, 1]
        coh_means, coh_errs = [], [[], []]
        for b in bands_names:
            c_mean = overall["band_summary"][b]["coh"]["mean_sd_ci"]["mean"]
            if error=="ci95":
                c_lo, c_hi = overall["band_summary"][b]["coh"]["mean_sd_ci"]["ci95"]
            else:
                c_lo, c_hi = overall["band_summary"][b]["coh"]["ci_boot"]
            coh_means.append(c_mean)
            coh_errs[0].append(c_mean - c_lo)
            coh_errs[1].append(c_hi - c_mean)

        ax_coh.bar(x, coh_means, width, yerr=coh_errs, capsize=3,
                   color=colors["coherence"], alpha=alpha_values["pred"],
                   label="Coherence", edgecolor=colors["coherence_edge"], hatch=patterns["pred"])
        ax_coh.set_xticks(x)
        ax_coh.set_xticklabels(bands_names, rotation=0, ha="right", fontsize=12)
        ax_coh.set_ylabel("Mean Coherence", fontsize=12)
        ax_coh.set_title(f"{label} - Coherence", fontsize=12, fontweight="bold")
        ax_coh.set_ylim(coh_min, coh_max)
        ax_coh.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_coh.spines['top'].set_visible(False)
        ax_coh.spines['right'].set_visible(False)

    handles = [
        plt.Rectangle((0,0),1,1, color=colors["true"], edgecolor=colors["true_edge"], hatch=patterns["true"], alpha=alpha_values["true"]),
        plt.Rectangle((0,0),1,1, color=colors["pred"], edgecolor=colors["pred_edge"], hatch=patterns["pred"], alpha=alpha_values["pred"]),
        plt.Rectangle((0,0),1,1, color=colors["coherence"], edgecolor=colors["coherence_edge"], hatch=patterns["pred"], alpha=alpha_values["pred"]),
    ]
    labels = ["Ground Truth RBP", "Predicted RBP", "Coherence"]
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 0.97))  # slightly above top

    plt.subplots_adjust(top=0.93, hspace=0.4, wspace=0.2)  # more space between plots
    if save_fig:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_metrics_grid(
    df_metrics,
    bands=("theta", "alpha", "beta"),
    xlabel="Model",
    unit_col="trial",
    save_fig=False,
    base_dir="./",
    experiment_name="",
):
    """
    Plots three separate grids for trial-level metrics (IEEE/Nature style):

      1. RMSE and Pearson r (per trial).
      2. Relative band power (RBP) per frequency band.
      3. Magnitude-squared coherence per frequency band.

    Parameters
    ----------
    df_metrics : pandas.DataFrame
        Must contain the following columns:

        - ``"model"``      : condition / method label.
        - ``unit_col``     : unit of observation (default ``"trial"``),
          one row per trial per model. Must be an integer index.
        - ``"rmse"``       : trial-level RMSE.
        - ``"r"``          : trial-level Pearson correlation.
        - ``"rbp_pred_<band>"`` for each band in ``bands``.
        - ``"coh_<band>"``      for each band in ``bands``.
    bands : tuple of str, default=("theta", "alpha", "beta")
        Frequency bands to plot. Gamma is not included by default
        because the video Nyquist limit is 25 Hz.
    xlabel : str, default="Model"
        Label for the categorical x-axis.
    unit_col : str, default="trial"
        Name of the column that identifies the observational unit.
        Use ``"trial"`` for trial-level data or ``"fold"`` for
        fold-level data (legacy).
    save_fig : bool, default=False
        If True, save each figure to ``base_dir`` using
        ``experiment_name`` as prefix.
    base_dir : str, default="./"
        Output directory for saved figures.
    experiment_name : str, default=""
        Prefix for saved figure filenames.

    Notes
    -----
    - Box plots summarize the trial-to-trial distribution of the
      metric for each model. Swarm plots overlay individual values.
    - With 40 trials per model, the swarm plot can be dense. If the
      figure becomes cluttered, reduce the swarm size or replace it
      with a strip plot with ``jitter=True``.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch

    labels = df_metrics["model"].unique()
    model_keys = list("abcdefghij")
    palette = {label: colors[k] for label, k in zip(labels, model_keys)}

    # ----------------------------------------------------------------
    # 1. RMSE & r
    # ----------------------------------------------------------------
    metrics_basic = ["rmse", "r"]
    fig, axes = plt.subplots(1, len(metrics_basic), figsize=(8 * len(metrics_basic), 7))
    if len(metrics_basic) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics_basic):
        sns.boxplot(
            x="model", y=m, hue="model",
            data=df_metrics, palette=palette,
            ax=ax, dodge=False, legend=False,
        )
        sns.swarmplot(
            x="model", y=m,
            data=df_metrics, color="k", alpha=0.5, size=3,
            ax=ax,
        )
        ax.set_title(f"{m.upper()} per trial", fontsize=14, fontweight="bold")
        ax.set_ylabel(m.upper(), fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)

    legend_handles = [Patch(facecolor=palette[label], label=label) for label in labels]
    fig.legend(
        handles=legend_handles,
        loc="upper center", ncol=len(labels),
        fontsize=12, title=xlabel, frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    if save_fig:
        plt.savefig(
            os.path.join(base_dir, f"{experiment_name}_RMSE_and_R.pdf"),
            bbox_inches="tight",
        )
    plt.show()
    plt.close()

    # ----------------------------------------------------------------
    # 2. RBP per band
    # ----------------------------------------------------------------
    rbp_cols = [f"rbp_pred_{b}" for b in bands]
    missing = [c for c in rbp_cols if c not in df_metrics.columns]
    if missing:
        raise KeyError(f"Missing RBP columns in df_metrics: {missing}")

    df_rbp = df_metrics.melt(
        id_vars=["model", unit_col],
        value_vars=rbp_cols,
        var_name="band", value_name="value",
    )
    df_rbp["band"] = df_rbp["band"].str.replace("rbp_pred_", "", regex=False)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sns.boxplot(
        x="band", y="value", hue="model",
        data=df_rbp, palette=palette, ax=ax, dodge=True,
    )
    sns.stripplot(
        x="band", y="value", hue="model",
        data=df_rbp, palette=palette, dodge=True,
        alpha=0.4, size=2.5, ax=ax,
    )
    ax.set_title("Relative Band Power (Predicted) per Band", fontsize=12, fontweight="bold")
    ax.set_ylabel("RBP")
    ax.set_xlabel("Frequency Band")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title=xlabel, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            os.path.join(base_dir, f"{experiment_name}_RBP_per_band.pdf"),
            bbox_inches="tight",
        )
    plt.show()
    plt.close()

    # ----------------------------------------------------------------
    # 3. Coherence per band
    # ----------------------------------------------------------------
    coh_cols = [f"coh_{b}" for b in bands]
    missing = [c for c in coh_cols if c not in df_metrics.columns]
    if missing:
        raise KeyError(f"Missing coherence columns in df_metrics: {missing}")

    df_coh = df_metrics.melt(
        id_vars=["model", unit_col],
        value_vars=coh_cols,
        var_name="band", value_name="value",
    )
    df_coh["band"] = df_coh["band"].str.replace("coh_", "", regex=False)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sns.boxplot(
        x="band", y="value", hue="model",
        data=df_coh, palette=palette, ax=ax, dodge=True,
    )
    sns.stripplot(
        x="band", y="value", hue="model",
        data=df_coh, palette=palette, dodge=True,
        alpha=0.4, size=2.5, ax=ax,
    )
    ax.set_title("Coherence per Band", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coherence")
    ax.set_xlabel("Frequency Band")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title=xlabel, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            os.path.join(base_dir, f"{experiment_name}_Coherence_per_band.pdf"),
            bbox_inches="tight",
        )
    plt.show()
    plt.close()

# ============================================================
# Coupling heatmap grid (subject × channel × reference)
# ============================================================
def plot_coupling_heatmap_grid(
    df,
    channels=("Fz", "Pz", "Oz"),
    references=("heog", "veog"),
    metrics=("r_broadband_4_45", "coh_theta"),
    metric_titles={
        "r_broadband_4_45": "Broadband Pearson r: 4\u201345 Hz",
        "coh_theta": "Theta coherence: 4\u20138 Hz",
    },
    cmaps={
        "r_broadband_4_45": "RdBu_r",
        "coh_theta": "RdYlGn",
    },
    value_ranges={
        "r_broadband_4_45": (-1.0, 1.0),
        "coh_theta": (0.0, 1.0),
    },
    figsize=(11.2, 7.6),
    dpi=400,
    title="EEG\u2013EOG coupling across subjects and channels",
    subtitle=None,
    save_fig=False,
    file_name="heatmap_all_channels.pdf",
):
    """
    Two side-by-side heatmaps of subject × (channel × EOG reference).

    Panels
    ------
    a  Broadband Pearson r
    b  Theta coherence

    Column hierarchy
    ----------------
    Channel
        ├── hEOG
        └── vEOG
    """

    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import TwoSlopeNorm, Normalize

    _INK, _MUTED, _RULE = "#1a1a1a", "#3a3a3a", "#2b2b2b"
    REF_DISPLAY = {"heog": "hEOG", "veog": "vEOG", "zemg": "zEMG"}

    def _fmt_signed(v, nd=2):
        if not np.isfinite(v): return "\u2014"
        body = f"{abs(v):.{nd}f}"
        return f"\u2212{body}" if v < 0 else f"{v:.{nd}f}"

    def _fmt_unsigned(v, nd=2):
        if not np.isfinite(v): return "\u2014"
        return f"{v:.{nd}f}"

    def _text_color_for(value, cmap, norm, lum_thresh=0.52):
        r, g, b, _ = cmap(norm(value))
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "white" if luminance < lum_thresh else "black"

    all_subjects = sorted(df["Subject"].unique())
    n_s = len(all_subjects)
    col_specs = [(ch, ref) for ch in channels for ref in references]
    n_cols = len(col_specs)
    n_ref = len(references)
    group_bounds = [n_ref * (gi + 1) - 0.5 for gi in range(len(channels) - 1)]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        1, len(metrics), figure=fig,
        left=0.085, right=0.885, top=0.835, bottom=0.095,
        wspace=0.34,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(metrics))]
    df_idx = df.set_index(["Subject", "Target", "Reference"])

    for k, metric in enumerate(metrics):
        ax = axes[k]
        cmap = mpl.colormaps[cmaps[metric]]
        vmin, vmax = value_ranges[metric]
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if metric == "r_broadband_4_45" else Normalize(vmin=vmin, vmax=vmax)

        M = np.full((n_s, n_cols), np.nan)
        for si, sid in enumerate(all_subjects):
            for ci, (ch, ref) in enumerate(col_specs):
                try:
                    v = df_idx.loc[(sid, ch, ref), metric]
                except KeyError:
                    continue
                v = np.asarray(v.to_numpy() if isinstance(v, pd.Series) else v, dtype=float)
                v = v[np.isfinite(v)]
                M[si, ci] = float(v.mean()) if v.size else np.nan

        im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest", origin="upper")

        for si in range(n_s):
            for ci in range(n_cols):
                v = M[si, ci]
                if not np.isfinite(v): continue
                txt = _fmt_signed(v) if metric == "r_broadband_4_45" else _fmt_unsigned(v)
                text_color = _text_color_for(v, cmap, norm)
                ax.text(ci, si, txt, ha="center", va="center", fontsize=7.0, fontweight="normal", color=text_color)

        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_s, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7, linestyle="-", alpha=1.0)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)

        for xb in group_bounds:
            ax.axvline(xb, color="white", linewidth=1.4, zorder=3)
            ax.axvline(xb, color=_RULE, linewidth=0.4, zorder=4)

        ax.set_xticks([])
        ax.set_yticks(np.arange(n_s))
        ax.set_yticklabels(all_subjects, fontsize=7.2, color=_INK)
        ax.tick_params(axis="y", length=0, pad=3)

        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
            sp.set_color(_RULE)

        xform = ax.get_xaxis_transform()

        # Channel labels and grouping lines
        for gi, ch in enumerate(channels):
            start = gi * n_ref
            end = start + n_ref - 1
            center = (start + end) / 2

            ax.text(
                center, 1.065, ch, transform=xform,
                ha="center", va="bottom", fontsize=7.8,
                fontweight="bold", color=_INK, clip_on=False,
            )
            ax.plot(
                [start - 0.42, end + 0.42], [1.035, 1.035],
                transform=xform, color=_RULE, linewidth=0.8, clip_on=False,
            )

        # Reference labels
        for ci, (ch, ref) in enumerate(col_specs):
            ax.text(
                ci, 1.005, REF_DISPLAY.get(ref.lower(), ref.upper()),
                transform=xform, ha="center", va="bottom",
                fontsize=6.8, color=_MUTED, clip_on=False,
            )

        # Panel title
        letter = "ab"[k] if k < 26 else str(k + 1)
        ax.text(
            0.000, 1.105, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10.0,
            fontweight="bold", color=_INK, clip_on=False,
        )
        ax.text(
            0.035, 1.105, metric_titles.get(metric, metric),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.8, color=_INK, clip_on=False,
        )

        # Colorbar
        pos = ax.get_position()
        cbar_pad, cbar_w = 0.012, 0.014
        cax = fig.add_axes([pos.x1 + cbar_pad, pos.y0, cbar_w, pos.height])
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_linewidth(0.5)
        cb.outline.set_edgecolor(_RULE)
        cb.ax.tick_params(labelsize=6.8, length=2.0, width=0.5, pad=1.8)

        if metric == "r_broadband_4_45":
            cb.set_label("Broadband Pearson r (4\u201345 Hz)", fontsize=7.8, labelpad=5, color=_INK)
            cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        else:
            cb.set_label("Theta coherence (4\u20138 Hz)", fontsize=7.8, labelpad=5, color=_INK)
            cb.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])

    # Shared y-axis label
    y_mid = (axes[0].get_position().y0 + axes[0].get_position().y1) / 2
    fig.text(
        0.038, y_mid, "Subject",
        ha="center", va="center", rotation="vertical",
        fontsize=8.5, color=_INK,
    )

    # Shared x-axis label
    x_mid = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.text(
        x_mid, 0.035, "Target EEG channel  ×  EOG reference",
        ha="center", va="bottom", fontsize=8.5, color=_INK,
    )

    # Main title first, then panel titles
    if title:
        fig.text(
            0.5, 0.955, title,
            ha="center", va="bottom",
            fontsize=11.0, fontweight="bold", color=_INK,
        )

    if subtitle:
        fig.text(
            0.5, 0.925, subtitle,
            ha="center", va="bottom",
            fontsize=7.8, color=_MUTED,
        )

    if save_fig:
        fig.savefig(
            file_name, dpi=dpi, bbox_inches="tight",
            facecolor="white", pad_inches=0.03,
        )

    return fig

def plot_supplementary_coupling_alpha_beta(
    df,
    channels=("Fz", "Pz", "Oz"),
    references=("heog", "veog"),
    metrics=("coh_alpha", "coh_beta"),
    metric_titles={
        "coh_alpha": "Alpha coherence: 8\u201313 Hz",
        "coh_beta": "Beta coherence: 13\u201325 Hz",
    },
    cmaps={
        "coh_alpha": "RdYlGn",
        "coh_beta": "RdYlGn",
    },
    value_ranges={
        "coh_alpha": (0.0, 1.0),
        "coh_beta": (0.0, 1.0),
    },
    figsize=(11.2, 7.6),
    dpi=400,
    title="EEG\u2013EOG spectral coupling across subjects and channels",
    subtitle=None,
    save_fig=False,
    file_name="fig_S_EEG_EOG_alpha_beta_coupling.pdf",
):
    """
    Two side-by-side supplementary heatmaps of subject × (channel × EOG reference).

    Panels
    ------
    a  Alpha coherence
    b  Beta coherence

    Column hierarchy
    ----------------
    Channel
        ├── hEOG
        └── vEOG
    """

    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import Normalize

    _INK, _MUTED, _RULE = "#1a1a1a", "#3a3a3a", "#2b2b2b"
    REF_DISPLAY = {"heog": "hEOG", "veog": "vEOG", "zemg": "zEMG"}

    def _fmt_unsigned(v, nd=2):
        if not np.isfinite(v): return "\u2014"
        return f"{v:.{nd}f}"

    def _text_color_for(value, cmap, norm, lum_thresh=0.52):
        r, g, b, _ = cmap(norm(value))
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "white" if luminance < lum_thresh else "black"

    all_subjects = sorted(df["Subject"].unique())
    n_s = len(all_subjects)
    col_specs = [(ch, ref) for ch in channels for ref in references]
    n_cols = len(col_specs)
    n_ref = len(references)
    group_bounds = [n_ref * (gi + 1) - 0.5 for gi in range(len(channels) - 1)]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        1, len(metrics), figure=fig,
        left=0.085, right=0.885, top=0.835, bottom=0.095,
        wspace=0.34,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(metrics))]
    df_idx = df.set_index(["Subject", "Target", "Reference"])

    for k, metric in enumerate(metrics):
        ax = axes[k]
        cmap = mpl.colormaps[cmaps[metric]]
        vmin, vmax = value_ranges[metric]
        norm = Normalize(vmin=vmin, vmax=vmax)

        M = np.full((n_s, n_cols), np.nan)

        for si, sid in enumerate(all_subjects):
            for ci, (ch, ref) in enumerate(col_specs):
                try:
                    v = df_idx.loc[(sid, ch, ref), metric]
                except KeyError:
                    continue
                v = np.asarray(v.to_numpy() if isinstance(v, pd.Series) else v, dtype=float)
                v = v[np.isfinite(v)]
                M[si, ci] = float(v.mean()) if v.size else np.nan

        im = ax.imshow(
            M, aspect="auto", cmap=cmap, norm=norm,
            interpolation="nearest", origin="upper",
        )

        # In-cell values
        for si in range(n_s):
            for ci in range(n_cols):
                v = M[si, ci]
                if not np.isfinite(v): continue
                ax.text(
                    ci, si, _fmt_unsigned(v),
                    ha="center", va="center",
                    fontsize=7.0, fontweight="normal",
                    color=_text_color_for(v, cmap, norm),
                )

        # Cell separators
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_s, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7, linestyle="-", alpha=1.0)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)

        # Channel-group separators
        for xb in group_bounds:
            ax.axvline(xb, color="white", linewidth=1.4, zorder=3)
            ax.axvline(xb, color=_RULE, linewidth=0.4, zorder=4)

        # Axes
        ax.set_xticks([])
        ax.set_yticks(np.arange(n_s))
        ax.set_yticklabels(all_subjects, fontsize=7.2, color=_INK)
        ax.tick_params(axis="y", length=0, pad=3)

        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
            sp.set_color(_RULE)

        xform = ax.get_xaxis_transform()

        # Channel labels + grouping lines
        for gi, ch in enumerate(channels):
            start = gi * n_ref
            end = start + n_ref - 1
            center = (start + end) / 2

            ax.text(
                center, 1.065, ch,
                transform=xform, ha="center", va="bottom",
                fontsize=7.8, fontweight="bold",
                color=_INK, clip_on=False,
            )

            ax.plot(
                [start - 0.42, end + 0.42], [1.035, 1.035],
                transform=xform, color=_RULE,
                linewidth=0.8, clip_on=False,
            )

        # Reference labels
        for ci, (ch, ref) in enumerate(col_specs):
            ax.text(
                ci, 1.005,
                REF_DISPLAY.get(ref.lower(), ref.upper()),
                transform=xform, ha="center", va="bottom",
                fontsize=6.8, color=_MUTED, clip_on=False,
            )

        # Panel title
        letter = "ab"[k]
        ax.text(
            0.000, 1.105, letter,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10.0, fontweight="bold",
            color=_INK, clip_on=False,
        )
        ax.text(
            0.035, 1.105,
            metric_titles.get(metric, metric),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.8, color=_INK, clip_on=False,
        )

        # Colorbar
        pos = ax.get_position()
        cbar_pad, cbar_w = 0.012, 0.014
        cax = fig.add_axes([pos.x1 + cbar_pad, pos.y0, cbar_w, pos.height])
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_linewidth(0.5)
        cb.outline.set_edgecolor(_RULE)
        cb.ax.tick_params(labelsize=6.8, length=2.0, width=0.5, pad=1.8)
        cb.set_label(
            metric_titles.get(metric, metric),
            fontsize=7.8, labelpad=5, color=_INK,
        )
        cb.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])

    # Shared y-axis label
    y_mid = (axes[0].get_position().y0 + axes[0].get_position().y1) / 2
    fig.text(
        0.038, y_mid, "Subject",
        ha="center", va="center", rotation="vertical",
        fontsize=8.5, color=_INK,
    )

    # Shared x-axis label
    x_mid = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.text(
        x_mid, 0.035,
        "Target EEG channel  ×  EOG reference",
        ha="center", va="bottom",
        fontsize=8.5, color=_INK,
    )

    # Main title
    if title:
        fig.text(
            0.5, 0.955, title,
            ha="center", va="bottom",
            fontsize=11.0, fontweight="bold",
            color=_INK,
        )

    if subtitle:
        fig.text(
            0.5, 0.925, subtitle,
            ha="center", va="bottom",
            fontsize=7.8, color=_MUTED,
        )

    if save_fig:
        fig.savefig(
            file_name, dpi=dpi, bbox_inches="tight",
            facecolor="white", pad_inches=0.03,
        )

    return fig

def plot_population_level_coupling(
    df_wide,
    df_population,
    references=("heog", "veog"),
    channels=("Fz", "Pz", "Oz"),
    title="Population-level EEG\u2013EOG coupling aggregation",
    subtitle="N = 18 participants",
    save_fig=False,
    file_name="fig_coupling_main.pdf",
):
    """
    Nature-style main figure: population-level EEG-EOG coupling aggregation.

    Input
    -----
    df_wide : one row per Subject x Target x Reference
              columns: Subject, Target, Reference,
                       r_broadband_4_45, coh_theta, coh_alpha, coh_beta

    df_population : population-level aggregation table
                    columns: Target, Reference, Metric,
                             Mean, CI_low, CI_high

    Panel a : Broadband Pearson r (4-45 Hz), Fisher-z aggregated
    Panel b : Theta-band coherence (4-8 Hz)

    Notes
    -----
    - y-limits are computed dynamically from the data (dots and
      population CIs) so that no error bar is clipped.
    - Population mean and 95% bootstrap CI are read directly from
      `df_population` to guarantee figure/table consistency.
    - Fisher-z is NOT applied to coherence; it is applied only
      to the broadband Pearson r.
    """
    from matplotlib.lines import Line2D

    # ---------------- Reference style ----------------
    REF_STYLE = {
        "heog": dict(label="hEOG", color="#4C72B0", dark="#2F5A9F",
                     offset=-0.17, marker="o"),
        "veog": dict(label="vEOG", color="#C44E52", dark="#9A2C22",
                     offset=+0.17, marker="o"),
    }

    # Map df_wide column names to the corresponding display labels
    # used in df_population["Metric"]. This ensures that the point
    # estimate and CI shown in the figure are the SAME quantities
    # computed in the aggregation cell, rather than a second,
    # independent bootstrap.
    METRIC_LABEL_MAP = {
        "r_broadband_4_45": "r (4\u201345 Hz)",
        "coh_theta":        r"$\theta$-coherence",
        "coh_alpha":        r"$\alpha$-coherence",
        "coh_beta":         r"$\beta$-coherence",
    }

    # ---------------- Panel spec ----------------
    PANELS = [
        dict(
            col="r_broadband_4_45", letter="a",
            subtitle="Broadband Pearson $r$ (4\u201345 Hz)",
            ylabel="Fisher-$z$ aggregated Pearson $r$",
            refline=0.0,
        ),
        dict(
            col="coh_theta", letter="b",
            subtitle="Theta coherence (4\u20138 Hz)",
            ylabel="Coherence (Theta)",
            refline=None,
        ),
    ]

    # ---------------- Figure ----------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.2, 3.1),
        gridspec_kw={"wspace": 0.32},
    )

    x_centers = np.arange(len(channels))
    rng = np.random.default_rng(0)

    for ax, panel in zip(axes, PANELS):
        col = panel["col"]
        metric_label = METRIC_LABEL_MAP[col]

        # subtle alternating channel-group background bands
        for i in range(len(channels)):
            if i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5,
                           color="0.965", zorder=0, lw=0)

        # zero reference line (thin dotted)
        if panel["refline"] is not None:
            ax.axhline(panel["refline"], color="0.55",
                       ls=(0, (4, 3)), lw=0.55, zorder=1)

        # ------------- data -------------
        vals_collected = []   # for dynamic y-limit computation

        for ref in references:
            s = REF_STYLE[ref]
            for i, ch in enumerate(channels):

                # ---- individual subject dots ----
                vals = (
                    df_wide.loc[
                        (df_wide["Target"] == ch) &
                        (df_wide["Reference"] == ref),
                        col,
                    ].dropna().to_numpy(float)
                )
                if vals.size == 0:
                    continue

                vals_collected.extend(vals.tolist())

                xpos = x_centers[i] + s["offset"]

                jitter = rng.normal(0, 0.030, vals.size)
                ax.scatter(
                    np.full(vals.size, xpos) + jitter, vals,
                    s=7.5, alpha=0.32,
                    color=s["color"],
                    edgecolors=s["dark"],
                    linewidths=0.25,
                    zorder=2,
                )

                # ---- population mean + bootstrap CI (from df_population) ----
                row = df_population[
                    (df_population["Target"] == ch) &
                    (df_population["Reference"] == ref) &
                    (df_population["Metric"] == metric_label)
                ]
                if row.empty:
                    continue
                row = row.iloc[0]

                m  = float(row["Mean"])
                lo = float(row["CI_low"])
                hi = float(row["CI_high"])

                vals_collected.extend([lo, hi])

                ax.errorbar(
                    xpos, m, yerr=[[m - lo], [hi - m]],
                    fmt="o", ms=4.2,
                    color="black",
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    capsize=1.9, capthick=0.85, elinewidth=0.85,
                    zorder=5,
                )

        # ------------- dynamic y-limits -------------
        # Compute from the actual data (dots + population CIs), with
        # 8% padding on both sides. This prevents any error bar from
        # being clipped by the x-axis.
        if vals_collected:
            lo_data = float(np.nanmin(vals_collected))
            hi_data = float(np.nanmax(vals_collected))
        else:
            lo_data, hi_data = 0.0, 1.0

        span = hi_data - lo_data
        if span <= 0:
            span = 1.0
        pad = 0.08 * span
        lo_lim = lo_data - pad
        hi_lim = hi_data + pad

        # Ensure the reference line stays inside the view
        if panel["refline"] is not None:
            ref_val = panel["refline"]
            if ref_val < lo_lim:
                lo_lim = ref_val - 0.05 * span
            if ref_val > hi_lim:
                hi_lim = ref_val + 0.05 * span

        # ------------- axes cosmetics -------------
        ax.set_xticks(x_centers)
        ax.set_xticklabels(channels, fontsize=8)
        ax.set_xlim(-0.55, len(channels) - 0.45)
        ax.set_ylim(lo_lim, hi_lim)
        ax.set_xlabel("EEG target channel", fontsize=8, labelpad=3)
        ax.set_ylabel(panel["ylabel"], fontsize=8, labelpad=4)
        ax.tick_params(axis="both", length=2.5, width=0.6,
                       labelsize=7.5, pad=2.5)
        ax.grid(axis="y", ls=":", lw=0.5, color="0.85",
                zorder=0, alpha=1.0)

        ax.set_title(
            f"$\\mathbf{{{panel['letter']}}}$   {panel['subtitle']}",
            loc="left", fontsize=8.5, color="0.15", pad=7,
        )

    # ---------------- Shared top legend ----------------
    handles = [
        Line2D([0], [0], marker=REF_STYLE["heog"]["marker"], color="w",
               markerfacecolor=REF_STYLE["heog"]["color"],
               markeredgecolor=REF_STYLE["heog"]["dark"],
               markeredgewidth=0.3, markersize=4.5,
               label=REF_STYLE["heog"]["label"]),
        Line2D([0], [0], marker=REF_STYLE["veog"]["marker"], color="w",
               markerfacecolor=REF_STYLE["veog"]["color"],
               markeredgecolor=REF_STYLE["veog"]["dark"],
               markeredgewidth=0.3, markersize=4.5,
               label=REF_STYLE["veog"]["label"]),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='black', markersize=4.5,
               markeredgecolor='white', markeredgewidth=0.6,
               label='Population estimate (95% bootstrap CI)'),
    ]

    # ================================================================
    # Header spacing
    # ----------------------------------------------------------------
    # The three header elements are anchored with the SAME va="bottom"
    # convention, so that HEADER_GAP is the actual gap between:
    #
    #   title  (top)
    #     ^ HEADER_GAP
    #   subtitle / N=18  (middle)
    #     ^ HEADER_GAP
    #   legend  (bottom)
    #
    # To move the whole block up/down  -> change Y_LEGEND_BOTTOM.
    # To change all three gaps at once -> change HEADER_GAP.
    # ================================================================
    Y_LEGEND_BOTTOM = 0.985      # bottom of the legend block (fig fraction)
    HEADER_GAP      = 0.06      # equal gap between the three elements

    y_legend_bottom = Y_LEGEND_BOTTOM
    y_sub_bottom    = y_legend_bottom + HEADER_GAP
    y_title_bottom  = y_sub_bottom    + HEADER_GAP

    fig.legend(
        handles=handles,
        loc="lower center",                      # anchor = bottom of legend
        bbox_to_anchor=(0.5, y_legend_bottom),   # equal-gap position
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.4,
        columnspacing=1.6,
        borderaxespad=0.0,
    )

    # ---------------- Global title + subtitle ----------------
    # Both use the same anchor (va="bottom"), so the gap to the
    # legend below and to each other is identical (HEADER_GAP).
    if subtitle:
        fig.text(
            0.5, y_sub_bottom, subtitle,
            ha="center", va="bottom",
            fontsize=7.2, color="0.35",
        )

    fig.text(
        0.5, y_title_bottom, title,
        ha="center", va="bottom",
        fontsize=9.0, fontweight="normal", color="0.10",
    )

    # ---------------- Layout reserve ----------------
    # Reserve vertical space for title + subtitle + legend above panels
    fig.tight_layout(rect=[0, 0, 1, 0.83])

    if save_fig:
        fig.savefig(file_name, dpi=400, bbox_inches="tight",
                    facecolor="white", pad_inches=0.02)

    return fig, axes

def plot_supplementary_population_level_coupling(
    df_wide,
    df_population,
    references=("heog", "veog"),
    channels=("Fz", "Pz", "Oz"),
    bands=("theta", "alpha", "beta"),
    band_titles={
        "theta": "Theta (4\u20138 Hz)",
        "alpha": "Alpha (8\u201313 Hz)",
        "beta":  "Beta (13\u201325 Hz)",
    },
    title="Population-level EEG\u2013EOG coherence across frequency bands",
    subtitle="N = 18 participants",
    figsize=(13.5, 3.1),
    save_fig=False,
    file_name="fig_S_coupling_bands.pdf",
):
    """
    Supplementary figure: EEG-EOG coherence for theta, alpha, and
    beta bands, shown as three panels in a single row.

    Parameters
    ----------
    df_wide : DataFrame
        Wide-format coupling table (one row per Subject x Target x
        Reference) with columns: Subject, Target, Reference,
        r_broadband_4_45, coh_theta, coh_alpha, coh_beta.
    df_population : DataFrame
        Population-level aggregation table with columns:
        Target, Reference, Metric, Mean, CI_low, CI_high,
        Cohen_dz, Holm_p.
    references : tuple of str
    channels : tuple of str
    bands : tuple of str
    band_titles : dict
        Display labels for each band (full names, not symbols).
    title : str
    subtitle : str
    figsize : tuple
    save_fig : bool
    file_name : str
    """
    from matplotlib.lines import Line2D

    # ---- metric label mapping ----
    # df_wide uses column names like "coh_theta"; df_population
    # uses display labels such as "$\\theta$-coherence".
    METRIC_LABEL_MAP = {
        "theta": r"$\theta$-coherence",
        "alpha": r"$\alpha$-coherence",
        "beta":  r"$\beta$-coherence",
    }

    # ---- reference style ----
    REF_STYLE = {
        "heog": dict(
            label="hEOG",
            color="#4C72B0", dark="#2F5A9F",
            offset=-0.17, marker="o",
        ),
        "veog": dict(
            label="vEOG",
            color="#C44E52", dark="#9A2C22",
            offset=+0.17, marker="o",
        ),
    }

    # ---- figure: one row, one column per band ----
    n_cols = len(bands)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"wspace": 0.14},
    )
    if n_cols == 1:
        axes = [axes]

    x_centers = np.arange(len(channels))
    rng = np.random.default_rng(0)

    for col_idx, band in enumerate(bands):
        ax = axes[col_idx]
        col = f"coh_{band}"
        metric_label = METRIC_LABEL_MAP[band]

        # subtle alternating channel-group background bands
        for i in range(len(channels)):
            if i % 2 == 0:
                ax.axvspan(
                    i - 0.5, i + 0.5,
                    color="0.965", zorder=0, lw=0,
                )

        # ------------- data -------------
        for ref in references:
            s = REF_STYLE[ref]
            for i, ch in enumerate(channels):

                # ---- individual subject dots ----
                vals = (
                    df_wide.loc[
                        (df_wide["Target"] == ch) &
                        (df_wide["Reference"] == ref),
                        col,
                    ].dropna().to_numpy(float)
                )
                if vals.size == 0:
                    continue

                xpos = x_centers[i] + s["offset"]

                jitter = rng.normal(0, 0.030, vals.size)
                ax.scatter(
                    np.full(vals.size, xpos) + jitter, vals,
                    s=7.5, alpha=0.32,
                    color=s["color"],
                    edgecolors=s["dark"],
                    linewidths=0.25,
                    zorder=2,
                )

                # ---- population mean + bootstrap CI (from df_population) ----
                row = df_population[
                    (df_population["Target"] == ch) &
                    (df_population["Reference"] == ref) &
                    (df_population["Metric"] == metric_label)
                ]
                if row.empty:
                    continue
                row = row.iloc[0]

                m  = float(row["Mean"])
                lo = float(row["CI_low"])
                hi = float(row["CI_high"])

                ax.errorbar(
                    xpos, m,
                    yerr=[[m - lo], [hi - m]],
                    fmt="o", ms=4.2,
                    color="black",
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    capsize=1.9, capthick=0.85, elinewidth=0.85,
                    zorder=5,
                )

        # ------------- axes cosmetics -------------
        ax.set_xticks(x_centers)
        ax.set_xticklabels(channels, fontsize=8)
        ax.set_xlim(-0.55, len(channels) - 0.45)
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(
            axis="both", length=2.5, width=0.6,
            labelsize=7.5, pad=2.5,
        )
        ax.grid(
            axis="y", ls=":", lw=0.5, color="0.85",
            zorder=0, alpha=1.0,
        )

        # y-label only on the leftmost panel
        if col_idx == 0:
            ax.set_ylabel("Coherence", fontsize=8, labelpad=4)

        # x-label on every panel (they are side by side)
        ax.set_xlabel(
            "EEG target channel", fontsize=8, labelpad=3,
        )

        # panel letter + band name (full word, not symbol)
        letter = chr(ord("a") + col_idx)
        ax.set_title(
            f"$\\mathbf{{{letter}}}$   {band_titles.get(band, band)}",
            loc="left", fontsize=8.5, color="0.15", pad=7,
        )

    # ---- shared legend (top, single row) ----
    handles = [
        Line2D(
            [0], [0],
            marker=REF_STYLE["heog"]["marker"], color="w",
            markerfacecolor=REF_STYLE["heog"]["color"],
            markeredgecolor=REF_STYLE["heog"]["dark"],
            markeredgewidth=0.3, markersize=4.5,
            label=REF_STYLE["heog"]["label"],
        ),
        Line2D(
            [0], [0],
            marker=REF_STYLE["veog"]["marker"], color="w",
            markerfacecolor=REF_STYLE["veog"]["color"],
            markeredgecolor=REF_STYLE["veog"]["dark"],
            markeredgewidth=0.3, markersize=4.5,
            label=REF_STYLE["veog"]["label"],
        ),
        Line2D(
            [0], [0],
            marker='o', color='w',
            markerfacecolor='black', markersize=4.5,
            markeredgecolor='white', markeredgewidth=0.6,
            label='Population estimate (95% bootstrap CI)',
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handletextpad=0.4,
        columnspacing=1.6,
        borderaxespad=0.0,
    )

    # ---- global title + subtitle ----
    fig.suptitle(
        title,
        fontsize=9.5, fontweight="normal",
        y=1.13, color="0.10",
        ha="center",
    )
    if subtitle:
        fig.text(
            0.5, 1.045, subtitle,
            ha="center", va="bottom",
            fontsize=7.5, color="0.35",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    if save_fig:
        fig.savefig(
            file_name, dpi=400, bbox_inches="tight",
            facecolor="white", pad_inches=0.02,
        )

    return fig, axes

# ============================================================
# Helper: dynamic y-limits for condition-level panels
# ============================================================
def _compute_ylim_condition_panels(
    df_subject,
    subject_col,
    df_population,
    population_col,
    channels,
    references,
    predictors=("Video model", "EOG regression"),
):
    """
    Compute y-limits for a condition-level panel from the union of
    participant-level dots and population-level CIs. Adds 8% padding.
    """
    vals = []

    for pred in predictors:
        for ch in channels:
            sub = df_subject[
                (df_subject["EEG_target"] == ch) &
                (df_subject["Predictor"] == pred)
            ][subject_col].dropna().to_numpy(float)
            if sub.size:
                vals.extend(sub.tolist())

            pop = df_population[
                (df_population["Target"] == ch) &
                (df_population["Predictor"] == pred)
            ]
            if not pop.empty:
                vals.extend([
                    float(pop[population_col + "_CI_low"].iloc[0]),
                    float(pop[population_col + "_CI_high"].iloc[0]),
                ])

    if not vals:
        return (-0.55, 1.05)

    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    span = hi - lo if hi > lo else 1.0
    pad = 0.08 * span
    return (lo - pad, hi + pad)

def plot_main_figure_video_vs_eog(
    df_performance,
    df_spectral,
    df_population_conditions,
    df_population_delta_z_r,
    df_population_spectral,
    subjects,
    channels=("Fz", "Pz", "Oz"),
    bands=("theta", "alpha", "beta"),
    band_colors={
        "theta": "#d62728",   # red
        "alpha": "#9467bd",   # purple
        "beta":  "#8c564b",   # brown
    },
    save_fig=False,
    file_name="fig_main_video_vs_eog.pdf",
):
    """
    Main figure for the Video vs EOG comparison across EEG target
    channels.

    Parameters
    ----------
    df_performance : DataFrame
        Participant-level prediction table. Columns:
            Subject, EEG_target, Predictor, r_fisher_z_aggregated.
    df_spectral : DataFrame
        Participant-level spectral table. Columns:
            Subject, EEG_target, Predictor,
            coh_theta, coh_alpha, coh_beta,
            rbp_err_theta, rbp_err_alpha, rbp_err_beta.
    df_population_conditions : DataFrame
        Population-level condition estimates for Panel a.
    df_population_delta_z_r : DataFrame
        Population-level paired contrast Delta_z_r for Panel b.
    df_population_spectral : DataFrame
        Population-level spectral contrast for Panels c and d.
        Rows with Metric == "Coherence" feed Panel c;
        rows with Metric == "RBP error" feed Panel d.
    subjects : list of int
    channels : tuple of str
    bands : tuple of str
    band_colors : dict
    save_fig : bool
    file_name : str
    """
    from matplotlib.lines import Line2D

    # ---------------- style constants ----------------
    COLOR_VIDEO = "#1f77b4"
    COLOR_EOG   = "#ff7f0e"
    COLOR_DZ    = "#2ca02c"
    COLOR_PAIR  = "0.78"

    # ---------------- figure: 2 x 2 ----------------
    fig, axes = plt.subplots(
        2, 2,
        figsize=(13.0, 8.5),
        gridspec_kw={"wspace": 0.30, "hspace": 0.50},
    )

    x_centers = np.arange(len(channels))
    off = {"Video model": -0.12, "EOG regression": 0.12}
    rng = np.random.default_rng(0)

    # ============================================================
    # PANEL A — Fisher-z aggregated Pearson r
    # ============================================================
    ax = axes[0, 0]
    ax.axhline(0.0, color="0.55", ls=(0, (4, 3)), lw=0.6, zorder=1)

    for i, ch in enumerate(channels):

        # ---- paired lines ----
        for sid in subjects:
            mask_v = (
                (df_performance["Subject"].astype(int) == int(sid)) &
                (df_performance["EEG_target"] == ch) &
                (df_performance["Predictor"] == "Video model")
            )
            mask_e = (
                (df_performance["Subject"].astype(int) == int(sid)) &
                (df_performance["EEG_target"] == ch) &
                (df_performance["Predictor"] == "EOG regression")
            )
            if mask_v.any() and mask_e.any():
                vv = df_performance.loc[mask_v, "r_fisher_z_aggregated"].values[0]
                ee = df_performance.loc[mask_e, "r_fisher_z_aggregated"].values[0]
                if np.isfinite(vv) and np.isfinite(ee):
                    ax.plot(
                        [x_centers[i] + off["Video model"],
                         x_centers[i] + off["EOG regression"]],
                        [vv, ee],
                        color=COLOR_PAIR, lw=0.6, alpha=0.55, zorder=2,
                    )

        # ---- individual dots ----
        for pred, color in (("Video model", COLOR_VIDEO),
                            ("EOG regression", COLOR_EOG)):
            vals = df_performance[
                (df_performance["EEG_target"] == ch) &
                (df_performance["Predictor"] == pred)
            ]["r_fisher_z_aggregated"].dropna().to_numpy(float)
            if vals.size == 0:
                continue
            jitter = rng.normal(0, 0.030, vals.size)
            ax.scatter(
                np.full(vals.size, x_centers[i] + off[pred]) + jitter, vals,
                s=18, alpha=0.55, color=color,
                edgecolors="white", linewidths=0.35, zorder=3,
            )

        # ---- population mean + 95% bootstrap CI ----
        for pred in ("Video model", "EOG regression"):
            sub = df_population_conditions[
                (df_population_conditions["Target"] == ch) &
                (df_population_conditions["Predictor"] == pred)
            ]
            if sub.empty:
                continue
            sub = sub.iloc[0]
            m     = float(sub["r"])
            ci_lo = float(sub["r_CI_low"])
            ci_hi = float(sub["r_CI_high"])
            ax.errorbar(
                x_centers[i] + off[pred], m,
                yerr=[[m - ci_lo], [ci_hi - m]],
                fmt="o", ms=7,
                color="black", markeredgecolor="white",
                markeredgewidth=0.5,
                capsize=2.2, capthick=0.9, elinewidth=0.9,
                zorder=5,
            )

    # ---- dynamic y-limits ----
    a_lo, a_hi = _compute_ylim_condition_panels(
        df_performance, "r_fisher_z_aggregated",
        df_population_conditions, "r",
        channels=channels,
        references=None,
        predictors=("Video model", "EOG regression"),
    )
    ax.set_ylim(a_lo, a_hi)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(channels)
    ax.set_xlim(-0.55, len(channels) - 0.45)
    ax.set_xlabel("EEG target channel", labelpad=3)
    ax.set_ylabel("Fisher-$z$ aggregated Pearson $r$")
    ax.set_title("a   Video vs EOG prediction", loc="left", pad=6)
    ax.grid(axis="y", ls=":", lw=0.5, color="0.9", zorder=0)

    # ============================================================
    # PANEL B — Δz(r) = Video − EOG
    # ============================================================
    ax = axes[0, 1]
    ax.axhline(0.0, color="0.55", ls=(0, (4, 3)), lw=0.6, zorder=1)

    delta_vals = []
    for i, ch in enumerate(channels):

        subj_deltas = []
        for sid in subjects:
            mask_v = (
                (df_performance["Subject"].astype(int) == int(sid)) &
                (df_performance["EEG_target"] == ch) &
                (df_performance["Predictor"] == "Video model")
            )
            mask_e = (
                (df_performance["Subject"].astype(int) == int(sid)) &
                (df_performance["EEG_target"] == ch) &
                (df_performance["Predictor"] == "EOG regression")
            )
            if mask_v.any() and mask_e.any():
                rv = df_performance.loc[mask_v, "r_fisher_z_aggregated"].values[0]
                re = df_performance.loc[mask_e, "r_fisher_z_aggregated"].values[0]
                if np.isfinite(rv) and np.isfinite(re):
                    zv = np.arctanh(np.clip(rv, -0.9999, 0.9999))
                    ze = np.arctanh(np.clip(re, -0.9999, 0.9999))
                    subj_deltas.append(zv - ze)

        subj_deltas = np.array(subj_deltas, dtype=float)
        subj_deltas = subj_deltas[np.isfinite(subj_deltas)]
        if subj_deltas.size == 0:
            continue

        delta_vals.extend(subj_deltas.tolist())

        jitter = rng.normal(0, 0.045, subj_deltas.size)
        ax.scatter(
            np.full(subj_deltas.size, x_centers[i]) + jitter, subj_deltas,
            s=18, alpha=0.55, color=COLOR_DZ,
            edgecolors="white", linewidths=0.35, zorder=3,
        )

        row = df_population_delta_z_r[
            df_population_delta_z_r["Target"] == ch
        ]
        if row.empty:
            continue
        row = row.iloc[0]

        m     = float(row["Delta_z_r"])
        ci_lo = float(row["CI_low"])
        ci_hi = float(row["CI_high"])
        p_h   = float(row["Holm_p"])

        delta_vals.extend([ci_lo, ci_hi])

        ax.errorbar(
            x_centers[i], m,
            yerr=[[m - ci_lo], [ci_hi - m]],
            fmt="o", ms=7,
            color="black", markeredgecolor="white",
            markeredgewidth=0.5,
            capsize=2.2, capthick=0.9, elinewidth=0.9,
            zorder=5,
        )

        if np.isfinite(p_h):
            ax.text(
                x_centers[i] + 0.22, m,
                r"$P_{\mathrm{Holm}}$" + f" = {p_h:.1e}",
                fontsize=7, va="center", ha="left", color="0.15",
            )

    if delta_vals:
        lo_d = float(np.nanmin(delta_vals))
        hi_d = float(np.nanmax(delta_vals))
    else:
        lo_d, hi_d = -1.0, 0.0
    span = hi_d - lo_d if hi_d > lo_d else 1.0
    pad = 0.10 * span
    ax.set_ylim(lo_d - pad, hi_d + pad)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(channels)
    ax.set_xlim(-0.55, len(channels) - 0.45)
    ax.set_xlabel("EEG target channel", labelpad=3)
    ax.set_ylabel(r"$\Delta z(r)$: Video $-$ EOG")
    ax.set_title("b   Incremental correlation over EOG", loc="left", pad=6)
    ax.grid(axis="y", ls=":", lw=0.5, color="0.9", zorder=0)

    # ============================================================
    # Helper: draw a Δ-panel (coherence or RBP)
    # ============================================================
    def _draw_delta_band_panel(
        ax, subject_col_prefix, population_metric,
        panel_letter, panel_title, ylabel,
    ):
        ax.axhline(0.0, color="0.55", ls=(0, (4, 3)), lw=0.6, zorder=1)

        n_b = len(bands)
        bw = 0.24
        b_offs = {
            b: (i - (n_b - 1) / 2) * bw
            for i, b in enumerate(bands)
        }

        all_vals = []
        for i, ch in enumerate(channels):
            for b in bands:

                subject_col = f"{subject_col_prefix}_{b}"

                subj_deltas = []
                for sid in subjects:
                    mask_v = (
                        (df_spectral["Subject"].astype(int) == int(sid)) &
                        (df_spectral["EEG_target"] == ch) &
                        (df_spectral["Predictor"] == "Video model")
                    )
                    mask_e = (
                        (df_spectral["Subject"].astype(int) == int(sid)) &
                        (df_spectral["EEG_target"] == ch) &
                        (df_spectral["Predictor"] == "EOG regression")
                    )
                    if mask_v.any() and mask_e.any():
                        cv = df_spectral.loc[mask_v, subject_col].values[0]
                        ce = df_spectral.loc[mask_e, subject_col].values[0]
                        if np.isfinite(cv) and np.isfinite(ce):
                            subj_deltas.append(cv - ce)

                subj_deltas = np.array(subj_deltas, dtype=float)
                subj_deltas = subj_deltas[np.isfinite(subj_deltas)]
                if subj_deltas.size == 0:
                    continue

                all_vals.extend(subj_deltas.tolist())

                xpos = x_centers[i] + b_offs[b]

                jitter = rng.normal(0, 0.018, subj_deltas.size)
                ax.scatter(
                    np.full(subj_deltas.size, xpos) + jitter, subj_deltas,
                    s=18, alpha=0.50, color=band_colors[b],
                    edgecolors="white", linewidths=0.30, zorder=3,
                )

                row = df_population_spectral[
                    (df_population_spectral["Target"] == ch) &
                    (df_population_spectral["Band"] == b) &
                    (df_population_spectral["Metric"] == population_metric)
                ]
                if row.empty:
                    continue
                row = row.iloc[0]

                m     = float(row["Effect"])
                ci_lo = float(row["CI_low"])
                ci_hi = float(row["CI_high"])

                all_vals.extend([ci_lo, ci_hi])

                ax.errorbar(
                    xpos, m,
                    yerr=[[m - ci_lo], [ci_hi - m]],
                    fmt="o", ms=6,
                    color="black", markeredgecolor="white",
                    markeredgewidth=0.45,
                    capsize=2.0, capthick=0.85, elinewidth=0.85,
                    zorder=5,
                )

        if all_vals:
            lo_v = float(np.nanmin(all_vals))
            hi_v = float(np.nanmax(all_vals))
        else:
            lo_v, hi_v = -0.5, 0.5
        span = hi_v - lo_v if hi_v > lo_v else 1.0
        pad = 0.10 * span
        ax.set_ylim(lo_v - pad, hi_v + pad)

        ax.set_xticks(x_centers)
        ax.set_xticklabels(channels)
        ax.set_xlim(-0.55, len(channels) - 0.45)
        ax.set_xlabel("EEG target channel", labelpad=3)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel_letter}   {panel_title}", loc="left", pad=6)
        ax.grid(axis="y", ls=":", lw=0.5, color="0.9", zorder=0)

    # ============================================================
    # PANEL C — ΔCoh = Video − EOG, per channel × band
    # ============================================================
    _draw_delta_band_panel(
        ax           = axes[1, 0],
        subject_col_prefix = "coh",
        population_metric  = "Coherence",
        panel_letter = "c",
        panel_title  = "Incremental coherence over EOG",
        ylabel       = r"$\Delta$Coherence: Video $-$ EOG",
    )

    # ============================================================
    # PANEL D — ΔRBP error = Video − EOG, per channel × band
    # ============================================================
    _draw_delta_band_panel(
        ax           = axes[1, 1],
        subject_col_prefix = "rbp_err",
        population_metric  = "RBP error",
        panel_letter = "d",
        panel_title  = "Incremental RBP error over EOG",
        ylabel       = r"$\Delta$RBP error: Video $-$ EOG",
    )

    # ============================================================
    # Shared top legend (horizontal)
    # ============================================================
    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLOR_VIDEO, markersize=6,
               markeredgecolor='white', markeredgewidth=0.4,
               label="Video model"),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLOR_EOG, markersize=6,
               markeredgecolor='white', markeredgewidth=0.4,
               label="EOG regression"),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLOR_DZ, markersize=6,
               markeredgecolor='white', markeredgewidth=0.4,
               label=r"$\Delta z(r)$ (Video $-$ EOG)"),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=band_colors["theta"], markersize=5,
               markeredgecolor='white', markeredgewidth=0.4,
               label=r"Theta band"),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=band_colors["alpha"], markersize=5,
               markeredgecolor='white', markeredgewidth=0.4,
               label=r"Alpha band"),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=band_colors["beta"], markersize=5,
               markeredgecolor='white', markeredgewidth=0.4,
               label=r"Beta band"),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='black', markersize=6,
               markeredgecolor='white', markeredgewidth=0.5,
               label="Population estimate (95% bootstrap CI)"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=7,
        frameon=False,
        fontsize=7.5,
        handletextpad=0.35,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    # ============================================================
    # Global title with N
    # ============================================================
    fig.suptitle(
        f"Video vs EOG prediction across EEG target channels "
        f"(N = {len(subjects)} participants)",
        fontsize=11, y=1.065, color="0.10",
    )

    # ============================================================
    # Layout
    # ============================================================
    fig.tight_layout(rect=[0, 0, 1, 0.955])

    if save_fig:
        fig.savefig(file_name, dpi=400, bbox_inches="tight",
                    facecolor="white", pad_inches=0.05)

    return fig, axes


def plot_supplementary_spectral_heatmaps(
    df_spectral, subjects, channels=("Fz", "Pz", "Oz"), bands=("theta", "alpha", "beta"),
    save_fig=False, file_name="fig_S_spectral_heatmaps.pdf", annotate=True, fmt=".2f"
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    coherence_cols = {"theta": "coh_theta", "alpha": "coh_alpha", "beta": "coh_beta"}
    rbp_cols = {"theta": "rbp_err_theta", "alpha": "rbp_err_alpha", "beta": "rbp_err_beta"}
    band_labels = {"theta": "Theta", "alpha": "Alpha", "beta": "Beta"}

    subject_list = [int(s) for s in subjects]
    df = df_spectral.copy()
    df["Subject"] = df["Subject"].astype(int)

    spectral_columns = [(band, ch) for band in bands for ch in channels]
    x_labels = [ch for band, ch in spectral_columns]
    y_labels = [f"S{sid:02d}" for sid in subject_list]

    def build_matrix(column_map, predictor):
        matrix = np.full((len(subject_list), len(spectral_columns)), np.nan, dtype=float)
        for r, sid in enumerate(subject_list):
            for c, (band, ch) in enumerate(spectral_columns):
                col = column_map[band]
                mask = (df["Subject"] == sid) & (df["EEG_target"] == ch) & (df["Predictor"] == predictor)
                values = df.loc[mask, col].dropna().to_numpy(float)
                if values.size: matrix[r, c] = values[0]
        return matrix

    coh_video = build_matrix(coherence_cols, "Video model")
    coh_eog = build_matrix(coherence_cols, "EOG regression")
    rbp_video = build_matrix(rbp_cols, "Video model")
    rbp_eog = build_matrix(rbp_cols, "EOG regression")

    delta_coh = coh_video - coh_eog
    delta_rbp = rbp_video - rbp_eog
    _rbp_finite = np.concatenate([
        rbp_video[np.isfinite(rbp_video)].ravel(),
        rbp_eog[np.isfinite(rbp_eog)].ravel(),
    ])
    
    shared_rbp_vmax = float(np.max(_rbp_finite)) if _rbp_finite.size else 1.0
    shared_rbp_vmax = max(shared_rbp_vmax, 1e-6)

    fig, axes = plt.subplots(
        2, 3, figsize=(15.2, 9.8),
        gridspec_kw={"wspace": 0.18, "hspace": 0.48}
    )
    axes = np.asarray(axes)

    def draw_heatmap(ax, matrix, title, cbar_label, mode, annotate_cells=True, vmax_override=None):
        finite = matrix[np.isfinite(matrix)]
        vmax = float(np.max(finite)) if finite.size else 1.0
        if vmax_override is not None:
            vmax = vmax_override

        if mode == "coherence":
            im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
        elif mode == "rbp":
            im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis_r", vmin=0.0, vmax=max(vmax, 1e-6))
        elif mode == "delta_coherence":
            max_abs = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 1e-6)
            im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="RdBu_r",
                           norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs))
        elif mode == "delta_rbp":
            max_abs = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 1e-6)
            im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="RdBu",
                           norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs))
        else:
            raise ValueError(f"Unknown heatmap mode: {mode}")

        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=30, fontsize=7.3)
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=7.0)
        ax.tick_params(axis="both", length=0)
        ax.grid(False)
        ax.minorticks_off()

        if annotate_cells:
            cmap, norm = im.cmap, im.norm
            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    value = matrix[r, c]
                    if not np.isfinite(value): continue
                    rr, gg, bb, _ = cmap(norm(value))
                    luminance = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb
                    ax.text(c, r, format(value, fmt), ha="center", va="center",
                            fontsize=5.8, color="white" if luminance < 0.52 else "black")

        # Panel title → band labels → grouping line → channel labels
        ax.set_title(title, loc="left", fontsize=9.5, pad=4, y=1.18)

        for band_i, band in enumerate(bands):
            start = band_i * len(channels)
            end = start + len(channels) - 1
            center = (start + end) / 2
            ax.text(center, 1.085, band_labels[band], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9.0, fontweight="bold", clip_on=False)
            ax.plot([start - 0.42, end + 0.42], [1.045, 1.045],
                    transform=ax.get_xaxis_transform(), color="black", linewidth=0.9, clip_on=False)

        for sep in (len(channels) - 0.5, 2 * len(channels) - 0.5):
            ax.axvline(sep, color="black", linewidth=1.0, alpha=0.65)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label(cbar_label, fontsize=7.5)
        cbar.ax.tick_params(labelsize=6.5, length=2)
        return im

    draw_heatmap(axes[0, 0], coh_video, "a   Coherence: Video model", "Coherence", "coherence", annotate)
    draw_heatmap(axes[0, 1], coh_eog, "b   Coherence: EOG regression", "Coherence", "coherence", annotate)
    draw_heatmap(axes[0, 2], delta_coh, r"c   $\Delta$Coherence: Video model $-$ EOG",
                 r"$\Delta$Coherence (Video $-$ EOG)", "delta_coherence", annotate)

    draw_heatmap(axes[1, 0], rbp_video, "d   RBP error: Video model", "RBP error", "rbp", annotate,
                 vmax_override=shared_rbp_vmax)
    draw_heatmap(axes[1, 1], rbp_eog, "e   RBP error: EOG regression", "RBP error", "rbp", annotate,
                 vmax_override=shared_rbp_vmax)
    draw_heatmap(axes[1, 2], delta_rbp, r"f   $\Delta$RBP error: Video model $-$ EOG",
                 r"$\Delta$RBP error (Video $-$ EOG)", "delta_rbp", annotate)
    axes[1, 1].set_xlabel("EEG target channel × frequency band", labelpad=8, fontsize=8.5)

    fig.suptitle(
        "Participant-level spectral characteristics of Video- and EOG-based predictions",
        fontsize=11, y=0.995
    )

    fig.text(
        0.5, 0.008,
        r"Brighter/yellower colors indicate better performance: for coherence, higher values are better; "
        r"for RBP error, lower values are better. In the $\Delta$ panels, red indicates the Video model "
        r"outperforms the EOG model (positive $\Delta$Coherence or negative $\Delta$RBP error).",
        ha="center", va="bottom", fontsize=7.4
    )

    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.90])

    if save_fig:
        fig.savefig(file_name, dpi=400, bbox_inches="tight", facecolor="white", pad_inches=0.05)

    return fig, axes