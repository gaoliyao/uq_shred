#!/usr/bin/env python3
"""
UQ-SHRED Visualization Utilities

Shared plotting functions for UQ-SHRED and UQ-DA-SHRED experiments.
Covers: quantile-band snapshots, space-time heatmaps, temporal traces,
calibration diagrams, error-vs-uncertainty scatter, and training curves.

Usage:
    from uq_viz import (
        plot_quantile_snapshots, plot_spacetime_overview,
        plot_temporal_traces, plot_calibration_single,
        plot_calibration_dual, plot_error_vs_uncertainty,
        plot_training_curve,
    )
"""

import numpy as np
import matplotlib.pyplot as plt


 
# CORE BUILDING BLOCKS
 

def _fill_quantile_bands(ax, xvals, quantiles_at_slice, color="C0"):
    """Fill 95/70/50% quantile bands on a single axis.

    Parameters
    ----------
    ax : matplotlib Axes
    xvals : 1-D array (spatial grid or time grid)
    quantiles_at_slice : dict {q: 1-D array} with keys including
        0.025, 0.15, 0.25, 0.5, 0.75, 0.85, 0.975
    color : matplotlib color string
    """
    ax.fill_between(xvals,
                    quantiles_at_slice[0.025], quantiles_at_slice[0.975],
                    alpha=0.15, color=color, label="95% CI")
    ax.fill_between(xvals,
                    quantiles_at_slice[0.15], quantiles_at_slice[0.85],
                    alpha=0.25, color=color, label="70% CI")
    ax.fill_between(xvals,
                    quantiles_at_slice[0.25], quantiles_at_slice[0.75],
                    alpha=0.4, color=color, label="50% CI")


def _compute_observed_coverage(samples_np, y_true_np, levels=None):
    """Compute observed coverage at each confidence level.

    Parameters
    ----------
    samples_np : (n_samples, N, D) or (n_samples, N)
    y_true_np  : (N, D) or (N,)
    levels     : list of confidence levels in (0, 1)

    Returns
    -------
    levels, observed : lists of floats
    """
    if levels is None:
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    observed = []
    for conf in levels:
        alpha = (1 - conf) / 2
        lower = np.percentile(samples_np, 100 * alpha, axis=0)
        upper = np.percentile(samples_np, 100 * (1 - alpha), axis=0)
        within = (y_true_np >= lower) & (y_true_np <= upper)
        observed.append(within.mean())
    return levels, observed


 
# 1. QUANTILE-BAND SPATIAL SNAPSHOTS  (single-field, e.g. 1D RDE)
 

def plot_quantile_snapshots(gt_field, pred_quantiles, times, sensor_locs,
                            x_grid, save_path, field_names=None,
                            plot_times=None, n_snaps=5):
    """Spatial snapshots with 95/70/50% quantile bands.

    Parameters
    ----------
    gt_field        : (T, n_ch, mx)
    pred_quantiles  : dict {q: (T, n_ch, mx)}
    times           : (T,) time array
    sensor_locs     : 1-D array of sensor indices on the spatial grid
    x_grid          : (mx,) spatial coordinates
    save_path       : pathlib.Path or str
    field_names     : list of channel names (default: ["Channel 0", ...])
    plot_times      : physical times to show (or None → evenly spaced)
    n_snaps         : number of snapshots if plot_times is None
    """
    T_total, n_ch, mx = gt_field.shape
    if field_names is None:
        field_names = [f"Channel {i}" for i in range(n_ch)]
    median = pred_quantiles[0.5]

    if plot_times is None:
        snap_idx = np.linspace(0, T_total - 1, n_snaps, dtype=int)
    else:
        snap_idx = [np.argmin(np.abs(times - t)) for t in plot_times]
    n_snaps = len(snap_idx)

    fig, axes = plt.subplots(n_ch, n_snaps,
                             figsize=(4 * n_snaps, 3.5 * n_ch), squeeze=False)
    fig.suptitle("UQ-SHRED Reconstruction: Spatial Snapshots with Quantile Bands",
                 fontsize=14, fontweight="bold", y=1.02)

    for col, ti in enumerate(snap_idx):
        for row in range(n_ch):
            ax = axes[row, col]
            q_slice = {q: v[ti, row] for q, v in pred_quantiles.items()}
            _fill_quantile_bands(ax, x_grid, q_slice, color="C0")
            ax.plot(x_grid, gt_field[ti, row], "k-", lw=1.2, label="Ground Truth")
            ax.plot(x_grid, median[ti, row], "C0--", lw=1.0, label="Median", alpha=0.8)

            if len(sensor_locs) <= 30:
                ax.scatter(x_grid[sensor_locs], gt_field[ti, row, sensor_locs],
                           c="red", s=15, zorder=5,
                           label="Sensors" if col == 0 else "")
            ax.set_title(f"t = {times[ti]:.2f}", fontsize=10)
            if col == 0:
                ax.set_ylabel(field_names[row])
            if row == n_ch - 1:
                ax.set_xlabel("x")
            if row == 0 and col == n_snaps - 1:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


 
# 2. SPACE-TIME HEATMAPS  (GT / median / error / CI width)
 

def plot_spacetime_overview(gt_field, pred_quantiles, times, x_grid, save_path,
                            field_names=None):
    """Space-time heatmaps: GT, median, |error|, 95% CI width.

    Parameters
    ----------
    gt_field       : (T, n_ch, mx)
    pred_quantiles : dict {q: (T, n_ch, mx)}
    times          : (T,)
    x_grid         : (mx,)
    save_path      : pathlib.Path or str
    field_names    : list of channel names
    """
    n_ch = gt_field.shape[1]
    if field_names is None:
        field_names = [f"Channel {i}" for i in range(n_ch)]
    median = pred_quantiles[0.5]

    fig, axes = plt.subplots(n_ch, 4, figsize=(20, 4 * n_ch))
    if n_ch == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("UQ-SHRED: Space-Time Overview", fontsize=14, fontweight="bold")

    for row in range(n_ch):
        gt_st = gt_field[:, row]
        med_st = median[:, row]
        err_st = np.abs(gt_st - med_st)
        width_st = (pred_quantiles[0.975][:, row] -
                    pred_quantiles[0.025][:, row])

        extent = [x_grid[0], x_grid[-1], times[0], times[-1]]
        kw = dict(aspect="auto", origin="lower", extent=extent)

        im0 = axes[row, 0].imshow(gt_st, **kw)
        axes[row, 0].set_title(f"{field_names[row]} – Ground Truth")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        im1 = axes[row, 1].imshow(med_st, **kw)
        axes[row, 1].set_title(f"{field_names[row]} – Median")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        im2 = axes[row, 2].imshow(err_st, **kw, cmap="Reds")
        axes[row, 2].set_title(f"{field_names[row]} – |Error|")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        im3 = axes[row, 3].imshow(width_st, **kw, cmap="Oranges")
        axes[row, 3].set_title(f"{field_names[row]} – 95% CI Width")
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046)

        for a in axes[row]:
            a.set_ylabel("time")
            a.set_xlabel("x")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


 
# 3. TEMPORAL TRACES AT SELECTED SPATIAL LOCATIONS
 

def plot_temporal_traces(gt_field, pred_quantiles, times, x_grid, save_path,
                         field_names=None, n_locs=4, spatial_idx=None):
    """Time series at selected spatial points with quantile bands.

    Parameters
    ----------
    gt_field       : (T, n_ch, mx)
    pred_quantiles : dict {q: (T, n_ch, mx)}
    times          : (T,)
    x_grid         : (mx,)
    save_path      : pathlib.Path or str
    field_names    : list of channel names
    n_locs         : number of spatial locations (used if spatial_idx is None)
    spatial_idx    : explicit array of spatial indices to plot
    """
    n_ch = gt_field.shape[1]
    mx = gt_field.shape[2]
    if field_names is None:
        field_names = [f"Channel {i}" for i in range(n_ch)]
    if spatial_idx is None:
        spatial_idx = np.linspace(0, mx - 1, n_locs + 2, dtype=int)[1:-1]
    median = pred_quantiles[0.5]

    fig, axes = plt.subplots(n_ch, len(spatial_idx),
                             figsize=(5 * len(spatial_idx), 3 * n_ch),
                             squeeze=False)
    fig.suptitle("UQ-SHRED: Temporal Reconstruction at Selected Spatial Points",
                 fontsize=14, fontweight="bold", y=1.02)

    for col, xi in enumerate(spatial_idx):
        for row in range(n_ch):
            ax = axes[row, col]
            q_slice = {q: v[:, row, xi] for q, v in pred_quantiles.items()}
            _fill_quantile_bands(ax, times, q_slice, color="C0")
            ax.plot(times, gt_field[:, row, xi], "k-", lw=1.2, label="Truth")
            ax.plot(times, median[:, row, xi], "C0--", lw=0.8, label="Median")

            ax.set_title(f"x = {x_grid[xi]:.2f}", fontsize=10)
            if col == 0:
                ax.set_ylabel(field_names[row])
            if row == n_ch - 1:
                ax.set_xlabel("time")
            if row == 0 and col == len(spatial_idx) - 1:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


 
# 4. DUAL SNAPSHOT PLOTS  (LF vs LF+HF, for DA-SHRED)
 

def plot_dual_snapshots(gt, q_lf, q_lfhf, times, N, save_path, n_snaps=8):
    """Side-by-side LF-only (green) vs LF+HF (blue) quantile-band snapshots.

    Parameters
    ----------
    gt      : (T, N) ground truth
    q_lf    : dict {q: (T, N)} LF-only quantiles
    q_lfhf  : dict {q: (T, N)} LF+HF quantiles
    times   : (T,) time array
    N       : spatial dimension
    save_path : pathlib.Path or str
    """
    T_total = gt.shape[0]
    snap_idx = np.linspace(0, T_total - 1, n_snaps, dtype=int)
    x = np.arange(N)

    fig, axes = plt.subplots(2, n_snaps, figsize=(3.5 * n_snaps, 7), squeeze=False)
    fig.suptitle("UQ-DA-SHRED: LF-only vs LF+HF with Quantile Bands",
                 fontsize=13, fontweight="bold", y=1.02)

    for col, ti in enumerate(snap_idx):
        # Top row: LF only
        ax = axes[0, col]
        q_slice = {q: v[ti] for q, v in q_lf.items()}
        _fill_quantile_bands(ax, x, q_slice, color="C2")
        ax.plot(x, gt[ti], "b-", lw=1.5, label="GT")
        ax.plot(x, q_lf[0.5][ti], "C2--", lw=1.0, label="LF Median")
        ax.set_title(f"t={times[ti]:.1f}", fontsize=9)
        if col == 0:
            ax.set_ylabel("LF only")
        if col == n_snaps - 1:
            ax.legend(fontsize=6, loc="upper right")

        # Bottom row: LF + HF
        ax2 = axes[1, col]
        q_slice2 = {q: v[ti] for q, v in q_lfhf.items()}
        _fill_quantile_bands(ax2, x, q_slice2, color="C0")
        ax2.plot(x, gt[ti], "b-", lw=1.5, label="GT")
        ax2.plot(x, q_lfhf[0.5][ti], "C0--", lw=1.0, label="LF+HF Median")
        if col == 0:
            ax2.set_ylabel("LF + HF")
        ax2.set_xlabel("x")
        if col == n_snaps - 1:
            ax2.legend(fontsize=6, loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


 
# 5. DUAL TEMPORAL TRACES  (LF vs LF+HF)
 

def plot_dual_timeseries(gt, q_lf, q_lfhf, times, N, save_path, n_locs=4):
    """Temporal traces: LF-only (top, green) vs LF+HF (bottom, blue).

    Parameters
    ----------
    gt      : (T, N)
    q_lf    : dict {q: (T, N)}
    q_lfhf  : dict {q: (T, N)}
    times   : (T,)
    N       : spatial dimension
    save_path : pathlib.Path or str
    """
    spatial_idx = np.linspace(0, N - 1, n_locs + 2, dtype=int)[1:-1]
    fig, axes = plt.subplots(2, len(spatial_idx),
                             figsize=(5 * len(spatial_idx), 6), squeeze=False)
    fig.suptitle("UQ-DA-SHRED: Temporal Traces with Quantile Bands",
                 fontsize=13, fontweight="bold", y=1.02)

    for col, xi in enumerate(spatial_idx):
        # LF only
        ax = axes[0, col]
        q_slice = {q: v[:, xi] for q, v in q_lf.items()}
        _fill_quantile_bands(ax, times, q_slice, color="C2")
        ax.plot(times, gt[:, xi], "b-", lw=1.2, label="GT")
        ax.plot(times, q_lf[0.5][:, xi], "C2--", lw=0.8, label="LF Median")
        ax.set_title(f"x = {xi}", fontsize=10)
        if col == 0:
            ax.set_ylabel("LF only")
        if col == len(spatial_idx) - 1:
            ax.legend(fontsize=7)

        # LF + HF
        ax2 = axes[1, col]
        q_slice2 = {q: v[:, xi] for q, v in q_lfhf.items()}
        _fill_quantile_bands(ax2, times, q_slice2, color="C0")
        ax2.plot(times, gt[:, xi], "b-", lw=1.2, label="GT")
        ax2.plot(times, q_lfhf[0.5][:, xi], "C0--", lw=0.8, label="LF+HF Median")
        if col == 0:
            ax2.set_ylabel("LF + HF")
        ax2.set_xlabel("time")
        if col == len(spatial_idx) - 1:
            ax2.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


 
# 6. CALIBRATION DIAGRAMS
 

def plot_calibration_single(samples_np, y_true_np, save_path, label="UQ-SHRED",
                            color="C0"):
    """Calibration diagram for a single model.

    Parameters
    ----------
    samples_np : (n_samples, N, D) ensemble predictions
    y_true_np  : (N, D) ground truth
    save_path  : pathlib.Path or str
    """
    levels, observed = _compute_observed_coverage(samples_np, y_true_np)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(levels, observed, f"{color}o-", label=label)
    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Observed Coverage")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Calibration Diagram")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_calibration_dual(samples_lf, samples_lfhf, y_true, save_path):
    """Side-by-side calibration diagrams for LF-only and LF+HF.

    Parameters
    ----------
    samples_lf   : (n_samples, N, D) LF-only ensemble
    samples_lfhf : (n_samples, N, D) LF+HF ensemble
    y_true       : (N, D) ground truth
    save_path    : pathlib.Path or str
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, samples, label, color in [
        (axes[0], samples_lf, "LF only", "C2"),
        (axes[1], samples_lfhf, "LF + HF", "C0"),
    ]:
        levels, obs = _compute_observed_coverage(samples, y_true)
        ax.plot([0, 1], [0, 1], "k--")
        ax.plot(levels, obs, f"{color}o-", label=label)
        ax.set_xlabel("Expected Coverage")
        ax.set_ylabel("Observed Coverage")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(f"Calibration — {label}")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


 
# 7. ERROR VS UNCERTAINTY SCATTER
 

def plot_error_vs_uncertainty(gt, med_lf, std_lf, med_lfhf, std_lfhf, save_path):
    """Side-by-side error vs predicted std scatter with binned trend.

    Parameters
    ----------
    gt       : (T, N)  ground truth
    med_lf   : (T, N)  LF-only median
    std_lf   : (T, N)  LF-only predicted std
    med_lfhf : (T, N)  LF+HF median
    std_lfhf : (T, N)  LF+HF predicted std
    save_path : pathlib.Path or str
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Error vs. Uncertainty Correlation",
                 fontsize=13, fontweight="bold")

    for ax, med, std, label, color in [
        (axes[0], med_lf, std_lf, "LF only", "C2"),
        (axes[1], med_lfhf, std_lfhf, "LF + HF", "C0"),
    ]:
        err = np.abs(gt - med).ravel()
        unc = std.ravel()
        n_plot = min(5000, len(err))
        idx = np.random.RandomState(42).choice(len(err), n_plot, replace=False)

        ax.scatter(unc[idx], err[idx], alpha=0.15, s=8, c=color, edgecolors="none")

        # Binned mean trend line
        bin_edges = np.percentile(unc[idx], np.linspace(0, 100, 21))
        bin_centers, bin_means = [], []
        for i in range(20):
            mask = (unc[idx] >= bin_edges[i]) & (unc[idx] < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append(unc[idx][mask].mean())
                bin_means.append(err[idx][mask].mean())
        ax.plot(bin_centers, bin_means, "C3-o", lw=2, ms=4, label="Binned mean")

        r = np.corrcoef(unc[idx], err[idx])[0, 1]
        ax.set_xlabel("Predicted Std")
        ax.set_ylabel("|Error|")
        ax.set_title(f"{label}  (r = {r:.3f})")
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


 
# 8. TRAINING CURVE
 

def plot_training_curve(val_errors, save_path, title="Training Curve",
                        xlabel="Validation check", ylabel="Relative validation error"):
    """Simple log-scale training/validation curve.

    Parameters
    ----------
    val_errors : list of floats
    save_path  : pathlib.Path or str
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(val_errors, "C0-")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale("log")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


 
# 9. SHARED UQ METRICS
 

def calibration_scores(samples_np, y_true_np,
                       levels=(0.5, 0.7, 0.9, 0.95, 0.99)):
    """Observed coverage at each confidence level.

    Parameters
    ----------
    samples_np : (n_samples, N, D) or (n_samples, N) — ensemble predictions
    y_true_np  : (N, D) or (N,)                      — ground truth

    Returns
    -------
    dict  {confidence_level: observed_coverage}
    """
    results = {}
    for conf in levels:
        alpha = (1 - conf) / 2
        lower = np.percentile(samples_np, 100 * alpha, axis=0)
        upper = np.percentile(samples_np, 100 * (1 - alpha), axis=0)
        within = (y_true_np >= lower) & (y_true_np <= upper)
        results[conf] = float(within.mean())
    return results


def compute_crps(samples_np, y_true_np):
    """Approximate CRPS via MAE minus half pairwise spread.

    Parameters
    ----------
    samples_np : (n_samples, N, D) or (n_samples, N)
    y_true_np  : (N, D) or (N,)
    """
    mae = np.abs(samples_np - y_true_np[None]).mean()
    n = min(50, samples_np.shape[0])
    rng = np.random.RandomState(0)
    i1 = rng.choice(samples_np.shape[0], n, replace=False)
    i2 = rng.choice(samples_np.shape[0], n, replace=False)
    return mae - np.abs(samples_np[i1] - samples_np[i2]).mean() / 2


def compute_sharpness(samples_np, conf=0.95):
    """Mean CI width at a given confidence level.

    Parameters
    ----------
    samples_np : (n_samples, N, D) or (n_samples, N)
    conf       : confidence level in (0, 1)
    """
    alpha = (1 - conf) / 2
    lo = np.percentile(samples_np, 100 * alpha, axis=0)
    hi = np.percentile(samples_np, 100 * (1 - alpha), axis=0)
    return float((hi - lo).mean())
