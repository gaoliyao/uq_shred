"""
Shared utilities for SST ablation studies.

All ablation scripts import this module first — it sets sys.path and os.chdir
so that project imports (processdata, models, uq_viz) and Data/ paths work.

Run scripts from the project root:
    python ablation/abl_lags.py
"""

import sys
import os
from pathlib import Path

# --- path setup (must happen before any project imports) ---
ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RDE_experiments" / "High_fidelity_RDE"))

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler

from processdata import load_data, TimeSeriesDataset
from models import SHRED, UQ_SHRED, fit, fit_uq
from uq_viz import calibration_scores, compute_crps, compute_sharpness


def load_sst_datasets(num_sensors, lags, device, seed=42):
    """Load and preprocess SST data for the given num_sensors and lags.

    Returns:
        train_ds, valid_ds, test_ds : TimeSeriesDataset objects on device
        sc                          : fitted MinMaxScaler
        m                           : number of spatial dimensions
    """
    rng = np.random.RandomState(seed)
    load_X = load_data("SST")
    n, m = load_X.shape

    sensor_locations = rng.choice(m, size=num_sensors, replace=False)

    train_indices = rng.choice(n - lags, size=1000, replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    rest = np.arange(n - lags)[mask != 0]
    valid_indices = rest[::2]
    test_indices  = rest[1::2]

    sc = MinMaxScaler()
    sc.fit(load_X[train_indices])
    X = sc.transform(load_X)

    all_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_in)):
        all_in[i] = X[i:i + lags, sensor_locations]

    def make_ds(idx):
        xi = torch.tensor(all_in[idx], dtype=torch.float32).to(device)
        yi = torch.tensor(X[idx + lags - 1], dtype=torch.float32).to(device)
        return TimeSeriesDataset(xi, yi)

    return make_ds(train_indices), make_ds(valid_indices), make_ds(test_indices), sc, m


def compute_uq_metrics(samples_np, truth_np):
    """Compute reconstruction and UQ metrics from an ensemble.

    Args:
        samples_np : (n_samples, N, D) numpy array — ensemble predictions
        truth_np   : (N, D) numpy array — ground truth

    Returns:
        dict with keys: rmse_uq, rel_error_uq, coverage_95, crps, sharpness_95, corr
    """
    median_np = np.median(samples_np, axis=0)
    std_np    = samples_np.std(axis=0)

    rmse_uq    = np.sqrt(np.mean((truth_np - median_np) ** 2))
    rel_uq     = np.linalg.norm(median_np - truth_np) / np.linalg.norm(truth_np)
    calib      = calibration_scores(samples_np, truth_np)
    crps_val   = compute_crps(samples_np, truth_np)
    sharp_95   = compute_sharpness(samples_np, 0.95)
    corr       = np.corrcoef(std_np.ravel(), np.abs(truth_np - median_np).ravel())[0, 1]

    return {
        "rmse_uq":      float(rmse_uq),
        "rel_error_uq": float(rel_uq),
        "coverage_95":  float(calib[0.95]),
        "crps":         float(crps_val),
        "sharpness_95": float(sharp_95),
        "corr":         float(corr),
    }


def plot_ablation(param_values, results, param_name, param_label, save_dir, log_x=False):
    """Save a 2×3 summary figure and a CSV for the ablation sweep.

    Args:
        param_values : list of swept parameter values (x-axis)
        results      : list of dicts, one per param value. Each dict must contain
                       rmse_shred, rmse_uq, rel_error_uq, coverage_95, crps,
                       sharpness_95, corr.
        param_name   : short name for filenames (e.g. 'lags')
        param_label  : human-readable axis label (e.g. 'Temporal Lag')
        save_dir     : Path where outputs are saved
        log_x        : whether to use log scale on x-axis
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    xs = param_values
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Ablation: {param_label}  (SST)", fontsize=14, fontweight="bold")

    def _plot(ax, key, ylabel, title, ref_key=None, ref_label=None):
        ys = [r[key] for r in results]
        ax.plot(xs, ys, "C0o-", lw=1.5, ms=6, label="UQ-SHRED")
        if ref_key is not None:
            ys_ref = [r[ref_key] for r in results]
            ax.plot(xs, ys_ref, "C3s--", lw=1.5, ms=6, label=ref_label)
            ax.legend(fontsize=8)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(param_label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], "rmse_uq",      "RMSE",          "Reconstruction RMSE",
          ref_key="rmse_shred", ref_label="SHRED")
    _plot(axes[0, 1], "rel_error_uq", "Relative Error", "Relative Error (UQ-SHRED median)")
    _plot(axes[0, 2], "coverage_95",  "Coverage",       "95% CI Coverage")
    axes[0, 2].axhline(0.95, color="k", linestyle="--", alpha=0.5, label="Ideal (0.95)")
    axes[0, 2].legend(fontsize=8)
    _plot(axes[1, 0], "crps",         "CRPS",           "CRPS  (lower = better)")
    _plot(axes[1, 1], "sharpness_95", "Mean CI Width",  "95% CI Sharpness  (lower = sharper)")
    _plot(axes[1, 2], "corr",         "Pearson r",      "Corr(σ, |error|)  (higher = better)")

    plt.tight_layout()
    fig_path = save_dir / f"ablation_{param_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {fig_path}")

    # CSV
    csv_path = save_dir / f"ablation_{param_name}.csv"
    fieldnames = [param_name] + list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for v, r in zip(param_values, results):
            w.writerow({param_name: v, **r})
    print(f"  CSV:    {csv_path}")
