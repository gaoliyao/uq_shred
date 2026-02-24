"""
UQ-SHRED Experiment: Neuropixel LFP Dataset (Allen Institute)

Trains SHRED (baseline) and UQ-SHRED on local field potential data recorded
from 3 mouse visual cortex regions (VISam, VISpm, VISp).

2 sensors observe, model reconstructs all 15 channels.

Produces: training_curve, calibration, error_vs_unc, timeseries_per_region.

Usage:
    python UQ_neuro.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "RDE_experiments" / "High_fidelity_RDE"))

import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler

from processdata import load_data, TimeSeriesDataset
from models import SHRED, UQ_SHRED, fit, fit_uq
from uq_viz import (
    plot_calibration_single, plot_training_curve,
    calibration_scores, compute_crps, compute_sharpness,
)

np.random.seed(42)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

dataset_name = "NEURO"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path(f"results/{dataset_name}_{timestamp}")
results_dir.mkdir(parents=True, exist_ok=True)
print(f"Results: {results_dir}")

config = {"dataset": dataset_name, "device": device, "seed": 42, "timestamp": timestamp}

# Region labels for the 15 channels (5 per region)
REGION_LABELS = (
    [f"VISam-{i}" for i in range(5)] +
    [f"VISpm-{i}" for i in range(5)] +
    [f"VISp-{i}"  for i in range(5)]
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [1] DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[1] Loading data …")

num_sensors = 2
lags = 52   # ~100 ms at 500 Hz

load_X = load_data(dataset_name)
n, m = load_X.shape
print(f"  {dataset_name}: {load_X.shape}  ({n} timesteps, {m} channels)")

# Pick 2 random sensor locations
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
print(f"  Sensor locations: {sensor_locations} ({[REGION_LABELS[i] for i in sensor_locations]})")

config.update({
    "num_sensors": num_sensors,
    "lags": lags,
    "data_shape": list(load_X.shape),
    "sensor_locations": sensor_locations.tolist(),
    "sensor_labels": [REGION_LABELS[i] for i in sensor_locations],
})

# Time-sequential split: train first 60%, then valid 20%, test 20%
n_windows = n - lags
n_train = int(0.6 * n_windows)
n_valid = int(0.2 * n_windows)
train_indices = np.arange(0, n_train)
valid_indices = np.arange(n_train, n_train + n_valid)
test_indices  = np.arange(n_train + n_valid, n_windows)
print(f"  Train: {len(train_indices)}, Valid: {len(valid_indices)}, Test: {len(test_indices)}")
config["split"] = {
    "train": len(train_indices),
    "valid": len(valid_indices),
    "test":  len(test_indices),
}

sc = MinMaxScaler()
sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)

all_data_in = np.zeros((n_windows, lags, num_sensors))
for i in range(n_windows):
    all_data_in[i] = transformed_X[i:i + lags, sensor_locations]

train_data_in  = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in  = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in   = torch.tensor(all_data_in[test_indices],  dtype=torch.float32).to(device)
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out  = torch.tensor(transformed_X[test_indices  + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset  = TimeSeriesDataset(test_data_in,  test_data_out)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [2] TRAIN SHRED (BASELINE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[2] Training SHRED …")
shred = SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=128, l2=128, dropout=0.01).to(device)
fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=20)

shred.eval()
with torch.no_grad():
    shred_recon = shred(test_dataset.X)
    shred_error = (torch.linalg.norm(shred_recon - test_dataset.Y) /
                   torch.linalg.norm(test_dataset.Y)).item()
print(f"  SHRED relative error: {shred_error:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [3] TRAIN UQ-SHRED
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[3] Training UQ-SHRED …")
uq_shred = UQ_SHRED(num_sensors, m, hidden_size=64, hidden_layers=2,
                    l1=128, l2=128, dropout=0.1, noise_dim=50).to(device)
uq_errors = fit_uq(uq_shred, train_dataset, valid_dataset,
                   batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=20)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [4] INFERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[4] Inference (n_samples=100) …")
uq_shred.eval()
samples = uq_shred.sample(test_dataset.X, n_samples=100)
samples_np = samples.cpu().numpy()

mean_recon   = samples.mean(dim=0)
median_recon = torch.tensor(np.median(samples_np, axis=0), dtype=torch.float32).to(device)
std_recon    = samples.std(dim=0)

uq_mean_error   = (torch.linalg.norm(mean_recon   - test_dataset.Y) /
                   torch.linalg.norm(test_dataset.Y)).item()
uq_median_error = (torch.linalg.norm(median_recon - test_dataset.Y) /
                   torch.linalg.norm(test_dataset.Y)).item()

shred_recon_np  = sc.inverse_transform(shred_recon.cpu().numpy())
test_truth_np   = sc.inverse_transform(test_dataset.Y.cpu().numpy())
samples_orig    = np.array([sc.inverse_transform(s) for s in samples_np])
median_orig     = np.median(samples_orig, axis=0)
std_orig        = samples_orig.std(axis=0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [5] METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'='*70}")
print("  METRICS")
print(f"{'='*70}")

calib    = calibration_scores(samples_orig, test_truth_np)
crps_val = compute_crps(samples_orig, test_truth_np)
sharp_95 = compute_sharpness(samples_orig, 0.95)
sharp_70 = compute_sharpness(samples_orig, 0.70)
sharp_50 = compute_sharpness(samples_orig, 0.50)

err_flat = np.abs(test_truth_np - median_orig).ravel()
std_flat = std_orig.ravel()
corr = np.corrcoef(std_flat, err_flat)[0, 1]

rmse_shred = np.sqrt(np.mean((test_truth_np - shred_recon_np) ** 2))
rmse_uq    = np.sqrt(np.mean((test_truth_np - median_orig) ** 2))

print(f"\n  {'':25s} {'RMSE':>12s}  {'Rel Error':>10s}")
print(f"  {'SHRED':25s} {rmse_shred:12.8f}  {shred_error:10.4f}")
print(f"  {'UQ-SHRED (median)':25s} {rmse_uq:12.8f}  {uq_median_error:10.4f}")
print(f"  {'UQ-SHRED (mean)':25s} {'---':>12s}  {uq_mean_error:10.4f}")
print(f"\n  Calibration: " + ", ".join(f"{k*100:.0f}%→{v*100:.1f}%" for k, v in calib.items()))
print(f"  CRPS:      {crps_val:.6f}")
print(f"  Sharpness: 95%={sharp_95:.6f}, 70%={sharp_70:.6f}, 50%={sharp_50:.6f}")
print(f"  Corr(σ, |e|): {corr:.4f}")

metrics_text = (
    f"{'='*70}\n"
    f"UQ-SHRED RESULTS: {dataset_name}\n"
    f"Timestamp: {timestamp}\n"
    f"{'='*70}\n\n"
    f"DATA\n"
    f"  Sensors (n={num_sensors}): {[REGION_LABELS[i] for i in sensor_locations]}\n"
    f"  Channels reconstructed: {m}\n\n"
    f"RECONSTRUCTION\n"
    f"  RMSE (SHRED):          {rmse_shred:.8f}\n"
    f"  RMSE (UQ-SHRED med):   {rmse_uq:.8f}\n"
    f"  Rel error (SHRED):     {shred_error:.4f}\n"
    f"  Rel error (UQ mean):   {uq_mean_error:.4f}\n"
    f"  Rel error (UQ median): {uq_median_error:.4f}\n\n"
    f"CALIBRATION\n"
    + "".join(f"  {k*100:.0f}% CI → {v*100:.1f}%\n" for k, v in calib.items()) +
    f"\nSHARPNESS\n"
    f"  95% CI width: {sharp_95:.6f}\n"
    f"  70% CI width: {sharp_70:.6f}\n"
    f"  50% CI width: {sharp_50:.6f}\n\n"
    f"CRPS: {crps_val:.6f}\n"
    f"Corr(σ, |error|): {corr:.4f}\n"
    f"{'='*70}\n"
)
with open(results_dir / "metrics.txt", "w") as f:
    f.write(metrics_text)
with open(results_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [6] PLOTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'='*70}")
print("  GENERATING PLOTS")
print(f"{'='*70}")

# 1. Training curve
plot_training_curve(uq_errors.tolist(), results_dir / "training_curve.png",
                    title="UQ-SHRED Training Curve (Neuro)")

# 2. Calibration
plot_calibration_single(samples_orig, test_truth_np,
                        results_dir / "uq_shred_calibration.png", label="UQ-SHRED")

# 3. Error vs uncertainty
fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle("Error vs. Uncertainty", fontsize=13, fontweight="bold")
n_plot = min(5000, len(err_flat))
idx_plot = np.random.RandomState(42).choice(len(err_flat), n_plot, replace=False)
ax.scatter(std_flat[idx_plot], err_flat[idx_plot], alpha=0.2, s=8, c="C0", edgecolors="none")
bin_edges = np.percentile(std_flat[idx_plot], np.linspace(0, 100, 21))
bin_centers, bin_means = [], []
for i in range(20):
    mask = (std_flat[idx_plot] >= bin_edges[i]) & (std_flat[idx_plot] < bin_edges[i + 1])
    if mask.sum() > 0:
        bin_centers.append(std_flat[idx_plot][mask].mean())
        bin_means.append(err_flat[idx_plot][mask].mean())
ax.plot(bin_centers, bin_means, "C3-o", lw=2, ms=4, label="Binned mean")
ax.set_xlabel("Predicted Std")
ax.set_ylabel("|Error|")
ax.set_title(f"UQ-SHRED  (r = {corr:.3f})")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(results_dir / "uq_shred_error_vs_unc.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {results_dir}/uq_shred_error_vs_unc.png")
plt.close(fig)

# 4. Per-region time-series with CI (3 rows × 5 cols: one per channel)
q_levels = [0.025, 0.15, 0.25, 0.5, 0.75, 0.85, 0.975]
quantiles_ts = {q: np.percentile(samples_orig, 100 * q, axis=0) for q in q_levels}
T_show = len(test_truth_np)
times_ts = np.arange(T_show)

region_names = ["VISam", "VISpm", "VISp"]
fig, axes = plt.subplots(3, 5, figsize=(20, 9), squeeze=False)
fig.suptitle("UQ-SHRED: Reconstruction per Channel (2 sensors → 15 channels)",
             fontsize=14, fontweight="bold")

for region_idx, region_name in enumerate(region_names):
    for ch in range(5):
        col_idx = region_idx * 5 + ch  # global channel index (0–14)
        ax = axes[region_idx, ch]
        ax.fill_between(times_ts,
                        quantiles_ts[0.025][:, col_idx],
                        quantiles_ts[0.975][:, col_idx],
                        alpha=0.15, color="C0", label="95% CI")
        ax.fill_between(times_ts,
                        quantiles_ts[0.15][:, col_idx],
                        quantiles_ts[0.85][:, col_idx],
                        alpha=0.25, color="C0", label="70% CI")
        ax.fill_between(times_ts,
                        quantiles_ts[0.25][:, col_idx],
                        quantiles_ts[0.75][:, col_idx],
                        alpha=0.4, color="C0", label="50% CI")
        ax.plot(times_ts, test_truth_np[:, col_idx], "k-", lw=1.2, label="Truth")
        ax.plot(times_ts, shred_recon_np[:, col_idx], "r-", lw=0.8, alpha=0.7, label="SHRED")
        ax.plot(times_ts, quantiles_ts[0.5][:, col_idx], "C0--", lw=0.8, label="UQ median")
        ax.set_title(f"{region_name}-ch{ch}", fontsize=8)
        if region_idx == 2:
            ax.set_xlabel("time step", fontsize=7)
        if ch == 0:
            ax.set_ylabel(region_name + "\nV", fontsize=7)
        ax.tick_params(labelsize=6)
        if region_idx == 0 and ch == 4:
            ax.legend(fontsize=5, loc="upper right")

plt.tight_layout()
fig.savefig(results_dir / "uq_shred_timeseries_channels.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {results_dir}/uq_shred_timeseries_channels.png")
plt.close(fig)

# 5. Per-region summary: median ± 95% CI for each region averaged across channels
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle("UQ-SHRED: Region-Averaged Reconstruction", fontsize=14, fontweight="bold")

for region_idx, region_name in enumerate(region_names):
    ch_slice = slice(region_idx * 5, region_idx * 5 + 5)
    truth_region  = test_truth_np[:, ch_slice].mean(axis=1)
    shred_region  = shred_recon_np[:, ch_slice].mean(axis=1)
    median_region = np.median(samples_orig, axis=0)[:, ch_slice].mean(axis=1)
    lower_region  = np.percentile(samples_orig, 2.5, axis=0)[:, ch_slice].mean(axis=1)
    upper_region  = np.percentile(samples_orig, 97.5, axis=0)[:, ch_slice].mean(axis=1)

    ax = axes[region_idx]
    ax.fill_between(times_ts, lower_region, upper_region, alpha=0.3, color="C0", label="95% CI")
    ax.plot(times_ts, truth_region, "k-", lw=1.5, label="Truth")
    ax.plot(times_ts, shred_region, "r-", lw=1.0, alpha=0.8, label="SHRED")
    ax.plot(times_ts, median_region, "C0--", lw=1.0, label="UQ median")
    ax.set_ylabel(f"{region_name}\nV (avg)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")

axes[-1].set_xlabel("time step")
plt.tight_layout()
fig.savefig(results_dir / "uq_shred_region_summary.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {results_dir}/uq_shred_region_summary.png")
plt.close(fig)

print(f"\n{'='*70}")
print(f"  UQ-SHRED ({dataset_name}) Complete!")
print(f"  Results: {results_dir}")
print(f"{'='*70}")
