"""
UQ-SHRED Experiment: Solar Data (SUN)

Trains SHRED (baseline) and UQ-SHRED on solar imagery data.
Produces: training_curve, calibration, error_vs_unc, timeseries, snapshots.

Usage:
    python UQ_sun.py
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

dataset_name = "SUN"
IMG_H, IMG_W = 271, 271
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path(f"results/{dataset_name}_{timestamp}")
results_dir.mkdir(parents=True, exist_ok=True)
print(f"Results: {results_dir}")

config = {"dataset": dataset_name, "device": device, "seed": 42, "timestamp": timestamp}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [1] DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[1] Loading data …")

num_sensors = 3
lags = 75

load_X = load_data(dataset_name)
n, m = load_X.shape
print(f"  {dataset_name}: {load_X.shape}  (spatial: {IMG_H}x{IMG_W}={m})")

sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
config.update({"num_sensors": num_sensors, "lags": lags, "data_shape": list(load_X.shape),
               "sensor_locations": sensor_locations.tolist()})

# Random split: 70% train, remainder split evenly to valid/test
train_indices = np.random.choice(n - lags, size=int(0.7 * (n - lags)), replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask != 0)[0]]
valid_indices = valid_test_indices[::2]
test_indices  = valid_test_indices[1::2]
print(f"  Train: {len(train_indices)}, Valid: {len(valid_indices)}, Test: {len(test_indices)}")
config["split"] = {"train": len(train_indices), "valid": len(valid_indices), "test": len(test_indices)}

sc = MinMaxScaler()
sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)

all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
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
shred = SHRED(num_sensors, m, hidden_size=128, hidden_layers=2, l1=350, l2=400, dropout=0.01).to(device)
fit(shred, train_dataset, valid_dataset, batch_size=20, num_epochs=1000, lr=1e-3, verbose=True, patience=50)

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
uq_shred = UQ_SHRED(num_sensors, m, hidden_size=128, hidden_layers=2,
                    l1=350, l2=400, dropout=0.1, noise_dim=100).to(device)
uq_errors = fit_uq(uq_shred, train_dataset, valid_dataset,
                   batch_size=20, num_epochs=1000, lr=1e-3, verbose=True, patience=50)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [4] INFERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[4] Inference (n_samples=50) …")
uq_shred.eval()
samples = uq_shred.sample(test_dataset.X, n_samples=50)
samples_np = samples.cpu().numpy()

mean_recon   = samples.mean(dim=0)
median_recon = torch.tensor(np.median(samples_np, axis=0), dtype=torch.float32).to(device)
std_recon    = samples.std(dim=0)

uq_mean_error   = (torch.linalg.norm(mean_recon   - test_dataset.Y) /
                   torch.linalg.norm(test_dataset.Y)).item()
uq_median_error = (torch.linalg.norm(median_recon - test_dataset.Y) /
                   torch.linalg.norm(test_dataset.Y)).item()

# Inverse-transform; sort by original time index for temporal coherence
shred_recon_np = sc.inverse_transform(shred_recon.cpu().numpy())
test_truth_np  = sc.inverse_transform(test_dataset.Y.cpu().numpy())
samples_orig   = np.array([sc.inverse_transform(s) for s in samples_np])

sort_order        = np.argsort(test_indices)
test_truth_sorted = test_truth_np[sort_order]
samples_sorted    = samples_orig[:, sort_order, :]
median_orig       = np.median(samples_orig, axis=0)
std_orig          = samples_orig.std(axis=0)
median_sorted     = np.median(samples_sorted, axis=0)
std_sorted        = samples_sorted.std(axis=0)


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

print(f"\n  {'':25s} {'RMSE':>10s}  {'Rel Error':>10s}")
print(f"  {'SHRED':25s} {rmse_shred:10.6f}  {shred_error:10.4f}")
print(f"  {'UQ-SHRED (median)':25s} {rmse_uq:10.6f}  {uq_median_error:10.4f}")
print(f"  {'UQ-SHRED (mean)':25s} {'---':>10s}  {uq_mean_error:10.4f}")
print(f"\n  Calibration: " + ", ".join(f"{k*100:.0f}%→{v*100:.1f}%" for k, v in calib.items()))
print(f"  CRPS:      {crps_val:.6f}")
print(f"  Sharpness: 95%={sharp_95:.6f}, 70%={sharp_70:.6f}, 50%={sharp_50:.6f}")
print(f"  Corr(σ, |e|): {corr:.4f}")

metrics_text = (
    f"{'='*70}\n"
    f"UQ-SHRED RESULTS: {dataset_name}\n"
    f"Timestamp: {timestamp}\n"
    f"{'='*70}\n\n"
    f"RECONSTRUCTION\n"
    f"  RMSE (SHRED):          {rmse_shred:.6f}\n"
    f"  RMSE (UQ-SHRED med):   {rmse_uq:.6f}\n"
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
                    title="UQ-SHRED Training Curve")

# 2. Calibration
plot_calibration_single(samples_orig, test_truth_np,
                        results_dir / "uq_shred_calibration.png", label="UQ-SHRED")

# 3. Error vs uncertainty
err = np.abs(test_truth_np - median_orig).ravel()
unc = std_orig.ravel()
n_plot = min(5000, len(err))
idx_plot = np.random.RandomState(42).choice(len(err), n_plot, replace=False)

fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle("Error vs. Uncertainty", fontsize=13, fontweight="bold")
ax.scatter(unc[idx_plot], err[idx_plot], alpha=0.15, s=8, c="C0", edgecolors="none")
bin_edges = np.percentile(unc[idx_plot], np.linspace(0, 100, 21))
bin_centers, bin_means = [], []
for i in range(20):
    mask = (unc[idx_plot] >= bin_edges[i]) & (unc[idx_plot] < bin_edges[i + 1])
    if mask.sum() > 0:
        bin_centers.append(unc[idx_plot][mask].mean())
        bin_means.append(err[idx_plot][mask].mean())
ax.plot(bin_centers, bin_means, "C3-o", lw=2, ms=4, label="Binned mean")
ax.set_xlabel("Predicted Std")
ax.set_ylabel("|Error|")
ax.set_title(f"UQ-SHRED  (r = {corr:.3f})")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(results_dir / "uq_shred_error_vs_unc.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {results_dir}/uq_shred_error_vs_unc.png")
plt.close(fig)

# 4. Timeseries at 4 spatial locations (time-sorted test set)
q_levels = [0.025, 0.15, 0.25, 0.5, 0.75, 0.85, 0.975]
quantiles = {q: np.percentile(samples_sorted, 100 * q, axis=0) for q in q_levels}
spatial_idx = np.linspace(0, m - 1, 6, dtype=int)[1:-1]  # 4 interior points
T_show = min(100, len(test_truth_sorted))
times = np.arange(T_show)

fig, axes = plt.subplots(1, len(spatial_idx), figsize=(5 * len(spatial_idx), 4), squeeze=False)
fig.suptitle("UQ-SHRED: Temporal Reconstruction at Selected Spatial Locations",
             fontsize=14, fontweight="bold", y=1.02)
for col, xi in enumerate(spatial_idx):
    ax = axes[0, col]
    ax.fill_between(times, quantiles[0.025][:T_show, xi], quantiles[0.975][:T_show, xi],
                    alpha=0.15, color="C0", label="95% CI")
    ax.fill_between(times, quantiles[0.15][:T_show, xi], quantiles[0.85][:T_show, xi],
                    alpha=0.25, color="C0", label="70% CI")
    ax.fill_between(times, quantiles[0.25][:T_show, xi], quantiles[0.75][:T_show, xi],
                    alpha=0.4, color="C0", label="50% CI")
    ax.plot(times, test_truth_sorted[:T_show, xi], "k-", lw=1.2, label="Truth")
    ax.plot(times, quantiles[0.5][:T_show, xi], "C0--", lw=0.8, label="Median")
    ax.set_title(f"Spatial idx {xi}", fontsize=10)
    ax.set_xlabel("time step")
    if col == 0:
        ax.set_ylabel("intensity")
    if col == len(spatial_idx) - 1:
        ax.legend(fontsize=7, loc="upper right")
plt.tight_layout()
fig.savefig(results_dir / "uq_shred_timeseries.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {results_dir}/uq_shred_timeseries.png")
plt.close(fig)

# 5. Spatial snapshots (SUN-specific: 2D images)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("UQ-SHRED: Spatial Snapshot (t=0)", fontsize=14, fontweight="bold")
truth_img  = test_truth_sorted[0].reshape(IMG_H, IMG_W)
median_img = median_sorted[0].reshape(IMG_H, IMG_W)
std_img    = std_sorted[0].reshape(IMG_H, IMG_W)
vmin, vmax = truth_img.min(), truth_img.max()
im0 = axes[0].imshow(truth_img, cmap="plasma", vmin=vmin, vmax=vmax)
axes[0].set_title("Ground Truth")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(median_img, cmap="plasma", vmin=vmin, vmax=vmax)
axes[1].set_title("UQ-SHRED Median")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046)
im2 = axes[2].imshow(std_img, cmap="viridis")
axes[2].set_title("Uncertainty σ")
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046)
plt.tight_layout()
fig.savefig(results_dir / "uq_shred_snapshots.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {results_dir}/uq_shred_snapshots.png")
plt.close(fig)

print(f"\n{'='*70}")
print(f"  UQ-SHRED ({dataset_name}) Complete!")
print(f"  Results: {results_dir}")
print(f"{'='*70}")
