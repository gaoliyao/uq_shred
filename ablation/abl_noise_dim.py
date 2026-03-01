"""
Ablation 3: Noise Dimension

Fixed : num_sensors=3, lags=52, n_samples=100, num_epochs=200
Sweep : noise_dim in [10, 50, 100, 200, 500]

Usage:
    python ablation/abl_noise_dim.py
"""

from utils import load_sst_datasets, compute_uq_metrics, plot_ablation
from models import SHRED, UQ_SHRED, fit, fit_uq

import numpy as np
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Fixed
NUM_SENSORS = 3
LAGS        = 52
N_SAMPLES   = 100
NUM_EPOCHS  = 200
PATIENCE    = 5

# Sweep
NOISE_DIM_VALUES = [10, 50, 100, 200, 500]

# Load data once (same sensors/lags for all runs)
train_ds, valid_ds, test_ds, sc, m = load_sst_datasets(NUM_SENSORS, LAGS, device)
truth_np = sc.inverse_transform(test_ds.Y.cpu().numpy())

# SHRED baseline trained once (independent of noise_dim)
print("\nTraining SHRED baseline ...")
shred = SHRED(NUM_SENSORS, m, hidden_size=64, hidden_layers=2,
              l1=350, l2=400, dropout=0.1).to(device)
fit(shred, train_ds, valid_ds, batch_size=64, num_epochs=NUM_EPOCHS,
    lr=1e-3, verbose=False, patience=PATIENCE)
shred.eval()
with torch.no_grad():
    shred_pred_np = sc.inverse_transform(shred(test_ds.X).cpu().numpy())
rmse_shred = float(np.sqrt(np.mean((truth_np - shred_pred_np) ** 2)))
print(f"  RMSE_SHRED={rmse_shred:.6f}")

results = []
for noise_dim in NOISE_DIM_VALUES:
    print(f"\n=== noise_dim={noise_dim} ===")

    uq_model = UQ_SHRED(NUM_SENSORS, m, hidden_size=64, hidden_layers=2,
                        l1=350, l2=400, dropout=0.1, noise_dim=noise_dim).to(device)
    fit_uq(uq_model, train_ds, valid_ds, batch_size=64, num_epochs=NUM_EPOCHS,
           lr=1e-3, verbose=False, patience=PATIENCE)
    samples_np = np.array([sc.inverse_transform(s)
                           for s in uq_model.sample(test_ds.X, N_SAMPLES).cpu().numpy()])

    metrics = compute_uq_metrics(samples_np, truth_np)
    metrics["rmse_shred"] = rmse_shred
    results.append(metrics)
    print(f"  RMSE_UQ={metrics['rmse_uq']:.6f}  Cov95={metrics['coverage_95']:.3f}  "
          f"CRPS={metrics['crps']:.6f}")

save_dir = Path(__file__).parent / "results" / "abl3_noise_dim"
plot_ablation(NOISE_DIM_VALUES, results, "noise_dim", "Noise Dimension", save_dir, log_x=True)
print("\nDone.")
