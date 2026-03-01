"""
Ablation 1: Temporal Lag Parameter

Fixed : num_sensors=3, noise_dim=100, n_samples=100, num_epochs=200
Sweep : lags in [1, 3, 10, 50, 100]

Usage:
    python ablation/abl_lags.py
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
NOISE_DIM   = 100
N_SAMPLES   = 100
NUM_EPOCHS  = 200
PATIENCE    = 5

# Sweep
LAG_VALUES = [1, 3, 10, 50, 100]

results = []
for lags in LAG_VALUES:
    print(f"\n=== lags={lags} ===")
    train_ds, valid_ds, test_ds, sc, m = load_sst_datasets(NUM_SENSORS, lags, device)

    # SHRED baseline
    shred = SHRED(NUM_SENSORS, m, hidden_size=64, hidden_layers=2,
                  l1=350, l2=400, dropout=0.1).to(device)
    fit(shred, train_ds, valid_ds, batch_size=64, num_epochs=NUM_EPOCHS,
        lr=1e-3, verbose=False, patience=PATIENCE)
    shred.eval()
    with torch.no_grad():
        shred_pred_np = sc.inverse_transform(shred(test_ds.X).cpu().numpy())
    truth_np   = sc.inverse_transform(test_ds.Y.cpu().numpy())
    rmse_shred = np.sqrt(np.mean((truth_np - shred_pred_np) ** 2))

    # UQ-SHRED
    uq_model = UQ_SHRED(NUM_SENSORS, m, hidden_size=64, hidden_layers=2,
                        l1=350, l2=400, dropout=0.1, noise_dim=NOISE_DIM).to(device)
    fit_uq(uq_model, train_ds, valid_ds, batch_size=64, num_epochs=NUM_EPOCHS,
           lr=1e-3, verbose=False, patience=PATIENCE)
    samples_np = np.array([sc.inverse_transform(s)
                           for s in uq_model.sample(test_ds.X, N_SAMPLES).cpu().numpy()])

    metrics = compute_uq_metrics(samples_np, truth_np)
    metrics["rmse_shred"] = float(rmse_shred)
    results.append(metrics)
    print(f"  RMSE_SHRED={rmse_shred:.6f}  RMSE_UQ={metrics['rmse_uq']:.6f}  "
          f"Cov95={metrics['coverage_95']:.3f}  CRPS={metrics['crps']:.6f}")

save_dir = Path(__file__).parent / "results" / "abl1_lags"
plot_ablation(LAG_VALUES, results, "lags", "Temporal Lag", save_dir, log_x=False)
print("\nDone.")
