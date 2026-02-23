"""
UQ-SHRED Experiment on 1D RDE Data

Train UQ-SHRED on ground truth RDE data, then run inference on a simulation
ensemble member to produce uncertainty-quantified reconstruction with quantile
bands (95%, 70%, 50%).

Data layout assumed (same as lapis_1drde.py):
    rde_data/
    ├── ground_truth/gt.npz      # keys: x, times, P, T  (each P,T are (T, mx))
    └── ensemble/run_000.npz …   # same keys per run

Usage:
    python uq_shred_rde_experiment.py --data_dir ./rde_data
    python uq_shred_rde_experiment.py --data_dir ./rde_data --ds_factor 75 --n_sensors 16
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from torch.utils.data import DataLoader

from uq_viz import (
    plot_quantile_snapshots, plot_spacetime_overview,
    plot_temporal_traces, plot_calibration_single, plot_training_curve,
    calibration_scores,
)


 
# 1. DATA LOADING  (mirrors lapis_1drde.py conventions)
 

def load_rde_data(data_dir: Path):
    """Load RDE ground truth and ensemble data.

    Returns
    -------
    sim_fields : list of (T, 2, mx) arrays  – ensemble members [P; T]
    gt_field   : (T, 2, mx) array           – ground truth
    x_grid     : (mx,) array
    times      : (T,) array
    """
    gt = np.load(data_dir / "ground_truth" / "gt.npz")
    gt_P = gt["P"].astype(np.float32)
    gt_T = gt["T"].astype(np.float32)
    x_grid = gt["x"].astype(np.float32)
    times = gt["times"].astype(np.float32)
    gt_field = np.stack([gt_P, gt_T], axis=1)
    print(f"  Ground truth: T={gt_P.shape[0]}, mx={gt_P.shape[1]}")

    ens_dir = data_dir / "ensemble"
    sim_fields = []
    for fpath in sorted(ens_dir.glob("run_*.npz")):
        d = np.load(fpath)
        sim_fields.append(np.stack([
            d["P"].astype(np.float32),
            d["T"].astype(np.float32)
        ], axis=1))
        print(f"  Loaded {fpath.name}: T={d['P'].shape[0]}, mx={d['P'].shape[1]}")

    print(f"  {len(sim_fields)} ensemble members + 1 GT")
    return sim_fields, gt_field, x_grid, times


def clip_time(fields_list, gt_field, times, t_start):
    """Discard frames with t < t_start."""
    if t_start <= 0:
        return fields_list, gt_field, times
    mask = times >= t_start
    return [f[mask] for f in fields_list], gt_field[mask], times[mask]


def downsample_field(field, factor):
    """Downsample (T, 2, mx) → (T, 2, mx_coarse) by strided subsampling."""
    if factor <= 1:
        return field, np.arange(field.shape[2])
    idx = np.arange(0, field.shape[2], factor)
    return field[:, :, idx], idx


def upsample_field(field_coarse, mx_full, coarse_idx):
    """Upsample (T, 2, mx_coarse) → (T, 2, mx_full) via linear interp."""
    if field_coarse.shape[2] == mx_full:
        return field_coarse
    T, n_ch, _ = field_coarse.shape
    x_c = coarse_idx.astype(np.float64)
    x_f = np.arange(mx_full, dtype=np.float64)
    out = np.zeros((T, n_ch, mx_full), dtype=field_coarse.dtype)
    for ch in range(n_ch):
        for t in range(T):
            out[t, ch] = np.interp(x_f, x_c, field_coarse[t, ch])
    return out


 
# 2. SENSOR PLACEMENT
 

def place_sensors_hybrid(sim_fields, mx, n_sensors, seed=42):
    """Hybrid sensor placement: half uniform, half variance-weighted."""
    rng = np.random.RandomState(seed)
    n_uni = n_sensors // 2
    n_var = n_sensors - n_uni
    uni_locs = np.linspace(0, mx - 1, n_uni, dtype=int)

    # Variance map
    var_maps = []
    for f in sim_fields:
        var_maps.append(np.var(f[:, 0], axis=0) + np.var(f[:, 1], axis=0))
    variance = np.mean(var_maps, axis=0)
    weights = variance + 1e-8

    min_spacing = max(1, mx // (2 * n_sensors))
    mask = np.ones(mx, dtype=bool)
    for u in uni_locs:
        lo, hi = max(0, u - min_spacing), min(mx, u + min_spacing + 1)
        mask[lo:hi] = False

    selected = []
    for _ in range(n_var):
        w = weights * mask
        if w.sum() < 1e-12:
            remaining = np.where(mask)[0]
            if len(remaining) == 0:
                break
            idx = rng.choice(remaining)
        else:
            w = w / w.sum()
            idx = rng.choice(mx, p=w)
        selected.append(idx)
        lo, hi = max(0, idx - min_spacing), min(mx, idx + min_spacing + 1)
        mask[lo:hi] = False

    sensor_locs = np.sort(np.unique(np.concatenate([
        uni_locs, np.array(selected, dtype=int)])))
    return sensor_locs


def extract_sensor_measurements(field, sensor_locs):
    """Extract (T, 2*n_sensors) sensor time series from (T, 2, mx) field."""
    P_sens = field[:, 0, sensor_locs]
    T_sens = field[:, 1, sensor_locs]
    return np.concatenate([P_sens, T_sens], axis=1)


 
# 3. DATASET CONSTRUCTION
 

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


def build_frame_dataset(field_full, field_coarse, sensor_locs, lags,
                        scaler_sensors=None, scaler_states=None, fit=True):
    """Build frame-by-frame SHRED dataset from a single trajectory.

    Input  : (N, lags, 2*n_sensors)   – lagged sensor windows
    Target : (N, 2*mx_coarse)          – flattened coarse state at current time

    Parameters
    ----------
    field_full   : (T, 2, mx)       – for sensor extraction (full grid)
    field_coarse : (T, 2, mx_coarse) – SHRED prediction targets
    sensor_locs  : (n_sensors,)
    lags         : int
    scaler_sensors, scaler_states : pre-fit scalers (for test/inference sets)
    fit          : whether to fit the scalers on this data
    """
    n_channels = 2 * len(sensor_locs)
    state_dim = 2 * field_coarse.shape[2]
    T = field_full.shape[0]
    N = T - lags
    assert N > 0, f"Not enough timesteps ({T}) for lags={lags}"

    sens = extract_sensor_measurements(field_full, sensor_locs)  # (T, n_channels)

    X = np.zeros((N, lags, n_channels), dtype=np.float32)
    Y = np.zeros((N, state_dim), dtype=np.float32)
    for i in range(N):
        X[i] = sens[i:i + lags]
        Y[i] = field_coarse[i + lags - 1].ravel()

    # Scaling
    if scaler_sensors is None:
        scaler_sensors = MinMaxScaler()
    if scaler_states is None:
        scaler_states = MinMaxScaler()

    if fit:
        scaler_sensors.fit(X.reshape(-1, n_channels))
        scaler_states.fit(Y)

    X_scaled = scaler_sensors.transform(X.reshape(-1, n_channels)).reshape(X.shape)
    Y_scaled = scaler_states.transform(Y)

    return X_scaled, Y_scaled, scaler_sensors, scaler_states


 
# 4. UQ-SHRED MODEL  (from uq_shred codebase)
 

class UQ_SHRED(torch.nn.Module):
    """SHRED + engression noise for uncertainty quantification."""

    def __init__(self, input_size, output_size, hidden_size=64, hidden_layers=2,
                 l1=350, l2=400, dropout=0.0, noise_dim=50):
        super().__init__()
        self.noise_dim = noise_dim
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        self.lstm = torch.nn.LSTM(
            input_size=input_size + noise_dim, hidden_size=hidden_size,
            num_layers=hidden_layers, batch_first=True)

        self.linear1 = torch.nn.Linear(hidden_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, noise=None):
        batch_size, lags, _ = x.shape
        device = x.device

        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=device)

        noise_expanded = noise.unsqueeze(1).expand(-1, lags, -1)
        x_noisy = torch.cat([x, noise_expanded], dim=-1)

        h_0 = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=device)
        _, (h_out, _) = self.lstm(x_noisy, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)

        out = torch.nn.functional.relu(self.dropout(self.linear1(h_out)))
        out = torch.nn.functional.relu(self.dropout(self.linear2(out)))
        out = self.linear3(out)
        return out

    def sample(self, x, n_samples=100):
        self.eval()
        with torch.no_grad():
            samples = torch.stack([self.forward(x) for _ in range(n_samples)])
        return samples  # (n_samples, batch, output_dim)

    def reconstruct(self, x, n_samples=100):
        samples = self.sample(x, n_samples)
        return samples.mean(dim=0), samples.std(dim=0)

    def reconstruct_quantiles(self, x, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975],
                              n_samples=200):
        samples = self.sample(x, n_samples)
        return {q: torch.quantile(samples, q, dim=0) for q in quantiles}


 
# 5. ENERGY LOSS & TRAINING
 

def energy_loss(y_true, y_pred1, y_pred2, beta=1):
    """Energy score loss for engression."""
    s1 = (torch.norm(y_pred1 - y_true, dim=1).pow(beta).mean() +
          torch.norm(y_pred2 - y_true, dim=1).pow(beta).mean()) / 2
    s2 = torch.norm(y_pred1 - y_pred2, dim=1).pow(beta).mean()
    return s1 - s2 / 2


def fit_uq(model, train_dataset, valid_dataset, batch_size=64, num_epochs=2000,
           lr=1e-3, verbose=True, patience=7, beta=1):
    """Train UQ-SHRED with energy loss."""
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_error_list = []
    patience_counter = 0
    best_params = deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        for data in train_loader:
            model.train()
            x, y = data[0], data[1]
            y_pred1 = model(x)
            y_pred2 = model(x)
            optimizer.zero_grad()
            loss = energy_loss(y, y_pred1, y_pred2, beta=beta)
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_mean, _ = model.reconstruct(valid_dataset.X, n_samples=20)
                val_error = torch.linalg.norm(val_mean - valid_dataset.Y)
                val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error.item())

            if verbose:
                print(f"  Epoch {epoch:4d}: val_error = {val_error_list[-1]:.6f}")

            if val_error_list[-1] <= min(val_error_list):
                patience_counter = 0
                best_params = deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_params)
    return val_error_list

 


 
# 6. MAIN EXPERIMENT
 

def main():
    parser = argparse.ArgumentParser(description="UQ-SHRED on 1D RDE data")
    parser.add_argument("--data_dir", type=str, default="./rde_data",
                        help="Path to RDE data directory")
    parser.add_argument("--results_dir", type=str, default="./uq_shred_results",
                        help="Where to save results")
    parser.add_argument("--ds_factor", type=int, default=75,
                        help="Spatial downsample factor")
    parser.add_argument("--t_clip", type=float, default=10.0,
                        help="Clip times before this value")
    parser.add_argument("--n_sensors", type=int, default=16,
                        help="Number of sensors")
    parser.add_argument("--lags", type=int, default=5,
                        help="Number of lag timesteps for SHRED input")
    parser.add_argument("--noise_dim", type=int, default=50,
                        help="Engression noise dimension")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Max training epochs")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of Monte Carlo samples for inference")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--infer_run", type=int, default=0, #change to 0 for closer trajectory, to 1 for further trajectory
                        help="Which ensemble member to run inference on (0-indexed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

     
    # 1. Load data
     
    print("\n[1] Loading data …")
    sim_fields, gt_field, x_grid, times = load_rde_data(data_dir)
    mx_full = gt_field.shape[2]

    # Time clipping
    if args.t_clip > 0:
        print(f"  Clipping t < {args.t_clip} …")
        sim_fields, gt_field, times = clip_time(sim_fields, gt_field, times, args.t_clip)
        print(f"  After clip: T={gt_field.shape[0]}, "
              f"t ∈ [{times[0]:.2f}, {times[-1]:.2f}]")

    T_total = gt_field.shape[0]

    # Spatial downsampling
    gt_coarse, coarse_idx = downsample_field(gt_field, args.ds_factor)
    sim_coarse = [downsample_field(f, args.ds_factor)[0] for f in sim_fields]
    mx_coarse = gt_coarse.shape[2]
    state_dim = 2 * mx_coarse
    print(f"  mx: {mx_full} → {mx_coarse} (ds={args.ds_factor}), state_dim={state_dim}")

     
    # 2. Place sensors & build datasets
     
    print("\n[2] Placing sensors …")
    # Use all fields (GT + ensemble) for sensor placement
    all_fields_for_sensors = [gt_field] + sim_fields
    sensors = place_sensors_hybrid(all_fields_for_sensors, mx_full,
                                   args.n_sensors, seed=args.seed)
    n_channels = 2 * len(sensors)
    print(f"  {len(sensors)} sensors placed, n_channels = {n_channels}")

    print("\n[3] Building datasets …")
    # --- Training data: ground truth trajectory ---
    X_train_all, Y_train_all, sc_sens, sc_state = build_frame_dataset(
        gt_field, gt_coarse, sensors, args.lags, fit=True)

    N_total = X_train_all.shape[0]
    # Split GT into train (80%) and validation (20%)
    perm = np.random.permutation(N_total)
    n_train = int(0.8 * N_total)
    train_idx = perm[:n_train]
    valid_idx = perm[n_train:]

    X_tr = torch.tensor(X_train_all[train_idx], dtype=torch.float32).to(device)
    Y_tr = torch.tensor(Y_train_all[train_idx], dtype=torch.float32).to(device)
    X_va = torch.tensor(X_train_all[valid_idx], dtype=torch.float32).to(device)
    Y_va = torch.tensor(Y_train_all[valid_idx], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(X_tr, Y_tr)
    valid_dataset = TimeSeriesDataset(X_va, Y_va)
    print(f"  Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")

    # --- Inference data: one ensemble member ---
    infer_idx = args.infer_run
    assert infer_idx < len(sim_fields), \
        f"--infer_run={infer_idx} but only {len(sim_fields)} ensemble members"
    print(f"\n  Inference target: ensemble run {infer_idx}")

    X_test_all, Y_test_all, _, _ = build_frame_dataset(
        sim_fields[infer_idx], sim_coarse[infer_idx], sensors, args.lags,
        scaler_sensors=sc_sens, scaler_states=sc_state, fit=False)

    X_te = torch.tensor(X_test_all, dtype=torch.float32).to(device)
    Y_te = torch.tensor(Y_test_all, dtype=torch.float32).to(device)
    test_dataset = TimeSeriesDataset(X_te, Y_te)
    print(f"  Test (inference) samples: {len(test_dataset)}")

     
    # 4. Train UQ-SHRED
     
    print(f"\n[4] Training UQ-SHRED (noise_dim={args.noise_dim}) …")
    model = UQ_SHRED(
        input_size=n_channels, output_size=state_dim,
        hidden_size=64, hidden_layers=2, l1=350, l2=400,
        dropout=0.0, noise_dim=args.noise_dim
    ).to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    val_errors = fit_uq(model, train_dataset, valid_dataset,
                        batch_size=args.batch_size, num_epochs=args.epochs,
                        lr=1e-3, verbose=True, patience=7, beta=1)

     
    # 5. Inference with UQ
     
    print(f"\n[5] Running inference ({args.n_samples} samples) …")
    model.eval()

    # Process in chunks to avoid OOM for large test sets
    chunk_size = 256
    N_test = len(test_dataset)
    all_samples = []

    with torch.no_grad():
        for start in range(0, N_test, chunk_size):
            end = min(start + chunk_size, N_test)
            x_chunk = X_te[start:end]
            # (n_samples, chunk_len, state_dim)
            chunk_samples = model.sample(x_chunk, n_samples=args.n_samples)
            all_samples.append(chunk_samples.cpu())

    # Concatenate: (n_samples, N_test, state_dim)
    all_samples = torch.cat(all_samples, dim=1)
    print(f"  Samples shape: {all_samples.shape}")

    # Inverse-transform samples and ground truth to original scale
    samples_np = all_samples.numpy()  # (n_samples, N_test, state_dim)
    y_true_scaled = Y_te.cpu().numpy()  # (N_test, state_dim)

    # Inverse transform each sample
    samples_orig = np.array([sc_state.inverse_transform(s) for s in samples_np])
    y_true_orig = sc_state.inverse_transform(y_true_scaled)

    # Compute quantiles
    quantile_levels = [0.025, 0.15, 0.25, 0.5, 0.75, 0.85, 0.975]
    quantiles_flat = {}  # {q: (N_test, state_dim)}
    for q in quantile_levels:
        quantiles_flat[q] = np.percentile(samples_orig, 100 * q, axis=0)

    # Reshape to spatial: (N_test, state_dim) → (N_test, 2, mx_coarse)
    N_infer = y_true_orig.shape[0]
    gt_test_coarse = y_true_orig.reshape(N_infer, 2, mx_coarse)
    quantiles_coarse = {q: v.reshape(N_infer, 2, mx_coarse) for q, v in quantiles_flat.items()}

    # Upsample to full grid
    gt_test_full = upsample_field(gt_test_coarse, mx_full, coarse_idx)
    quantiles_full = {q: upsample_field(v, mx_full, coarse_idx)
                      for q, v in quantiles_coarse.items()}

    # The times for the test frames: offset by lags
    test_times = times[args.lags:args.lags + N_infer]

     
    # 6. Metrics
     
    print("\n[6] Computing metrics …")
    median_full = quantiles_full[0.5]
    err_P = np.sqrt(np.mean((gt_test_full[:, 0] - median_full[:, 0]) ** 2))
    err_T = np.sqrt(np.mean((gt_test_full[:, 1] - median_full[:, 1]) ** 2))
    print(f"  RMSE  Pressure: {err_P:.6f}")
    print(f"  RMSE  Temperature: {err_T:.6f}")

    # Calibration (numpy-based via uq_viz)
    calib = calibration_scores(samples_orig, y_true_orig,
                               levels=[0.5, 0.7, 0.9, 0.95, 0.99])
    print("  Calibration (expected → observed):")
    for level, obs in calib.items():
        print(f"    {level*100:.0f}% CI → {obs*100:.1f}% coverage")

     
    # 7. Visualization
     
    print("\n[7] Generating plots …")
    field_names = ["Pressure (P)", "Temperature (T)"]

    plot_quantile_snapshots(
        gt_test_full, quantiles_full, test_times, sensors, x_grid,
        results_dir / "uq_shred_rde_snapshots.png",
        field_names=field_names)

    plot_spacetime_overview(
        gt_test_full, quantiles_full, test_times, x_grid,
        results_dir / "uq_shred_rde_spacetime.png",
        field_names=field_names)

    plot_temporal_traces(
        gt_test_full, quantiles_full, test_times, x_grid,
        results_dir / "uq_shred_rde_timeseries.png",
        field_names=field_names)

    plot_calibration_single(samples_orig, y_true_orig,
                            results_dir / "calibration.png")

    plot_training_curve(val_errors, results_dir / "training_curve.png",
                        title="UQ-SHRED Training Curve")

    # Save raw predictions
    np.savez_compressed(
        results_dir / "uq_shred_predictions.npz",
        quantiles={str(q): v for q, v in quantiles_full.items()},
        gt=gt_test_full, times=test_times, x_grid=x_grid,
        sensor_locs=sensors, calibration=calib)

    print("\n" + "=" * 60)
    print("  UQ-SHRED RDE Experiment Complete!")
    print(f"  Results saved to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
