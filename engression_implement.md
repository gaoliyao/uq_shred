# UQ-SHRED Implementation Plan

**Goal:** Add uncertainty quantification to SHRED using engression framework.

## Architecture

```
Input: (batch, lags, num_sensors)
           ↓
    [sensors, ε]  ← concat noise (same ε across all lags)
           ↓
      LSTM encoder → hidden state
           ↓
      SDN decoder → reconstruction
           ↓
Output: (batch, output_dim)
```

**Key change:** Concatenate noise to sensor input, train with energy loss.

---

## UQ_SHRED Class

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy


class UQ_SHRED(nn.Module):
    """UQ-SHRED: SHRED with uncertainty quantification via engression.
    
    Same interface as SHRED, with added noise_dim parameter.
    """
    
    def __init__(self, input_size, output_size, hidden_size=64, hidden_layers=2,
                 l1=350, l2=400, dropout=0.0, noise_dim=50):
        super(UQ_SHRED, self).__init__()
        
        # Noise dimension
        self.noise_dim = noise_dim
        
        # LSTM takes sensors + noise
        self.lstm = nn.LSTM(
            input_size=input_size + noise_dim,  # <-- wider input
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True
        )
        
        # Decoder (same as SHRED)
        self.linear1 = nn.Linear(hidden_size, l1)
        self.linear2 = nn.Linear(l1, l2)
        self.linear3 = nn.Linear(l2, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
    
    def forward(self, x, noise=None):
        """Forward pass with noise injection.
        
        Args:
            x: (batch, lags, num_sensors)
            noise: (batch, noise_dim) or None (will generate if None)
        
        Returns:
            output: (batch, output_dim)
        """
        batch_size, lags, _ = x.shape
        device = x.device
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=device)
        
        # Expand noise across all lags: (batch, noise_dim) -> (batch, lags, noise_dim)
        noise_expanded = noise.unsqueeze(1).expand(-1, lags, -1)
        
        # Concatenate: (batch, lags, num_sensors + noise_dim)
        x_noisy = torch.cat([x, noise_expanded], dim=-1)
        
        # LSTM
        h_0 = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=device)
        
        _, (h_out, _) = self.lstm(x_noisy, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)
        
        # Decoder
        output = self.linear1(h_out)
        output = self.dropout(output)
        output = F.relu(output)
        
        output = self.linear2(output)
        output = self.dropout(output)
        output = F.relu(output)
        
        output = self.linear3(output)
        
        return output
    
    def sample(self, x, n_samples=100):
        """Generate multiple samples for UQ.
        
        Args:
            x: (batch, lags, num_sensors)
            n_samples: number of samples
        
        Returns:
            samples: (n_samples, batch, output_dim)
        """
        self.eval()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x)  # noise generated inside
                samples.append(out)
        return torch.stack(samples, dim=0)
    
    def predict(self, x, n_samples=100, return_std=True):
        """Predict with uncertainty.
        
        Args:
            x: (batch, lags, num_sensors)
            n_samples: number of samples for UQ
            return_std: whether to return std
        
        Returns:
            mean: (batch, output_dim)
            std: (batch, output_dim) if return_std=True
        """
        samples = self.sample(x, n_samples)  # (n_samples, batch, output_dim)
        mean = samples.mean(dim=0)
        if return_std:
            std = samples.std(dim=0)
            return mean, std
        return mean
    
    def predict_quantiles(self, x, quantiles=[0.025, 0.5, 0.975], n_samples=100):
        """Predict quantiles for confidence intervals.
        
        Args:
            x: (batch, lags, num_sensors)
            quantiles: list of quantiles
            n_samples: number of samples
        
        Returns:
            dict of quantile -> (batch, output_dim)
        """
        samples = self.sample(x, n_samples)  # (n_samples, batch, output_dim)
        results = {}
        for q in quantiles:
            results[q] = torch.quantile(samples, q, dim=0)
        return results
```

---

## Energy Loss Function

```python
def energy_loss(y_true, y_pred1, y_pred2, beta=1):
    """Energy score loss for engression.
    
    Args:
        y_true: (batch, dim) ground truth
        y_pred1: (batch, dim) first sample
        y_pred2: (batch, dim) second sample
        beta: power parameter (1 = L1-like, 2 = L2-like)
    
    Returns:
        loss: scalar
    """
    # Prediction error: both samples should be close to truth
    s1 = (torch.norm(y_pred1 - y_true, dim=1).pow(beta).mean() + 
          torch.norm(y_pred2 - y_true, dim=1).pow(beta).mean()) / 2
    
    # Diversity: samples should be different
    s2 = torch.norm(y_pred1 - y_pred2, dim=1).pow(beta).mean()
    
    return s1 - s2 / 2
```

---

## Training Function

```python
def fit_uq(model, train_dataset, valid_dataset, batch_size=64, num_epochs=4000,
           lr=1e-3, verbose=False, patience=5, beta=1):
    """Training function for UQ_SHRED with energy loss.
    
    Same interface as original fit(), but uses energy loss.
    """
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()
    
    for epoch in range(1, num_epochs + 1):
        
        for k, data in enumerate(train_loader):
            model.train()
            x, y = data[0], data[1]
            
            # Two forward passes with different noise
            y_pred1 = model(x)  # generates noise internally
            y_pred2 = model(x)  # different noise
            
            optimizer.zero_grad()
            loss = energy_loss(y, y_pred1, y_pred2, beta=beta)
            loss.backward()
            optimizer.step()
        
        # Validation (using mean prediction)
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_mean, val_std = model.predict(valid_dataset.X, n_samples=50)
                val_error = torch.linalg.norm(val_mean - valid_dataset.Y)
                val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error)
            
            if verbose:
                print(f'Epoch {epoch}: val_error={val_error:.4f}, mean_std={val_std.mean():.4f}')
            
            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = deepcopy(model.state_dict())
            else:
                patience_counter += 1
            
            if patience_counter == patience:
                model.load_state_dict(best_params)
                return torch.tensor(val_error_list).cpu()
    
    model.load_state_dict(best_params)
    return torch.tensor(val_error_list).detach().cpu().numpy()
```

---

## Usage Example

```python
from processdata import load_data, TimeSeriesDataset
from sklearn.preprocessing import MinMaxScaler

# Load data (same as original)
num_sensors = 3
lags = 52
load_X = load_data('SST')
n, m = load_X.shape
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

# Prepare datasets (same as original)
# ... (same preprocessing as SHRED)

# Create UQ-SHRED model
uq_shred = UQ_SHRED(
    input_size=num_sensors,
    output_size=m,
    hidden_size=64,
    hidden_layers=2,
    l1=350,
    l2=400,
    dropout=0.1,
    noise_dim=50  # <-- only new parameter
).to(device)

# Train with energy loss
val_errors = fit_uq(
    uq_shred, train_dataset, valid_dataset,
    batch_size=64, num_epochs=1000, lr=1e-3, verbose=True
)

# Inference with UQ
mean, std = uq_shred.predict(test_dataset.X, n_samples=100)

# Or get confidence intervals
quantiles = uq_shred.predict_quantiles(test_dataset.X, quantiles=[0.025, 0.5, 0.975])
lower = quantiles[0.025]
median = quantiles[0.5]
upper = quantiles[0.975]
```

---

## Summary of Changes from SHRED

| Component | SHRED | UQ_SHRED |
|-----------|-------|----------|
| LSTM input | num_sensors | num_sensors + noise_dim |
| forward() | (x) | (x, noise=None) |
| Loss | MSELoss | energy_loss |
| predict() | single output | mean + std |
| New methods | — | sample(), predict_quantiles() |

**Lines changed:** ~50 lines of new code, architecture nearly identical.

---

## Naming Convention

```python
# Reconstruction (current state from sensors)
mean, std = model.reconstruct(x, n_samples=100)
quantiles = model.reconstruct_quantiles(x, quantiles=[0.025, 0.5, 0.975])

# Prediction (future forecasting - separate LSTM, implement later)
# mean, std = model.predict(x, horizon=10)
```

---

## Evaluation Metrics

### 1. Reconstruction Quality

```python
def relative_error(y_true, y_pred):
    """Same metric as original SHRED."""
    return torch.linalg.norm(y_pred - y_true) / torch.linalg.norm(y_true)
```

### 2. Calibration (Is UQ trustworthy?)

```python
def coverage(y_true, lower, upper):
    """Fraction of true values within confidence interval."""
    within = (y_true >= lower) & (y_true <= upper)
    return within.float().mean()

def calibration_scores(model, x, y_true, confidence_levels=[0.5, 0.7, 0.9, 0.95, 0.99], n_samples=100):
    """Compute coverage at multiple confidence levels."""
    samples = model.sample(x, n_samples)  # (n_samples, batch, dim)
    
    results = {}
    for conf in confidence_levels:
        alpha = (1 - conf) / 2
        lower = torch.quantile(samples, alpha, dim=0)
        upper = torch.quantile(samples, 1 - alpha, dim=0)
        results[conf] = coverage(y_true, lower, upper).item()
    
    return results  # {0.9: 0.87, 0.95: 0.93, ...} ideally matches keys
```

### 3. CRPS (Continuous Ranked Probability Score)

```python
def crps(y_true, samples):
    """Proper scoring rule for probabilistic predictions.
    
    Args:
        y_true: (batch, dim)
        samples: (n_samples, batch, dim)
    
    Returns:
        crps: scalar (lower is better)
    """
    n_samples = samples.shape[0]
    
    # Term 1: E[|Y - Ŷ|]
    mae = torch.abs(samples - y_true.unsqueeze(0)).mean()
    
    # Term 2: E[|Ŷ - Ŷ'|] / 2
    # Compute pairwise differences (subsample for efficiency)
    idx1 = torch.randperm(n_samples)[:min(50, n_samples)]
    idx2 = torch.randperm(n_samples)[:min(50, n_samples)]
    diff = torch.abs(samples[idx1] - samples[idx2]).mean()
    
    return mae - diff / 2
```

### 4. Sharpness (Is UQ tight?)

```python
def sharpness(lower, upper):
    """Average width of confidence intervals (smaller is sharper)."""
    return (upper - lower).mean()

def uncertainty_error_correlation(y_true, y_pred_mean, y_pred_std):
    """Correlation between predicted uncertainty and actual error.
    
    Should be positive: high uncertainty where errors are high.
    """
    errors = torch.abs(y_true - y_pred_mean).flatten()
    stds = y_pred_std.flatten()
    
    # Pearson correlation
    corr = torch.corrcoef(torch.stack([errors, stds]))[0, 1]
    return corr.item()
```

---

## Figures for Paper

### Figure 1: Full Reconstruction with Uncertainty Bands

```python
def plot_reconstruction_with_uq(model, test_dataset, sc, sensor_idx=0, n_samples=100):
    """Reconstruction plot with shaded uncertainty region."""
    import matplotlib.pyplot as plt
    
    model.eval()
    samples = model.sample(test_dataset.X, n_samples)  # (n_samples, batch, dim)
    
    # Inverse transform
    samples_np = samples.cpu().numpy()
    samples_orig = np.array([sc.inverse_transform(s) for s in samples_np])
    y_true = sc.inverse_transform(test_dataset.Y.cpu().numpy())
    
    mean = samples_orig.mean(axis=0)
    std = samples_orig.std(axis=0)
    lower = np.percentile(samples_orig, 2.5, axis=0)
    upper = np.percentile(samples_orig, 97.5, axis=0)
    
    # Plot single sensor
    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.arange(len(mean))
    
    ax.fill_between(t, lower[:, sensor_idx], upper[:, sensor_idx], 
                    alpha=0.3, label='95% CI')
    ax.plot(t, mean[:, sensor_idx], 'b-', label='Mean reconstruction')
    ax.plot(t, y_true[:, sensor_idx], 'k--', alpha=0.7, label='Ground truth')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.set_title(f'UQ-SHRED Reconstruction (Sensor {sensor_idx})')
    
    return fig
```

### Figure 2: Multi-Sensor Uncertainty Plot

```python
def plot_sensor_grid(model, test_dataset, sc, sensor_indices, n_samples=100):
    """Grid of sensor-level reconstructions with UQ."""
    import matplotlib.pyplot as plt
    
    n_sensors = len(sensor_indices)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 3*n_sensors), sharex=True)
    
    samples = model.sample(test_dataset.X, n_samples).cpu().numpy()
    samples_orig = np.array([sc.inverse_transform(s) for s in samples])
    y_true = sc.inverse_transform(test_dataset.Y.cpu().numpy())
    
    mean = samples_orig.mean(axis=0)
    lower = np.percentile(samples_orig, 2.5, axis=0)
    upper = np.percentile(samples_orig, 97.5, axis=0)
    t = np.arange(len(mean))
    
    for i, (ax, idx) in enumerate(zip(axes, sensor_indices)):
        ax.fill_between(t, lower[:, idx], upper[:, idx], alpha=0.3)
        ax.plot(t, mean[:, idx], 'b-', linewidth=1)
        ax.plot(t, y_true[:, idx], 'k--', alpha=0.7, linewidth=1)
        ax.set_ylabel(f'Sensor {idx}')
    
    axes[-1].set_xlabel('Time')
    fig.suptitle('UQ-SHRED: Sensor-Level Reconstruction')
    plt.tight_layout()
    
    return fig
```

### Figure 3: Calibration Diagram

```python
def plot_calibration(model, test_dataset, n_samples=200):
    """Calibration plot: expected vs observed coverage."""
    import matplotlib.pyplot as plt
    
    confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    observed = calibration_scores(model, test_dataset.X, test_dataset.Y, 
                                   confidence_levels, n_samples)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    expected = confidence_levels
    obs = [observed[c] for c in confidence_levels]
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(expected, obs, 'bo-', label='UQ-SHRED')
    ax.fill_between(expected, 
                    [e - 0.05 for e in expected], 
                    [e + 0.05 for e in expected], 
                    alpha=0.2, color='gray', label='±5% tolerance')
    
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.set_title('Calibration Diagram')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    return fig
```

### Figure 4: Spatial Uncertainty Map (for SST)

```python
def plot_spatial_uncertainty(model, test_dataset, grid_shape, n_samples=100):
    """Heatmap of average uncertainty across spatial locations."""
    import matplotlib.pyplot as plt
    
    samples = model.sample(test_dataset.X, n_samples)
    std = samples.std(dim=0).mean(dim=0).cpu().numpy()  # avg over time, keep space
    
    # Reshape to spatial grid
    std_map = std.reshape(grid_shape)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(std_map, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Avg. Std Dev')
    ax.set_title('Spatial Distribution of Reconstruction Uncertainty')
    
    return fig
```

### Figure 5: Uncertainty-Error Correlation

```python
def plot_uncertainty_vs_error(model, test_dataset, n_samples=100):
    """Scatter plot: does uncertainty predict error?"""
    import matplotlib.pyplot as plt
    
    samples = model.sample(test_dataset.X, n_samples)
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    
    errors = torch.abs(test_dataset.Y - mean).flatten().cpu().numpy()
    stds = std.flatten().cpu().numpy()
    
    # Subsample for visualization
    idx = np.random.choice(len(errors), min(5000, len(errors)), replace=False)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(stds[idx], errors[idx], alpha=0.1, s=1)
    
    # Trend line
    z = np.polyfit(stds[idx], errors[idx], 1)
    p = np.poly1d(z)
    x_line = np.linspace(stds.min(), stds.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.2f})')
    
    corr = np.corrcoef(stds[idx], errors[idx])[0, 1]
    ax.set_xlabel('Predicted Uncertainty (σ)')
    ax.set_ylabel('Actual Error |y - ŷ|')
    ax.set_title(f'Uncertainty vs Error (correlation={corr:.3f})')
    ax.legend()
    
    return fig
```

### Figure 6: SHRED vs UQ-SHRED Comparison

```python
def plot_comparison(shred_model, uq_shred_model, test_dataset, sc, sensor_idx=0, n_samples=100):
    """Side-by-side comparison of deterministic vs UQ reconstruction."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    y_true = sc.inverse_transform(test_dataset.Y.cpu().numpy())
    t = np.arange(len(y_true))
    
    # SHRED (deterministic)
    with torch.no_grad():
        shred_pred = sc.inverse_transform(shred_model(test_dataset.X).cpu().numpy())
    
    axes[0].plot(t, y_true[:, sensor_idx], 'k-', label='Ground truth')
    axes[0].plot(t, shred_pred[:, sensor_idx], 'b-', label='SHRED')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].set_title('SHRED (Deterministic)')
    
    # UQ-SHRED
    samples = uq_shred_model.sample(test_dataset.X, n_samples).cpu().numpy()
    samples_orig = np.array([sc.inverse_transform(s) for s in samples])
    mean = samples_orig.mean(axis=0)
    lower = np.percentile(samples_orig, 2.5, axis=0)
    upper = np.percentile(samples_orig, 97.5, axis=0)
    
    axes[1].fill_between(t, lower[:, sensor_idx], upper[:, sensor_idx], alpha=0.3)
    axes[1].plot(t, y_true[:, sensor_idx], 'k-', label='Ground truth')
    axes[1].plot(t, mean[:, sensor_idx], 'b-', label='UQ-SHRED mean')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].set_title('UQ-SHRED (With Uncertainty)')
    
    plt.tight_layout()
    return fig
```

---

## Ablation Studies

### 1. noise_dim Sensitivity

```python
noise_dims = [10, 25, 50, 100, 200]
# Train model for each, compare CRPS and calibration
```

### 2. n_samples for Inference

```python
n_samples_list = [10, 25, 50, 100, 200, 500]
# Measure: (1) mean estimate stability, (2) CI stability, (3) inference time
```

### 3. Energy Loss vs MSE

```python
# Train two models:
# - UQ_SHRED with energy loss (proper engression)
# - UQ_SHRED with MSE loss (naive baseline)
# Compare calibration — energy loss should be better calibrated
```

---

## Summary Table for Paper

| Metric | SHRED | UQ-SHRED | Notes |
|--------|-------|----------|-------|
| Relative Error | X.XX | X.XX | Mean reconstruction |
| CRPS | — | X.XX | Lower is better |
| Coverage (90%) | — | XX% | Target: 90% |
| Coverage (95%) | — | XX% | Target: 95% |
| Sharpness | — | X.XX | CI width |
| UQ-Error Corr | — | X.XX | Higher is better |
