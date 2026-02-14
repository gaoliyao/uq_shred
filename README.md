# UQ-SHRED

**Uncertainty Quantification for SHRED using Engression**

## Overview

UQ-SHRED combines [SHRED](https://github.com/Jan-Williams/pyshred) (SHallow REcurrent Decoder) with [engression](https://arxiv.org/abs/2307.00835) for principled uncertainty quantification in spatiotemporal reconstruction and forecasting from sparse sensor measurements.

**Core idea:** Inject noise into sensor input, train with energy loss, and sample multiple forward passes to get calibrated uncertainty estimates.

## Architecture

```
Input: (batch, lags, num_sensors)
         |
   [sensors, noise]  <-- noise: (batch, noise_dim), same across all lags
         |
   LSTM(input_size + noise_dim) -> hidden
         |
   SDN decoder -> reconstruction
         |
Output: (batch, output_dim)
```

One model, multiple forward passes with different noise = multiple samples for UQ.

## Files

| File | Purpose |
|------|---------|
| `models.py` | SHRED, UQ_SHRED, UQ_Forecaster, fit(), fit_uq() |
| `uq.py` | Energy loss, calibration, CRPS, sharpness, plotting |
| `processdata.py` | Data loading, TimeSeriesDataset |
| `forecasts.py` | Deterministic forecast pipeline |
| `uq_shred_demo.ipynb` | Full demo notebook with comparisons and figures |
| `example.py` | Minimal SHRED example |

## Quick Start

```python
from models import UQ_SHRED, fit_uq
from processdata import load_data, TimeSeriesDataset
import torch

# Train UQ-SHRED (same interface as SHRED, with noise_dim)
model = UQ_SHRED(input_size=3, output_size=100, noise_dim=50)
fit_uq(model, train_dataset, valid_dataset, num_epochs=200)

# Reconstruct with uncertainty
mean, std = model.reconstruct(x, n_samples=100)

# Get samples and quantiles
samples = model.sample(x, n_samples=200)
quantiles = model.reconstruct_quantiles(x, quantiles=[0.025, 0.5, 0.975])
```

### Temporal Forecasting with UQ

```python
from models import UQ_Forecaster, fit_uq

forecaster = UQ_Forecaster(input_size=num_sensors, noise_dim=50)
fit_uq(forecaster, train_forecast_dataset, valid_forecast_dataset)

# Forecast with uncertainty (mean + std over horizon)
mean_traj, std_traj = forecaster.forecast(x, horizon=50, n_samples=100)
```

## Demo Notebook

`uq_shred_demo.ipynb` runs the full pipeline on SST data:

1. **Part 1:** Train standard SHRED (deterministic baseline)
2. **Part 2:** Train UQ-SHRED, compare reconstruction (mean vs median), evaluate calibration/CRPS/sharpness
3. **Part 3:** Train SHRED and UQ forecasters, compare forecast predictions (mean vs median) with uncertainty growth

All figures and metrics are saved to `results/{dataset}_{YYYYMMDD_HHMMSS}/`.

### Generated Figures

| Figure | Description |
|--------|-------------|
| `shred_reconstruction.png` | SHRED baseline reconstruction |
| `uq_shred_reconstruction.png` | UQ-SHRED reconstruction with mean, median, 95% CI |
| `uq_shred_multisensor.png` | Multi-sensor reconstruction grid |
| `shred_vs_uqshred.png` | 3-panel: SHRED vs UQ-SHRED vs overlay comparison |
| `calibration_diagram.png` | Expected vs observed coverage |
| `uncertainty_vs_error.png` | Uncertainty-error correlation |
| `forecast_comparison.png` | 3-panel: SHRED vs UQ forecast vs overlay comparison |
| `uncertainty_growth.png` | Uncertainty and error growth over forecast horizon |
| `metrics.txt` | Summary table of all metrics |

## Evaluation Metrics

- **Relative Error** — reconstruction/forecast accuracy (SHRED vs UQ mean vs UQ median)
- **Calibration** — 95% CI should contain ~95% of true values
- **Sharpness** — CI width (smaller is better)
- **CRPS** — proper scoring rule (lower is better)
- **Uncertainty-Error Correlation** — high uncertainty where errors are high

## Data

SST data (`Data/SST_data.mat`) is tracked via git-lfs.

## References

- Engression: [arXiv:2307.00835](https://arxiv.org/abs/2307.00835)
- SHRED: [Jan-Williams/pyshred](https://github.com/Jan-Williams/pyshred)
- SINDy-SHRED: [arXiv:2501.13329](https://arxiv.org/abs/2501.13329)
