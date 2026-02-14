# CLAUDE.md - UQ-SHRED Developer Notes

## Project Overview

UQ-SHRED = SHRED + Engression for uncertainty quantification.

**Core idea:** Inject noise into sensor input, train with energy loss → calibrated UQ.

---

## Architecture

```
Input: (batch, lags, num_sensors)
          ↓
   [sensors, ε]  ← noise: (batch, noise_dim), same across all lags
          ↓
   LSTM(input_size + noise_dim) → hidden
          ↓
   SDN decoder → reconstruction
          ↓
Output: (batch, output_dim)
```

**Key:** One model, multiple forward passes with different noise = multiple samples.

---

## Files

| File | Purpose |
|------|---------|
| `models.py` | SHRED, SDN, UQ_SHRED, fit(), fit_uq() |
| `uq.py` | energy_loss, metrics, plotting |
| `processdata.py` | Data loading, TimeSeriesDataset |
| `engression_implement.md` | Full implementation plan |

---

## Critical Implementation Details

### 1. Noise Injection

```python
# Noise shape: (batch, noise_dim)
# Expand to: (batch, lags, noise_dim) — SAME noise across all timesteps
noise_expanded = noise.unsqueeze(1).expand(-1, lags, -1)
x_noisy = torch.cat([x, noise_expanded], dim=-1)
```

**Why same noise across lags?** The noise represents "which sample from P(Y|X)" — should be consistent within one sequence.

### 2. Energy Loss

```python
loss = E[||Ŷ - Y||] - E[||Ŷ - Ŷ'||] / 2
```

- First term: predictions close to truth
- Second term: predictions spread out (diversity)
- **Both needed** — without s2, model ignores noise and collapses to deterministic

### 3. Training vs Inference

| Phase | Forward passes | Notes |
|-------|---------------|-------|
| Training | 2 per batch | Different noise → y_pred1, y_pred2 |
| Inference | K (~100) | Sample K times → mean/std/quantiles |

### 4. Device Handling

```python
# Noise must be on same device as input
noise = torch.randn(batch_size, self.noise_dim, device=x.device)

# LSTM hidden states too
h_0 = torch.zeros(..., device=device)
```

---

## Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `noise_dim` | 50 | Start here, tune 25-100 |
| `n_samples` | 100 | For inference; 20 for validation (speed) |
| `beta` | 1 | Energy loss power; 1=L1, 2=L2 |
| `lr` | 1e-3 | Same as SHRED |
| `patience` | 5 | Early stopping |

---

## Common Pitfalls

### ❌ Noise generated once and reused
```python
# WRONG
noise = torch.randn(...)
for _ in range(n_samples):
    out = model(x, noise)  # Same output every time!
```

```python
# CORRECT
for _ in range(n_samples):
    out = model(x)  # Noise generated inside forward()
```

### ❌ Forgetting to call model.eval() for inference
```python
# WRONG - dropout still active
samples = [model(x) for _ in range(100)]

# CORRECT
model.eval()
with torch.no_grad():
    samples = [model(x) for _ in range(100)]
```

### ❌ Using MSE loss instead of energy loss
MSE loss → model ignores noise → no UQ. **Must use energy_loss.**

### ❌ Too few samples for quantiles
```python
# WRONG - 10 samples for 2.5% quantile is unreliable
quantiles = model.reconstruct_quantiles(x, n_samples=10)

# CORRECT
quantiles = model.reconstruct_quantiles(x, n_samples=200)
```

---

## Evaluation Checklist

1. **Reconstruction error** — should match SHRED baseline
2. **Calibration** — 95% CI should contain ~95% of true values
3. **Sharpness** — CI shouldn't be unnecessarily wide
4. **CRPS** — proper scoring rule (lower is better)
5. **Uncertainty-error correlation** — high σ where errors are high

---

## Quick Test

```python
from models import UQ_SHRED, fit_uq
from processdata import load_data, TimeSeriesDataset
import torch

# Minimal test
model = UQ_SHRED(input_size=3, output_size=100, noise_dim=50)
x = torch.randn(8, 52, 3)  # (batch, lags, sensors)

# Check stochasticity
y1 = model(x)
y2 = model(x)
assert not torch.allclose(y1, y2), "Model should give different outputs!"

# Check sampling
samples = model.sample(x, n_samples=10)
assert samples.shape == (10, 8, 100), f"Wrong shape: {samples.shape}"

print("✓ All checks passed")
```

---

## References

- Engression paper: [arXiv:2307.00835](https://arxiv.org/abs/2307.00835)
- Engression code: `engression_repo/` (cloned)
- Original SHRED: [Jan-Williams/pyshred](https://github.com/Jan-Williams/pyshred)

---

## TODO

- [ ] Run on SST data and compare to SHRED baseline
- [ ] Generate paper figures (calibration, reconstruction + UQ)
- [ ] Ablation: noise_dim sensitivity
- [ ] Ablation: energy loss vs MSE
- [ ] Forward prediction with UQ (future work)
