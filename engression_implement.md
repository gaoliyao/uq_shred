# Engression Implementation Guide

**Code cloned to:** `uqshred/engression_repo/`

## Core Idea

**Engression = Energy-based distributional regression**

Instead of predicting a point estimate E[Y|X], predict the full conditional distribution P(Y|X) using a stochastic neural network trained with energy score loss.

## Neural Network Architecture (from `models.py`)

### StoLayer (Stochastic Layer) — The Building Block

```python
class StoLayer(nn.Module):
    def forward(self, x):
        # Concatenate input with fresh Gaussian noise
        eps = torch.randn(x.size(0), self.noise_dim) * self.noise_std
        out = torch.cat([x, eps], dim=1)  # [batch, in_dim + noise_dim]
        
        # Linear → BatchNorm (optional) → Activation
        out = self.layer(out)  # Linear: (in_dim + noise_dim) → out_dim
        return out
```

**Key insight:** Each forward pass concatenates *new* random noise, so same input → different outputs.

### StoNet (Full Network)

```
Input X (in_dim)
    ↓
StoLayer: [X, noise] → Linear(in_dim + noise_dim, hidden_dim) → BN → ReLU
    ↓
StoLayer: [h, noise] → Linear(hidden_dim + noise_dim, hidden_dim) → BN → ReLU  (repeat n-2 times)
    ↓
Linear(hidden_dim, out_dim)  ← NOTE: final layer has NO noise injection
    ↓
Output Ŷ (out_dim)
```

**Architecture choices:**
- `num_layer=2`: input layer + output layer (no intermediate)
- `hidden_dim=100`: neurons per hidden layer
- `noise_dim=100`: Gaussian noise dimensions concatenated
- `add_bn=True`: BatchNorm after linear layers
- `resblock=False`: use StoResBlock instead for skip connections

### StoResBlock (Residual Variant)

```python
def forward(self, x):
    eps1 = torch.randn(x.size(0), noise_dim)
    h = relu(bn(linear1(cat([x, eps1]))))  # layer1
    
    eps2 = torch.randn(x.size(0), noise_dim)  # FRESH noise
    out = linear2(cat([h, eps2]))  # layer2
    
    return out + x  # skip connection (or linear projection if dims differ)
```

## Loss Function (Energy Score)

```python
def energy_loss(y_true, y_pred1, y_pred2, beta=1):
    """
    y_true: ground truth
    y_pred1, y_pred2: two independent samples from model (same x, different noise)
    """
    s1 = ||y_pred1 - y_true||^beta + ||y_pred2 - y_true||^beta  # prediction error
    s2 = ||y_pred1 - y_pred2||^beta  # sample diversity
    return s1/2 - s2/2
```

**Intuition:** 
- s1: predictions should be close to truth
- s2: samples should spread out (capture variance)
- Balance → calibrated uncertainty

## Minimal Implementation

```python
class StoLayer(nn.Module):
    def __init__(self, in_dim, out_dim, noise_dim=100):
        super().__init__()
        self.fc = nn.Linear(in_dim + noise_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.noise_dim = noise_dim
    
    def forward(self, x):
        noise = torch.randn(x.size(0), self.noise_dim, device=x.device)
        return F.relu(self.bn(self.fc(torch.cat([x, noise], dim=1))))

class Engressor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=100, noise_dim=100, n_layers=2):
        super().__init__()
        layers = [StoLayer(in_dim, hidden_dim, noise_dim)]
        for _ in range(n_layers - 1):
            layers.append(StoLayer(hidden_dim, hidden_dim, noise_dim))
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim + noise_dim, out_dim)
        self.noise_dim = noise_dim
    
    def forward(self, x):
        h = self.layers(x)
        noise = torch.randn(h.size(0), self.noise_dim, device=h.device)
        return self.out(torch.cat([h, noise], dim=1))

# Training loop
for x, y in dataloader:
    y1 = model(x)  # sample 1
    y2 = model(x)  # sample 2 (different noise)
    loss = energy_loss(y, y1, y2)
    loss.backward()
    optimizer.step()
```

## Prediction Modes

```python
# Mean: average multiple samples
y_mean = torch.stack([model(x) for _ in range(100)]).mean(0)

# Quantiles: sort samples
samples = torch.stack([model(x) for _ in range(1000)], dim=1)
q025 = samples.quantile(0.025, dim=1)
q975 = samples.quantile(0.975, dim=1)

# Sample: single forward pass
y_sample = model(x)
```

## Integration with SHRED

**Option A: Engression Decoder**
```
Sensors → LSTM → z (latent) → Engression → P(reconstruction | z)
```

**Option B: Engression on Latent**
```
Sensors → LSTM → Engression(z) → Decoder → reconstruction
```

**Recommended for UQ-SHRED:** Option A — replace the SDN decoder with an Engressor

## Key Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| noise_dim | 100 | Higher = more expressive |
| hidden_dim | 100 | Match problem complexity |
| n_layers | 2 | 2-3 usually sufficient |
| beta | 1 | Energy score power (1 = L1, 2 = L2) |
| n_samples | 100 | For mean/quantile prediction |

## Quick Start

```bash
pip install engression
```

```python
from engression import engression

model = engression(X_train, Y_train, num_epochs=500, device="cuda")
y_mean = model.predict(X_test, target="mean")
y_quantiles = model.predict(X_test, target=[0.025, 0.5, 0.975])
```

## References

- Paper: [arXiv:2307.00835](https://arxiv.org/abs/2307.00835)
- Code: [github.com/xwshen51/engression](https://github.com/xwshen51/engression)
