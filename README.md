# UQ-SHRED

**Uncertainty Quantification for SHRED using Engression**

## Overview

Combining SHRED (SHallow REcurrent Decoder) with engression for principled uncertainty quantification in latent dynamics discovery.

## Key Ideas

- **SHRED:** Learn latent representations from sparse sensor measurements
- **Engression:** Energy-based regression for distributional predictions
- **UQ-SHRED:** Uncertainty-aware latent state estimation and forecasting

## TODO

- [ ] Literature review on engression
- [ ] Implement engression layer for SHRED decoder
- [ ] Uncertainty propagation through SINDy dynamics
- [ ] Experiments on synthetic + real data

## Engression Summary

**Paper:** "Engression: Extrapolation through the Lens of Distributional Regression" (arXiv:2307.00835)

**Key ideas:**
- Neural network-based **distributional regression** — estimates full conditional distribution P(Y|X)
- **Generative:** Can sample from the fitted conditional distribution
- Works for **high-dimensional outcomes**
- Enables **extrapolation** for "pre-additive noise" models (noise added to covariates before nonlinear transform)
- Available in R and Python

**Why it fits SHRED:**
- SHRED decoder: latent z → high-dim reconstruction
- Engression can make this distributional: z → P(reconstruction | z)
- Enables UQ for both reconstruction and forecasting
- Extrapolation capability useful for out-of-distribution patient states

## References

- Engression: https://arxiv.org/abs/2307.00835
- SINDy-SHRED: https://arxiv.org/abs/2501.13329

---
*Created: 2026-02-14*
