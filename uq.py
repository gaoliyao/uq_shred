"""
UQ utilities for UQ-SHRED: loss function, metrics, and plotting.
"""

import torch
import numpy as np


# === Loss Function ===

def energy_loss(y_true, y_pred1, y_pred2, beta=1):
    """Energy score loss for engression.
    
    Args:
        y_true: (batch, dim) ground truth
        y_pred1: (batch, dim) first sample
        y_pred2: (batch, dim) second sample
        beta: power parameter (1=L1, 2=L2)
    
    Returns:
        loss: scalar
    """
    s1 = (torch.norm(y_pred1 - y_true, dim=1).pow(beta).mean() + 
          torch.norm(y_pred2 - y_true, dim=1).pow(beta).mean()) / 2
    s2 = torch.norm(y_pred1 - y_pred2, dim=1).pow(beta).mean()
    return s1 - s2 / 2


# === Metrics ===

def coverage(y_true, lower, upper):
    """Fraction of true values within CI."""
    within = (y_true >= lower) & (y_true <= upper)
    return within.float().mean().item()


def calibration_scores(samples, y_true, levels=[0.5, 0.7, 0.9, 0.95, 0.99]):
    """Coverage at multiple confidence levels.
    
    Args:
        samples: (n_samples, batch, dim)
        y_true: (batch, dim)
        levels: confidence levels
    
    Returns:
        dict: {level: observed_coverage}
    """
    results = {}
    for conf in levels:
        alpha = (1 - conf) / 2
        lower = torch.quantile(samples, alpha, dim=0)
        upper = torch.quantile(samples, 1 - alpha, dim=0)
        results[conf] = coverage(y_true, lower, upper)
    return results


def crps(samples, y_true):
    """Continuous Ranked Probability Score (lower is better).
    
    Args:
        samples: (n_samples, batch, dim)
        y_true: (batch, dim)
    """
    mae = torch.abs(samples - y_true.unsqueeze(0)).mean()
    n = min(50, samples.shape[0])
    idx1, idx2 = torch.randperm(samples.shape[0])[:n], torch.randperm(samples.shape[0])[:n]
    diff = torch.abs(samples[idx1] - samples[idx2]).mean()
    return (mae - diff / 2).item()


def sharpness(samples, conf=0.95):
    """Average CI width (smaller is sharper).
    
    Args:
        samples: (n_samples, batch, dim)
        conf: confidence level
    """
    alpha = (1 - conf) / 2
    lower = torch.quantile(samples, alpha, dim=0)
    upper = torch.quantile(samples, 1 - alpha, dim=0)
    return (upper - lower).mean().item()


# === Plotting ===

def plot_reconstruction(samples, y_true, sc=None, idx=0, ax=None):
    """Plot reconstruction with uncertainty band for single location.
    
    Args:
        samples: (n_samples, batch, dim) 
        y_true: (batch, dim)
        sc: sklearn scaler for inverse transform (optional)
        idx: spatial index to plot
        ax: matplotlib axis (creates new if None)
    """
    import matplotlib.pyplot as plt
    
    samples_np = samples.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    if sc is not None:
        samples_np = np.array([sc.inverse_transform(s) for s in samples_np])
        y_true_np = sc.inverse_transform(y_true_np)
    
    mean = samples_np.mean(axis=0)[:, idx]
    lower = np.percentile(samples_np, 2.5, axis=0)[:, idx]
    upper = np.percentile(samples_np, 97.5, axis=0)[:, idx]
    truth = y_true_np[:, idx]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    t = np.arange(len(mean))
    ax.fill_between(t, lower, upper, alpha=0.3, label='95% CI')
    ax.plot(t, mean, 'b-', label='Mean')
    ax.plot(t, truth, 'k--', alpha=0.7, label='Truth')
    ax.legend()
    ax.set_xlabel('Time')
    
    return ax


def plot_calibration(samples, y_true, ax=None):
    """Calibration diagram: expected vs observed coverage."""
    import matplotlib.pyplot as plt
    
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    observed = calibration_scores(samples, y_true, levels)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    obs = [observed[l] for l in levels]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.plot(levels, obs, 'bo-', label='UQ-SHRED')
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.legend()
    
    return ax
