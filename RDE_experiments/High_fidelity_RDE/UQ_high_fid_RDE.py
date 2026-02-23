"""
UQ-DA-SHRED: Uncertainty Quantification for Multiscale DA-SHRED
 

Strategy: Pre-train deterministic DA-SHRED backbone normally, then inject noise
into both LF and HF pathways and fine-tune end-to-end with energy loss on U_real.

The DA-SHRED backbone handles the hard problem (learning to reconstruct across
the sim→real domain gap), and the engression fine-tuning handles the easier
problem (learning to spread predictions to quantify uncertainty).

Data: DA_data/Koch_model_processed.npy, DA_data/high_fidelity_sim_processed.npy

Usage:
    python uq_dashred.py
    python uq_dashred.py --data_dir DA_data --noise_dim 50 --n_samples 200
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
import copy

from uq_viz import (
    plot_dual_snapshots, plot_dual_timeseries,
    plot_calibration_dual, plot_error_vs_uncertainty,
    plot_training_curve,
    calibration_scores, compute_crps, compute_sharpness,
)

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


 
# 1. DATASETS
 

class TimeSeriesDataset(Dataset):
    def __init__(self, U, sensor_indices, lags, scaler=None, fit_scaler=False):
        self.U = U; self.sensor_indices = sensor_indices; self.lags = lags
        self.S = U[:, sensor_indices]
        if scaler is None:
            self.scaler_U = MinMaxScaler(); self.scaler_S = MinMaxScaler()
        else:
            self.scaler_U, self.scaler_S = scaler
        if fit_scaler:
            self.U_scaled = self.scaler_U.fit_transform(self.U)
            self.S_scaled = self.scaler_S.fit_transform(self.S)
        else:
            self.U_scaled = self.scaler_U.transform(self.U)
            self.S_scaled = self.scaler_S.transform(self.S)
        self.valid_indices = np.arange(lags, len(U))

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        return (torch.tensor(self.S_scaled[t - self.lags:t], dtype=torch.float32),
                torch.tensor(self.U_scaled[t], dtype=torch.float32))

    def get_scalers(self): return (self.scaler_U, self.scaler_S)


 
# 2. DETERMINISTIC DA-SHRED COMPONENTS (from forward_lapis_rde_v3.py)
 

def frequency_sparsity_bandlimited(signal, max_freq):
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)
    resolvable = magnitudes[:, :max_freq + 1]
    unresolvable = magnitudes[:, max_freq + 1:] if magnitudes.shape[-1] > max_freq + 1 else None
    l1 = torch.sum(resolvable, dim=-1); l2 = torch.sqrt(torch.sum(resolvable ** 2, dim=-1) + 1e-8)
    loss = l1 / (l2 + 1e-8)
    if unresolvable is not None and unresolvable.shape[-1] > 0:
        oob = torch.sum(unresolvable ** 2, dim=-1) / (torch.sum(magnitudes ** 2, dim=-1) + 1e-8)
        return torch.mean(loss) + 100.0 * torch.mean(oob)
    return torch.mean(loss)


class SHRED(nn.Module):
    def __init__(self, num_sensors, lags, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_sensors, hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_size))

    def encode(self, x):
        _, (h_n, _) = self.lstm(x); return self.norm(h_n[-1])

    def forward(self, x):
        z = self.encode(x); return self.decoder(z), z


class HF_SHRED(nn.Module):
    def __init__(self, num_sensors, lags, hidden_size, output_size,
                 velocity_correction='deformation', use_time_derivatives=True,
                 use_lag_attention=True):
        super().__init__()
        self.hidden_size = hidden_size; self.output_size = output_size
        self.num_sensors = num_sensors; self.lags = lags
        self.velocity_correction = velocity_correction
        self.use_time_derivatives = use_time_derivatives
        self.use_lag_attention = use_lag_attention
        input_size = num_sensors * 3 if use_time_derivatives else num_sensors
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_size)
        if use_lag_attention:
            self.lag_attention = nn.Sequential(
                nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, lags), nn.Softmax(dim=-1))
            self.lstm_all_steps = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_size))
        if velocity_correction == 'deformation':
            self.deformation_net = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, output_size))
            self.amplitude_net = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, output_size), nn.Softplus())
        self.scale = nn.Parameter(torch.tensor(0.5))

    def compute_time_derivatives(self, x):
        dx = torch.zeros_like(x); d2x = torch.zeros_like(x)
        dx[:, 1:-1, :] = (x[:, 2:, :] - x[:, :-2, :]) / 2.0
        dx[:, 0, :] = x[:, 1, :] - x[:, 0, :]; dx[:, -1, :] = x[:, -1, :] - x[:, -2, :]
        d2x[:, 1:-1, :] = x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :]
        d2x[:, 0, :] = d2x[:, 1, :]; d2x[:, -1, :] = d2x[:, -2, :]
        return torch.cat([x, dx, d2x], dim=-1)

    def encode(self, x):
        x_emb = self.compute_time_derivatives(x) if self.use_time_derivatives else x
        if self.use_lag_attention:
            all_h, _ = self.lstm_all_steps(x_emb)
            _, (h_f, _) = self.lstm(x_emb); h_f = self.norm(h_f[-1])
            attn = self.lag_attention(h_f)
            z = torch.sum(attn.unsqueeze(-1) * all_h, dim=1)
            self.last_attention_weights = attn.detach()
        else:
            _, (h_n, _) = self.lstm(x_emb); z = self.norm(h_n[-1])
        return z

    def apply_deformation(self, signal, shift_field, amplitude_field=None):
        B, N = signal.shape
        x_base = torch.linspace(0, N - 1, N, device=signal.device).unsqueeze(0).expand(B, -1)
        x_w = (x_base + shift_field) % N
        xf = x_w.long() % N; xc = (xf + 1) % N
        wc = x_w - x_w.floor(); wf = 1 - wc
        out = wf * torch.gather(signal, 1, xf) + wc * torch.gather(signal, 1, xc)
        if amplitude_field is not None: out = out * amplitude_field
        return out

    def decode_from_latent(self, z):
        base = self.scale * self.decoder(z)
        if self.velocity_correction == 'deformation':
            shift = torch.tanh(self.deformation_net(z)) * 10
            amp = self.amplitude_net(z) * 0.5 + 0.5
            return self.apply_deformation(base, shift, amp)
        return base

    def forward(self, x):
        z = self.encode(x); return self.decode_from_latent(z), z


class LatentGAN(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim))
        with torch.no_grad():
            self.generator[-1].weight.mul_(0.1); self.generator[-1].bias.zero_()
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1))

    def forward(self, z): return z + self.generator(z)


class SparseFreqDASHRED(nn.Module):
    """Standard (deterministic) DA-SHRED."""
    def __init__(self, lf_shred, num_sensors, lags, hidden_size, output_size, sensor_indices):
        super().__init__()
        self.lf_lstm = copy.deepcopy(lf_shred.lstm)
        self.lf_norm = copy.deepcopy(lf_shred.norm)
        self.lf_decoder = copy.deepcopy(lf_shred.decoder)
        self.gan = LatentGAN(lf_shred.hidden_size)
        self.hf_shred = HF_SHRED(num_sensors, lags, hidden_size, output_size,
                                  velocity_correction='deformation',
                                  use_time_derivatives=True, use_lag_attention=True)
        self.register_buffer('sensor_indices', torch.tensor(sensor_indices, dtype=torch.long))
        self.lags = lags; self.num_sensors = num_sensors; self.output_size = output_size

    def encode_lf(self, x):
        _, (h_n, _) = self.lf_lstm(x); return self.lf_norm(h_n[-1])

    def decode_lf(self, z): return self.lf_decoder(z)

    def forward(self, sensor_history, use_gan=True):
        z_lf = self.encode_lf(sensor_history)
        if use_gan: z_lf = self.gan(z_lf)
        u_lf = self.decode_lf(z_lf)
        residual_history = torch.zeros_like(sensor_history)
        sensors_lf_pred = u_lf[:, self.sensor_indices]
        for lag in range(self.lags):
            residual_history[:, lag, :] = sensor_history[:, lag, :] - sensors_lf_pred.detach()
        u_hf, z_hf = self.hf_shred(residual_history)
        return u_lf + u_hf, u_lf, u_hf, z_lf, z_hf


 
# 3. UQ-DA-SHRED MODEL (noise-injected version)
 

class UQ_SparseFreqDASHRED(nn.Module):
    """DA-SHRED with noise injection for UQ via engression.

    Noise is concatenated to sensor input before BOTH the LF and HF LSTMs,
    so the same noise vector coherently influences the full LF+HF pipeline.
    """

    def __init__(self, num_sensors, lags, hidden_size, output_size, sensor_indices,
                 noise_dim=50):
        super().__init__()
        self.noise_dim = noise_dim
        self.hidden_size = hidden_size
        self.lags = lags
        self.num_sensors = num_sensors
        self.output_size = output_size

        # LF pathway: LSTM(sensors + noise) → decoder
        self.lf_lstm = nn.LSTM(num_sensors + noise_dim, hidden_size,
                               num_layers=2, batch_first=True, dropout=0.1)
        self.lf_norm = nn.LayerNorm(hidden_size)
        self.lf_decoder = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_size))

        # GAN for domain transfer
        self.gan = LatentGAN(hidden_size)

        # HF pathway: time derivatives of residual sensors + same noise
        hf_input_size = num_sensors * 3 + noise_dim
        self.hf_lstm = nn.LSTM(hf_input_size, hidden_size, num_layers=2,
                               batch_first=True, dropout=0.1)
        self.hf_norm = nn.LayerNorm(hidden_size)
        self.hf_lag_attention = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.Tanh(),
            nn.Linear(64, lags), nn.Softmax(dim=-1))
        self.hf_lstm_all_steps = nn.LSTM(hf_input_size, hidden_size, num_layers=1,
                                          batch_first=True)
        self.hf_decoder = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_size))
        self.hf_deformation_net = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, output_size))
        self.hf_amplitude_net = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Linear(64, output_size), nn.Softplus())
        self.hf_scale = nn.Parameter(torch.tensor(0.5))

        self.register_buffer('sensor_indices',
                             torch.tensor(sensor_indices, dtype=torch.long))

    def _inject_noise(self, x, noise):
        noise_exp = noise.unsqueeze(1).expand(-1, x.shape[1], -1)
        return torch.cat([x, noise_exp], dim=-1)

    def _compute_time_derivatives(self, x):
        dx = torch.zeros_like(x); d2x = torch.zeros_like(x)
        dx[:, 1:-1, :] = (x[:, 2:, :] - x[:, :-2, :]) / 2.0
        dx[:, 0, :] = x[:, 1, :] - x[:, 0, :]; dx[:, -1, :] = x[:, -1, :] - x[:, -2, :]
        d2x[:, 1:-1, :] = x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :]
        d2x[:, 0, :] = d2x[:, 1, :]; d2x[:, -1, :] = d2x[:, -2, :]
        return torch.cat([x, dx, d2x], dim=-1)

    def _apply_deformation(self, signal, shift_field, amplitude_field=None):
        B, N = signal.shape
        x_base = torch.linspace(0, N - 1, N, device=signal.device).unsqueeze(0).expand(B, -1)
        x_w = (x_base + shift_field) % N
        xf = x_w.long() % N; xc = (xf + 1) % N
        wc = x_w - x_w.floor(); wf = 1 - wc
        out = wf * torch.gather(signal, 1, xf) + wc * torch.gather(signal, 1, xc)
        if amplitude_field is not None: out = out * amplitude_field
        return out

    def encode_lf(self, x, noise=None):
        B = x.shape[0]
        if noise is None:
            noise = torch.randn(B, self.noise_dim, device=x.device)
        x_noisy = self._inject_noise(x, noise)
        _, (h_n, _) = self.lf_lstm(x_noisy)
        return self.lf_norm(h_n[-1])

    def decode_lf(self, z):
        return self.lf_decoder(z)

    def encode_hf(self, residual_history, noise):
        x_emb = self._compute_time_derivatives(residual_history)
        noise_exp = noise.unsqueeze(1).expand(-1, residual_history.shape[1], -1)
        x_noisy = torch.cat([x_emb, noise_exp], dim=-1)
        all_h, _ = self.hf_lstm_all_steps(x_noisy)
        _, (h_f, _) = self.hf_lstm(x_noisy)
        h_f = self.hf_norm(h_f[-1])
        attn = self.hf_lag_attention(h_f)
        z = torch.sum(attn.unsqueeze(-1) * all_h, dim=1)
        return z

    def decode_hf(self, z):
        base = self.hf_scale * self.hf_decoder(z)
        shift = torch.tanh(self.hf_deformation_net(z)) * 10
        amp = self.hf_amplitude_net(z) * 0.5 + 0.5
        return self._apply_deformation(base, shift, amp)

    def forward(self, sensor_history, noise=None, use_gan=True):
        B = sensor_history.shape[0]
        if noise is None:
            noise = torch.randn(B, self.noise_dim, device=sensor_history.device)

        z_lf = self.encode_lf(sensor_history, noise)
        if use_gan:
            z_lf = self.gan(z_lf)
        u_lf = self.decode_lf(z_lf)

        residual_history = torch.zeros_like(sensor_history)
        sensors_lf_pred = u_lf[:, self.sensor_indices]
        for lag in range(self.lags):
            residual_history[:, lag, :] = sensor_history[:, lag, :] - sensors_lf_pred.detach()

        z_hf = self.encode_hf(residual_history, noise)
        u_hf = self.decode_hf(z_hf)

        return u_lf + u_hf, u_lf, u_hf, z_lf, z_hf

    def sample(self, sensor_history, n_samples=100, use_gan=True):
        self.eval()
        with torch.no_grad():
            samples_total, samples_lf = [], []
            for _ in range(n_samples):
                ut, ul, _, _, _ = self.forward(sensor_history, use_gan=use_gan)
                samples_total.append(ut)
                samples_lf.append(ul)
        return torch.stack(samples_total), torch.stack(samples_lf)

    def init_from_deterministic(self, det_dashred):
        """Initialize UQ model weights from a trained deterministic DA-SHRED."""
        with torch.no_grad():
            det_sd = det_dashred.state_dict()
            uq_sd = self.state_dict()

            for name, param in det_sd.items():
                uq_name = name
                if name.startswith('hf_shred.'):
                    suffix = name[len('hf_shred.'):]
                    if suffix.startswith('lstm_all_steps'):
                        uq_name = 'hf_lstm_all_steps' + suffix[len('lstm_all_steps'):]
                    elif suffix.startswith('lstm'):
                        uq_name = 'hf_lstm' + suffix[len('lstm'):]
                    elif suffix.startswith('norm'):
                        uq_name = 'hf_norm' + suffix[len('norm'):]
                    elif suffix.startswith('lag_attention'):
                        uq_name = 'hf_lag_attention' + suffix[len('lag_attention'):]
                    elif suffix.startswith('decoder'):
                        uq_name = 'hf_decoder' + suffix[len('decoder'):]
                    elif suffix.startswith('deformation_net'):
                        uq_name = 'hf_deformation_net' + suffix[len('deformation_net'):]
                    elif suffix.startswith('amplitude_net'):
                        uq_name = 'hf_amplitude_net' + suffix[len('amplitude_net'):]
                    elif suffix.startswith('scale'):
                        uq_name = 'hf_scale'
                    else:
                        continue

                if uq_name not in uq_sd:
                    continue

                uq_param = uq_sd[uq_name]
                if param.shape == uq_param.shape:
                    uq_param.copy_(param)
                elif 'weight_ih_l0' in uq_name:
                    S = param.shape[1]
                    uq_param[:, :S].copy_(param)

        print("  Initialized UQ model from deterministic DA-SHRED weights")


 
# 4. TRAINING FUNCTIONS
 

# --- Deterministic DA-SHRED training (standard backbone) ---

def train_lf_shred(model, train_loader, epochs=200, lr=1e-3):
    print("\n=== Stage 1: Train LF-SHRED on Simulation ===")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train(); total_loss = 0
        for s, t in train_loader:
            s, t = s.to(device), t.to(device); optimizer.zero_grad()
            p, _ = model(s); loss = F.mse_loss(p, t); loss.backward(); optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
    return model


def train_gan(dashred, z_sim, z_real, epochs=300, lr=1e-4):
    print("\n=== Stage 2: Train GAN ===")
    z_s = torch.tensor(z_sim, dtype=torch.float32).to(device)
    z_r = torch.tensor(z_real, dtype=torch.float32).to(device)
    opt_g = optim.Adam(dashred.gan.generator.parameters(), lr=lr)
    opt_d = optim.Adam(dashred.gan.discriminator.parameters(), lr=lr)
    bs = 32; nb = min(len(z_s), len(z_r)) // bs
    for epoch in range(epochs):
        ps, pr = torch.randperm(len(z_s)), torch.randperm(len(z_r))
        for i in range(nb):
            zs = z_s[ps[i*bs:(i+1)*bs]]; zr = z_r[pr[i*bs:(i+1)*bs]]
            opt_d.zero_grad(); zf = dashred.gan(zs)
            dl = (F.binary_cross_entropy_with_logits(dashred.gan.discriminator(zr), torch.ones(len(zr),1,device=device)) +
                  F.binary_cross_entropy_with_logits(dashred.gan.discriminator(zf.detach()), torch.zeros(len(zs),1,device=device)))
            dl.backward(); opt_d.step()
            opt_g.zero_grad(); zf = dashred.gan(zs)
            gl = F.binary_cross_entropy_with_logits(dashred.gan.discriminator(zf), torch.ones(len(zs),1,device=device))
            gl.backward(); opt_g.step()
        if (epoch+1) % 100 == 0: print(f"  Epoch {epoch+1}/{epochs}, G:{gl.item():.4f}, D:{dl.item():.4f}")


def train_hf_sparse(dashred, loader, sensor_indices, epochs=500, lr=1e-3,
                    lambda_sparse=0.1, stage_name="Stage 3"):
    max_freq = len(sensor_indices) // 2
    print(f"\n=== {stage_name}: HF-SHRED (Bandlimited Sparsity) ===")
    params = list(dashred.hf_shred.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)
    sidx = torch.tensor(sensor_indices, dtype=torch.long).to(device)
    with torch.no_grad():
        ss, _ = next(iter(loader)); ss = ss.to(device)
        _, ul, _, _, _ = dashred(ss, use_gan=True)
        rs = (ss[:, -1, :] - ul[:, sidx]).abs().mean().item()
    max_hf = rs * 3; warmup = 100
    for epoch in range(epochs):
        dashred.train(); esl = 0
        cl = 0.0 if epoch < warmup else lambda_sparse * min(1.0, (epoch - warmup) / (epochs - warmup) * 2)
        for s, _ in loader:
            s = s.to(device); optimizer.zero_grad()
            _, ul, uh, _, _ = dashred(s, use_gan=True)
            sc = s[:, -1, :]; sl = ul[:, sidx].detach(); rt = sc - sl; sh = uh[:, sidx]
            sl2 = F.mse_loss(sh, rt)
            sp = frequency_sparsity_bandlimited(uh, max_freq)
            mp = F.relu(uh.abs().mean() - max_hf) ** 2
            loss = sl2 + cl * sp + mp; loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step(); esl += sl2.item()
        esl /= len(loader); scheduler.step(esl)
        if (epoch+1) % 100 == 0: print(f"  Epoch {epoch+1}/{epochs}, Sensor: {esl:.6f}")
    return dashred


# --- UQ fine-tuning (energy loss) ---

def energy_loss(y_true, y_pred1, y_pred2, beta=1):
    s1 = (torch.norm(y_pred1 - y_true, dim=1).pow(beta).mean() +
          torch.norm(y_pred2 - y_true, dim=1).pow(beta).mean()) / 2
    s2 = torch.norm(y_pred1 - y_pred2, dim=1).pow(beta).mean()
    return s1 - s2 / 2


def train_uq_endtoend(uq_model, loader_real, epochs=1000, lr=1e-4, patience=10):
    """Fine-tune full UQ-DA-SHRED end-to-end with energy loss on U_real."""
    print(f"\n=== UQ Fine-tuning: energy loss on U_real ({epochs} epochs) ===")
    optimizer = optim.Adam(uq_model.parameters(), lr=lr)
    val_errors = []
    patience_counter = 0
    best_params = deepcopy(uq_model.state_dict())

    for epoch in range(1, epochs + 1):
        uq_model.train(); total_loss = 0; n_batches = 0
        for s, t in loader_real:
            s, t = s.to(device), t.to(device)
            optimizer.zero_grad()
            ut1, _, _, _, _ = uq_model(s, use_gan=True)
            ut2, _, _, _, _ = uq_model(s, use_gan=True)
            loss = energy_loss(t, ut1, ut2)
            loss.backward(); optimizer.step()
            total_loss += loss.item(); n_batches += 1

        if epoch % 20 == 0 or epoch == 1:
            uq_model.eval()
            with torch.no_grad():
                all_preds = []
                for s, t in loader_real:
                    s = s.to(device)
                    preds = torch.stack([uq_model(s, use_gan=True)[0]
                                         for _ in range(20)]).mean(0)
                    all_preds.append((preds, t.to(device)))
                pred_cat = torch.cat([p for p, _ in all_preds])
                true_cat = torch.cat([t for _, t in all_preds])
                ve = (torch.linalg.norm(pred_cat - true_cat) /
                      torch.linalg.norm(true_cat)).item()
                val_errors.append(ve)

            if epoch % 100 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}: val_error = {ve:.6f}, "
                      f"loss = {total_loss/n_batches:.6f}")

            if ve <= min(val_errors):
                patience_counter = 0
                best_params = deepcopy(uq_model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    uq_model.load_state_dict(best_params)
    return val_errors


 
# 5. METRICS & VISUALIZATION  (imported from uq_viz)
 


 
# 7. MAIN
 

def main():
    parser = argparse.ArgumentParser(description="UQ-DA-SHRED")
    parser.add_argument("--data_dir", type=str, default="DA_data")
    parser.add_argument("--results_dir", type=str, default="uq_dashred_results")
    parser.add_argument("--noise_dim", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--epochs_lf", type=int, default=300)
    parser.add_argument("--epochs_gan", type=int, default=400)
    parser.add_argument("--epochs_hf", type=int, default=500)
    parser.add_argument("--epochs_hf_ft", type=int, default=200)
    parser.add_argument("--epochs_uq", type=int, default=1000)
    parser.add_argument("--lags", type=int, default=25)
    parser.add_argument("--num_sensors", type=int, default=25)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--lambda_sparse", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    #  ==
    # [1] LOAD DATA
    #  ==
    print("[1] Loading data …")
    U_sim = np.load(os.path.join(args.data_dir, "Koch_model_processed.npy"), allow_pickle=True)
    U_real = np.load(os.path.join(args.data_dir, "high_fidelity_sim_processed.npy"), allow_pickle=True)
    N = U_sim.shape[1]
    print(f"  U_sim: {U_sim.shape}, U_real: {U_real.shape}, N={N}")

    sensor_indices = np.linspace(0, N - 1, args.num_sensors, dtype=int)

    U_comb = np.vstack([U_sim, U_real])
    tmp = TimeSeriesDataset(U_comb, sensor_indices, args.lags, fit_scaler=True)
    scaler_U, scaler_S = tmp.get_scalers()

    n_train = int(0.8 * len(U_sim))
    tr_sim = TimeSeriesDataset(U_sim[:n_train], sensor_indices, args.lags, scaler=(scaler_U, scaler_S))
    tr_real = TimeSeriesDataset(U_real[:n_train], sensor_indices, args.lags, scaler=(scaler_U, scaler_S))
    full_real = TimeSeriesDataset(U_real, sensor_indices, args.lags, scaler=(scaler_U, scaler_S))

    ld_sim = DataLoader(tr_sim, batch_size=args.batch_size, shuffle=True)
    ld_real = DataLoader(tr_real, batch_size=args.batch_size, shuffle=True)
    ld_real_full = DataLoader(full_real, batch_size=args.batch_size, shuffle=False)

    #  ==
    # [2] TRAIN DETERMINISTIC DA-SHRED BACKBONE
    #  ==
    print("\n" + "=" * 70)
    print("[2] TRAIN DETERMINISTIC DA-SHRED BACKBONE")
    print("=" * 70)

    lf_shred = SHRED(args.num_sensors, args.lags, args.hidden_size, N).to(device)
    lf_shred = train_lf_shred(lf_shred, ld_sim, epochs=args.epochs_lf)

    det_dashred = SparseFreqDASHRED(lf_shred, args.num_sensors, args.lags,
                                     args.hidden_size, N, sensor_indices).to(device)

    print("\n  Extracting latents for GAN …")
    det_dashred.eval(); Zs, Zr = [], []
    with torch.no_grad():
        for s, _ in ld_sim: Zs.append(det_dashred.encode_lf(s.to(device)).cpu())
        for s, _ in ld_real: Zr.append(det_dashred.encode_lf(s.to(device)).cpu())
    train_gan(det_dashred, torch.cat(Zs).numpy(), torch.cat(Zr).numpy(),
              epochs=args.epochs_gan)

    det_dashred = train_hf_sparse(det_dashred, ld_real, sensor_indices,
                                   epochs=args.epochs_hf, lambda_sparse=args.lambda_sparse)
    det_dashred = train_hf_sparse(det_dashred, ld_real, sensor_indices,
                                   epochs=args.epochs_hf_ft, lr=1e-4,
                                   lambda_sparse=args.lambda_sparse * 0.1,
                                   stage_name="Stage 3b: Fine-tune")

    #  ==
    # [3] INITIALIZE UQ MODEL & FINE-TUNE WITH ENGRESSION
    #  ==
    print("\n" + "=" * 70)
    print("[3] UQ FINE-TUNING WITH ENGRESSION")
    print("=" * 70)

    uq_model = UQ_SparseFreqDASHRED(
        args.num_sensors, args.lags, args.hidden_size, N,
        sensor_indices, noise_dim=args.noise_dim).to(device)
    uq_model.init_from_deterministic(det_dashred)
    print(f"  Total params: {sum(p.numel() for p in uq_model.parameters()):,}")

    val_errors = train_uq_endtoend(uq_model, ld_real, epochs=args.epochs_uq,
                                    lr=1e-4, patience=10)

    #  ==
    # [4] INFERENCE
    #  ==
    print(f"\n{'='*70}")
    print(f"[4] INFERENCE ({args.n_samples} samples)")
    print(f"{'='*70}")

    uq_model.eval()
    all_total, all_lf = [], []
    with torch.no_grad():
        for s, t in ld_real_full:
            s = s.to(device)
            st, sl = uq_model.sample(s, n_samples=args.n_samples, use_gan=True)
            all_total.append(st.cpu()); all_lf.append(sl.cpu())

    samples_total = torch.cat(all_total, dim=1).numpy()
    samples_lf = torch.cat(all_lf, dim=1).numpy()

    samples_total_orig = np.array([scaler_U.inverse_transform(s) for s in samples_total])
    samples_lf_orig = np.array([scaler_U.inverse_transform(s) for s in samples_lf])
    N_frames = samples_total.shape[1]
    gt_orig = U_real[args.lags:][:N_frames]

    print(f"  LF samples: {samples_lf_orig.shape}")
    print(f"  LF+HF samples: {samples_total_orig.shape}")

    # Quantiles (including 70% = 15th–85th)
    q_levels = [0.025, 0.15, 0.25, 0.5, 0.75, 0.85, 0.975]
    q_lf = {q: np.percentile(samples_lf_orig, 100*q, axis=0) for q in q_levels}
    q_lfhf = {q: np.percentile(samples_total_orig, 100*q, axis=0) for q in q_levels}
    std_lf = samples_lf_orig.std(axis=0)
    std_lfhf = samples_total_orig.std(axis=0)
    times = np.arange(N_frames) * args.dt + args.lags * args.dt

    #  ==
    # [5] METRICS
    #  ==
    print(f"\n{'='*70}")
    print("  METRICS")
    print(f"{'='*70}")

    med_lf = q_lf[0.5]; med_lfhf = q_lfhf[0.5]
    rmse_lf = np.sqrt(np.mean((gt_orig - med_lf) ** 2))
    rmse_lfhf = np.sqrt(np.mean((gt_orig - med_lfhf) ** 2))
    print(f"\n  {'':20s} {'RMSE':>10s}")
    print(f"  {'UQ-LF (median)':20s} {rmse_lf:10.6f}")
    print(f"  {'UQ-LF+HF (median)':20s} {rmse_lfhf:10.6f}")

    for label, samples in [("LF", samples_lf_orig), ("LF+HF", samples_total_orig)]:
        calib = calibration_scores(samples, gt_orig)
        crps = compute_crps(samples, gt_orig)
        s95 = compute_sharpness(samples, 0.95)
        s70 = compute_sharpness(samples, 0.70)
        s50 = compute_sharpness(samples, 0.50)
        print(f"\n  {label}:")
        print(f"    Calibration: " +
              ", ".join(f"{k*100:.0f}%→{v*100:.1f}%" for k, v in calib.items()))
        print(f"    CRPS: {crps:.6f}")
        print(f"    Sharpness: 95% CI={s95:.6f}, 70% CI={s70:.6f}, 50% CI={s50:.6f}")

    #  ==
    # [6] PLOTS
    #  ==
    print(f"\n{'='*70}")
    print("  GENERATING PLOTS")
    print(f"{'='*70}")

    plot_dual_snapshots(gt_orig, q_lf, q_lfhf, times, N,
                        results_dir / "uq_dashred_snapshots.png")
    plot_dual_timeseries(gt_orig, q_lf, q_lfhf, times, N,
                         results_dir / "uq_dashred_timeseries.png")
    plot_calibration_dual(samples_lf_orig, samples_total_orig, gt_orig,
                          results_dir / "uq_dashred_calibration.png")
    plot_error_vs_uncertainty(gt_orig, med_lf, std_lf, med_lfhf, std_lfhf,
                              results_dir / "uq_dashred_error_vs_unc.png")
    plot_training_curve(val_errors, results_dir / "training_curve.png",
                        title="UQ End-to-End Fine-tuning Curve")

    print(f"\n{'='*70}")
    print(f"  UQ-DA-SHRED Complete!")
    print(f"  Results: {results_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
