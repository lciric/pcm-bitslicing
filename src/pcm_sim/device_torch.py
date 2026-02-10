"""PCM device physics — PyTorch implementation for DNN inference (Figure 5)."""

import math
import torch

# Config shared by Figure 5
N_STD_W = 2.0       # Joshi 2020: α=2.0 for CIFAR-10
NOISE_STD = 0.038   # 3.8% noise injection during training
T0 = 20.0           # Reference read time (s)
G_MAX = 25.0        # Max conductance (µS)
T_READ = 250e-9     # Read pulse duration (s)
GDC_NU = 0.049      # Global drift compensation exponent
N_MC_RUNS = 100     # Default Monte Carlo runs

_PROG_COEFFS = (-1.1731, 1.9650, 0.2635)


def programming_noise_std(g_norm):
    """State-dependent programming noise σ (normalised conductance)."""
    g = torch.abs(g_norm)
    c0, c1, c2 = _PROG_COEFFS
    return (c0 * g ** 2 + c1 * g + c2) / G_MAX


def sample_drift_exponent(g_norm):
    """Sample drift exponent ν(G_T) — stored once per chip programming."""
    log_g = torch.log(torch.abs(g_norm) + 1e-9)
    mu = torch.clamp(-0.0155 * log_g + 0.0244, 0.049, 0.1)
    sigma = torch.clamp(-0.0125 * log_g - 0.0059, 0.008, 0.045)
    nu = mu + sigma * torch.randn_like(g_norm)
    return torch.clamp(nu, 0.049, 0.1)


def read_noise_qs(g_norm):
    """1/f read noise magnitude Q_s(G)."""
    return torch.minimum(
        0.0088 * torch.reciprocal(torch.abs(g_norm) + 1e-9) ** 0.65,
        torch.tensor(0.2, device=g_norm.device),
    )


def f_integral(t):
    """Time-dependent 1/f noise integral."""
    return math.sqrt(max(math.log((t + T_READ) / (2 * T_READ)), 0))
