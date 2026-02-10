"""PCM device physics — doped-GST mushroom-type, Nandakumar et al. (2020).

All functions operate on numpy arrays normalised to G_max = 25 µS.
The polynomial coefficients and drift/noise models are calibrated from
a million-device array in 90 nm CMOS technology.
"""

import numpy as np

# Programming noise polynomial (µS) — fitted on iterative program-and-verify
_PROG_POLY = np.poly1d([-1.1731, 1.9650, 0.2635])

# Read pulse duration for 1/f noise integral
T_READ = 250e-9  # seconds


def prog_std(G):
    """State-dependent programming noise σ_prog(G_T) in µS."""
    return _PROG_POLY(np.abs(G))


def nu_mean(G):
    """Mean drift exponent µ_ν(G_T)."""
    return np.clip(-0.0155 * np.log(np.abs(G) + 1e-15) + 0.0244, 0.049, 0.1)


def nu_std(G):
    """Std of drift exponent σ_ν(G_T)."""
    return np.clip(-0.0125 * np.log(np.abs(G) + 1e-15) - 0.0059, 0.008, 0.045)


def qs_f(G):
    """1/f read noise magnitude Q_s(G_T)."""
    return np.minimum(0.0088 * np.reciprocal(np.abs(G) + 1e-15) ** 0.65, 0.2)


def f_integral(t):
    """Time-dependent 1/f noise integral."""
    return np.sqrt(np.log((t + T_READ) / (2 * T_READ)))


def bitget(A, k):
    """Extract bit *k* (1-indexed) from integer array *A*."""
    return (np.uint64(A) >> np.uint64(k - 1)) & np.uint64(1)
