"""Monte Carlo MVM trial engines for PCM crossbar simulations.

Each ``run_trial_*`` function executes a single Monte Carlo trial
(input vector + weight matrix → analogue MVM with PCM noise) and
returns the relative error η.
"""

import math
import numpy as np
from . import device
from .slicing import slice_weights, quantize_ternary

# ──────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────

def _make_data(N, N_INP, N_STD, N_STD_W, RANGE_INP, RANGE_W):
    """Generate Gaussian input vector and weight matrix, quantised to INT."""
    scale_inp = RANGE_INP / N_STD
    scale_w = RANGE_W / N_STD_W
    hp_w = np.random.normal(0, 1, (N, N))
    hp_in = np.random.normal(0, 1, N)
    lp_in = np.clip(np.rint(hp_in * scale_inp), -RANGE_INP, RANGE_INP)
    lp_w = np.clip(np.rint(hp_w * scale_w), -RANGE_W, RANGE_W)
    hp_out = lp_w.astype(float) @ (lp_in.astype(float) / (scale_inp * scale_w))
    return lp_in, lp_w, hp_out, scale_inp, scale_w


def _input_slice(lp_in, N_INP, n_bs_inp):
    """Bit-slice the input vector into a single slice (n_bs_inp=1 default)."""
    bits = math.ceil((N_INP - 1) / n_bs_inp)
    k = 1
    S = np.zeros_like(lp_in, dtype=float)
    for _ in range(bits):
        S += 2 ** (_ ) * device.bitget(np.abs(lp_in), k)
        k += 1
    return np.sign(lp_in) * S


def _mvm_at_time(cond_T, cond_P, nus, sl_in, sl_cal, N, base_w,
                  n_bs_w, T0, t_val, scale_out, RANGE_OUT, cal_ref=None):
    """Compute MVM at a given time with drift + 1/f noise + GDC."""
    RANGE_BS_COND = cond_T[0].max() if cond_T else 1.0  # not used for qs_f normalisation
    lp_out = np.zeros(N)
    lp_out_cal = np.zeros(N)

    for j in range(n_bs_w):
        cond = cond_P[j] * (t_val / T0) ** (-nus[j]) if t_val > T0 else cond_P[j].copy()
        G_t = cond_T[j] / (np.max(np.abs(cond_T[j])) + 1e-15)
        cond += cond * device.qs_f(G_t) * device.f_integral(t_val) * np.random.normal(0, 1, cond.shape)
        adc = np.clip(np.rint(cond @ (sl_in * scale_out)), -RANGE_OUT, RANGE_OUT)
        adc_cal = np.clip(np.rint(cond @ (sl_cal * scale_out)), -RANGE_OUT, RANGE_OUT)
        lp_out += adc * base_w ** j
        lp_out_cal += adc_cal * base_w ** j

    if cal_ref is None:
        cal_ref = np.sum(np.abs(lp_out_cal))
    beta = cal_ref / (np.sum(np.abs(lp_out_cal)) + 1e-15)
    lp_out *= beta
    return lp_out, cal_ref


# ──────────────────────────────────────────────────────────────────
# Figure 4a — η(n_W) at t0 and 1 month
# ──────────────────────────────────────────────────────────────────

def run_trial_4a(seed, algorithm, w_mode, base, n_bs_w, *,
                 N=128, N_STD=4.0, N_STD_W=3.0, N_INP=9, N_W=9, N_OUT=8, T0=20.0):
    """Single trial for Figure 4a: returns {'t0': η, '1month': η}."""
    np.random.seed(seed)
    RANGE_INP = 2 ** (N_INP - 1) - 1
    RANGE_W = 2 ** (N_W - 1) - 1
    RANGE_OUT = 2 ** (N_OUT - 1) - 1
    RANGE_BS_COND = (25.0 * (T0 / 20.0) ** (-0.049)) * 1e-6
    MONTH = 86400 * 30
    n_bs_inp = 1

    # Base & range
    if w_mode == "positional":
        base_w = 2 ** math.ceil((N_W - 1) / n_bs_w)
        range_bs_w = base_w - 1
    elif w_mode == "equal":
        base_w = 1
        range_bs_w = RANGE_W / n_bs_w
    else:
        base_w = base
        range_bs_w = RANGE_W / ((1 - base_w ** n_bs_w) / (1 - base_w))

    base_inp = 2 ** math.ceil((N_INP - 1) / n_bs_inp)
    range_bs_inp = base_inp - 1
    scale_inp = RANGE_INP / N_STD
    scale_w = RANGE_W / N_STD_W
    scale_out = ((RANGE_OUT / N_STD) / (range_bs_inp * RANGE_BS_COND * math.sqrt(N))
                 * ((N_STD - 1.5) / n_bs_inp + 1.5)
                 * ((N_STD_W - 1.5) / n_bs_w + 1.5))

    # Data
    lp_in, lp_w, hp_out, _, _ = _make_data(N, N_INP, N_STD, N_STD_W, RANGE_INP, RANGE_W)
    lp_cal = np.clip(np.rint(np.ones(N) * scale_inp), -RANGE_INP, RANGE_INP)
    sl_in = _input_slice(lp_in, N_INP, n_bs_inp)
    sl_cal = _input_slice(lp_cal, N_INP, n_bs_inp)

    # Slicing
    cond_T, cond_P = slice_weights(lp_w, algorithm, w_mode, n_bs_w, base_w,
                                    range_bs_w, RANGE_BS_COND, N)
    nus = [np.abs(device.nu_mean(cond_T[j] / RANGE_BS_COND)
                  + device.nu_std(cond_T[j] / RANGE_BS_COND)
                  * np.random.normal(0, 1, (N, N)))
           for j in range(n_bs_w)]

    errors = {}
    cal_ref = None
    for t_name, t_val in [("t0", T0), ("1month", float(MONTH))]:
        lp_out, cal_ref = _mvm_at_time(
            cond_T, cond_P, nus, sl_in, sl_cal, N, base_w, n_bs_w,
            T0, t_val, scale_out, RANGE_OUT, cal_ref)
        final = lp_out / (scale_out * scale_inp * scale_w) * (range_bs_w / RANGE_BS_COND)
        errors[t_name] = np.linalg.norm(final - hp_out) / np.linalg.norm(hp_out)
    return errors


# ──────────────────────────────────────────────────────────────────
# Figure 4b — η(b_W) at t0 and 1 month
# ──────────────────────────────────────────────────────────────────

def run_trial_4b(seed, algorithm, base, n_bs_w, *,
                 N=128, N_STD=4.0, N_STD_W=3.0, N_INP=9, N_W=9, N_OUT=8, T0=20.0):
    """Single trial for Figure 4b: returns {'t0': η, '1month': η}."""
    np.random.seed(seed)
    RANGE_INP = 2 ** (N_INP - 1) - 1
    RANGE_W = 2 ** (N_W - 1) - 1
    RANGE_OUT = 2 ** (N_OUT - 1) - 1
    RANGE_BS_COND = (25.0 * (T0 / 20.0) ** (-0.049)) * 1e-6
    MONTH = 86400 * 30
    n_bs_inp = 1

    if base == 1:
        w_mode = "equal"
        base_w = 1
        range_bs_w = RANGE_W / n_bs_w
    else:
        w_mode = "varying"
        base_w = base
        range_bs_w = RANGE_W / ((1 - base_w ** n_bs_w) / (1 - base_w))

    base_inp = 2 ** math.ceil((N_INP - 1) / n_bs_inp)
    range_bs_inp = base_inp - 1
    scale_inp = RANGE_INP / N_STD
    scale_w = RANGE_W / N_STD_W
    scale_out = ((RANGE_OUT / N_STD) / (range_bs_inp * RANGE_BS_COND * math.sqrt(N))
                 * ((N_STD - 1.5) / n_bs_inp + 1.5)
                 * ((N_STD_W - 1.5) / n_bs_w + 1.5))

    lp_in, lp_w, hp_out, _, _ = _make_data(N, N_INP, N_STD, N_STD_W, RANGE_INP, RANGE_W)
    lp_cal = np.clip(np.rint(np.ones(N) * scale_inp), -RANGE_INP, RANGE_INP)
    sl_in = _input_slice(lp_in, N_INP, n_bs_inp)
    sl_cal = _input_slice(lp_cal, N_INP, n_bs_inp)

    cond_T, cond_P = slice_weights(lp_w, algorithm, w_mode, n_bs_w, base_w,
                                    range_bs_w, RANGE_BS_COND, N)
    nus = [np.abs(device.nu_mean(cond_T[j] / RANGE_BS_COND)
                  + device.nu_std(cond_T[j] / RANGE_BS_COND)
                  * np.random.normal(0, 1, (N, N)))
           for j in range(n_bs_w)]

    errors = {}
    cal_ref = None
    for t_name, t_val in [("t0", T0), ("1month", float(MONTH))]:
        lp_out, cal_ref = _mvm_at_time(
            cond_T, cond_P, nus, sl_in, sl_cal, N, base_w, n_bs_w,
            T0, t_val, scale_out, RANGE_OUT, cal_ref)
        final = lp_out / (scale_out * scale_inp * scale_w) * (range_bs_w / RANGE_BS_COND)
        errors[t_name] = np.linalg.norm(final - hp_out) / np.linalg.norm(hp_out)
    return errors


# ──────────────────────────────────────────────────────────────────
# Figure 4c — η(t) time series
# ──────────────────────────────────────────────────────────────────

def run_trial_4c(seed, algorithm, w_mode, base, n_bs_w, time_points, *,
                 N=128, N_STD=4.0, N_STD_W=3.0, N_INP=9, N_W=9, N_OUT=8, T0=20.0):
    """Single trial for Figure 4c: returns list of η for each time point."""
    np.random.seed(seed)
    RANGE_INP = 2 ** (N_INP - 1) - 1
    RANGE_W = 2 ** (N_W - 1) - 1
    RANGE_OUT = 2 ** (N_OUT - 1) - 1
    RANGE_BS_COND = (25.0 * (T0 / 20.0) ** (-0.049)) * 1e-6
    n_bs_inp = 1

    if w_mode == "equal":
        base_w = 1
        range_bs_w = RANGE_W / n_bs_w
    else:
        base_w = base
        range_bs_w = RANGE_W / ((1 - base_w ** n_bs_w) / (1 - base_w))

    base_inp = 2 ** math.ceil((N_INP - 1) / n_bs_inp)
    range_bs_inp = base_inp - 1
    scale_inp = RANGE_INP / N_STD
    scale_w = RANGE_W / N_STD_W
    scale_out = ((RANGE_OUT / N_STD) / (range_bs_inp * RANGE_BS_COND * math.sqrt(N))
                 * ((N_STD - 1.5) / n_bs_inp + 1.5)
                 * ((N_STD_W - 1.5) / n_bs_w + 1.5))

    lp_in, lp_w, hp_out, _, _ = _make_data(N, N_INP, N_STD, N_STD_W, RANGE_INP, RANGE_W)
    lp_cal = np.clip(np.rint(np.ones(N) * scale_inp), -RANGE_INP, RANGE_INP)
    sl_in = _input_slice(lp_in, N_INP, n_bs_inp)
    sl_cal = _input_slice(lp_cal, N_INP, n_bs_inp)

    cond_T, cond_P = slice_weights(lp_w, algorithm, w_mode, n_bs_w, base_w,
                                    range_bs_w, RANGE_BS_COND, N)
    nus = [np.abs(device.nu_mean(cond_T[j] / RANGE_BS_COND)
                  + device.nu_std(cond_T[j] / RANGE_BS_COND)
                  * np.random.normal(0, 1, (N, N)))
           for j in range(n_bs_w)]

    errors = []
    cal_ref = None
    for t_val in time_points:
        lp_out, cal_ref = _mvm_at_time(
            cond_T, cond_P, nus, sl_in, sl_cal, N, base_w, n_bs_w,
            T0, t_val, scale_out, RANGE_OUT, cal_ref)
        final = lp_out / (scale_out * scale_inp * scale_w) * (range_bs_w / RANGE_BS_COND)
        errors.append(np.linalg.norm(final - hp_out) / np.linalg.norm(hp_out))
    return errors


# ──────────────────────────────────────────────────────────────────
# BitNet extension — ternary + standard MVM at multiple time points
# ──────────────────────────────────────────────────────────────────

def run_trial_bitnet(seed, algo, n_bs_w, ternary=False, time_points=None, *,
                     N=128, N_STD=4.0, N_STD_W=3.0, N_INP=9, N_W=9, N_OUT=8, T0=20.0):
    """Single trial for BitNet PCM extension: returns list of η per time point."""
    if time_points is None:
        time_points = np.logspace(np.log10(T0), np.log10(86400 * 365), 10)

    np.random.seed(seed)
    RANGE_INP = 2 ** (N_INP - 1) - 1
    RANGE_W = 2 ** (N_W - 1) - 1
    RANGE_OUT = 2 ** (N_OUT - 1) - 1
    RANGE_BS_COND = (25.0 * (T0 / 20.0) ** (-0.049)) * 1e-6
    n_bs_inp = 1

    scale_inp = RANGE_INP / N_STD
    scale_w = RANGE_W / N_STD_W
    hp_w = np.random.normal(0, 1, (N, N))
    hp_in = np.random.normal(0, 1, N)
    lp_in = np.clip(np.rint(hp_in * scale_inp), -RANGE_INP, RANGE_INP)

    if ternary:
        w_t, alpha = quantize_ternary(hp_w)
        lp_w = np.clip(np.rint(w_t * alpha * scale_w), -RANGE_W, RANGE_W)
    else:
        lp_w = np.clip(np.rint(hp_w * scale_w), -RANGE_W, RANGE_W)

    hp_out = lp_w.astype(float) @ (lp_in.astype(float) / (scale_inp * scale_w))

    # Equal significance, b_W = 1
    range_bs_w = RANGE_W / n_bs_w
    base_inp = 2 ** math.ceil((N_INP - 1) / n_bs_inp)
    range_bs_inp = base_inp - 1
    scale_out = ((RANGE_OUT / N_STD) / (range_bs_inp * RANGE_BS_COND * math.sqrt(N))
                 * ((N_STD - 1.5) / n_bs_inp + 1.5)
                 * ((N_STD_W - 1.5) / n_bs_w + 1.5))

    lp_cal = np.clip(np.rint(np.ones(N) * scale_inp), -RANGE_INP, RANGE_INP)
    sl_in = _input_slice(lp_in, N_INP, n_bs_inp)
    sl_cal = _input_slice(lp_cal, N_INP, n_bs_inp)

    cond_T, cond_P = slice_weights(lp_w, algo, "equal", n_bs_w, 1,
                                    range_bs_w, RANGE_BS_COND, N)
    nus = [np.abs(device.nu_mean(cond_T[j] / RANGE_BS_COND)
                  + device.nu_std(cond_T[j] / RANGE_BS_COND)
                  * np.random.normal(0, 1, (N, N)))
           for j in range(n_bs_w)]

    errors = []
    cal_ref = None
    for t_val in time_points:
        lp_out, cal_ref = _mvm_at_time(
            cond_T, cond_P, nus, sl_in, sl_cal, N, 1, n_bs_w,
            T0, t_val, scale_out, RANGE_OUT, cal_ref)
        final = lp_out / (scale_out * scale_inp * scale_w) * (range_bs_w / RANGE_BS_COND)
        errors.append(np.linalg.norm(final - hp_out) / np.linalg.norm(hp_out))
    return errors
