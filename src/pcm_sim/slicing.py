"""Weight slicing algorithms for analog PCM crossbar arrays.

Implements Equal-fill, Max-fill, Max-fill with EC (dependent),
and Digital (positional) slicing from Le Gallo et al. (2022) §2.
Also supports ternary weight quantisation for BitNet-style models.
"""

import math
import numpy as np
from . import device


def slice_weights(lp_w, algorithm, w_mode, n_slices, base_w, range_bs_w,
                  range_bs_cond, N=None):
    """Slice an integer weight matrix into conductance targets + programmed values.

    Parameters
    ----------
    lp_w : ndarray (N, N)
        Signed integer weight matrix (after quantisation).
    algorithm : str
        'EqualFill', 'MaxFill', or 'dependent' (Max-fill with EC).
    w_mode : str
        'equal' (b_W=1), 'varying' (b_W>1), or 'positional' (digital bit extraction).
    n_slices : int
        Number of weight slices n_W.
    base_w : float
        Base of the weight slices b_W.
    range_bs_w : float
        Slice value range r_{s_W}.
    range_bs_cond : float
        Conductance range G_max (in S) at the read time.
    N : int, optional
        Crossbar dimension (inferred from lp_w if None).

    Returns
    -------
    cond_T : list of ndarray
        Target conductance arrays (LSB-first).
    cond_P : list of ndarray
        Programmed conductance arrays (with programming noise, LSB-first).
    """
    if N is None:
        N = lp_w.shape[0]
    signs = np.sign(lp_w).astype(float)
    tmp = np.abs(lp_w).astype(float)
    cT, cP = [], []

    def _pn(S, ct):
        mask = (S != 0).astype(float)
        return mask * np.random.normal(0, 1, lp_w.shape) * device.prog_std(ct / range_bs_cond) * 1e-6

    # --- Digital (positional) slicing ---
    if w_mode == "positional":
        N_W = int(round(math.log2(np.max(np.abs(lp_w)) + 1))) + 1
        bits_per_slice = math.ceil((N_W - 1) / n_slices)
        abs_int = np.abs(lp_w)
        k = 1
        for _ in range(n_slices):
            S = np.zeros_like(lp_w, dtype=float)
            for b in range(bits_per_slice):
                if k <= N_W - 1:
                    S += 2 ** b * device.bitget(abs_int, k)
                    k += 1
            ct = signs * S * range_bs_cond / range_bs_w
            eps = _pn(S, ct)
            cT.append(ct)
            cP.append(ct + eps)
        return cT, cP

    # --- Equal significance (base=1) ---
    if w_mode == "equal":
        if algorithm == "EqualFill":
            B = tmp / n_slices
            for _ in range(n_slices):
                ct = signs * B * range_bs_cond / range_bs_w
                eps = _pn(B, ct)
                cT.append(ct)
                cP.append(ct + eps)
        else:  # MaxFill or dependent
            for i in range(n_slices):
                if i == n_slices - 1:
                    S = tmp
                else:
                    A = np.floor(tmp / range_bs_w)
                    A[A >= 1] = 1
                    A[A < 1] = 0
                    S = range_bs_w * A
                ct = signs * S * range_bs_cond / range_bs_w
                eps = _pn(S, ct)
                cT.append(ct)
                cP.append(ct + eps)
                if i < n_slices - 1:
                    tmp = tmp - S
                    if algorithm == "dependent":
                        tmp -= signs * eps * range_bs_w / range_bs_cond

    # --- Varying significance (base>1) ---
    else:
        _cT, _cP = [], []
        if algorithm == "EqualFill":
            factor = (1 - base_w ** n_slices) / (1 - base_w)
            for _ in range(n_slices):
                S = tmp / factor
                ct = signs * S * range_bs_cond / range_bs_w
                eps = _pn(S, ct)
                _cT.append(ct)
                _cP.append(ct + eps)
            cT, cP = _cT[::-1], _cP[::-1]
        else:  # MaxFill or dependent
            for i in range(n_slices, 0, -1):
                if i == 1:
                    S = tmp
                else:
                    lc = range_bs_w * (1 - base_w ** (i - 1)) / (1 - base_w)
                    A = np.ceil(tmp / lc)
                    A[A <= 1] = 0
                    A[A > 1] = 1
                    S = (tmp / base_w ** (i - 1)) * A
                    S = np.minimum(S, range_bs_w)
                ct = signs * S * range_bs_cond / range_bs_w
                eps = _pn(S, ct)
                _cT.append(ct)
                _cP.append(ct + eps)
                if i > 1:
                    tmp = tmp - base_w ** (i - 1) * S
                    if algorithm == "dependent":
                        tmp -= base_w ** (i - 1) * signs * eps * range_bs_w / range_bs_cond
            cT, cP = _cT[::-1], _cP[::-1]

    return cT, cP


def quantize_ternary(W):
    """Quantize weight matrix to {-1, 0, +1} using threshold α · mean(|W|).

    Parameters
    ----------
    W : ndarray
        Floating-point weight matrix.

    Returns
    -------
    W_t : ndarray
        Ternary weight matrix in {-1, 0, +1}.
    alpha : float
        Scaling factor (mean of magnitudes above threshold).
    """
    alpha = 0.7 * np.mean(np.abs(W))
    W_t = np.zeros_like(W)
    W_t[W > alpha] = 1
    W_t[W < -alpha] = -1
    alpha = np.mean(np.abs(W[np.abs(W) > alpha])) if np.any(np.abs(W) > alpha) else 1.0
    return W_t, alpha
