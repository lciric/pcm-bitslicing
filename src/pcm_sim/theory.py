"""Analytical MVM error predictions — Le Gallo et al. (2022) equation (8)."""

import math


def equal_fill_theory(eta_s, base, n_w):
    """Predicted η from equal-fill slicing (equation 8).

    Parameters
    ----------
    eta_s : float
        Single-slice MVM error (baseline at n_W=1).
    base : int or float
        Base of the weight slices b_W.
    n_w : int
        Number of weight slices n_W.

    Returns
    -------
    float
        Predicted relative MVM error η.
    """
    if n_w == 1:
        return eta_s
    if base > 1:
        return eta_s * math.sqrt(
            (1 - base) * (1 + base ** n_w)
            / ((1 + base) * (1 - base ** n_w))
        )
    return eta_s / math.sqrt(n_w)
