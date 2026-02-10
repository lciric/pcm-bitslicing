#!/usr/bin/env python3
"""BitNet PCM extension — ternary vs. standard weight quantisation.

Compares MVM error over time for {-1, 0, +1} ternary weights (BitNet-style)
against full-precision INT-9 weights on the same PCM crossbar simulator.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pcm_sim.engine import run_trial_bitnet

# ── configuration ───────────────────────────────────────────────
N_TRIALS = 200
T0 = 20.0
YEAR = 365 * 86400
NUM_T = 12
TIME_POINTS = np.logspace(np.log10(T0), np.log10(YEAR), NUM_T)

CONFIGS = [
    ("Max-fill EC", "dependent", 8),
    ("Max-fill",    "MaxFill",   8),
    ("Equal-fill",  "EqualFill", 4),
    ("$n_W$=1",     "EqualFill", 1),
]

COLORS = {
    "Max-fill EC": "#2ca02c",
    "Max-fill":    "#1f77b4",
    "Equal-fill":  "#d62728",
    "$n_W$=1":     "#ff7f0e",
}

# ── sweep ───────────────────────────────────────────────────────
results = {}
t_start = time.time()

for label, algo, nw in CONFIGS:
    for ternary in [False, True]:
        key = f"{label} ({'ternary' if ternary else 'standard'})"
        print(f"{key} ...", end=" ", flush=True)
        seeds = np.random.randint(0, 2**31, N_TRIALS)
        all_err = Parallel(n_jobs=-1)(
            delayed(run_trial_bitnet)(int(s), algo, nw, ternary, TIME_POINTS)
            for s in seeds)
        arr = np.array(all_err)
        results[key] = {"mean": arr.mean(0), "std": arr.std(0)}
        print(f"η(t0)={arr[:, 0].mean():.4f} → η(1yr)={arr[:, -1].mean():.4f}")

print(f"\nTotal: {(time.time() - t_start) / 60:.1f} min")

# ── plot ────────────────────────────────────────────────────────
DAY, MONTH = 86400, 30 * 86400
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, wtype, title in [
    (ax1, "standard", "Standard INT-9 weights"),
    (ax2, "ternary",  "Ternary {-1, 0, +1} weights"),
]:
    for label, _, _ in CONFIGS:
        key = f"{label} ({wtype})"
        m = results[key]["mean"]
        s = results[key]["std"]
        ax.semilogx(TIME_POINTS, m, "o-", color=COLORS[label],
                     label=label, ms=4, lw=1.5)
        ax.fill_between(TIME_POINTS, m - s, m + s,
                        color=COLORS[label], alpha=0.12)

    for t, tl in [(DAY, "1 day"), (MONTH, "1 month"), (YEAR, "1 year")]:
        ax.axvline(t, color="grey", ls=":", alpha=0.4)
    ax.set_xlabel("Time after programming (s)")
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

ax1.set_ylabel("Relative MVM error")
fig.suptitle(
    f"BitNet PCM extension — 128×128 crossbar, {N_TRIALS} trials, mean ± 1σ",
    fontsize=13, y=1.01)
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/bitnet_pcm_extension.png", dpi=150, bbox_inches="tight")
print("Saved: figures/bitnet_pcm_extension.png")
