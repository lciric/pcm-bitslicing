#!/usr/bin/env python3
"""Figure 4(c) — Relative MVM error vs. time after programming.

Reproduces Le Gallo et al. (2022), Fig. 4c.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pcm_sim.engine import run_trial_4c

# ── configuration ───────────────────────────────────────────────
N_TRIALS = 200
T0 = 20.0
YEAR = 365 * 86400
NUM_T = 12
TIME_POINTS = np.logspace(np.log10(T0), np.log10(YEAR), NUM_T)

CONFIGS = []
for n_w in [1, 2, 4, 8]:
    CONFIGS.append((f"Max-fill EC, equal, $n_W$={n_w}", "dependent", "equal", 1, n_w))
    CONFIGS.append((f"Max-fill, equal, $n_W$={n_w}",    "MaxFill",   "equal", 1, n_w))
    CONFIGS.append((f"Equal-fill, equal, $n_W$={n_w}",  "EqualFill", "equal", 1, n_w))
CONFIGS.append(("Max-fill EC, varying $b_W$=2, $n_W$=2", "dependent", "varying", 2, 2))

COLOR_MAP = {
    "Max-fill EC, equal":  "#2ca02c",
    "Max-fill, equal":     "#1f77b4",
    "Equal-fill, equal":   "#d62728",
    "Max-fill EC, varying": "#9467bd",
}
NW_LS = {1: "-", 2: "--", 4: "-.", 8: ":"}

# ── sweep ───────────────────────────────────────────────────────
results = {}
t_start = time.time()

for label, algo, wm, base, nw in CONFIGS:
    print(f"{label} ...", end=" ", flush=True)
    seeds = np.random.randint(0, 2**31, N_TRIALS)
    all_err = Parallel(n_jobs=-1)(
        delayed(run_trial_4c)(int(s), algo, wm, base, nw, TIME_POINTS) for s in seeds)
    arr = np.array(all_err)  # (N_TRIALS, NUM_T)
    results[label] = {"mean": arr.mean(0), "std": arr.std(0)}
    print(f"η(t0)={arr[:, 0].mean():.4f} → η(1yr)={arr[:, -1].mean():.4f}")

print(f"\nTotal: {(time.time() - t_start) / 60:.1f} min")

# ── plot ────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

DAY, MONTH = 86400, 30 * 86400

for ax, configs_sub, title in [
    (ax1, [c for c in CONFIGS if "EC" in c[0]], "Max-fill with EC"),
    (ax2, [c for c in CONFIGS if "EC" not in c[0]], "Max-fill & Equal-fill"),
]:
    for label, algo, wm, base, nw in configs_sub:
        for key, col_key in COLOR_MAP.items():
            if key in label:
                color = col_key
                break
        m = results[label]["mean"]
        s = results[label]["std"]
        ax.semilogx(TIME_POINTS, m, NW_LS[nw], color=color, lw=1.5,
                     label=label)
        ax.fill_between(TIME_POINTS, m - s, m + s, color=color, alpha=0.1)

    for t, tl in [(DAY, "1 day"), (MONTH, "1 month"), (YEAR, "1 year")]:
        ax.axvline(t, color="grey", ls=":", alpha=0.4)
        ax.text(t, ax.get_ylim()[1], f" {tl}", fontsize=7, va="top", color="grey")
    ax.set_xlabel("Time after programming (s)")
    ax.set_ylabel("Relative MVM error")
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

plt.suptitle("Figure 4(c)", fontsize=14, y=1.01)
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/figure_4c.png", dpi=150, bbox_inches="tight")
print("Saved: figures/figure_4c.png")
