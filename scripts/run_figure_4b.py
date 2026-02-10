#!/usr/bin/env python3
"""Figure 4(b) — Relative MVM error vs. base of weight slices b_W.

Two panels: at t₀ = 20 s (left) and at 1 month (right).
Reproduces Le Gallo et al. (2022), Fig. 4b.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pcm_sim.engine import run_trial_4b

# ── configuration ───────────────────────────────────────────────
N_TRIALS = 300
BASE_LIST = [1, 2, 3, 4, 5, 6, 7, 8]

CONFIGS = [
    ("Equal-fill, $n_W$=2",       "EqualFill", 2),
    ("Max-fill, $n_W$=2",         "MaxFill",   2),
    ("Max-fill with EC, $n_W$=2", "dependent",  2),
    ("Equal-fill, $n_W$=4",       "EqualFill", 4),
    ("Max-fill, $n_W$=4",         "MaxFill",   4),
    ("Max-fill with EC, $n_W$=4", "dependent",  4),
]

COLORS = {
    "Equal-fill, $n_W$=2":       "#d62728",
    "Max-fill, $n_W$=2":         "#1f77b4",
    "Max-fill with EC, $n_W$=2": "#2ca02c",
    "Equal-fill, $n_W$=4":       "#ff7f0e",
    "Max-fill, $n_W$=4":         "#9467bd",
    "Max-fill with EC, $n_W$=4": "#8c564b",
}
LSTYLES = {2: "-o", 4: "--s"}

# ── sweep ───────────────────────────────────────────────────────
results = {}
t_start = time.time()

for label, algo, nw in CONFIGS:
    results[label] = {k: np.zeros(len(BASE_LIST)) for k in
                      ["mean_t0", "std_t0", "mean_1m", "std_1m"]}
    for i, bw in enumerate(BASE_LIST):
        print(f"{label}, b_W={bw} ...", end=" ", flush=True)
        seeds = np.random.randint(0, 2**31, N_TRIALS)
        res = Parallel(n_jobs=-1)(
            delayed(run_trial_4b)(int(s), algo, bw, nw) for s in seeds)
        t0 = np.array([r["t0"] for r in res])
        m1 = np.array([r["1month"] for r in res])
        results[label]["mean_t0"][i] = np.mean(t0)
        results[label]["std_t0"][i]  = np.std(t0)
        results[label]["mean_1m"][i] = np.mean(m1)
        results[label]["std_1m"][i]  = np.std(m1)
        print(f"η(t0)={results[label]['mean_t0'][i]:.4f}")

print(f"\nTotal: {(time.time() - t_start) / 60:.1f} min")

# ── plot ────────────────────────────────────────────────────────
bw = np.array(BASE_LIST)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, sfx, title in [
    (ax1, "t0", "at $t_0$"),
    (ax2, "1m", "at 1 month"),
]:
    for label, algo, nw in CONFIGS:
        ls = LSTYLES[nw]
        ax.errorbar(bw, results[label][f"mean_{sfx}"],
                    yerr=results[label][f"std_{sfx}"],
                    fmt=ls, color=COLORS[label], ms=5, capsize=3,
                    capthick=0.8, elinewidth=0.8, lw=1.5, label=label)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Base of weight slices ($b_W$)")
    ax.set_ylabel("Relative MVM error")
    ax.set_xticks(bw)
    ax.grid(alpha=0.3)

handles, labels_ = ax2.get_legend_handles_labels()
fig.legend(handles, labels_, loc="center right",
           bbox_to_anchor=(1.25, 0.5), fontsize=9, frameon=True)
plt.suptitle("Figure 4(b)", fontsize=14, y=1.01)
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/figure_4b.png", dpi=150, bbox_inches="tight")
print("Saved: figures/figure_4b.png")
