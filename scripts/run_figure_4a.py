#!/usr/bin/env python3
"""Figure 4(a) — Relative MVM error vs. number of weight slices.

Two panels: at t₀ = 20 s (left) and at 1 month (right).
Reproduces Le Gallo et al. (2022), Fig. 4a.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pcm_sim.engine import run_trial_4a
from pcm_sim.theory import equal_fill_theory

# ── configuration ───────────────────────────────────────────────
N_TRIALS = 200
N_W_LIST = [1, 2, 4, 8]

CONFIGS = [
    ("Digital slicing",           "MaxFill",   "positional", 2),
    ("$b_W$=2, Equal-fill",       "EqualFill", "varying",    2),
    ("$b_W$=2, Max-fill",         "MaxFill",   "varying",    2),
    ("$b_W$=2, Max-fill with EC", "dependent", "varying",    2),
    ("$b_W$=1, Equal-fill",       "EqualFill", "equal",      1),
    ("$b_W$=1, Max-fill",         "MaxFill",   "equal",      1),
    ("$b_W$=1, Max-fill with EC", "dependent", "equal",      1),
]

STYLES = {
    "Digital slicing":           ("#1f77b4", "D", 7),
    "$b_W$=2, Equal-fill":       ("#2ca02c", "o", 6),
    "$b_W$=2, Max-fill":         ("#17becf", "o", 6),
    "$b_W$=2, Max-fill with EC": ("#c5b200", "o", 6),
    "$b_W$=1, Equal-fill":       ("#ff9896", "o", 6),
    "$b_W$=1, Max-fill":         ("#d62728", "o", 6),
    "$b_W$=1, Max-fill with EC": ("#8b0000", "o", 6),
}

# ── sweep ───────────────────────────────────────────────────────
results = {}
t_start = time.time()

for label, algo, wm, base in CONFIGS:
    results[label] = {k: np.zeros(len(N_W_LIST)) for k in
                      ["mean_t0", "std_t0", "mean_1m", "std_1m"]}
    for i, nw in enumerate(N_W_LIST):
        print(f"{label}, n_W={nw} ...", end=" ", flush=True)
        seeds = np.random.randint(0, 2**31, N_TRIALS)
        res = Parallel(n_jobs=-1)(
            delayed(run_trial_4a)(int(s), algo, wm, base, nw) for s in seeds)
        t0 = np.array([r["t0"] for r in res])
        m1 = np.array([r["1month"] for r in res])
        results[label]["mean_t0"][i] = np.mean(t0)
        results[label]["std_t0"][i]  = np.std(t0)
        results[label]["mean_1m"][i] = np.mean(m1)
        results[label]["std_1m"][i]  = np.std(m1)
        print(f"η(t0)={results[label]['mean_t0'][i]:.4f}  "
              f"η(1mo)={results[label]['mean_1m'][i]:.4f}")

print(f"\nTotal: {(time.time() - t_start) / 60:.1f} min")

# ── theory ──────────────────────────────────────────────────────
eta_s_t0 = results["$b_W$=1, Equal-fill"]["mean_t0"][0]
eta_s_1m = results["$b_W$=1, Equal-fill"]["mean_1m"][0]
th = {
    "bw1_t0": [equal_fill_theory(eta_s_t0, 1, nw) for nw in N_W_LIST],
    "bw2_t0": [equal_fill_theory(eta_s_t0, 2, nw) for nw in N_W_LIST],
    "bw1_1m": [equal_fill_theory(eta_s_1m, 1, nw) for nw in N_W_LIST],
    "bw2_1m": [equal_fill_theory(eta_s_1m, 2, nw) for nw in N_W_LIST],
}

# ── plot ────────────────────────────────────────────────────────
nw = np.array(N_W_LIST)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, sfx, title, th1, th2 in [
    (ax1, "t0", "at $t_0$",  th["bw1_t0"], th["bw2_t0"]),
    (ax2, "1m", "at 1 month", th["bw1_1m"], th["bw2_1m"]),
]:
    ax.plot(nw, th2, "-", color="#2ca02c", lw=2, label="Equal-fill theory, $b_W$=2")
    ax.plot(nw, th1, "-", color="#ff7f0e", lw=2, label="Equal-fill theory, $b_W$=1")
    for label in STYLES:
        c, m, ms = STYLES[label]
        ax.errorbar(nw, results[label][f"mean_{sfx}"],
                    yerr=results[label][f"std_{sfx}"],
                    color=c, ls="None", marker=m, ms=ms,
                    capsize=3, capthick=0.8, elinewidth=0.8, label=label)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Number of weight slices ($n_W$)")
    ax.set_ylabel("Relative MVM error")
    ax.set_xticks(nw)
    ax.grid(alpha=0.3)

handles, labels_ = ax2.get_legend_handles_labels()
fig.legend(handles, labels_, loc="center right",
           bbox_to_anchor=(1.22, 0.5), fontsize=9, frameon=True)
plt.suptitle("Figure 4(a)", fontsize=14, y=1.01)
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/figure_4a.png", dpi=150, bbox_inches="tight")
print("Saved: figures/figure_4a.png")
