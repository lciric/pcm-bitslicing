#!/usr/bin/env python3
"""Figure 5 — CIFAR-10 accuracy with PCM bit-slicing.

5(a): Accuracy vs. time — Max-fill b_W=1, n_W ∈ {1,2,4,8}.
5(c): Accuracy vs. n_W — 5 algorithm configs at t₀ and 1 month.

Requires GPU for reasonable runtime (~2–4 h total with N_MC_RUNS=100).
Reproduces Le Gallo et al. (2022), Figs. 5a & 5c.
"""

import sys, os, copy, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from pcm_sim import device_torch as phys
from pcm_sim.training import train_noise_aware
from pcm_sim.utils import evaluate, eval_mc

# ── device ──────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── data ────────────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data", train=True, download=True, transform=transform_train),
    batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data", train=False, download=True, transform=transform_test),
    batch_size=128, shuffle=False, num_workers=2)

# ── baseline ────────────────────────────────────────────────────
model_baseline = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True
).to(DEVICE)
baseline_acc = evaluate(model_baseline, test_loader)
print(f"Digital baseline: {baseline_acc:.2f}%")

# ── noise-aware training ────────────────────────────────────────
MODEL_PATH = "figures/resnet32_robust.pth"
os.makedirs("figures", exist_ok=True)

if os.path.exists(MODEL_PATH):
    print(f"Loading {MODEL_PATH}")
    model_robust = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=False)
    model_robust.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model_robust = model_robust.to(DEVICE)
else:
    print("Training noise-aware model (15 epochs) ...")
    model_robust = copy.deepcopy(model_baseline)
    model_robust = train_noise_aware(model_robust, train_loader, device=DEVICE)
    torch.save(model_robust.state_dict(), MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")

robust_acc = evaluate(model_robust, test_loader)
print(f"Robust digital accuracy: {robust_acc:.2f}%")

# ── Figure 5(a) — accuracy vs. time ────────────────────────────
DAY, MONTH, YEAR = 86400, 30 * 86400, 365 * 86400
time_points = np.logspace(np.log10(20), np.log10(YEAR), 12)
slice_counts = [1, 2, 4, 8]

print("\n=== Figure 5(a): Max-fill b_W=1 ===")
results_5a = {n: {"mean": [], "std": []} for n in slice_counts}

for n_slices in slice_counts:
    print(f"n_W={n_slices}:")
    for t in time_points:
        mean, std = eval_mc(model_robust, test_loader, n_slices,
                            "max_fill", 1, t, phys.N_MC_RUNS)
        results_5a[n_slices]["mean"].append(mean)
        results_5a[n_slices]["std"].append(std)
        print(f"  t={t:.0f}s  acc={mean:.2f}±{std:.2f}%")

# Plot 5a
plt.figure(figsize=(10, 6))
colors_5a = {1: "#d62728", 2: "#ff7f0e", 4: "#2ca02c", 8: "#1f77b4"}
for n in slice_counts:
    m = np.array(results_5a[n]["mean"])
    s = np.array(results_5a[n]["std"])
    plt.semilogx(time_points, m, "o-", color=colors_5a[n],
                 label=f"$n_W={n}$", ms=5, lw=2)
    plt.fill_between(time_points, m - s, m + s, color=colors_5a[n], alpha=0.2)

plt.axhline(robust_acc, color="gray", ls="--", alpha=0.7, label="Digital baseline")
for t, tl in [(DAY, "1 day"), (MONTH, "1 month"), (YEAR, "1 year")]:
    plt.axvline(t, color="gray", ls=":", alpha=0.3)
plt.xlabel("Time after programming (s)")
plt.ylabel("CIFAR-10 test accuracy (%)")
plt.title("Figure 5(a): Max-fill ($b_W$=1) with GDC")
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/figure_5a.png", dpi=150, bbox_inches="tight")
print("Saved: figures/figure_5a.png")
np.savez("figures/results_5a.npz", results_5a=results_5a,
         time_points=time_points, robust_acc=robust_acc)

# ── Figure 5(c) — accuracy vs. n_W at t₀ and 1 month ──────────
configs_5c = [
    ("equal_fill", 1, "#d62728", "Equal-fill ($b_W=1$)"),
    ("max_fill",   1, "#1f77b4", "Max-fill ($b_W=1$)"),
    ("dependent",  1, "#2ca02c", "Max-fill EC ($b_W=1$)"),
    ("max_fill",   2, "#9467bd", "Max-fill ($b_W=2$)"),
    ("dependent",  2, "#8c564b", "Max-fill EC ($b_W=2$)"),
]
t_targets = {"$t_0$": phys.T0, "1 month": MONTH}
slice_values = [1, 2, 4, 8]

print("\n=== Figure 5(c) ===")
results_5c = {}
for algo, base, col, label in configs_5c:
    results_5c[label] = {t: {"mean": [], "std": []} for t in t_targets}

for algo, base, col, label in configs_5c:
    print(f"\n{label}:")
    for t_name, t_val in t_targets.items():
        for n in slice_values:
            mean, std = eval_mc(model_robust, test_loader, n, algo, base,
                                t_val, phys.N_MC_RUNS)
            results_5c[label][t_name]["mean"].append(mean)
            results_5c[label][t_name]["std"].append(std)
            print(f"  {t_name} n={n}: {mean:.1f}% ± {std:.1f}%")

# Plot 5c
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
for ax, t_name in zip([ax1, ax2], t_targets.keys()):
    for algo, base, col, label in configs_5c:
        m = np.array(results_5c[label][t_name]["mean"])
        s = np.array(results_5c[label][t_name]["std"])
        ax.errorbar(slice_values[:len(m)], m, yerr=s, fmt="o-",
                    color=col, label=label, ms=8, capsize=5, lw=2)
    ax.set_xlabel("Number of slices ($n_W$)")
    ax.set_title(f"Performance at {t_name}", fontsize=13)
    ax.set_xticks(slice_values)
    ax.grid(True, alpha=0.4, ls="--")

ax1.set_ylabel("Test accuracy (%)")
ax2.legend(loc="best", fontsize=10, frameon=True)
plt.suptitle("Figure 5(c): CIFAR-10 accuracy vs. bit-slicing configuration",
             fontsize=14)
plt.tight_layout()
plt.savefig("figures/figure_5c.png", dpi=150, bbox_inches="tight")
print("Saved: figures/figure_5c.png")
np.savez("figures/results_5c.npz", results_5c=results_5c)

# ── summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Digital baseline:  {baseline_acc:.1f}%")
print(f"Robust baseline:   {robust_acc:.1f}%")
print(f"\n--- Figure 5(a): Max-fill b=1 ---")
for n in slice_counts:
    t0 = results_5a[n]["mean"][0]
    yr = results_5a[n]["mean"][-1]
    print(f"  n_W={n}: {t0:.1f}% (t0) → {yr:.1f}% (1yr), drop={t0 - yr:.1f}%")
print(f"\n--- Figure 5(c) at 1 month ---")
for _, _, _, label in configs_5c:
    m = results_5c[label]["1 month"]["mean"]
    print(f"  {label}: n=1 {m[0]:.1f}% → n=8 {m[-1]:.1f}%")
