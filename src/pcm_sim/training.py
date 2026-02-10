"""Noise-aware training — Joshi et al. (2020).

Injects Gaussian noise proportional to max |W| during forward propagation
and clips weights to ±α·σ_W.  Used to produce HWA-trained ResNet-32 for
Figure 5 of Le Gallo et al. (2022).
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from . import device_torch as phys


def train_noise_aware(model, train_loader, *, epochs=15, lr=0.01, device=None):
    """Hardware-aware fine-tuning with additive weight noise.

    Parameters
    ----------
    model : nn.Module
        Pre-trained model (e.g. ResNet-32).
    train_loader : DataLoader
        CIFAR-10 training loader.
    epochs : int
    lr : float
    device : torch.device

    Returns
    -------
    nn.Module
        Fine-tuned model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sch = optim.lr_scheduler.MultiStepLR(opt, [10], 0.1)
    crit = nn.CrossEntropyLoss()

    for ep in range(epochs):
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            # Inject noise, clip, forward, then restore clean weights
            saved = {}
            for n, m in model.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    saved[n] = m.weight.data.clone()
                    w = m.weight.data
                    noise = torch.randn_like(w) * phys.NOISE_STD * w.abs().max()
                    limit = phys.N_STD_W * w.std()
                    m.weight.data = torch.clamp(w + noise, -limit, limit)

            out = model(x)
            loss = crit(out, y)
            loss.backward()

            for n, m in model.named_modules():
                if n in saved:
                    m.weight.data = saved[n]
            opt.step()

            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        sch.step()
        print(f"  Epoch {ep + 1:2d}/{epochs} | Acc: {100 * correct / total:.1f}%")

    return model


def recalibrate_bn(model, loader, *, device=None):
    """Re-estimate BatchNorm statistics using clean forward passes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Recalibrating BN", leave=False):
            model(x.to(device))
    model.eval()
