"""DNN inference utilities for PCM bit-slicing evaluation."""

import copy
import numpy as np
import torch
import torch.nn as nn

from .analog_layers import AnalogConv2d, AnalogLinear


def convert_to_analog(model, num_slices, algo, base=1):
    """Replace Conv2d / Linear layers with their PCM analog equivalents.

    Returns a new (deep-copied) model; the original is untouched.
    """
    model = copy.deepcopy(model)

    def _go(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                new = AnalogConv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    child.stride, child.padding, bias=child.bias is not None,
                    num_slices=num_slices, algo=algo, base=base)
                new.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new.bias.data = child.bias.data.clone()
                setattr(module, name, new)
            elif isinstance(child, nn.Linear):
                new = AnalogLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    num_slices=num_slices, algo=algo, base=base)
                new.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new.bias.data = child.bias.data.clone()
                setattr(module, name, new)
            else:
                _go(child)

    _go(model)
    return model


def set_drift_time(model, t):
    """Set the drift time for all analog layers in the model."""
    for m in model.modules():
        if hasattr(m, "drift_time"):
            m.drift_time = t


def evaluate(model, loader, *, max_batches=None):
    """Compute classification accuracy (%)."""
    model.eval()
    correct = total = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


def eval_mc(model_base, loader, n_slices, algo, base, t_drift, n_runs, *,
            max_batches=None):
    """Monte Carlo evaluation: programme n_runs independent chips.

    Returns
    -------
    mean : float
        Mean accuracy (%).
    std : float
        Std of accuracy (%).
    """
    device = next(model_base.parameters()).device
    accs = []
    for _ in range(n_runs):
        m = convert_to_analog(model_base, n_slices, algo, base).to(device)
        set_drift_time(m, t_drift)
        accs.append(evaluate(m, loader, max_batches=max_batches))
    return np.mean(accs), np.std(accs)
