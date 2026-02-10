"""Analog layers with stored PCM programming state for DNN inference.

Programming noise and drift exponent ν are sampled ONCE at 'chip programming'.
Only 1/f read noise is stochastic at each forward pass.  This is critical for
the Dependent (EC) algorithm: error correction compensates the *actual*
programming error of the previous slice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import device_torch as phys


class AnalogMixin:
    """Mixin providing PCM bit-slicing inference on any weight tensor."""

    def _init_analog(self, num_slices, algo, base):
        self.num_slices = num_slices
        self.algo = algo
        self.base = float(base)
        self.drift_time = phys.T0
        self._is_programmed = False
        self._g_prog, self._nu, self._powers = [], [], []
        self._signs, self._range_slice, self._scale = None, None, None

    # ── chip programming (called once) ──────────────────────────
    def _program(self, w_clean):
        device = w_clean.device
        std = w_clean.std()
        if std < 1e-9:
            std = torch.tensor(1.0, device=device)
        limit = phys.N_STD_W * std

        self._scale = 255.0 / limit
        w_int = torch.round(torch.clamp(w_clean, -limit, limit) * self._scale)
        self._signs = torch.sign(w_int)
        residual = torch.abs(w_int).float()

        if self.base == 1.0:
            self._range_slice = 255.0 / self.num_slices
        else:
            self._range_slice = 255.0 * (self.base - 1) / (self.base ** self.num_slices - 1)

        self._g_prog, self._nu, self._powers = [], [], []

        for i in range(self.num_slices, 0, -1):
            power = self.base ** (i - 1)
            self._powers.append(power)

            if self.algo == "equal_fill":
                S_target = residual / i
            else:
                if i == 1:
                    S_target = torch.clamp(residual, 0, self._range_slice)
                else:
                    cap = (self._range_slice * (1 - self.base ** (i - 1)) / (1 - self.base)
                           if self.base > 1 else self._range_slice * (i - 1))
                    needs = (residual > cap).float()
                    S_target = torch.clamp((residual / power) * needs, 0, self._range_slice)

            g_target = torch.clamp(S_target / (self._range_slice + 1e-9), 0, 1)
            noise = phys.programming_noise_std(g_target) * torch.randn_like(g_target)
            g_prog = torch.clamp(g_target + noise, 0, 1)
            self._g_prog.append(g_prog)
            self._nu.append(phys.sample_drift_exponent(g_target))

            S_actual = g_prog * self._range_slice
            if self.algo == "dependent":
                residual = residual - S_actual * power
            else:
                residual = residual - S_target * power
            residual = torch.clamp(residual, min=0)

        self._is_programmed = True

    # ── read at drift time t ────────────────────────────────────
    def _read(self, t):
        if not self._is_programmed:
            self._program(self.weight)

        w_recon = torch.zeros_like(self._signs, dtype=torch.float32)
        f_int = phys.f_integral(t) if t > phys.T0 else 0

        for g_prog, nu, power in zip(self._g_prog, self._nu, self._powers):
            g_drift = g_prog * ((t / phys.T0) ** (-nu)) if t > phys.T0 else g_prog
            if f_int > 0:
                qs = phys.read_noise_qs(g_prog)
                g_read = g_drift + g_drift * qs * f_int * torch.randn_like(g_drift)
            else:
                g_read = g_drift
            g_read = torch.clamp(g_read, 0, 1)
            w_recon += self._signs * g_read * self._range_slice * power

        if t > phys.T0:
            w_recon *= (t / phys.T0) ** phys.GDC_NU
        return w_recon / self._scale


class AnalogConv2d(nn.Conv2d, AnalogMixin):
    def __init__(self, in_ch, out_ch, ks, stride=1, padding=0,
                 bias=True, num_slices=8, algo="max_fill", base=1):
        super().__init__(in_ch, out_ch, ks, stride, padding, bias=bias)
        self._init_analog(num_slices, algo, base)

    def forward(self, x):
        return F.conv2d(x, self._read(self.drift_time), self.bias,
                        self.stride, self.padding)


class AnalogLinear(nn.Linear, AnalogMixin):
    def __init__(self, in_f, out_f, bias=True, num_slices=8, algo="max_fill", base=1):
        super().__init__(in_f, out_f, bias=bias)
        self._init_analog(num_slices, algo, base)

    def forward(self, x):
        return F.linear(x, self._read(self.drift_time), self.bias)
