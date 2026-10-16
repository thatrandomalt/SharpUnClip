# SPDX-FileCopyrightText: 2026 thatrandomalt
# SPDX-FileCopyrightText: 2019 Pariente Manuel (Asteroid: network structure mirrored here)
#
# SPDX-License-Identifier: MIT

"""Unlimit with De-limiter (Jeon & Lee, WASPAA 2023); weights and notices: THIRD_PARTY_NOTICES.md.

A Conv-TasNet-style network (Luo & Mesgarani 2019) that predicts one gain per sample,
shared by all channels. The layers mirror Asteroid's (Pariente et al. 2020, MIT) so the
published weights load without that dependency. Needs PyTorch.
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "delimiter.pth")


class GLN(nn.Module):  # global layer norm over (channels, time)
    def __init__(self, c):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(c)); self.beta = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        m = x.mean(dim=(1, 2), keepdim=True)
        v = ((x - m) ** 2).mean(dim=(1, 2), keepdim=True)
        return (x - m) / torch.sqrt(v + 1e-8) * self.gamma[:, None] + self.beta[:, None]


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.shared_block = nn.Sequential(
            nn.Conv1d(128, 512, 1), nn.PReLU(), GLN(512),
            nn.Conv1d(512, 512, 3, padding=d, dilation=d, groups=512), nn.PReLU(), GLN(512))
        self.res_conv = nn.Conv1d(512, 128, 1)
        self.skip_conv = nn.Conv1d(512, 128, 1)

    def forward(self, x):
        h = self.shared_block(x)
        return self.res_conv(h), self.skip_conv(h)


class Masker(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Sequential(GLN(1024), nn.Conv1d(1024, 128, 1))
        self.TCN = nn.ModuleList(Block(2 ** x) for _ in range(2) for x in range(5))
        self.mask_net = nn.Sequential(nn.PReLU(), nn.Conv1d(128, 512, 1))

    def forward(self, x):
        out, skip = self.bottleneck(x), 0
        for blk in self.TCN:
            r, s = blk(out)
            out, skip = out + r, skip + s
        return torch.relu(self.mask_net(skip))


class _FB(nn.Module):
    def __init__(self):
        super().__init__()
        self.filterbank = nn.Module()
        self.filterbank._filters = nn.Parameter(torch.empty(512, 1, 128))


class DeLimiter(nn.Module):
    """wav (b, 2, T) at 44.1 kHz -> per-sample gain (b, 1, T) in (0, 1)."""
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder, self.masker = _FB(), _FB(), Masker()

    def forward(self, wav):
        b, c, T = wav.shape
        spec = F.conv1d(wav.reshape(b * c, 1, T), self.encoder.filterbank._filters, stride=64)
        spec = torch.relu(spec).reshape(b, c * 512, -1)
        g = F.conv_transpose1d(self.masker(spec), self.decoder.filterbank._filters, stride=64)
        return torch.sigmoid(F.pad(g, (0, T - g.shape[-1])))


_model = {}


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2017 Facebook Inc. (Soumith Chintala)
# SPDX-License-Identifier: BSD-2-Clause
def resample(w, orig_freq, new_freq, lowpass_filter_width=6, rolloff=0.99):
    """w (..., n) from orig_freq to new_freq Hz: torchaudio.functional.resample with its defaults
    (Hann-windowed sinc), the same operations in the same order, so Unlimit needs no torchaudio, whose
    builds must match PyTorch's exactly (a mismatched one would not load next to AMD's ROCm PyTorch)."""
    gcd = math.gcd(int(orig_freq), int(new_freq))
    orig_freq, new_freq = int(orig_freq) // gcd, int(new_freq) // gcd
    if orig_freq == new_freq:
        return w
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    kw = dict(dtype=w.dtype, device=w.device)
    idx = torch.arange(-width, width + orig_freq, **kw)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, **kw)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    kernel = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernel *= window * (base_freq / orig_freq)
    shape = w.size()
    w = w.reshape(-1, shape[-1])
    n_wav, length = w.shape
    out = F.conv1d(F.pad(w, (width, width + orig_freq))[:, None], kernel, stride=orig_freq)
    out = out.transpose(1, 2).reshape(n_wav, -1)
    out = out[..., :int(torch.ceil(torch.as_tensor(new_freq * length / orig_freq)))]
    return out.view(shape[:-1] + out.shape[-1:])
# SPDX-SnippetEnd


@torch.no_grad()
def raw_gain(x, sr, tick=None, seg=6.0, fade=1.0, dev=None):
    """Per-sample gain of the model over its median for x (n, ch): the 100 % unlimit gain.
    The model runs at 44.1 kHz on crossfaded chunks; only the gain curve is resampled to sr.
    dev: the torch device; None = PyTorch's GPU if there is one, else the CPU."""
    dev = dev or ("cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    if torch.version.hip:  # ROCm: skip MIOpen's first-use tuning of every conv shape (as sharpunclip.torch_gpu)
        torch.backends.cudnn.enabled = False
    if dev not in _model:
        m = DeLimiter()
        m.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=True)
        _model[dev] = m.to(dev).eval()
    model = _model[dev]
    st = x if x.shape[1] == 2 else np.repeat(x[:, :1], 2, 1)  # mono -> both inputs
    w = torch.from_numpy(np.ascontiguousarray(st.T, np.float32)).to(dev)
    if sr != 44100:
        w = resample(w, sr, 44100)
    T = w.shape[-1]; S, H = int(seg * 44100), int((seg - fade) * 44100); Fd = S - H
    g = torch.zeros(T, device=dev); ws = torch.zeros(T, device=dev)
    ramp = torch.linspace(0, 1, Fd + 2, device=dev)[1:-1]
    starts = range(0, max(1, T - Fd), H)
    for i, s in enumerate(starts):
        if tick:
            tick(i / len(starts))
        e = min(T, s + S)
        win = torch.ones(e - s, device=dev)
        if s > 0:
            win[:Fd] = ramp[:min(Fd, e - s)]
        if e < T:
            win[-Fd:] = ramp.flip(0)
        g[s:e] += model(w[None, :, s:e])[0, 0] * win; ws[s:e] += win
    g = (g / ws.clamp_min(1e-8)).cpu().numpy().astype(np.float64)
    if sr != 44100:  # smooth curve: linear interpolation is enough
        g = np.interp(np.arange(len(x)) * (44100 / sr), np.arange(T), g)
    g = g[:len(x)]
    return g / np.median(g)


def gain(x, sr, amount=0.5, tick=None, seg=6.0, fade=1.0, dev=None):
    """Per-sample unlimit gain for x (n, ch): raw_gain raised to `amount`."""
    return raw_gain(x, sr, tick, seg, fade, dev) ** amount
