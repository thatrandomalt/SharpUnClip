#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 thatrandomalt
#
# SPDX-License-Identifier: MIT

"""SharpUnClip: undo clipping, limiting and compression.

Declip rebuilds clipped samples with frame-wise A-SPADE; unlimit (unlimit.py) and
uncompress are optional later stages. See METHODS.md for the methods.
"""
import argparse
import copy
import faulthandler
import functools
import json
import math
import os
import sys
import time
import traceback

import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None

# Apple GPUs: PyTorch ops that MPS lacks run on the CPU instead of failing (read when torch loads)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
GPU_DEVS = ("opencl", "cuda", "xpu", "mps")  # devices that mean "a GPU was asked for and found"


def mask_min_run(mask, min_run):
    """mask without its runs shorter than min_run samples."""
    if min_run <= 1:
        return mask
    d = np.diff(mask.astype(np.int8), prepend=np.int8(0), append=np.int8(0))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (ends - starts) >= min_run
    edges = np.zeros(len(mask) + 1, np.int8)
    edges[starts[keep]] = 1  # a run never ends where another starts
    edges[ends[keep]] = -1
    return np.cumsum(edges[:-1], dtype=np.int8) > 0


@functools.lru_cache(None)
def torch_gpu():
    """PyTorch's GPU, or None: "cuda" (NVIDIA, or AMD with a ROCm build), "xpu" (Intel), "mps" (Apple)."""
    try:
        import torch
    except Exception:
        return None
    if torch.cuda.is_available():
        if torch.version.hip:  # ROCm: MIOpen tunes each new conv shape on first use (minutes, looks hung);
            torch.backends.cudnn.enabled = False  # PyTorch's own GPU conv is fine for these small kernels
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def _hip():
    import torch
    return bool(torch.version.hip)


def spade_backend(gpu=False):
    """Return (array module, device). gpu: False/"cpu", True/"auto", "opencl" or "cuda".
    auto tries OpenCL, then PyTorch's GPU (CUDA, XPU or MPS); otherwise the CPU (PyTorch if
    installed, else NumPy). With a ROCm build of PyTorch, auto skips OpenCL: Mesa's OpenCL (Rusticl)
    loaded into the same process hid the ROCm GPU from PyTorch (seen on an RX 9070), so both engines
    run on PyTorch there."""
    gpu = "auto" if gpu is True else gpu or "cpu"
    tg = torch_gpu() if gpu != "cpu" else None  # before any OpenCL is loaded (see above)
    if gpu == "opencl" or gpu == "auto" and not (tg == "cuda" and _hip()):
        try:
            import sharpunclip_cl
            if sharpunclip_cl.device() is not None:
                return sharpunclip_cl, "opencl"
        except Exception:
            pass
    if tg and (gpu == "auto" or gpu == tg):
        import torch
        return torch, tg
    try:
        import torch
        return torch, "cpu"
    except Exception:
        return np, None


def backend_name(xp, dev, opt=None):
    """What does the heavy lifting: the SPADE backend, or the NMF engine's PyTorch device."""
    if opt and opt.get("declip", True) and opt.get("engine") == "NMF":
        try:
            import torch
        except Exception:
            return "NMF engine needs PyTorch"
        d = _nmf_device(dev)
        if d != "cpu":
            return "NMF on " + _torch_gpu_name(d)
        if dev in GPU_DEVS:  # a GPU was found (OpenCL), but this PyTorch build can't use it
            return f"NMF on CPU (PyTorch {torch.__version__} has no GPU support for this GPU; see README)"
        return "NMF on CPU"
    if dev == "opencl":
        return "OpenCL: " + xp.device().name.strip()
    if dev in ("cuda", "xpu", "mps"):
        return _torch_gpu_name(dev)
    return "CPU"


def _torch_gpu_name(dev):
    import torch
    kind = {"cuda": "ROCm" if torch.version.hip else "CUDA", "xpu": "Intel XPU", "mps": "Apple MPS"}[dev]
    if dev == "mps":
        return kind
    try:
        return f"{kind}: " + getattr(torch, dev).get_device_name()
    except Exception:  # a name is never worth failing a render for
        return kind


def _nmf_device(dev):
    """The NMF engine runs on PyTorch: its GPU when a GPU was asked for and one is there."""
    return (torch_gpu() or "cpu") if dev in GPU_DEVS else "cpu"


def _nmf_analysis(opt, y, sr, an=None):
    """(mode, analysis) for NMF auto detection. nmf_knee: 0 = as found (auto), 0.5-0.99 = soft knee at
    this fraction of each ceiling (no shoulder search), >= 1 = ceilings only (hard)."""
    k = float(opt.get("nmf_knee") or 0.0)
    an = copy.deepcopy(an)
    if k >= 1.0:
        return "hard", an
    if k > 0.0:
        if an is None:
            from betterdeclipper.auto import analyze
            from betterdeclipper.detect import estimate_lsb
            an = analyze(y, estimate_lsb(y), None, sr, search=False)
        an["shoulder"] = dict(knee=max(0.5, min(k, 0.99)))  # auto_constraints fills acc_min itself
    return "auto", an


def _nmf_constraints(y, sr, opt, cap, device, h):
    """The constraint set BetterDeclipper's declip() builds for these settings:
    (m_hi, m_lo, th_hi, th_lo, gcap)."""
    if opt["auto_detect"]:
        from betterdeclipper.auto import auto_constraints
        from betterdeclipper.detect import estimate_lsb
        from betterdeclipper.engine import fast_pass
        mode, an = _nmf_analysis(opt, y, sr, h.get("nmf_analysis"))
        m_hi, m_lo, th_hi, th_lo, rep = auto_constraints(y, estimate_lsb(y), fast_pass(sr, device), mode=mode,
                                                         sr=sr, analysis=an)
        gcap = rep.pop("gcap", None)
        if np.isfinite(cap):
            gcap = cap if gcap is None else np.minimum(gcap, cap)
        return m_hi, m_lo, th_hi, th_lo, gcap
    m = (np.stack(h["masks"], 1) if h.get("masks") is not None
         else np.stack([_clip_mask(y[:, c], sr, opt) for c in range(y.shape[1])], 1))
    m_hi, m_lo = m & (y > 0), m & (y < 0)
    return m_hi, m_lo, np.where(m_hi, y, np.inf), np.where(m_lo, y, -np.inf), None if np.isinf(cap) else cap


def _declip_nmf_spade(y, sr, opt, cap, dev, xp, progress, h):
    """NMF quality "normal+spade": BetterDeclipper's NMF and SPADE models plus our short-frame linked
    A-SPADE (512-sample frames, 16x overlap, rebuilt |x| <= 2.5|y|), all under one constraint set and
    fused 0.52 / 0.28 / 0.2 below 4 kHz (NMF alone above), as measured in METHODS 6.1."""
    from betterdeclipper import engine as E
    from betterdeclipper.methods.common import full_thresholds
    device = _nmf_device(dev)
    cons = _nmf_constraints(y, sr, opt, cap, device, h)
    m_hi, m_lo, th_hi, th_lo, gcap = cons
    clipped = m_hi | m_lo
    if not clipped.any():
        return y.copy(), dict(clipped_frac=0.0)
    TH, TL = full_thresholds(th_hi, y.shape), full_thresholds(th_lo, y.shape)
    G = None if gcap is None else np.broadcast_to(np.asarray(gcap, float), y.shape)
    part = lambda i: progress and (lambda f: progress((i + f) / 3))
    ests = []
    for i, kind in enumerate(("nmf", "spade")):
        p = part(i)
        ests.append(E.declip(y, sr, models=[(kind, 93, {})], constraints=cons, device=device, verbose=False,
                             progress=p and (lambda s, n, _t, p=p: p(s / n)), lam_refs=h.get("lam_refs"))[0])
    lo, hi = y.copy(), y.copy()
    lo[m_hi], hi[m_hi] = TH[m_hi], np.maximum(2.5 * y[m_hi], TH[m_hi])
    hi[m_lo], lo[m_lo] = TL[m_lo], np.minimum(2.5 * y[m_lo], TL[m_lo])
    with np.errstate(invalid="ignore"):
        ub = None if G is None else np.where(np.isinf(G), np.inf, G * TH)
        lb = None if G is None else np.where(np.isinf(G), -np.inf, G * TL)
    if G is not None:
        hi[m_hi], lo[m_lo] = np.minimum(hi[m_hi], ub[m_hi]), np.maximum(lo[m_lo], lb[m_lo])
    s = 1.0 / np.abs(y).max()
    p = part(2)
    ours = aspade(y * s, lo * s, hi * s, 512, 16, xp=xp, dev=dev, progress=p and (lambda f, d, o: p(f)),
                  group=2 if y.shape[1] == 2 else 1, unscale=s, frame0=h.get("start", 0) // 32)
    ests.append(np.clip(ours, lo, hi))
    out = E.fuse(y, ests, [0.52, 0.28, 0.2], [1.0, 0.0, 0.0], sr)
    out[~clipped] = y[~clipped]  # the engine's final consistency projection
    out = np.where(m_hi, np.maximum(out, TH), out)
    out = np.where(m_lo, np.minimum(out, TL), out)
    if G is not None:
        out = np.where(m_hi, np.minimum(out, ub), out)
        out = np.where(m_lo, np.maximum(out, lb), out)
    return out, dict(clipped_frac=float(clipped.mean()))


def declip_nmf(y, sr, opt, cap=np.inf, dev=None, progress=None, hints=None, xp=np):
    """Declip with BetterDeclipper's NMF / social-sparsity / A-SPADE fusion (betterdeclipper/).
    auto_detect uses its ceiling analysis; otherwise our clip_above/... masks become its
    constraints (flagged samples keep their sign, |in| <= |out| <= cap*|in|).
    progress(frac) is called between steps; raising from it cancels. hints: see process.
    xp, dev: the SPADE backend, for nmf_quality "normal+spade". Returns (out, info)."""
    import torch
    from betterdeclipper.engine import declip
    torch.set_flush_denormal(True)  # per thread: the engine sets it once, on the thread importing it
    h = {} if hints is None else hints
    if opt["nmf_quality"] == "normal+spade":
        return _declip_nmf_spade(y, sr, opt, cap, dev, xp, progress, h)
    kw = dict(preset=opt["nmf_quality"], device=_nmf_device(dev), verbose=False,
              progress=progress and (lambda i, n, _t: progress(i / n)), lam_refs=h.get("lam_refs"))
    if opt["auto_detect"]:
        kw["max_gain_db"] = 20 * math.log10(cap) if np.isfinite(cap) else None
        kw["mode"], kw["analysis"] = _nmf_analysis(opt, y, sr, h.get("nmf_analysis"))
    else:
        m = (np.stack(h["masks"], 1) if h.get("masks") is not None
             else np.stack([_clip_mask(y[:, c], sr, opt) for c in range(y.shape[1])], 1))
        m_hi, m_lo = m & (y > 0), m & (y < 0)
        kw["constraints"] = (m_hi, m_lo, np.where(m_hi, y, np.inf), np.where(m_lo, y, -np.inf),
                             None if np.isinf(cap) else cap)
    return declip(y, sr, **kw)


def aspade(y, lo, hi, w=1024, ov=8, red=2, eps=0.1, maxit=1000, xp=np, dev=None,
           batch=2048, progress=None, group=1, unscale=1.0, frame0=0):
    """Frame-wise A-SPADE (Kitić, Bertin & Gribonval 2015) with per-sample bounds
    lo <= x <= hi (lo == hi: reliable sample).

    y, lo, hi are (n, ch). Frames are solved in time-ordered batches, so memory stays
    bounded. progress(frac, done, out) is called after each batch with the output so far
    (final up to sample `done`) and with out=None in between; raising from it cancels.
    group > 1 solves that many channels per frame with one shared set of DFT bins.
    The output (and every progress snapshot) is divided by `unscale`.
    frame0: index of y's first frame in a longer signal; batches split where that signal's
    do, so an excerpt starting on the frame grid gives that signal's samples bit-exactly."""
    n, ch = y.shape; a = w // ov; pad = w
    L = n + 2 * pad; L += (-L) % a
    P = lambda v: np.pad(v, ((pad, L - n - pad), (0, 0)))
    yp, lop, hip = P(y), P(lo), P(hi)
    starts = np.arange(0, L - w + 1, a)
    g = np.hanning(w + 2)[1:-1]
    acc = np.zeros((L, ch)); ws = np.zeros(L)
    out = y / unscale
    fin = [0]  # samples before this are final in `out`

    def finish(done):
        # only the newly final samples; clamp again: fp32 backends can land slightly outside the bounds
        a0 = fin[0]
        if done > a0:
            out[a0:done] = np.clip(acc[pad + a0:pad + done] / np.maximum(ws[pad + a0:pad + done, None], 1e-12),
                                   lo[a0:done], hi[a0:done]) / unscale
            fin[0] = done
        return out

    cuts = np.r_[0, np.arange((-frame0) % batch or batch, len(starts), batch), len(starts)]
    nb = len(cuts) - 1
    for bi in range(nb):
        s = starts[cuts[bi]:cuts[bi + 1]]; B = len(s)
        seg = s[:, None] - s[0] + np.arange(w); m = s[-1] - s[0] + w

        def rows(v):
            return v[s[0] + seg].transpose(0, 2, 1).reshape(B * ch, w)
        Y, LO, HI = rows(yp), rows(lop), rows(hip)
        act = np.flatnonzero(np.any(LO != HI, axis=1))
        if group > 1 and act.size:  # keep each frame's channels together
            act = (np.unique(act // group)[:, None] * group + np.arange(group)).ravel()
        if act.size:
            tick = progress and (lambda f: progress((bi + f) / nb, None, None))
            Y[act] = _aspade_frames(Y[act], LO[act], HI[act], red * w, eps, maxit, xp, dev, tick, group)
        ws[s[0]:s[0] + m] += np.bincount(seg.ravel(), weights=np.tile(g, B), minlength=m)
        Yg = (Y.reshape(B, ch, w) * g).transpose(1, 0, 2)
        for c in range(ch):
            acc[s[0]:s[0] + m, c] += np.bincount(seg.ravel(), weights=Yg[c].ravel(), minlength=m)
        if progress:
            nxt = cuts[bi + 1]  # samples before the next frame start are final
            done = n if nxt >= len(starts) else min(n, max(0, starts[nxt] - pad))
            progress((bi + 1) / nb, done, finish(done))
    return finish(n)


def _aspade_frames(Y, LO, HI, N, eps, maxit, xp, dev, tick=None, group=1):
    if dev == "opencl":
        return xp.aspade_frames(Y, LO, HI, N, eps, maxit, tick, group)
    w = Y.shape[1]; sq = N ** 0.5
    if xp is np:
        T = lambda v: v
        live, ar = np.arange(len(Y) // group), np.arange(group)
        kth = lambda m, k: np.partition(m, m.shape[1] - k, axis=1)[:, m.shape[1] - k, None]
        rep, gmax = lambda m: np.repeat(m, group, 0), lambda v: v.max(1)
    else:  # torch, fp32
        T = lambda v: xp.as_tensor(v, dtype=xp.float32, device=dev)
        live, ar = xp.arange(len(Y) // group, device=dev), xp.arange(group, device=dev)
        kth = lambda m, k: xp.topk(m, k, dim=1, sorted=False).values.amin(1, keepdim=True)
        rep, gmax = lambda m: m.repeat_interleave(group, 0), lambda v: v.amax(1)
    Y, LO, HI = T(Y), T(LO), T(HI)
    A = lambda v: xp.fft.rfft(v, N) / sq          # tight-frame analysis
    As = lambda z: xp.fft.irfft(z, N)[:, :w] * sq  # adjoint
    x = Y; Z = A(x); u = Z * 0; k = 1; nb = Z.shape[1]
    for it in range(maxit):
        if tick and it % 25 == 0:
            tick(it / maxit)
        r = (live[:, None] * group + ar).reshape(-1)
        V = Z[r] + u[r]
        e = (abs(V) ** 2).reshape(-1, group, nb).sum(1)
        zb = V * rep(e >= kth(e, min(k, nb)))
        xl = xp.minimum(xp.maximum(As(zb - u[r]), LO[r]), HI[r])
        Zl = A(xl)
        d = Zl - zb
        u[r] += d
        x[r] = xl; Z[r] = Zl
        live = live[gmax((abs(d) ** 2).sum(1).reshape(-1, group)) > eps * eps]
        if len(live) == 0:
            break
        k += 1
    return x.cpu().numpy() if hasattr(x, "cpu") else x


NMF_QUALITIES = ("fast", "normal", "high", "best", "normal+spade")

# Defaults are the Accurate preset. The keys are also the GUI field names, the CLI's
# argparse dests and the keys of saved preset files. (One expert option is CLI-only and not
# listed here: opt["soft_clip"] = "cubic", see process().)
DEFAULTS = {
    "clip_above": 99.9,       # % of peak
    "not_clip_below": 99.8,   # % of peak
    "not_clip_amp": -30.0,    # dBFS
    "peak_ignore": 0.001,     # fraction of samples
    "peak_time": 0.4,         # s
    "reset_amp": -90.0,       # dBFS
    "reset_time": 0.2,        # s
    "dynamic": True,
    "input_gain": 0.0,        # dB
    "output_gain": 0.0,       # dB
    "limiter_peak": 0.0,      # dBFS
    "limiter": False,
    "declip": True,
    "remove_glitch": True,
    "linked_stereo": True,
    "soft": False,
    "diff_clips": True,
    "trans_amp": -40.0,       # dBFS
    "trans_time": 0.5,        # s
    "speed_quality": 5,
    "lookahead": 0,
    "max_restore": 0.0,       # x clipped level, 0 = no cap (SPADE: 2.5 is the most accurate)
    "engine": "NMF",          # declip engine: NMF (BetterDeclipper's model fusion) or SPADE
    "nmf_quality": "normal",  # NMF engine: fast, normal, high, best or normal+spade (adds our SPADE model)
    "auto_detect": True,      # NMF engine: its own ceiling analysis instead of clip_above etc.
    "nmf_knee": 0.0,          # NMF auto: 0 = as found, 0.5-0.99 = soft knee (x ceiling), 1 = ceilings only
    "unlimit": False,
    "unlimit_amount": 50.0,   # %
    "uncompress": False,
    "uc_threshold": -12.0,    # dBFS
    "uc_ratio": 2.0,
    "uc_attack": 10.0,        # ms
    "uc_release": 150.0,      # ms
    "uc_makeup": 0.0,         # dB
    "uc_link": False,
    "uc_amount": 50.0,        # %
    "process_rate": 0.0,      # Hz, 0 = the file's rate
}
PRESETS = {
    "Accurate": dict(DEFAULTS),
    "Accurate SPADE": dict(DEFAULTS, engine="SPADE", max_restore=2.5),
    "Unmaster": dict(DEFAULTS, auto_detect=False, clip_above=75.0, not_clip_below=74.9, unlimit=True,
                     unlimit_amount=15.0),
    "Unmaster SPADE": dict(DEFAULTS, engine="SPADE", clip_above=93.0, not_clip_below=92.9, max_restore=2.5,
                           unlimit=True),
    "Uncompress": dict(DEFAULTS, declip=False, uncompress=True),
    "SharpUnClip": dict(DEFAULTS, engine="SPADE", clip_above=78.0, not_clip_below=77.0, speed_quality=3,
                        lookahead=1, max_restore=0.0),
}


def settings_from_cfg(cfg):
    """Settings dict from GUI/CLI/preset values (unknown keys are ignored)."""
    out = dict(DEFAULTS)
    for k, v in cfg.items():
        k = k.removeprefix("pa_")  # older preset files
        if k in out and v is not None:
            out[k] = bool(int(v)) if isinstance(DEFAULTS[k], bool) else v
    return out


def _db2lin(db):
    return 10.0 ** (float(db) / 20.0)


def cubic_inverse(y, c):
    """Undo the cubic soft clipper y = c*(w - w^3/6.75), w = u/c, flat at +-c from |w| = 1.5 (PeakEater's
    "cubic"). Unit slope at 0, so c is its only parameter; |y| >= c comes back as a flat ceiling at 1.5c."""
    s = np.clip(y / c, -1.0, 1.0)
    return 3.0 * c * np.cos(np.arccos(-s) / 3.0 - 2.0 * np.pi / 3.0)


def uncompress_gain(y, sr, threshold, ratio, attack_ms, release_ms, makeup=0.0, link=False, tick=None):
    """Gain in dB, (n, ch), that inverts a feed-forward compressor: peak detector, hard
    knee, gain smoothed in dB with branching attack/release (Giannoulis et al. 2012),
    inverted sample by sample from the smoother state (Gorlow & Reiss 2013)."""
    k9 = math.log(9)  # attack/release are 10-90% rise times
    aA = math.exp(-k9 / (max(float(attack_ms), 0.01) / 1000 * sr))
    aR = math.exp(-k9 / (max(float(release_ms), 0.01) / 1000 * sr))
    k, T = 1 - 1 / max(float(ratio), 1.0), float(threshold)
    det = np.abs(y).max(1, keepdims=True) if link else np.abs(y)
    gains = np.empty(det.shape)
    nch = det.shape[1]
    for ch in range(nch):
        Ys = (20 * np.log10(np.maximum(det[:, ch], 1e-12)) - makeup).tolist()
        s, g = 0.0, [0.0] * len(Ys)
        for i, Y in enumerate(Ys):
            if tick and i % sr == 0:
                tick((ch + i / len(Ys)) / nch)
            X = Y - aR * s                   # below threshold the gain only releases
            if X <= T:
                s = aR * s
            else:                            # use whichever branch is self-consistent
                a = aA
                X = (Y - a * s - (1 - a) * k * T) / (1 - (1 - a) * k)
                if not -k * (X - T) < s:
                    a = aR
                    X = (Y - a * s - (1 - a) * k * T) / (1 - (1 - a) * k)
                s = a * s + (1 - a) * (-k * (X - T))
            g[i] = s
        gains[:, ch] = g
    return -(gains + makeup)


def peak_steps(a, sr, opt):
    """Measured peak of a = |x| as a staircase: peaks(t) = values[k] for starts[k] <= t < starts[k+1].
    diff_clips: one peak per section, split at gaps quieter than trans_amp for at least
    trans_time; gap samples take the next section's peak, a trailing gap the last one's.
    Otherwise one peak per peak_time block."""
    n = len(a)
    reset_amp = _db2lin(opt["reset_amp"])
    ignore = min(0.999, max(0.0, float(opt["peak_ignore"])))

    def peak(seg):
        return float(np.quantile(seg, 1.0 - ignore)) if seg.size > 1 else float(seg[0])
    if opt["diff_clips"]:
        trans_win = max(1, int(round(float(sr) * max(0.0, float(opt["trans_time"])))))
        sections = ~mask_min_run(a < _db2lin(opt["trans_amp"]), trans_win)
        d = np.diff(np.concatenate(([0], sections.astype(np.int8), [0])))
        cstarts, cends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if cstarts.size == 0:
            return np.array([0]), np.array([reset_amp])
        values = [max(peak(a[s:e]), reset_amp) for s, e in zip(cstarts, cends)]
        return np.r_[0, cends[:-1]], np.array(values)
    win = max(1, int(round(float(sr) * max(0.001, float(opt["peak_time"])))))
    reset_win = max(1, int(round(float(sr) * max(0.001, float(opt["reset_time"])))))
    starts = np.arange(0, n, win)
    values = np.empty(len(starts))
    for k, s in enumerate(starts):
        aa = a[s:s + win]
        pk = peak(aa)
        if aa.max() < reset_amp and aa.size >= reset_win:
            pk = reset_amp
        values[k] = max(pk, reset_amp)
    return starts, values


def expand_steps(starts, values, n):
    return np.repeat(values, np.diff(np.r_[starts, n]))


def flag_mask(a, peaks, opt):
    """Which samples of a = |x| are clipped, given the measured peaks(t)."""
    # dynamic: clipped above clip_above; otherwise above not_clip_below
    high = float(opt["clip_above"]) / 100.0
    low = float(opt["not_clip_below"]) / 100.0
    mask = a >= peaks * (high if opt["dynamic"] else min(low, high))
    if opt["soft"]:
        mask |= a >= (1.0 - 0.08) * peaks
    mask &= a >= _db2lin(opt["not_clip_amp"])
    if opt["remove_glitch"]:
        mask = mask_min_run(mask, 2)
    return mask


def _clip_mask(y0, sr, opt):
    """Which samples of one channel are clipped."""
    a = np.abs(y0)
    return flag_mask(a, expand_steps(*peak_steps(a, sr, opt), len(a)), opt)


def process(x, sr, opt=None, xp=np, dev=None, progress=None, info=None, hints=None):
    """Declip, then unlimit, then uncompress x ((n,) or (n, ch)), as enabled in `opt`
    (see DEFAULTS). progress: see aspade. A dict passed as `info` receives "flagged", the
    fraction of samples declip treated as clipped, and "limited", the fraction the limiter cut.
    hints: optional dict of precomputed file-global inputs (GUI previews):
      masks        list of per-channel bool arrays: manual-detection flags for this x
      norm         SPADE level normalisation (1 / max|y0| of the whole file)
      start        index of x[0] in the whole file (SPADE: frame batches line up with the render's)
      nmf_analysis betterdeclipper analysis dict (refs, shoulder, source); NMF auto detection
      lam_refs     {win_ms: float}: file-global PnP lambda reference
      unlimit_g    (n,): 100 % unlimit gain, g / median(g) over the whole file
      unlimit_ref  (n, ch): the declip stage's input for this x (stage B of a split render; default y0)
      uc_gain      (n, ch): uncompress gain in dB for this stage input; present with value None,
                   it receives the computed gain
      soft_clip_c  opt["soft_clip"] == "cubic": the ceiling, max|y0| of the whole file"""
    opt = dict(DEFAULTS, **(opt or {}))
    pr = float(opt["process_rate"] or 0)
    if pr > 0 and abs(pr - sr) > 0.5:
        # Same samples, treated as pr Hz. The compressor times describe the real
        # compressor, so rescale them to the new time base.
        k = sr / pr
        opt = dict(opt, process_rate=0.0, uc_attack=float(opt["uc_attack"]) * k,
                   uc_release=float(opt["uc_release"]) * k)
        return process(x, pr, opt, xp, dev, progress, info, hints)
    h = {} if hints is None else hints
    x = np.asarray(x, dtype=np.float64)
    mono = x.ndim == 1
    y0 = (x[:, None] if mono else x) * _db2lin(opt["input_gain"])
    if y0.size and opt["declip"] and opt.get("soft_clip", "off") == "cubic":
        c = h.get("soft_clip_c") or float(np.abs(y0).max())
        if c > 0:
            y0 = cubic_inverse(y0, c)  # every later stage sees the un-soft-clipped signal
    out = y0.copy()
    ul = float(opt["unlimit_amount"]) / 100.0 if opt["unlimit"] else 0.0
    uc = float(opt["uc_amount"]) / 100.0 if opt["uncompress"] else 0.0
    # each active stage gets a share of the progress bar
    weights = [("declip", 8, bool(opt["declip"])), ("unlimit", 1, ul > 0), ("uncompress", 1, uc > 0)]
    total, span = sum(w for _, w, on in weights if on) or 1, {}
    acc = 0
    for name, w, on in weights:
        if on:
            span[name] = (acc / total, (acc + w) / total); acc += w

    def share(name):
        lo, hi = span[name]
        return lambda f: lo + (hi - lo) * f
    if y0.size and opt["declip"]:
        # flagged samples keep their sign: |in| <= |out| <= cap*|in|
        cap = float(opt["max_restore"])
        cap = max(cap, 1.0) if cap > 0 else np.inf
        at = share("declip")
        if opt["engine"] == "NMF":
            out, nmf_info = declip_nmf(y0, sr, opt, cap, dev, progress and (lambda f: progress(at(f), None, None)),
                                       h, xp)
            if info is not None:
                info["flagged"] = nmf_info["clipped_frac"]
            if progress:
                progress(at(1.0), len(out), out)
        else:
            lo, hi = y0.copy(), y0.copy()
            for c in range(y0.shape[1]):
                m = h["masks"][c] if h.get("masks") is not None else _clip_mask(y0[:, c], sr, opt)
                neg, pos = m & (y0[:, c] < 0), m & (y0[:, c] > 0)
                lo[neg, c] = cap * y0[neg, c]
                hi[pos, c] = cap * y0[pos, c]
            if info is not None:
                info["flagged"] = float(np.mean(lo != hi))
            if np.any(lo != hi):
                quality = max(0, min(6, int(opt["speed_quality"])))
                w = 512 << max(0, min(2, int(opt["lookahead"])))
                ov = (2, 4, 4, 8, 8, 16, 16)[quality]
                s = h.get("norm") or 1.0 / np.abs(y0).max()  # eps is absolute, so normalise the level
                prog = progress and (lambda f, d, o: progress(at(f), d, o))
                grp = 2 if opt["linked_stereo"] and y0.shape[1] == 2 else 1
                out = aspade(y0 * s, lo * s, hi * s, w, ov, xp=xp, dev=dev, progress=prog, group=grp, unscale=s,
                             frame0=h.get("start", 0) // (w // ov))
                out = np.clip(out, lo, hi)  # keeps untouched samples bit-exact
    if y0.size and ul > 0:
        at = share("unlimit")
        ref = h.get("unlimit_ref", y0)
        g = h.get("unlimit_g")
        if g is None:
            import unlimit  # needs PyTorch
            g = unlimit.gain(ref, sr, ul, progress and (lambda f: progress(at(f), None, None)))
        else:
            g = g ** ul
        # gain the input and add declip's correction unscaled (multiplying the declipped signal applied
        # the gain twice to the rebuilt peaks); exactly ref * g when declip is off
        out = ref * g[:, None] + (out - ref)
    if y0.size and uc > 0:
        at = share("uncompress")
        g = h.get("uc_gain")
        if g is None:
            g = uncompress_gain(out, sr, opt["uc_threshold"], opt["uc_ratio"], opt["uc_attack"],
                                opt["uc_release"], float(opt["uc_makeup"]), bool(opt["uc_link"]),
                                progress and (lambda f: progress(at(f), None, None)))
            if "uc_gain" in h:
                h["uc_gain"] = g
        out = out * 10 ** (g * uc / 20)
    if progress and (ul > 0 or uc > 0):
        progress(1.0, len(out), out)
    out *= _db2lin(opt["output_gain"])
    if opt["limiter"]:
        lim = _db2lin(opt["limiter_peak"])
        if lim > 0:
            if info is not None:
                info["limited"] = float(np.mean(np.abs(out) > lim))
            out = np.clip(out, -lim, lim)
    return out[:, 0] if mono else out


class RunLog:
    """Run log (appended, UTF-8), written line by line as the run goes, so a failed run still leaves
    its header and error, and a native crash or hang (GPU driver) faulthandler's stack dump.
    path None = off; a log that can't be opened never stops the run."""

    def __init__(self, path):
        try:
            self.f = open(path, "a", encoding="utf-8", errors="replace", buffering=1) if path else None
        except OSError:
            self.f = None
        if self.f:
            faulthandler.enable(self.f)

    def __call__(self, line):
        if self.f:
            self.f.write(line + "\n")

    def close(self):
        if self.f:
            faulthandler.disable()
            self.f.close()
            self.f = None


def env_info():
    """Python, PyTorch and GPU runtime versions, for logs and bug reports."""
    s = f"python={sys.version.split()[0]} platform={sys.platform}"
    try:
        import torch
        s += f" torch={torch.__version__} hip={torch.version.hip} cuda={torch.version.cuda} torch_gpu={torch_gpu()}"
    except Exception as e:
        s += f" torch=unavailable ({e})"
    return s


def main():
    D = DEFAULTS
    toggle = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser(description="SharpUnClip: SPADE audio declipper and unlimiter")
    p.add_argument("input", help="input audio (anything libsndfile reads: wav/flac/ogg/...)")
    p.add_argument("-o", "--out", help="output path (default input_declipped.wav)")
    p.add_argument("--preset", help=f"start from a preset ({', '.join(PRESETS)}) or a preset "
                                    ".json saved by the GUI; other flags override it")
    p.add_argument("--clip-above", type=float, default=D["clip_above"],
                   help="%% of peak; louder samples are rebuilt (99.9 = only samples at the ceiling)")
    p.add_argument("--not-clip-below", type=float, default=D["not_clip_below"],
                   help="%% of peak; quieter samples are never touched")
    p.add_argument("--max-restore", type=float, default=D["max_restore"],
                   help="cap rebuilt peaks at this many times the clipped level (0 = no cap; "
                        "2.5 is the most accurate on real songs)")
    p.add_argument("--engine", choices=("NMF", "SPADE"), type=str.upper, default=D["engine"],
                   help="declip engine: NMF (BetterDeclipper's model fusion, most accurate, needs "
                        "PyTorch; GPU via CUDA, ROCm, Intel XPU or Apple MPS) or SPADE (A-SPADE, GPU via OpenCL on any vendor)")
    p.add_argument("--nmf-quality", choices=NMF_QUALITIES, default=D["nmf_quality"],
                   help="NMF engine: speed/quality; normal+spade adds our short-frame SPADE as a third "
                        "model (more accurate on hard clipping, one extra SPADE pass)")
    p.add_argument("--soft-clip", choices=("off", "cubic"), default="off",
                   help="cubic = undo a cubic soft clipper (e.g. PeakEater's cubic curve) before declipping; "
                        "only for audio known to be soft clipped: on hard clips it is much worse")
    p.add_argument("--auto-detect", action=argparse.BooleanOptionalAction, default=D["auto_detect"],
                   help="NMF engine: find the clip/limiter ceilings automatically (off: use "
                        "--clip-above and the other detection flags)")
    p.add_argument("--nmf-knee", type=float, default=D["nmf_knee"],
                   help="NMF auto detection: soft knee as a fraction of each ceiling (0 = as found, "
                        "0.5-0.99 = rebuild from this level, 1 = ceilings only)")
    p.add_argument("--unlimit", dest="unlimit_cli", type=float, nargs="?", const=-1.0, metavar="AMOUNT",
                   help="undo a limiter after declipping, at AMOUNT %% (default 50, the most accurate "
                        "when unlimit runs alone; the Unmaster preset uses 15; 0 = off; needs PyTorch)")
    p.add_argument("--uncompress", dest="uncompress_cli", type=float, nargs="?", const=-1.0, metavar="AMOUNT",
                   help="undo a compressor last, at AMOUNT %% (default 50; 100 is exact only if "
                        "the model fits); give its settings with the --comp-* flags")
    p.add_argument("--comp-threshold", dest="uc_threshold", type=float, default=D["uc_threshold"],
                   help="the compressor's threshold, dBFS")
    p.add_argument("--comp-ratio", dest="uc_ratio", type=float, default=D["uc_ratio"])
    p.add_argument("--comp-attack", dest="uc_attack", type=float, default=D["uc_attack"], help="ms")
    p.add_argument("--comp-release", dest="uc_release", type=float, default=D["uc_release"], help="ms")
    p.add_argument("--comp-makeup", dest="uc_makeup", type=float, default=D["uc_makeup"], help="dB")
    p.add_argument("--process-rate", type=float, default=D["process_rate"], metavar="HZ",
                   help="process the samples as if they were HZ audio (e.g. half the file's rate = "
                        "slowed down), then write at the file's rate again; 0 = off")
    p.add_argument("--comp-link", dest="uc_link", action=toggle, default=D["uc_link"],
                   help="the compressor was stereo-linked")
    p.add_argument("--quality", dest="speed_quality", type=int, default=D["speed_quality"],
                   help="0-6: frame overlap 2..16x (higher = smoother, slower)")
    p.add_argument("--lookahead", type=int, default=D["lookahead"],
                   help="0/1/2: frame length 512/1024/2048")
    p.add_argument("--not-clip-amp", type=float, default=D["not_clip_amp"],
                   help="dBFS; quieter samples are never touched")
    p.add_argument("--peak-ignore", type=float, default=D["peak_ignore"],
                   help="fraction of loudest samples ignored when measuring the peak")
    p.add_argument("--peak-time", type=float, default=D["peak_time"],
                   help="peak measurement window in seconds (only with --no-diff-clips)")
    p.add_argument("--reset-amp", type=float, default=D["reset_amp"])
    p.add_argument("--reset-time", type=float, default=D["reset_time"],
                   help="no effect, kept for older presets")
    p.add_argument("--trans-amp", type=float, default=D["trans_amp"],
                   help="dBFS; amplitude below this held for --trans-time marks a transition")
    p.add_argument("--trans-time", type=float, default=D["trans_time"],
                   help="seconds a transition must last to reset the clip level")
    p.add_argument("--input-gain", type=float, default=D["input_gain"])
    p.add_argument("--output-gain", type=float, default=D["output_gain"])
    p.add_argument("--limiter-peak", type=float, default=D["limiter_peak"])
    p.add_argument("--dynamic", action=toggle, default=D["dynamic"],
                   help="only rebuild above --clip-above (off: also the band down to --not-clip-below)")
    p.add_argument("--declip", action=toggle, default=D["declip"])
    p.add_argument("--linked-stereo", action=toggle, default=D["linked_stereo"],
                   help="rebuild left and right from shared components (less watery stereo)")
    p.add_argument("--remove-glitch", action=toggle, default=D["remove_glitch"],
                   help="ignore single stray samples over the threshold")
    p.add_argument("--diff-clips", action=toggle, default=D["diff_clips"],
                   help="measure the clip level per section between silent gaps "
                        "(mixes/streams with different clip levels per song)")
    p.add_argument("--soft", action=toggle, default=D["soft"],
                   help="also treat near-ceiling (rolled-off/soft) tops as clipped; "
                        "may over-process clean slow peaks")
    p.add_argument("--limiter", action=toggle, default=D["limiter"],
                   help="hard-clip the output at --limiter-peak (cuts restored peaks back off)")
    p.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 1,
                   help="CPU threads (PyTorch CPU backend)")
    p.add_argument("--gpu", nargs="?", const="auto", choices=("auto",) + GPU_DEVS,
                   help="run on a GPU: auto (default) = OpenCL (any vendor) for the SPADE engine, then "
                        "PyTorch's GPU (cuda: NVIDIA or AMD ROCm, xpu: Intel, mps: Apple), which the NMF "
                        "engine always uses; falls back to CPU")
    p.add_argument("--log", metavar="FILE", help="append a run log (params, timings) to FILE")
    p.add_argument("--report", action="store_true", help="print per-channel output peak")
    pre, _ = p.parse_known_args()
    if pre.preset:
        if pre.preset in PRESETS:
            p.set_defaults(**PRESETS[pre.preset])
        elif os.path.isfile(pre.preset):
            with open(pre.preset) as f:
                p.set_defaults(**{k.removeprefix("pa_"): v for k, v in json.load(f).items()})
        else:
            raise SystemExit(f"unknown preset {pre.preset!r} (choose: {', '.join(PRESETS)}, or a .json file)")
    a = p.parse_args()
    for stage, amount in (("unlimit", a.unlimit_cli), ("uncompress", a.uncompress_cli)):
        if amount is not None:  # bare flag: on at the default amount; 0: off
            setattr(a, stage, amount != 0)
            if amount > 0:
                setattr(a, "unlimit_amount" if stage == "unlimit" else "uc_amount", amount)

    if sf is None:
        raise SystemExit("soundfile not installed (pip install soundfile)")

    threads = max(1, a.threads)
    opt = settings_from_cfg(vars(a))
    opt["soft_clip"] = a.soft_clip
    xp, dev = spade_backend(a.gpu)
    if a.gpu and dev not in GPU_DEVS:
        print("no GPU backend (pip install pyopencl), using CPU", file=sys.stderr)
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass
    print("using " + backend_name(xp, dev, opt), file=sys.stderr)

    t0 = time.time()
    log = RunLog(a.log)
    try:
        log(f"SharpUnClip {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"input={a.input} backend={backend_name(xp, dev, opt)} threads={threads}")
        log(env_info())
        x, sr = sf.read(a.input, always_2d=True)  # (n, ch)
        log(f"sr={sr} samples={x.shape[0]} channels={x.shape[1]}")
        log(" ".join(f"{k}={v}" for k, v in opt.items()))
        out_path = a.out or os.path.splitext(a.input)[0] + "_declipped.wav"
        shown, info = [-1], {}

        def progress(f, *_):
            if int(f * 100) != shown[0]:
                shown[0] = int(f * 100)
                print(f"\r{f:5.0%}", end="", file=sys.stderr, flush=True)
        out = process(x, sr, opt, xp=xp, dev=dev, progress=progress, info=info)
        print(file=sys.stderr)
        if opt["declip"] and info.get("flagged") == 0:
            print("declip: no clipping found, so it left the audio unchanged"
                  + (" (on a limited master try --no-auto-detect, or --preset Unmaster)"
                     if opt["engine"] == "NMF" and opt["auto_detect"] else ""), file=sys.stderr)
        if "flagged" in info:
            log(f"declip flagged={info['flagged']:.4%}")
        if a.report:
            for c in range(out.shape[1]):
                print(f"ch{c}: output peak={np.abs(out[:, c]).max():.3f}")
        sf.write(out_path, out, sr, subtype="FLOAT")  # restored peaks exceed 1.0
        log(f"output={out_path} peak={np.abs(out).max():.3f} elapsed={time.time()-t0:.2f}s")
    except BaseException:  # also Ctrl+C: the log says how the run ended
        log("error:\n" + traceback.format_exc())
        raise
    finally:
        log("")
        log.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
