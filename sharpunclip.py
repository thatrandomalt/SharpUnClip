#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 AG89
#
# SPDX-License-Identifier: MIT

"""SharpUnClip: undo clipping, limiting and compression.

Declip rebuilds clipped samples with frame-wise A-SPADE; unlimit (unlimit.py) and
uncompress are optional later stages. See METHODS.md for the methods.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None


def mask_min_run(mask, min_run):
    if min_run <= 1:
        return mask
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    keep = np.zeros_like(mask)
    for s, e in zip(starts, ends):
        if e - s >= min_run:
            keep[s:e] = True
    return keep


def spade_backend(gpu=False):
    """Return (array module, device). gpu: False/"cpu", True/"auto", "opencl" or "cuda".
    auto tries OpenCL, then CUDA; otherwise the CPU (PyTorch if installed, else NumPy)."""
    gpu = "auto" if gpu is True else gpu or "cpu"
    if gpu in ("auto", "opencl"):
        try:
            import sharpunclip_cl
            if sharpunclip_cl.device() is not None:
                return sharpunclip_cl, "opencl"
        except Exception:
            pass
    if gpu in ("auto", "cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return torch, "cuda"
        except Exception:
            pass
    try:
        import torch
        return torch, "cpu"
    except Exception:
        return np, None


def backend_name(xp, dev):
    if dev == "opencl":
        return "OpenCL: " + xp.device().name.strip()
    if dev == "cuda":
        return "CUDA: " + xp.cuda.get_device_name()
    return "CPU"


def aspade(y, lo, hi, w=1024, ov=8, red=2, eps=0.1, maxit=1000, xp=np, dev=None,
           batch=2048, progress=None, group=1):
    """Frame-wise A-SPADE (Kitić, Bertin & Gribonval 2015) with per-sample bounds
    lo <= x <= hi (lo == hi: reliable sample).

    y, lo, hi are (n, ch). Frames are solved in time-ordered batches, so memory stays
    bounded. progress(frac, done, out) is called after each batch with the output so far
    (final up to sample `done`) and with out=None in between; raising from it cancels.
    group > 1 solves that many channels per frame with one shared set of DFT bins."""
    n, ch = y.shape; a = w // ov; pad = w
    L = n + 2 * pad; L += (-L) % a
    P = lambda v: np.pad(v, ((pad, L - n - pad), (0, 0)))
    yp, lop, hip = P(y), P(lo), P(hi)
    starts = np.arange(0, L - w + 1, a)
    g = np.hanning(w + 2)[1:-1]
    acc = np.zeros((L, ch)); ws = np.zeros(L)

    def finish(done):
        # clamp again: fp32 backends can land slightly outside the bounds
        o = np.clip(acc[pad:pad + n] / np.maximum(ws[pad:pad + n, None], 1e-12), lo, hi)
        o[done:] = y[done:]
        return o

    nb = -(-len(starts) // batch)
    for bi in range(nb):
        s = starts[bi * batch:(bi + 1) * batch]; B = len(s)
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
            nxt = (bi + 1) * batch  # samples before the next frame start are final
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


# Defaults are the Accurate preset. The keys are also the GUI field names, the CLI's
# argparse dests and the keys of saved preset files.
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
    "max_restore": 2.5,       # x clipped level, 0 = no cap
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
    "Unmaster": dict(DEFAULTS, clip_above=93.0, not_clip_below=92.9, unlimit=True),
    "SharpUnClip": dict(DEFAULTS, clip_above=78.0, not_clip_below=77.0, speed_quality=3,
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


def _per_clip_peaks(a, peaks, n, sr, trans_amp_db, trans_time, ignore, reset_amp):
    """Fill `peaks` with one peak per section, splitting at gaps quieter than trans_amp
    for at least trans_time. Gap samples take the next section's peak."""
    trans_amp = _db2lin(trans_amp_db)
    trans_win = max(1, int(round(float(sr) * max(0.0, float(trans_time)))))
    low = (a < trans_amp).astype(np.int8)
    d = np.diff(np.concatenate(([0], low, [0])))
    tstarts = np.flatnonzero(d == 1)
    tends = np.flatnonzero(d == -1)
    trans = np.zeros(n, dtype=bool)
    longr = (tends - tstarts) >= trans_win
    for s, e in zip(tstarts[longr], tends[longr]):
        trans[s:e] = True
    ntrans = (~trans).astype(np.int8)
    d2 = np.diff(np.concatenate(([0], ntrans, [0])))
    cstarts = np.flatnonzero(d2 == 1)
    cends = np.flatnonzero(d2 == -1)
    if cstarts.size == 0:
        peaks[:] = reset_amp
        return
    clip_of = np.full(n, -1, dtype=np.int64)
    seg_peak = np.empty(cstarts.size, dtype=np.float64)
    for k in range(cstarts.size):
        s, e = int(cstarts[k]), int(cends[k])
        seg = a[s:e]
        pk = float(np.quantile(seg, 1.0 - ignore)) if seg.size > 1 else float(seg[0])
        seg_peak[k] = max(pk, reset_amp)
        clip_of[s:e] = k
    valid_idx = np.flatnonzero(clip_of >= 0)
    valid_ids = clip_of[valid_idx]
    pos = np.searchsorted(valid_idx, np.arange(n), side="left")
    # a trailing gap takes the last section's peak
    peaks[:] = seg_peak[valid_ids[np.minimum(pos, valid_ids.size - 1)]]


def _clip_mask(y0, sr, opt):
    """Which samples of one channel are clipped."""
    n = len(y0)
    a = np.abs(y0)
    peaks = np.empty(n, dtype=np.float64)

    win = max(1, int(round(float(sr) * max(0.001, float(opt["peak_time"])))))
    reset_win = max(1, int(round(float(sr) * max(0.001, float(opt["reset_time"])))))
    reset_amp = _db2lin(opt["reset_amp"])
    floor = _db2lin(opt["not_clip_amp"])
    low = float(opt["not_clip_below"]) / 100.0
    ignore = min(0.999, max(0.0, float(opt["peak_ignore"])))

    if opt["diff_clips"]:
        _per_clip_peaks(a, peaks, n, sr, float(opt["trans_amp"]), float(opt["trans_time"]),
                        ignore, reset_amp)
    else:
        for s in range(0, n, win):
            e = min(n, s + win)
            aa = a[s:e]
            pk = float(np.quantile(aa, 1.0 - ignore)) if aa.size > 1 else float(aa[0])
            if aa.max() < reset_amp and (e - s) >= reset_win:
                pk = reset_amp
            pk = max(pk, reset_amp)
            peaks[s:e] = pk

    # dynamic: clipped above clip_above; otherwise above not_clip_below
    high = float(opt["clip_above"]) / 100.0
    mask = a >= peaks * (high if opt["dynamic"] else min(low, high))
    if opt["soft"]:
        mask |= a >= (1.0 - 0.08) * peaks
    mask &= a >= floor

    if opt["remove_glitch"]:
        mask = mask_min_run(mask, 2)
    return mask


def process(x, sr, opt=None, xp=np, dev=None, progress=None):
    """Declip, then unlimit, then uncompress x ((n,) or (n, ch)), as enabled in `opt`
    (see DEFAULTS). progress: see aspade."""
    opt = dict(DEFAULTS, **(opt or {}))
    pr = float(opt["process_rate"] or 0)
    if pr > 0 and abs(pr - sr) > 0.5:
        # Same samples, treated as pr Hz. The compressor times describe the real
        # compressor, so rescale them to the new time base.
        k = sr / pr
        opt = dict(opt, process_rate=0.0, uc_attack=float(opt["uc_attack"]) * k,
                   uc_release=float(opt["uc_release"]) * k)
        return process(x, pr, opt, xp, dev, progress)
    x = np.asarray(x, dtype=np.float64)
    mono = x.ndim == 1
    y0 = (x[:, None] if mono else x) * _db2lin(opt["input_gain"])
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
        lo, hi = y0.copy(), y0.copy()
        for c in range(y0.shape[1]):
            m = _clip_mask(y0[:, c], sr, opt)
            neg, pos = m & (y0[:, c] < 0), m & (y0[:, c] > 0)
            lo[neg, c] = cap * y0[neg, c]
            hi[pos, c] = cap * y0[pos, c]
        if np.any(lo != hi):
            quality = max(0, min(6, int(opt["speed_quality"])))
            w = 512 << max(0, min(2, int(opt["lookahead"])))
            ov = (2, 4, 4, 8, 8, 16, 16)[quality]
            s = 1.0 / np.abs(y0).max()  # eps is absolute, so normalise the level
            at = share("declip")
            prog = progress and (lambda f, d, o: progress(at(f), d, None if o is None else o / s))
            grp = 2 if opt["linked_stereo"] and y0.shape[1] == 2 else 1
            out = aspade(y0 * s, lo * s, hi * s, w, ov, xp=xp, dev=dev, progress=prog, group=grp) / s
            out = np.clip(out, lo, hi)  # keeps untouched samples bit-exact
    if y0.size and ul > 0:
        import unlimit  # needs PyTorch
        at = share("unlimit")
        out = out * unlimit.gain(y0, sr, ul, progress and (lambda f: progress(at(f), None, None)))[:, None]
    if y0.size and uc > 0:
        at = share("uncompress")
        g = uncompress_gain(out, sr, opt["uc_threshold"], opt["uc_ratio"], opt["uc_attack"],
                            opt["uc_release"], float(opt["uc_makeup"]), bool(opt["uc_link"]),
                            progress and (lambda f: progress(at(f), None, None)))
        out = out * 10 ** (g * uc / 20)
    if progress and (ul > 0 or uc > 0):
        progress(1.0, len(out), out)
    out *= _db2lin(opt["output_gain"])
    if opt["limiter"]:
        lim = _db2lin(opt["limiter_peak"])
        if lim > 0:
            out = np.clip(out, -lim, lim)
    return out[:, 0] if mono else out


def _write_log(path, lines):
    with open(path, "a") as f:
        for ln in lines:
            f.write(ln + "\n")


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
    p.add_argument("--unlimit", dest="unlimit_cli", type=float, nargs="?", const=-1.0, metavar="AMOUNT",
                   help="undo a limiter after declipping, at AMOUNT %% (default 50, the most "
                        "accurate on real masters; 0 = off; needs PyTorch)")
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
                   help="peak measurement window in seconds")
    p.add_argument("--reset-amp", type=float, default=D["reset_amp"])
    p.add_argument("--reset-time", type=float, default=D["reset_time"],
                   help="reset duration in seconds")
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
    p.add_argument("--gpu", nargs="?", const="auto", choices=("auto", "opencl", "cuda"),
                   help="run on a GPU: auto (default) = OpenCL (any vendor) then CUDA; "
                        "falls back to CPU")
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
    xp, dev = spade_backend(a.gpu)
    if a.gpu and dev not in ("opencl", "cuda"):
        print("no GPU backend (pip install pyopencl), using CPU", file=sys.stderr)
    if dev == "cpu":
        xp.set_num_threads(threads)
    print("using " + backend_name(xp, dev), file=sys.stderr)

    t0 = time.time()
    log_lines = [f"SharpUnClip {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 f"input={a.input} backend={backend_name(xp, dev)} threads={threads}"]
    x, sr = sf.read(a.input, always_2d=True)  # (n, ch)
    log_lines.append(f"sr={sr} samples={x.shape[0]} channels={x.shape[1]}")
    opt = settings_from_cfg(vars(a))
    log_lines.append(" ".join(f"{k}={v}" for k, v in opt.items()))
    out_path = a.out or os.path.splitext(a.input)[0] + "_declipped.wav"
    out = process(x, sr, opt, xp=xp, dev=dev,
                  progress=lambda f, *_: print(f"\r{f:5.0%}", end="", file=sys.stderr, flush=True))
    print(file=sys.stderr)
    if a.report:
        for c in range(out.shape[1]):
            print(f"ch{c}: output peak={np.abs(out[:, c]).max():.3f}")
    sf.write(out_path, out, sr, subtype="FLOAT")  # restored peaks exceed 1.0
    log_lines.append(f"output={out_path} peak={np.abs(out).max():.3f} elapsed={time.time()-t0:.2f}s")
    if a.log:
        _write_log(a.log, log_lines)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
