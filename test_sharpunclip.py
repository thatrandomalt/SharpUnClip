# SPDX-FileCopyrightText: 2026 thatrandomalt
#
# SPDX-License-Identifier: MIT

import math, os, subprocess, sys, tempfile, types
import numpy as np
import soundfile as sf
import sharpunclip as U
import session

SPADE = dict(U.PRESETS["Accurate SPADE"])


def _backends():
    """Yield "cpu", PyTorch's GPU if present ("cuda", "xpu" or "mps"), and "opencl" once per OpenCL GPU
    (selected through SHARPUNCLIP_OPENCL_DEVICE while that name is being tested)."""
    yield "cpu"
    if U.torch_gpu():
        yield U.torch_gpu()
    try:
        import pyopencl as cl
        names = [d.name for p in cl.get_platforms() for d in p.get_devices()
                 if d.type & cl.device_type.GPU]
    except Exception:
        names = []
    for n in names:
        os.environ["SHARPUNCLIP_OPENCL_DEVICE"] = n
        try:
            yield "opencl"
        finally:
            del os.environ["SHARPUNCLIP_OPENCL_DEVICE"]



def test_declip_improves_hard_clip():
    sr = 16000
    t = np.arange(sr * 2) / sr
    rng = np.random.RandomState(2)
    k = 6 * np.exp(-t * 3) * np.sin(2 * np.pi * 55 * t)
    b = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 660 * t)
    s = k + b + 0.03 * rng.randn(t.size)
    src = 1.4 * s / np.max(np.abs(s))
    cl = np.clip(src, -0.9, 0.9)
    opt = dict(SPADE)
    opt["lookahead"] = 2  # 2048 frames: a 55 Hz kick needs several periods per frame
    opt["limiter"] = False
    out = U.process(cl, sr, opt)
    before = np.sum((cl - src) ** 2)
    after = np.sum((out - src) ** 2)
    gain = 10 * np.log10(before / after)
    assert gain > 1.0, f"SNR gain only {gain:.2f} dB"
    assert not np.any(np.isnan(out))


def test_declip_below_clip_above_is_restored():
    # Everything above peak*clip_above is rebuilt (never quieter), everything below
    # is kept, even when the clipped tops aren't perfectly flat.
    sr = 16000
    t = np.arange(sr) / sr
    src = 1.6 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 1700 * t)
    cl = np.clip(src, -1, 1) + 0.002 * np.sin(2 * np.pi * 3100 * t)  # no flat runs
    opt = dict(SPADE, clip_above=78.0, not_clip_below=77.0, peak_ignore=0.001,
              limiter=False)
    m = np.abs(cl) >= 0.78 * np.quantile(np.abs(cl), 0.999)
    for name in _backends():
        out = U.process(cl, sr, opt, *U.spade_backend(name))
        assert np.array_equal(out[~m], cl[~m]), name
        assert np.all(np.abs(out[m]) >= np.abs(cl[m]) - 1e-6), name
        gain = 10 * np.log10(np.sum((cl - src) ** 2) / np.sum((out - src) ** 2))
        assert gain > 3.0, f"SNR gain only {gain:.2f} dB on {name}"


def test_linked_stereo_uses_the_clean_channel():
    # With linked stereo a clean R guides the clipped L's rebuild; R stays bit-exact.
    sr = 16000
    t = np.arange(sr) / sr
    src = 1.6 * np.sin(2 * np.pi * 110 * t) + 0.5 * np.sin(2 * np.pi * 1300 * t + 1) \
        + 0.3 * np.sin(2 * np.pi * 2900 * t + 2)
    st = np.stack([np.clip(src, -1, 1), 0.5 * src], 1)
    opt = dict(SPADE, max_restore=0.0, diff_clips=False)
    for name in _backends():
        alone = U.process(st, sr, dict(opt, linked_stereo=False), *U.spade_backend(name))
        linked = U.process(st, sr, opt, *U.spade_backend(name))
        err = lambda o: 10 * np.log10(np.sum(src ** 2) / np.sum((o[:, 0] - src) ** 2))
        assert err(linked) > err(alone) + 3, f"{name}: linked {err(linked):.1f} vs alone {err(alone):.1f} dB"
        assert np.array_equal(linked[:, 1], st[:, 1]), name


def test_max_restore_cap_and_level_invariance():
    # |in| <= |out| <= cap*|in| on rebuilt samples, and the result doesn't depend on level.
    sr = 16000
    t = np.arange(sr) / sr
    src = 2.5 * np.sin(2 * np.pi * 70 * t) + 0.3 * np.sin(2 * np.pi * 900 * t)
    cl = np.clip(src, -1, 1)
    opt = dict(SPADE, clip_above=99.9, not_clip_below=99.8, max_restore=1.5)
    out = U.process(cl, sr, opt)
    m = np.abs(cl) >= 0.999
    assert np.all(np.abs(out[m]) <= 1.5 * np.abs(cl[m]) + 1e-9)
    assert np.all(np.abs(out[m]) >= np.abs(cl[m]) - 1e-9)
    quiet = U.process(0.25 * cl, sr, opt) / 0.25
    assert np.max(np.abs(quiet - out)) < 1e-6, np.max(np.abs(quiet - out))


def test_spade_progress_snapshots_are_final():
    # Each progress snapshot is already final up to `done`; reporting progress changes nothing.
    sr = 16000
    t = np.arange(3 * sr) / sr
    src = np.stack([1.6 * np.sin(2 * np.pi * 110 * t), 1.3 * np.sin(2 * np.pi * 165 * t + 1)], 1)
    cl = np.clip(src, -1, 1)
    opt = dict(SPADE, speed_quality=0)
    lo, hi = cl.copy(), cl.copy()
    m = np.abs(cl) >= 0.999
    lo[m & (cl < 0)] *= 2.5
    hi[m & (cl > 0)] *= 2.5
    for name in _backends():
        xp, dev = U.spade_backend(name)
        snaps = []
        out = U.aspade(cl * 0.5, lo * 0.5, hi * 0.5, 512, 2, xp=xp, dev=dev, batch=16, unscale=0.5,
                       progress=lambda f, d, o: o is not None and snaps.append((d, o[:d].copy())))
        assert len(snaps) > 5 and snaps[-1][0] == len(cl), (name, len(snaps))
        for d, o in snaps:
            assert np.array_equal(o, out[:d]), (name, d)
        seen = U.process(cl, sr, opt, xp, dev, progress=lambda f, d, o: None)
        assert np.array_equal(seen, U.process(cl, sr, opt, xp, dev)), name


def _kick_stereo(sr, secs=2, seed=2):
    t = np.arange(secs * sr) / sr
    rng = np.random.RandomState(seed)
    k = 6 * np.exp(-(t % 0.5) * 6) * np.sin(2 * np.pi * 55 * t)
    b = np.stack([0.4 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 660 * t),
                  0.3 * np.sin(2 * np.pi * 330 * t + 1) + 0.1 * np.sin(2 * np.pi * 990 * t)], 1)
    s = k[:, None] + b + 0.02 * rng.randn(t.size, 2)
    return 1.4 * s / np.max(np.abs(s))


def test_nmf_engine_auto():
    # NMF engine, automatic analysis: finds the ceiling, keeps everything below it, improves.
    if not _has_torch():
        return  # the NMF engine needs PyTorch
    sr = 16000
    src = _kick_stereo(sr)
    cl = np.clip(src, -0.9, 0.9)
    seen = []
    for dev in ("cpu",) + ((U.torch_gpu(),) if U.torch_gpu() else ()):
        out = U.process(cl, sr, dict(U.DEFAULTS, max_restore=0), dev=dev,
                        progress=lambda f, d, o: seen.append((f, d)))
        keep = np.abs(cl) < 0.9 * 0.999
        assert np.array_equal(out[keep], cl[keep]), dev
        gain = 10 * np.log10(np.sum((cl - src) ** 2) / np.sum((out - src) ** 2))
        assert gain > 3.0, f"{dev}: SNR gain only {gain:.2f} dB"
        assert seen[-1] == (1.0, len(cl)) and all(0 <= f <= 1 for f, _ in seen), seen[-3:]


def test_lsb_estimate_ignores_silent_intro():
    # A silent intro must not make 16-bit audio look like 8-bit (a bug in the original estimator).
    from betterdeclipper.detect import estimate_lsb
    rng = np.random.default_rng(0)
    noise = np.round(rng.uniform(-0.5, 0.5, (48000 * 5, 2)) * 32768) / 32768
    assert estimate_lsb(np.vstack([np.zeros((48000 * 3, 2)), noise])) == 1 / 32768
    assert estimate_lsb(np.zeros((48000, 2))) == 1e-7


def test_nmf_engine_manual_detection_bounds():
    # auto_detect off: our clip_above masks and the max_restore cap become its constraints.
    if not _has_torch():
        return  # the NMF engine needs PyTorch
    sr = 16000
    src = _kick_stereo(sr, seed=5)
    cl = np.clip(src, -0.9, 0.9)
    opt = dict(U.DEFAULTS, auto_detect=False, clip_above=95.0, not_clip_below=94.9, max_restore=1.2)
    out = U.process(cl, sr, opt)
    m = np.zeros(cl.shape, bool)
    for c in range(2):
        m[:, c] = U._clip_mask(cl[:, c], sr, U.settings_from_cfg(opt))
    assert m.any() and np.array_equal(out[~m], cl[~m])
    assert np.all(np.abs(out[m]) >= np.abs(cl[m]) - 1e-6)
    assert np.all(np.abs(out[m]) <= 1.2 * np.abs(cl[m]) + 1e-6)
    assert np.all(np.sign(out[m]) == np.sign(cl[m]))


def test_declip_improves_soft_clip():
    sr = 16000
    t = np.arange(sr * 2) / sr
    rng = np.random.RandomState(3)
    k = 6 * np.exp(-t * 3) * np.sin(2 * np.pi * 55 * t)
    b = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 660 * t)
    s = k + b + 0.03 * rng.randn(t.size)
    src = 1.4 * s / np.max(np.abs(s))
    lvl = 0.9
    cl = lvl * np.tanh(src / lvl)
    opt = dict(SPADE)
    opt["speed_quality"] = 3
    opt["limiter"] = False
    opt["soft"] = True
    out = U.process(cl, sr, opt)
    before = np.sum((cl - src) ** 2)
    after = np.sum((out - src) ** 2)
    gain = 10 * np.log10(before / after)
    assert gain > 1.0, f"soft-clip SNR gain only {gain:.2f} dB"
    assert not np.any(np.isnan(out))


def test_declip_diff_clips():
    sr = 16000
    t = np.arange(sr) / sr
    tone = 0.8 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)
    tone = tone / np.max(np.abs(tone))
    silence = np.zeros(int(0.3 * sr))
    # Clip B tops out below clip A's level, so only per-section peaks find it.
    # The trailing silence must not break the peak lookup.
    cl = np.concatenate([np.clip(tone, -0.9, 0.9), silence, np.clip(tone, -0.85, 0.85), silence])
    src = np.concatenate([tone, silence, tone, silence])
    opt = dict(SPADE)
    opt["speed_quality"] = 3
    opt["limiter"] = False
    opt["diff_clips"] = True
    opt["trans_amp"] = -40.0
    opt["trans_time"] = 0.2
    out = U.process(cl, sr, opt)
    before = np.sum((cl - src) ** 2)
    after = np.sum((out - src) ** 2)
    gain = 10 * np.log10(before / after)
    assert gain > 1.0, f"diff_clips SNR gain only {gain:.2f} dB"
    assert not np.any(np.isnan(out))


def test_unlimit_linked_gain():
    # Unlimit = one gain per sample, the same on L and R, body (median) level kept.
    try:
        import unlimit
    except ImportError:
        return  # PyTorch not installed: unlimit is optional
    sr = 48000
    t = np.arange(2 * sr) / sr
    beat = np.exp(-(t % 0.5) * 12) * np.sin(2 * np.pi * 60 * t)
    x = np.tanh(3 * np.stack([beat + 0.2 * np.sin(2 * np.pi * 440 * t)] * 2, 1)) * 0.9
    g = unlimit.gain(x, sr, 0.5, dev="cpu")  # process() without a GPU: the CPU
    assert g.shape == (len(x),) and np.all(np.isfinite(g)) and np.all(g > 0)
    assert abs(np.median(g) - 1) < 1e-9
    out = U.process(x, sr, dict(U.DEFAULTS, declip=False, unlimit=50))
    assert np.array_equal(out, x * g[:, None])  # unlimit alone: exactly the input times the gain


def test_unlimit_adds_declip_correction():
    # With declip on, unlimit gains the input and adds declip's correction unscaled: y*g + (declip - y).
    try:
        import unlimit
    except ImportError:
        return
    sr = 48000
    src = _kick_stereo(sr, secs=2)
    cl = np.clip(src, -0.9, 0.9)
    opt = dict(SPADE, unlimit=True, unlimit_amount=15.0)
    d = U.process(cl, sr, dict(opt, unlimit=False))
    g = unlimit.gain(cl, sr, 0.15, dev="cpu")  # process() without a GPU: the CPU
    out = U.process(cl, sr, opt)
    assert np.allclose(out, cl * g[:, None] + (d - cl), rtol=0, atol=1e-12)


def test_cubic_soft_clip_inverse():
    # soft_clip="cubic" undoes y = c*(w - w^3/6.75) exactly below the ceiling, then declips as usual.
    c = 0.8
    u = np.linspace(-1.5 * c, 1.5 * c, 4001)
    w = u / c
    assert np.max(np.abs(U.cubic_inverse(c * (w - w ** 3 / 6.75), c) - u)) < 1e-9
    sr = 16000
    src = _kick_stereo(sr, secs=1, seed=4)
    ww = np.clip(src / np.abs(src).max() * 2.0, -1.5, 1.5)
    soft = 0.9 * (ww - ww ** 3 / 6.75)
    opt = dict(SPADE, speed_quality=1)
    a = U.process(soft, sr, dict(opt, soft_clip="cubic"))
    b = U.process(U.cubic_inverse(soft, np.abs(soft).max()), sr, opt)
    assert np.array_equal(a, b)
    assert np.array_equal(U.process(soft, sr, opt), U.process(soft, sr, dict(opt, soft_clip="off")))


def test_nmf_normal_plus_spade():
    # NMF quality "normal+spade" (our short-frame SPADE as a third fused model): same invariants as the
    # engine (untouched samples bit-exact, rebuilt ones keep sign and grow, cap holds), and it improves.
    if not _has_torch():
        return  # the NMF engine needs PyTorch
    sr = 16000
    src = _kick_stereo(sr)
    cl = np.clip(src, -0.9, 0.9)
    out = U.process(cl, sr, dict(U.DEFAULTS, nmf_quality="normal+spade"))
    keep = np.abs(cl) < 0.9 * 0.999
    assert np.array_equal(out[keep], cl[keep])
    gain = 10 * np.log10(np.sum((cl - src) ** 2) / np.sum((out - src) ** 2))
    assert gain > 3.0, f"SNR gain only {gain:.2f} dB"
    opt = dict(U.DEFAULTS, nmf_quality="normal+spade", auto_detect=False, clip_above=95.0, not_clip_below=94.9,
               max_restore=1.2)
    out = U.process(cl[:, 0], sr, opt)  # mono, manual detection, capped
    m = U._clip_mask(cl[:, 0], sr, U.settings_from_cfg(opt))
    assert m.any() and np.array_equal(out[~m], cl[~m, 0])
    assert np.all(np.abs(out[m]) >= np.abs(cl[m, 0]) - 1e-9) and np.all(np.abs(out[m]) <= 1.2 * np.abs(cl[m, 0]) + 1e-9)
    assert np.all(np.sign(out[m]) == np.sign(cl[m, 0]))


def _compress(x, sr, T, R, attack_ms, release_ms, makeup=0.0):
    """The compressor model that uncompress inverts, per channel."""
    k9 = math.log(9)
    aA, aR = math.exp(-k9 / (attack_ms / 1000 * sr)), math.exp(-k9 / (release_ms / 1000 * sr))
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        X = 20 * np.log10(np.maximum(np.abs(x[:, ch]), 1e-12))
        s, g = 0.0, []
        for v in np.minimum(0, -(1 - 1 / R) * (X - T)).tolist():
            s = aA * s + (1 - aA) * v if v < s else aR * s + (1 - aR) * v
            g.append(s)
        out[:, ch] = x[:, ch] * 10 ** ((np.array(g) + makeup) / 20)
    return out


def _drums(sr, secs=1):
    t = np.arange(secs * sr) / sr
    hit = np.exp(-(t % 0.25) * 18) * np.sin(2 * np.pi * 70 * t) * 1.8
    return np.stack([hit + 0.2 * np.sin(2 * np.pi * 523 * t), hit + 0.2 * np.sin(2 * np.pi * 659 * t)], 1)


def test_uncompress_inverts_compressor():
    # 100% = exact inverse of the model; 50% = exactly half the correction in dB
    sr = 16000
    x = _drums(sr)
    y = _compress(x, sr, -12, 4, 5, 100, makeup=3)
    opt = dict(U.DEFAULTS, declip=False, uncompress=True, uc_threshold=-12, uc_ratio=4,
               uc_attack=5, uc_release=100, uc_makeup=3, uc_amount=100)
    full = U.process(y, sr, opt)
    assert np.max(np.abs(full - x)) < 1e-9, np.max(np.abs(full - x))
    # the compressor times stay correct at another process rate
    slow = U.process(y, sr, dict(opt, process_rate=sr / 2))
    assert np.max(np.abs(slow - x)) < 1e-9, np.max(np.abs(slow - x))
    half = U.process(y, sr, dict(opt, uc_amount=50))
    k = np.abs(y) > 1e-3
    assert np.allclose(np.log(np.abs(half[k] / y[k])), 0.5 * np.log(np.abs(full[k] / y[k])))


def test_cli_uncompress():
    sr = 16000
    x = _drums(sr)
    d = tempfile.mkdtemp()
    inp, outp = os.path.join(d, "comp.wav"), os.path.join(d, "out.wav")
    sf.write(inp, _compress(x, sr, -12, 4, 5, 100), sr, subtype="FLOAT")
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "sharpunclip.py"), inp, "-o", outp, "--no-declip",
                        "--uncompress", "100", "--comp-threshold", "-12", "--comp-ratio", "4",
                        "--comp-attack", "5", "--comp-release", "100"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = sf.read(outp, always_2d=True)[0]
    assert np.max(np.abs(out - x)) < 1e-5, np.max(np.abs(out - x))  # float32 WAV round trip


def test_cli_presets():
    sr = 16000
    t = np.arange(sr) / sr
    st = np.stack([np.clip(1.5 * np.sin(2 * np.pi * 220 * t), -1, 1)] * 2, axis=1)
    d = tempfile.mkdtemp()
    inp, outp = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
    sf.write(inp, st, sr, subtype="PCM_16")
    cli = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sharpunclip.py"), inp, "-o", outp]
    if _has_torch():  # Accurate: the NMF engine
        r = subprocess.run(cli + ["--preset", "Accurate", "--log", outp + ".log"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        log = open(outp + ".log").read()
        assert "clip_above=99.9" in log and "max_restore=0.0" in log and "engine=NMF" in log, log
        assert np.abs(sf.read(outp)[0]).max() > 1.05
    r = subprocess.run(cli + ["--engine", "spade", "--log", outp + ".log"], capture_output=True, text=True)
    assert r.returncode == 0 and "engine=SPADE" in open(outp + ".log").read(), r.stderr
    assert np.abs(sf.read(outp)[0]).max() > 1.05
    r = subprocess.run(cli + ["--preset", "nope"], capture_output=True, text=True)
    assert r.returncode != 0 and "unknown preset" in r.stderr


def test_cli_stereo_roundtrip():
    # CLI end to end: stereo in, float WAV out, restored peaks above full scale,
    # progress reported.
    sr = 16000
    t = np.arange(sr) / sr
    src = 1.5 * np.sin(2 * np.pi * 220 * t)
    st = np.stack([np.clip(src, -1, 1), np.clip(0.8 * src, -1, 1)], axis=1)
    d = tempfile.mkdtemp()
    inp, outp = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
    sf.write(inp, st, sr, subtype="PCM_16")
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "sharpunclip.py"), inp, "-o", outp,
                        "--clip-above", "95", "--not-clip-below", "94"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "100%" in r.stderr, r.stderr
    out, sr2 = sf.read(outp, always_2d=True)
    assert sr2 == sr and out.shape == st.shape and sf.info(outp).subtype == "FLOAT"
    assert np.abs(out).max() > 1.1, np.abs(out).max()


# The mask helpers before the peak_steps/flag_mask split, verbatim, as the reference.
def _old_mask_min_run(mask, min_run):
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


def _old_per_clip_peaks(a, peaks, n, sr, trans_amp_db, trans_time, ignore, reset_amp):
    trans_amp = U._db2lin(trans_amp_db)
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
    peaks[:] = seg_peak[valid_ids[np.minimum(pos, valid_ids.size - 1)]]


def _old_clip_mask(y0, sr, opt):
    n = len(y0)
    a = np.abs(y0)
    peaks = np.empty(n, dtype=np.float64)

    win = max(1, int(round(float(sr) * max(0.001, float(opt["peak_time"])))))
    reset_win = max(1, int(round(float(sr) * max(0.001, float(opt["reset_time"])))))
    reset_amp = U._db2lin(opt["reset_amp"])
    floor = U._db2lin(opt["not_clip_amp"])
    low = float(opt["not_clip_below"]) / 100.0
    ignore = min(0.999, max(0.0, float(opt["peak_ignore"])))

    if opt["diff_clips"]:
        _old_per_clip_peaks(a, peaks, n, sr, float(opt["trans_amp"]), float(opt["trans_time"]),
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

    high = float(opt["clip_above"]) / 100.0
    mask = a >= peaks * (high if opt["dynamic"] else min(low, high))
    if opt["soft"]:
        mask |= a >= (1.0 - 0.08) * peaks
    mask &= a >= floor

    if opt["remove_glitch"]:
        mask = _old_mask_min_run(mask, 2)
    return mask


def test_mask_min_run_identical():
    rng = np.random.default_rng(1)
    masks = [rng.random(5000) < p for p in (0.1, 0.5, 0.9)]
    masks += [np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], bool), np.array([0, 1, 1, 0, 0, 1, 0], bool),
              np.ones(4, bool), np.zeros(4, bool), np.zeros(0, bool)]
    for m in masks:
        for k in (1, 2, 3):
            assert np.array_equal(U.mask_min_run(m, k), _old_mask_min_run(m, k)), (k, m[:10])


def test_clip_mask_refactor_identical():
    # peak_steps/expand_steps/flag_mask flag exactly what the pre-split _clip_mask flagged
    sr = 8000
    rng = np.random.default_rng(7)
    t = np.arange(sr) / sr
    tone = np.clip(1.3 * np.sin(2 * np.pi * 110 * t) + 0.1 * rng.standard_normal(sr), -1, 1)
    hiss = lambda k: 1e-4 * rng.standard_normal(k)
    edges = hiss(sr)
    edges[0], edges[-1], edges[sr // 2:sr // 2 + 3] = 0.9, -0.8, 0.7  # 1-sample sections at both ends
    quiet = 0.03 * np.sin(2 * np.pi * 50 * t[:4000])  # above the -60 dB floor, below the -20 dB gap level
    signals = {
        "gaps": np.concatenate([hiss(2400), tone, hiss(2400), 0.8 * tone[:3000], hiss(50), tone[:2000],
                                hiss(2400)]),
        "silent": np.zeros(sr),
        "all gap": quiet,
        "single": tone,
        "edges": edges,
        "short quiet runs": tone * (rng.random(sr) > 0.05),
        # quiet leading/trailing gaps, an inner gap exactly 0.05 s long, a quieter last partial 0.4 s block
        "quiet gaps": np.concatenate([quiet, np.zeros(800), tone, 0.9 * np.ones(100), 0.02 * np.ones(400),
                                      0.5 * np.ones(100), 0.5 * tone, quiet[:3333]]),
    }
    opts = []
    for i in range(12):
        lo = 93 if i & 2 else 60  # Soft's 0.92 band only matters above the threshold
        opts.append(dict(U.DEFAULTS, diff_clips=bool(i & 1), soft=bool(i & 2), remove_glitch=bool(i & 4),
                         dynamic=i % 3 != 0, clip_above=rng.uniform(max(lo, 70), 99.99),
                         not_clip_below=rng.uniform(lo, 99.9), not_clip_amp=rng.uniform(-60, -10),
                         peak_ignore=rng.choice([0, 0.001, 0.01, 0.2]), peak_time=rng.choice([0.0001, 0.05, 0.4, 2.0]),
                         reset_amp=rng.uniform(-90, -40), reset_time=rng.choice([0.0001, 0.2]),
                         trans_amp=rng.uniform(-70, -20), trans_time=rng.choice([0, 0.0001, 0.05, 0.3])))
    quiet_opt = dict(U.DEFAULTS, trans_amp=-20.0, trans_time=0.05, not_clip_amp=-60.0, reset_amp=-35.0,
                     clip_above=95.0, peak_ignore=0.0, remove_glitch=False)
    opts += [quiet_opt, dict(quiet_opt, soft=True, clip_above=99.0), dict(quiet_opt, diff_clips=False),
             dict(quiet_opt, trans_amp=-50.0, reset_amp=-25.0)]  # a section quieter than Reset level
    for i, opt in enumerate(opts):
        for name, y in signals.items():
            assert np.array_equal(U._clip_mask(y, sr, opt), _old_clip_mask(y, sr, opt)), (i, name, opt)


def _has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def test_process_hints_equivalent():
    # precomputed masks and norm (GUI previews) give the same output
    sr = 16000
    x = np.clip(_kick_stereo(sr, 1), -0.9, 0.9)
    names = [n for n, p in U.PRESETS.items() if p["engine"] == "SPADE"] + (["Unmaster"] if _has_torch() else [])
    for name in names:
        opt = U.settings_from_cfg(dict(U.PRESETS[name], unlimit=False, input_gain=2.0, nmf_quality="fast"))
        y0 = x * U._db2lin(opt["input_gain"])
        h = dict(masks=[U._clip_mask(y0[:, c], sr, opt) for c in range(2)], norm=1 / np.abs(y0).max())
        be = U.spade_backend("cpu")
        assert np.array_equal(U.process(x, sr, opt, *be, hints=h), U.process(x, sr, opt, *be)), name


def test_spade_region_preview_exact():
    # span on the frame grid + file-global masks and norm + stage B = the render's samples, also
    # across one of its 2048-frame batches (at 65024), on every backend; a span 7 samples off is not exact
    sr = 48000
    n = int(1.5 * sr)
    t = np.arange(n) / sr
    rng = np.random.RandomState(4)
    s = np.stack([1.6 * np.sin(2 * np.pi * 110 * t) + 0.3 * rng.randn(n),
                  1.3 * np.sin(2 * np.pi * 165 * t + 1) + 0.2 * rng.randn(n)], 1)
    loud = (t < 0.27) | (t > 1.15)  # densely clipped only around the regions
    x = np.clip(s * np.where(loud, 1.0, 0.3)[:, None], -1, 1)
    song = types.SimpleNamespace(x=x, sr=sr)
    cfg = U.settings_from_cfg(U.PRESETS["Accurate SPADE"])
    y0 = x * U._db2lin(cfg["input_gain"])
    masks = [U._clip_mask(y0[:, c], sr, cfg) for c in range(2)]
    for be in _backends():
        xp, dev = U.spade_backend(be)
        full = U.process(x, sr, cfg, xp, dev)
        for r0, r1, off in ((0, 12000, 0), (60000, 70000, 0), (60000, 70000, 7)):
            S, E = session.preview_span(song, cfg, r0, r1, False)
            S -= off
            D = session.stage_a(song, session.stage_cfg_a(cfg, False), S, E, xp, dev,
                                dict(masks=[m[S:E] for m in masks], norm=1 / np.abs(y0).max()))
            B = session.stage_b(session.stage_cfg_b(cfg), D, sr, {})
            assert np.array_equal(B[r0 - S:r1 - S], full[r0:r1]) != bool(off), (be, r0, off)


def test_preview_span_nmf():
    # NMF spans: 1.5 s context (0.75 s fast), widened until the NMF rank reaches its cap (5.1 s stereo at 48 kHz)
    if not _has_torch():
        return
    sr = 48000
    song = types.SimpleNamespace(x=np.zeros((20 * sr, 2)), sr=sr)
    cfg = U.settings_from_cfg(U.PRESETS["Accurate"])
    assert session.preview_span(song, cfg, 10 * sr, 11 * sr, True) == (int(9.25 * sr), int(11.75 * sr))
    S, E = session.preview_span(song, cfg, 10 * sr, 11 * sr, False)
    assert 5.0 < (E - S) / sr < 5.2 and S < 9.5 * sr and E > 11.5 * sr
    S, E = session.preview_span(song, cfg, 0, sr, False)
    assert S == 0 and 5.0 < E / sr < 5.2


def test_stage_split_equals_one_call():
    # stage B (unlimit, uncompress, output) of stage A (declip, input gain) = one process() call
    sr = 16000
    x = np.clip(1.3 * _compress(_drums(sr, 2), sr, -12, 4, 5, 100), -1, 1)
    song = types.SimpleNamespace(x=x, sr=sr)
    base = dict(U.PRESETS["Uncompress"], unlimit=_has_torch(), input_gain=1.5, output_gain=-2.0)
    for cfg in (base, dict(base, declip=True, engine="SPADE", limiter=True)):
        cfg = U.settings_from_cfg(cfg)
        D = session.stage_a(song, session.stage_cfg_a(cfg, False), 0, len(x), np, None, {})
        hB = dict(uc_gain=None)
        if cfg["unlimit"]:
            import unlimit
            hB["unlimit_ref"] = x * U._db2lin(cfg["input_gain"])  # stage A's input: the gain applies to it
            hB["unlimit_g"] = unlimit.raw_gain(hB["unlimit_ref"], sr, dev="cpu")  # process() without a GPU
        B = session.stage_b(session.stage_cfg_b(cfg), D, sr, hB)
        assert np.array_equal(B, U.process(x, sr, cfg)), cfg["declip"]
        assert hB["uc_gain"].shape == x.shape
        assert np.array_equal(session.stage_b(session.stage_cfg_b(cfg), D, sr, hB), B)  # cached uc_gain
        P = session.stage_b(dict(session.stage_cfg_b(cfg), limiter=False), D, sr, hB)  # the limiter on its own
        assert np.array_equal(session.limit(P, sr, session.stage_cfg_b(cfg)), B)


def test_uncompress_warmup():
    # uncompress of a region warmed up over preview_span's lead-in is within 1e-3 dB of the render
    sr = 16000
    r0, r1 = int(8.5 * sr), 9 * sr
    for ratio in (2.0, 10.0):
        x = _compress(_drums(sr, 10), sr, -12, ratio, 5, 150)
        song = types.SimpleNamespace(x=x, sr=sr)
        cfg = U.settings_from_cfg(dict(U.PRESETS["Uncompress"], uc_ratio=ratio, uc_attack=5, uc_amount=100))
        S, E = session.preview_span(song, cfg, r0, r1, False)
        assert 0 < S < r0 and E == r1
        D = session.stage_a(song, session.stage_cfg_a(cfg, False), S, E, np, None, {})
        B = session.stage_b(session.stage_cfg_b(cfg), D, sr, {})[r0 - S:]
        full = U.process(x, sr, cfg)[r0:r1]
        k = np.abs(full) > 1e-4
        db = np.max(np.abs(20 * np.log10(np.abs(B[k]) / np.abs(full[k]))))
        assert db < 1e-3, (ratio, db)
    long = types.SimpleNamespace(x=np.zeros((60 * sr, 1)), sr=sr)  # warm-up capped at 10 s
    assert session.preview_span(long, dict(cfg, uc_ratio=20.0, uc_release=5000.0), 30 * sr, 31 * sr, False) == (20 * sr, 31 * sr)


def test_nmf_knee_flags():
    # NMF auto detection: a lower knee flags a superset, 1 = ceilings only; rebuilt samples never get quieter
    if not _has_torch():
        return
    from betterdeclipper.auto import auto_constraints
    from betterdeclipper.detect import estimate_lsb
    sr = 16000
    y = np.clip(0.95 * np.tanh(1.5 * _kick_stereo(sr, seed=3)), -0.85, 0.85)  # soft shoulder, then a plateau
    lsb = estimate_lsb(y)
    flags = {}
    for k in (0.7, 0.8, 0.9, 1.0):
        mode, an = U._nmf_analysis(dict(nmf_knee=k), y, sr)
        m_hi, m_lo, *_ = auto_constraints(y, lsb, None, mode, sr, an)
        flags[k] = m_hi | m_lo
    assert np.all(flags[0.7] >= flags[0.9]) and np.all(flags[0.9] >= flags[1.0])
    assert flags[0.7].sum() > flags[0.9].sum() > flags[1.0].sum() > 0
    m = flags[0.8]
    out = U.process(y, sr, dict(U.DEFAULTS, nmf_quality="fast", nmf_knee=0.8))
    assert np.array_equal(out[~m], y[~m]) and np.all(np.abs(out[m]) >= np.abs(y[m]) - 1e-6)


def test_lam_refs_override():
    # declip(lam_refs=lam_refs_for(...)) is the same run as declip() computing it itself
    if not _has_torch():
        return
    from betterdeclipper.engine import declip, lam_refs_for, PRESETS
    from betterdeclipper.methods.common import threshold_scale
    sr = 16000
    y = np.clip(_kick_stereo(sr, 1), -0.9, 0.9)
    m_hi, m_lo = y >= 0.9, y <= -0.9
    cons = (m_hi, m_lo, np.where(m_hi, y, np.inf), np.where(m_lo, y, -np.inf))
    ref = lam_refs_for(y, sr, threshold_scale(*cons[2:]), PRESETS["fast"], "cpu")
    assert lam_refs_for(y, float(sr), threshold_scale(*cons[2:]), PRESETS["fast"], "cpu") == ref  # Process at Hz
    run = lambda lr: declip(y, sr, preset="fast", constraints=cons, device="cpu", verbose=False, lam_refs=lr)[0]
    out = run(None)
    assert np.array_equal(run(ref), out)
    assert not np.array_equal(run({k: 2 * v for k, v in ref.items()}), out)


def test_unlimit_raw_gain():
    if not _has_torch():
        return
    import unlimit
    sr = 22050
    x = 0.5 * _drums(sr, 2)
    assert np.array_equal(unlimit.gain(x, sr, 0.5), unlimit.raw_gain(x, sr) ** 0.5)


def test_cli_nmf_knee():
    if not _has_torch():
        return
    sr = 16000
    t = np.arange(sr // 2) / sr
    d = tempfile.mkdtemp()
    inp, outp = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
    sf.write(inp, np.stack([np.clip(1.5 * np.sin(2 * np.pi * 220 * t), -1, 1)] * 2, 1), sr, subtype="PCM_16")
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sharpunclip.py"),
                        inp, "-o", outp, "--engine", "NMF", "--nmf-quality", "fast", "--nmf-knee", "1",
                        "--log", outp + ".log"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "nmf_knee=1.0" in open(outp + ".log").read()
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sharpunclip.py"),
                        inp, "-o", outp, "--nmf-quality", "normal+spade", "--soft-clip", "cubic",
                        "--log", outp + ".log2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    log = open(outp + ".log2").read()
    assert "nmf_quality=normal+spade" in log and "soft_clip=cubic" in log, log


def test_cli_log_on_failure():
    # A failed run still logs its header, the PyTorch/GPU versions and the error's traceback.
    d = tempfile.mkdtemp()
    lp = os.path.join(d, "run.log")
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sharpunclip.py"),
                        os.path.join(d, "missing.wav"), "--log", lp], capture_output=True, text=True)
    assert r.returncode != 0
    log = open(lp, encoding="utf-8").read()
    assert "input=" in log and "python=" in log and "error:" in log and "Traceback" in log, log


def test_nmf_preview_close():
    # NMF auto detection: the render reusing the GUI's analysis (ceilings, shoulder) is the CLI's, and a
    # region preview (that analysis, the whole song's lambda reference, the span widened until the NMF rank
    # reaches its cap) is within 30 dB of it
    if not _has_torch():
        return
    sr = 48000
    x = np.clip(_kick_stereo(sr, 8), -0.9, 0.9)
    song = session.Song(x, sr, None)
    cfg = U.settings_from_cfg(dict(U.PRESETS["Accurate"], nmf_quality="fast"))
    sh = song.shoulder(cfg, "cpu")
    full = U.process(x, sr, cfg)
    assert np.array_equal(U.process(x, sr, cfg, hints={"nmf_analysis": song.analysis(cfg, sh)}), full)
    r0, r1 = 3 * sr, 4 * sr
    S, E = session.preview_span(song, cfg, r0, r1, False)
    xp, dev = U.spade_backend("cpu")
    D = session.stage_a(song, session.stage_cfg_a(cfg, False), S, E, xp, dev, session.hints_a(song, cfg, S, E, dev, sh))
    B, ref = session.stage_b(session.stage_cfg_b(cfg), D, sr, {})[r0 - S:r1 - S], full[r0:r1]
    snr = 10 * np.log10(np.sum(ref ** 2) / np.sum((B - ref) ** 2))
    assert snr >= 30 and not np.array_equal(ref, x[r0:r1]), snr


def test_worker_run_pending():
    # a long job runs the queued urgent ones inside it, on its own thread; the others wait for it
    import queue, threading, time
    q, w, order = queue.Queue(), None, []
    gate = threading.Event()

    def long(ev):
        gate.wait(5)
        w.run_pending(1)
        order.append("long")
        return "long"
    w = session.Worker(q, threading.Lock())
    w.submit("long", 1, long, prio=3)
    while not w.running:
        time.sleep(0.005)
    w.submit("low", 2, lambda ev: order.append("low"), prio=2)
    w.submit("urgent", 3, lambda ev: order.append("urgent") or w.running[0], prio=0)
    gate.set()
    got = [q.get(timeout=5) for _ in range(3)]
    assert order == ["urgent", "long", "low"] and got[0] == ("job", "urgent", 3, "urgent"), (order, got)


def test_worker_latest_wins():
    # a new job of a kind replaces the queued one and cancels the running one: only the latest posts
    import queue, threading, time
    q, gate = queue.Queue(), threading.Event()
    w = session.Worker(q)
    w.submit("a", 1, lambda ev: gate.wait(5) and "slow")
    time.sleep(0.05)
    w.submit("a", 2, lambda ev: "replaced")
    w.submit("a", 3, lambda ev: "latest")
    w.submit("b", 4, lambda ev: 1 / 0)
    gate.set()
    assert [q.get(timeout=5) for _ in range(2)] == [("job", "a", 3, "latest"),
                                                    ("job_err", "b", 4, "ZeroDivisionError: division by zero")]
    time.sleep(0.05)
    assert q.empty()


def test_worker_pause_and_lock():
    # the heavy worker: paused, queued jobs wait; a job cancelled while it waits for the lock never starts
    import queue, threading, time
    q, lock, ran = queue.Queue(), threading.Lock(), []
    w = session.Worker(q, lock)
    w.pause(True)
    w.submit("a", 1, lambda ev: "x")
    time.sleep(0.1)
    assert q.empty() and w.jobs
    w.pause(False)
    assert q.get(timeout=5) == ("job", "a", 1, "x")
    with lock:
        w.submit("b", 2, lambda ev: ran.append(1))
        for _ in range(100):
            if w.running:
                break
            time.sleep(0.01)
        w.cancel()
    time.sleep(0.1)
    assert q.empty() and not ran and w.running is None


def test_estimates():
    # seconds of work per audio second: seeds, then an EMA (alpha 0.5) of what jobs took
    e = session.Estimates()
    spade = session.est_kind(U.PRESETS["SharpUnClip"], False, False)
    assert spade == ("SPADE", 3, 1, False) and abs(e.rate(spade) - 1.5 * 1.6) < 1e-12
    nmf = session.est_kind(U.PRESETS["Accurate"], True, True)
    assert nmf == ("NMF", "fast", True) and abs(e.seconds(nmf, 10) - 1.5) < 1e-12
    e.update(nmf, 10.0, 3.5)
    assert abs(e.rate(nmf) - 0.25) < 1e-12 and session.est_kind(U.PRESETS["Uncompress"], False, True) is None
    assert abs(e.seconds(("unlimit", False), 10) - (4.5 + 1.7)) < 1e-12  # + loading the model, once
    e.update(("unlimit", False), 10, 6.2)
    assert abs(e.seconds(("unlimit", False), 10) - 5.35) < 1e-12


def test_risk_cols():
    # the frame-risk strip: columns overlapping a SPADE frame [k*a - w, k*a) more than half flagged,
    # from Song.flags' counts, against a brute-force scan
    rng = np.random.default_rng(5)
    for _ in range(100):
        n = int(rng.integers(100, 9000))
        x = 0.1 * rng.standard_normal((n, 1))
        for _ in range(int(rng.integers(0, 6))):
            s = int(rng.integers(0, n))
            x[s:s + int(rng.integers(1, 3000))] = 1.0
        song = session.Song(x, 48000, None)
        cfg = dict(U.PRESETS["Accurate SPADE"], diff_clips=False, remove_glitch=False)
        M, cs = song.flags(cfg)[0][:, 0], song.flags(cfg)[3][0]
        w = 512 << int(rng.integers(0, 3))
        a = w // int(rng.choice(session.OV))
        t0, t1 = sorted(int(v) for v in rng.integers(0, n, 2))
        cols = int(rng.integers(1, 200))
        e = np.linspace(t0, max(t1, t0 + 1), cols + 1).astype(np.int64)
        ref = np.zeros(cols, bool)
        for k in range(1, (n + w) // a + 2):
            if 2 * M[max(0, k * a - w):k * a].sum() > w:
                ref |= (e[:-1] < k * a) & (e[1:] > k * a - w)
        assert np.array_equal(session.risk_cols(cs, w, a, e), ref), (n, w, a, t0, t1, cols)


def test_torch_gpu_pick():
    # Intel (XPU) and Apple (MPS) GPUs are picked like CUDA when CUDA is absent, for both engines.
    try:
        import torch
    except ImportError:
        return
    real = torch.cuda.is_available, getattr(torch, "xpu", None), torch.backends.mps.is_available
    try:
        torch.cuda.is_available = lambda: False
        torch.xpu = type("X", (), {"is_available": staticmethod(lambda: True)})
        U.torch_gpu.cache_clear()
        assert U.torch_gpu() == "xpu" and U.spade_backend("xpu") == (torch, "xpu")
        assert U._nmf_device("opencl") == "xpu" and U._nmf_device("cpu") == "cpu"
        torch.xpu.is_available = staticmethod(lambda: False)
        torch.backends.mps.is_available = lambda: True
        U.torch_gpu.cache_clear()
        assert U.torch_gpu() == "mps" and U._nmf_device("mps") == "mps" and U.spade_backend("cuda")[1] == "cpu"
    finally:
        torch.cuda.is_available, torch.backends.mps.is_available = real[0], real[2]
        if real[1] is None:
            del torch.xpu
        else:
            torch.xpu = real[1]
        U.torch_gpu.cache_clear()


def test_gpu_probe_order():
    # ROCm (simulated): loading Mesa's OpenCL hid the GPU from PyTorch on an RX 9070. PyTorch's GPU is
    # probed first, and auto then skips OpenCL (both engines on PyTorch); forced OpenCL keeps NMF on it.
    try:
        import torch
        import sharpunclip_cl
    except ImportError:
        return
    real = torch.cuda.is_available, torch.version.hip, sharpunclip_cl.device, torch.backends.cudnn.enabled

    def clobber():  # what loading Rusticl did
        torch.cuda.is_available = lambda: False
        return object()
    try:
        torch.cuda.is_available, torch.version.hip, sharpunclip_cl.device = (lambda: True), "7.1", clobber
        U.torch_gpu.cache_clear()
        assert U.spade_backend(True) == (torch, "cuda") and torch.cuda.is_available()  # OpenCL never loaded
        assert not torch.backends.cudnn.enabled  # MIOpen off on ROCm
        xp, dev = U.spade_backend("opencl")
        assert dev == "opencl" and not torch.cuda.is_available() and U._nmf_device(dev) == "cuda"
    finally:
        torch.cuda.is_available, torch.version.hip, sharpunclip_cl.device, torch.backends.cudnn.enabled = real
        U.torch_gpu.cache_clear()


def test_resample_matches_torchaudio():
    # Unlimit's resampler (no torchaudio needed) is torchaudio's default resample, bit for bit.
    try:
        import torch
        import torchaudio.functional as AF
        import unlimit
    except Exception:  # no torchaudio here (or none that loads): nothing to compare with
        return
    w = torch.from_numpy(np.random.default_rng(0).standard_normal((2, 9000)).astype(np.float32))
    for sr in (48000, 22050, 96000, 32000, 44100, 16000):
        assert torch.equal(unlimit.resample(w, sr, 44100), AF.resample(w, sr, 44100)), sr


def test_nmf_multi_device():
    # --gpus: the NMF engine's chunks shared between devices (here one device twice: two threads) give
    # the one-device result, in order; and the CLI accepts and logs it.
    if not _has_torch():
        return
    from betterdeclipper import engine as E
    sr = 16000
    cl = np.clip(_kick_stereo(sr, secs=3), -0.9, 0.9)
    for dev in ("cpu",) + ((U.torch_gpu(),) if U.torch_gpu() else ()):
        kw = dict(preset="fast", chunk_s=1.0, verbose=False)
        one = E.declip(cl, sr, device=dev, **kw)[0]
        two = E.declip(cl, sr, device=[dev, dev], **kw)[0]
        assert np.array_equal(one, two), (dev, np.abs(one - two).max())
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "in.wav")
    sf.write(inp, cl, sr, subtype="FLOAT")
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sharpunclip.py"),
                        inp, "-o", os.path.join(d, "out.wav"), "--nmf-quality", "fast", "--gpus", "all",
                        "--log", os.path.join(d, "run.log")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "gpus=all" in open(os.path.join(d, "run.log"), encoding="utf-8").read()


def test_spade_multi_device():
    # --gpus: SPADE's frame batches shared between devices (here one device twice: two threads) give
    # the one-device result, with progress in order; on every backend.
    sr = 16000
    cl = np.clip(_kick_stereo(sr, secs=2), -0.9, 0.9)
    m = np.abs(cl) >= 0.9 * 0.999
    lo, hi = cl.copy(), cl.copy()
    lo[m & (cl > 0)], hi[m & (cl > 0)] = 0.9, 2.5 * cl[m & (cl > 0)]
    hi[m & (cl < 0)], lo[m & (cl < 0)] = -0.9, 2.5 * cl[m & (cl < 0)]
    real = U._spade_devices
    for name in _backends():
        xp, dev = U.spade_backend(name)
        run = lambda p=None: U.aspade(cl, lo, hi, 512, 4, xp=xp, dev=dev, batch=32, group=2, progress=p)
        one = run()
        D = xp.device() if dev == "opencl" else dev
        done = []
        try:
            U._spade_devices = lambda xp_, dev_: [D, D]
            two = run(lambda f, d, o: d is None or done.append(d))
        finally:
            U._spade_devices = real
        assert np.array_equal(one, two), (name, np.abs(one - two).max())
        assert len(done) > 2 and done == sorted(done), (name, done)


def test_gpu_choice():
    # --gpu-name / the GUI's GPU list: one GPU for every engine, by part of its name; "NAME #2" is the
    # second GPU of that name. PyTorch gets its index (simulated: two identical CUDA cards); every real
    # OpenCL GPU is reached by the name list_gpus gives it.
    try:
        import torch
    except ImportError:
        return
    real = torch.cuda.is_available, torch.cuda.device_count, torch.cuda.get_device_name, torch.version.hip
    try:
        torch.cuda.is_available, torch.cuda.device_count = (lambda: True), (lambda: 2)
        torch.cuda.get_device_name, torch.version.hip = (lambda i=None: "Card A"), None
        for name, want in (("", "cuda"), ("card a", "cuda"), ("Card A #2", "cuda:1"), ("Card A #3", None), ("B", None)):
            U.set_gpu(name)
            assert U.torch_gpu() == want, (name, U.torch_gpu())
        U.set_gpu("Card A #2")
        assert U._nmf_device("opencl") == "cuda:1" and U.is_gpu("cuda:1") and not U.is_gpu("cpu")
    finally:
        torch.cuda.is_available, torch.cuda.device_count, torch.cuda.get_device_name, torch.version.hip = real
        U.set_gpu("")
    try:
        import sharpunclip_cl as C
        labels = U._numbered([C.label(d) for d in C.gpus()])
    except Exception:
        labels = []
    try:
        for lab in labels:
            U.set_gpu(lab)
            xp, dev = U.spade_backend("opencl")
            assert dev == "opencl" and C.label(C.device()) == lab.split(" #")[0], lab
    finally:
        U.set_gpu("")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    print("all tests passed")
