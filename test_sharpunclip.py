# SPDX-FileCopyrightText: 2026 AG89
#
# SPDX-License-Identifier: MIT

import math, os, subprocess, sys, tempfile
import numpy as np
import soundfile as sf
import sharpunclip as U


def _backends():
    """Yield "cpu", "cuda" if present, and "opencl" once per OpenCL GPU (selected
    through SHARPUNCLIP_OPENCL_DEVICE while that name is being tested)."""
    yield "cpu"
    if U.spade_backend("cuda")[1] == "cuda":
        yield "cuda"
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
    opt = dict(U.DEFAULTS)
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
    opt = dict(U.DEFAULTS, clip_above=78.0, not_clip_below=77.0, peak_ignore=0.001,
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
    opt = dict(U.DEFAULTS, max_restore=0.0, diff_clips=False)
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
    opt = dict(U.DEFAULTS, clip_above=99.9, not_clip_below=99.8, max_restore=1.5)
    out = U.process(cl, sr, opt)
    m = np.abs(cl) >= 0.999
    assert np.all(np.abs(out[m]) <= 1.5 * np.abs(cl[m]) + 1e-9)
    assert np.all(np.abs(out[m]) >= np.abs(cl[m]) - 1e-9)
    quiet = U.process(0.25 * cl, sr, opt) / 0.25
    assert np.max(np.abs(quiet - out)) < 1e-6, np.max(np.abs(quiet - out))


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
    opt = dict(U.DEFAULTS)
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
    opt = dict(U.DEFAULTS)
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
    g = unlimit.gain(x, sr, 0.5)
    assert g.shape == (len(x),) and np.all(np.isfinite(g)) and np.all(g > 0)
    assert abs(np.median(g) - 1) < 1e-9
    out = U.process(x, sr, dict(U.DEFAULTS, declip=False, unlimit=50))
    assert np.allclose(out, x * g[:, None])


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
    r = subprocess.run(cli + ["--preset", "Accurate", "--log", outp + ".log"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    log = open(outp + ".log").read()
    assert "clip_above=99.9" in log and "max_restore=2.5" in log, log
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    print("all tests passed")
