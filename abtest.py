# SPDX-FileCopyrightText: 2026 AG89
#
# SPDX-License-Identifier: MIT

"""Score SharpUnClip (or any tool) against a known original. This is how the numbers in
METHODS.md were measured.

  python abtest.py --orig original.wav --degraded clipped.flac [--args "--preset Accurate --gpu"]
      runs sharpunclip.py on the degraded file, then scores both against the original
  python abtest.py --orig original.wav --degraded clipped.flac --out processed.wav
      scores an existing result (made with any tool)
  python abtest.py --clean clean.wav [--clip-level 0.3] [--args "..."]
      hard-clips a clean file at clip-level x its peak, then runs the first mode
  python abtest.py --selftest

The original and degraded files must be sample-aligned renders of the same audio (export
the same project twice, once with the processing). The script checks alignment and polarity.

Scores (higher is better except HF LSD):
  SNR      sample-exact accuracy, dB; right for declipping, which keeps the level
  SI-SDR   scale-invariant SDR (Le Roux et al. 2019), dB; right when the overall level may
           differ (unlimit, uncompress)
  side     SNR of the stereo side signal (L-R): stereo swirl lowers it
  HF LSD   log-spectral distance to the original, dB, 2-16 kHz, only in frames where the
           processing changed samples: spectral junk ("watery" musical noise) raises it
"""
import argparse, os, shlex, subprocess, sys, tempfile
import numpy as np
import soundfile as sf

SHARPUNCLIP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sharpunclip.py")


def snr(ref, x):
    e = np.sum((ref - x) ** 2)
    return float("inf") if e == 0 else 10 * np.log10(np.sum(ref ** 2) / e)


def sisdr(ref, x):
    ref, x = ref.ravel(), x.ravel()
    a = np.dot(x, ref) / np.dot(ref, ref)
    e = np.sum((a * ref - x) ** 2)
    return float("inf") if e == 0 else 10 * np.log10(np.sum((a * ref) ** 2) / e)


def hf_lsd(ref, x, changed, sr, n=1024, hop=256):
    """Mean log-spectral distance (dB) over 2-16 kHz in frames containing changed samples."""
    win = np.hanning(n); fr = np.fft.rfftfreq(n, 1 / sr); band = (fr >= 2000) & (fr <= 16000)
    vals = []
    for s in range(0, len(ref) - n, hop):
        if changed[s:s + n].any():
            A = np.abs(np.fft.rfft(ref[s:s + n] * win[:, None], axis=0))[band]
            B = np.abs(np.fft.rfft(x[s:s + n] * win[:, None], axis=0))[band]
            vals.append(np.sqrt(np.mean((20 * np.log10((A + 1e-9) / (B + 1e-9))) ** 2)))
    return float(np.mean(vals)) if vals else float("nan")


def lag(ref, x, maxlag=4000, sec=20):
    """Offset of x relative to ref (samples), from a cross-correlation of a mid stretch."""
    s0 = len(ref) // 3; L = max(1, min(sec * 48000, len(ref) - s0, len(x) - s0))
    a, b = ref[s0:s0 + L].mean(1), x[s0:s0 + L].mean(1)
    N = 1 << (2 * L).bit_length()
    xc = np.fft.irfft(np.fft.rfft(b, N) * np.conj(np.fft.rfft(a, N)), N)
    return int(np.argmax(np.r_[xc[-maxlag:], xc[:maxlag + 1]])) - maxlag


def report(orig, degraded, out, sr):
    n = min(len(orig), len(degraded), len(out))
    orig, degraded, out = orig[:n], degraded[:n], out[:n]
    lg = lag(orig, degraded)
    big = np.abs(orig) > 0.03
    pol = np.mean(np.sign(orig[big]) == np.sign(degraded[big])) * 100
    print(f"alignment: degraded is {lg:+d} samples from the original; same polarity on {pol:.1f}% of samples")
    if lg != 0:
        print("  WARNING: not sample-aligned; scores are meaningless until the renders line up")
    if pol < 99:
        print("  NOTE: under 99% means the two renders differ beyond the processing (e.g. random or"
              "\n        free-running effects in the project); treat the scores as a lower bound")
    changed = (np.abs(out - degraded) > 1e-9).any(1)
    side = (lambda v: v[:, 0] - v[:, 1]) if orig.shape[1] == 2 else None
    rows = [("SNR dB", snr), ("SI-SDR dB", sisdr)]
    print(f"{'score':12s} {'degraded':>9s} {'processed':>10s} {'change':>8s}")
    for name, f in rows:
        a, b = f(orig, degraded), f(orig, out)
        print(f"{name:12s} {a:9.2f} {b:10.2f} {b - a:+8.2f}")
    if side:
        a, b = snr(side(orig), side(degraded)), snr(side(orig), side(out))
        print(f"{'side SNR dB':12s} {a:9.2f} {b:10.2f} {b - a:+8.2f}")
    if changed.any():
        a, b = hf_lsd(orig, degraded, changed, sr), hf_lsd(orig, out, changed, sr)
        print(f"{'HF LSD dB':12s} {a:9.2f} {b:10.2f} {b - a:+8.2f}   (lower is better)")
    print(f"processing changed {changed.mean() * 100:.2f}% of samples; peak {np.abs(out).max():.2f} "
          f"(original {np.abs(orig).max():.2f})")


def run_sharpunclip(degraded_path, args):
    out = os.path.join(tempfile.mkdtemp(), "processed.wav")
    subprocess.run([sys.executable, SHARPUNCLIP, degraded_path, "-o", out] + shlex.split(args or ""), check=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--orig"); p.add_argument("--degraded"); p.add_argument("--out")
    p.add_argument("--clean"); p.add_argument("--clip-level", type=float, default=0.3)
    p.add_argument("--args", help="arguments passed to sharpunclip.py, e.g. \"--preset Accurate --gpu\"")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if a.clean:
        x, sr = sf.read(a.clean, always_2d=True)
        a.orig, a.degraded = a.clean, os.path.join(tempfile.mkdtemp(), "clipped.wav")
        lvl = a.clip_level * np.abs(x).max()
        sf.write(a.degraded, np.clip(x, -lvl, lvl), sr, subtype="FLOAT")
    if not (a.orig and a.degraded):
        p.print_help(); return 1
    out_path = a.out or run_sharpunclip(a.degraded, a.args)
    orig, sr = sf.read(a.orig, always_2d=True)
    degraded, sr2 = sf.read(a.degraded, always_2d=True)
    out, sr3 = sf.read(out_path, always_2d=True)
    if not sr == sr2 == sr3:
        raise SystemExit(f"sample rates differ: {sr}, {sr2}, {sr3}")
    report(orig, degraded, out, sr)
    return 0


def selftest():
    sr = 16000; t = np.arange(2 * sr) / sr
    rng = np.random.default_rng(0)
    clean = np.stack([0.8 * np.sin(2 * np.pi * 300 * t), 0.7 * np.sin(2 * np.pi * 450 * t)], 1) \
        + 0.05 * rng.standard_normal((len(t), 2))
    clipped = np.clip(clean, -0.4, 0.4)
    better = clean + 0.01 * rng.standard_normal(clean.shape)
    changed = (np.abs(better - clipped) > 1e-9).any(1)
    assert snr(clean, clipped) < snr(clean, better) and sisdr(clean, clipped) < sisdr(clean, better)
    assert abs(sisdr(clean, 3 * better) - sisdr(clean, better)) < 1e-9  # scale-free
    assert hf_lsd(clean, better, changed, sr) < hf_lsd(clean, clipped, changed, sr)
    assert lag(clean, clipped) == 0 and lag(clean, np.roll(clipped, 37, 0)) == 37
    print(f"abtest selftest OK  (SNR {snr(clean, clipped):.1f} -> {snr(clean, better):.1f} dB, "
          f"lag detection ok)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
