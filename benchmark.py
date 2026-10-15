#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 thatrandomalt
#
# SPDX-License-Identifier: MIT

"""Benchmark every algorithm on the CPU and on the GPU (if there is one): time and accuracy on one
committed test song, so machines and versions can be compared.

  python benchmark.py                   all algorithms (a few minutes on a GPU, longer on a CPU)
  python benchmark.py --quick           one run per algorithm family, on the song's first 8 s
  python benchmark.py --cpu | --gpu     one device only
  python benchmark.py -t 8              8 CPU threads (default: all)
  python benchmark.py --json out.json   also write the results, e.g. to compare machines
  python benchmark.py --save-reference  make this machine's scores the reference (bench/reference.json)
  python benchmark.py --make-song       regenerate bench/song.flac (the committed file is the baseline:
                                        a regenerated one can differ in its last bits between NumPy versions)

The song is synthesized (drums, bass, chords, arpeggio; 16 s, stereo, 48 kHz), so its true original is
known. Each run degrades it the same way: hard-clipped so 3 % of its samples clip (11 dB under its peak),
limited 6 dB by a lookahead limiter, compressed 4:1 (the compressor model Uncompress inverts). Score = improvement over the degraded
file against the original, in dB (SNR for clipping, SI-SDR else); an output within 100 dB of the original
counts as exact (Uncompress with the true settings inverts its own model). vs ref = score minus the
reference's. CPU=GPU = how closely the GPU's output matches the CPU's, SNR in dB: the iterative engines
round differently on each device (about 40-60 dB); far lower means the GPU computes something else.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np
import soundfile as sf

import sharpunclip as U
from abtest import snr, sisdr
from test_sharpunclip import _compress

HERE = os.path.dirname(os.path.abspath(__file__))
SONG, REF = os.path.join(HERE, "bench", "song.flac"), os.path.join(HERE, "bench", "reference.json")
COMP = (-18.0, 4.0, 5.0, 100.0)  # threshold dB, ratio, attack ms, release ms
P = U.PRESETS
RUNS = [  # (case, name, settings, in --quick)
    ("clipped", "Accurate (NMF normal)", P["Accurate"], True),
    ("clipped", "NMF fast", dict(P["Accurate"], nmf_quality="fast"), False),
    ("clipped", "NMF normal+spade", dict(P["Accurate"], nmf_quality="normal+spade"), False),
    ("clipped", "Accurate SPADE", P["Accurate SPADE"], True),
    ("clipped", "SharpUnClip", P["SharpUnClip"], False),
    ("limited", "Unmaster", P["Unmaster"], True),
    ("limited", "Unmaster SPADE", P["Unmaster SPADE"], False),
    ("limited", "Unlimit only (50 %)", dict(U.DEFAULTS, declip=False, unlimit=True), False),
    ("compressed", "Uncompress preset", P["Uncompress"], False),
    ("compressed", "Uncompress, known settings", dict(P["Uncompress"], uc_threshold=COMP[0], uc_ratio=COMP[1],
                                                      uc_attack=COMP[2], uc_release=COMP[3], uc_amount=100.0), True),
]


def make_song(path):
    """8 bars at 120 BPM in A minor (Am F C G): kick, snare, hats, saw bass, detuned saw chords, a
    plucked arpeggio; L and R mixed differently. Fixed seed; peak 0.98; 24-bit FLAC."""
    from scipy.signal import butter, sosfilt
    sr, beat = 48000, 0.5
    n = int(32 * beat * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(2026)
    filt = lambda v, f, kind="low", o=2: sosfilt(butter(o, f, kind, fs=sr, output="sos"), v)
    hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)
    saw = lambda f: 2 * ((np.cumsum(f) / sr) % 1) - 1  # phase-continuous across pitch changes
    ph = t % beat
    kick = np.sin(2 * np.pi * (50 * ph + 120 * (1 - np.exp(-ph * 35)) / 35)) * np.exp(-ph * 7)
    sp = (t - beat) % (2 * beat)  # snare on beats 2 and 4
    snare = 0.5 * filt(rng.standard_normal(n), 1500, "high") * np.exp(-sp * 25) \
        + 0.5 * np.sin(2 * np.pi * 185 * sp) * np.exp(-sp * 35)
    hats = 0.18 * filt(rng.standard_normal(n), 8000, "high", 4) * np.exp(-(t % (beat / 2)) * 90)
    chords = np.array([(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)])  # MIDI notes
    ch = chords[(t // (4 * beat)).astype(int) % 4]  # (n, 3): the bar's chord
    bass = filt(saw(hz(ch[:, 0] - 24.0)), 300) * (0.55 + 0.45 * np.exp(-(t % (beat / 2)) * 8))
    pad = lambda d: 0.12 * sum(filt(saw(hz(ch[:, i] + 0.0) * d), 1800) for i in range(3))
    step = (t // (beat / 4)).astype(int)
    arp = 0.25 * np.sin(2 * np.pi * np.cumsum(hz(ch[np.arange(n), step % 3] + 12.0)) / sr) \
        * np.exp(-(t % (beat / 4)) * 18)
    x = np.stack([kick + 0.8 * snare + hats + 0.6 * bass + pad(1.003) + 0.8 * arp,
                  kick + 0.8 * snare + 0.8 * hats + 0.6 * bass + pad(0.997) + 0.5 * arp], 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, 0.98 * x / np.abs(x).max(), sr, subtype="PCM_24")


def limit(x, sr, ceiling=0.98, look_ms=5.0, release_ms=80.0):
    """A linked lookahead peak limiter: the gain reaches ceiling / peak before each peak, recovers with
    release_ms, and anything left over is clipped."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    L = max(1, int(look_ms * sr / 1000))
    g = np.minimum(1.0, ceiling / np.maximum(maximum_filter1d(np.abs(x).max(1), 2 * L + 1), 1e-12))
    a, h, out = math.exp(-1 / (release_ms * sr / 1000)), 1.0, np.empty_like(g)
    for i, v in enumerate(g.tolist()):  # falls at once, rises slowly
        h = v if v < h else a * h + (1 - a) * v
        out[i] = h
    return np.clip(x * uniform_filter1d(out, L)[:, None], -ceiling, ceiling)


def cases(x, sr):
    """{case: (degraded, original, score function, its name)}."""
    c, up = np.quantile(np.abs(x), 0.97), 2.0 * x  # 3 % of the samples clip
    return {"clipped": (np.clip(x, -c, c), x, snr, "SNR"),
            "limited": (limit(up, sr), up, sisdr, "SI-SDR"),
            "compressed": (_compress(x, sr, *COMP), x, sisdr, "SI-SDR")}


def uses_gpu(s):
    return s["declip"] or s["unlimit"]  # uncompress always runs on the CPU


def main():
    ap = argparse.ArgumentParser(description="Time and score every algorithm on the CPU and the GPU.")
    ap.add_argument("--quick", action="store_true", help="one run per family, on the song's first 8 s")
    dv = ap.add_mutually_exclusive_group()
    dv.add_argument("--cpu", action="store_true", help="the CPU only")
    dv.add_argument("--gpu", action="store_true", help="the GPU only")
    ap.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 1,
                    help="CPU threads (default: all, as the app)")
    ap.add_argument("--gpu-name", metavar="NAME", default=U.gpu_name(),
                    help="the GPU to use, by part of its name (python sharpunclip.py --list-gpus)")
    ap.add_argument("--gpus", metavar="N", default="1", help="share the work between N GPUs (or all)")
    ap.add_argument("--json", metavar="FILE", help="also write the results as JSON")
    ap.add_argument("--save-reference", action="store_true", help="store these scores as the reference")
    ap.add_argument("--make-song", action="store_true", help="regenerate the test song")
    a = ap.parse_args()
    U.set_gpu(a.gpu_name)
    U.set_gpus(0 if a.gpus.strip().lower() == "all" else a.gpus)
    if a.make_song or not os.path.exists(SONG):
        make_song(SONG)
        print(f"made {SONG}")
        if a.make_song:
            return
    x, sr = sf.read(SONG, always_2d=True)
    if a.quick:
        x = x[:8 * sr]
    song_id = hashlib.sha1(open(SONG, "rb").read()).hexdigest()[:12]
    mode = "quick" if a.quick else "full"
    ref = {}
    if os.path.exists(REF):
        r = json.load(open(REF, encoding="utf-8"))
        ref = r.get(mode, {}) if r.get("song") == song_id else {}
        if r.get("song") != song_id:
            print("note: bench/song.flac is not the reference's song: no vs ref column")

    threads = max(1, a.threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass
    devs = [] if a.gpu else [("CPU", *U.spade_backend(False))]
    gx, gd = U.spade_backend(True)
    if not a.cpu and U.is_gpu(gd):
        devs.append(("GPU", gx, gd))
    elif a.gpu:
        sys.exit("no GPU found (see README: Install)")
    print(f"song {os.path.relpath(SONG, HERE)} ({song_id}), {len(x) / sr:.1f} s, {mode}")
    print(U.env_info())
    print(f"CPU: {platform.processor() or platform.machine()}, {threads} threads")
    for name, xp, dev in devs:
        if name == "GPU":
            print(f"GPU: {U.backend_name(xp, dev, U.DEFAULTS)} | SPADE: {U.backend_name(xp, dev)}")
    for name, xp, dev in devs:  # warm-up, untimed: GPU kernels compile, models load
        for s in (P["Accurate"], P["Accurate SPADE"], RUNS[7][2]):
            U.process(x[:2 * sr], sr, s, xp=xp, dev=dev)

    C, rows, out = cases(x, sr), [], {}
    print(f"\n{'case':<11}{'algorithm':<29}{'device':<7}{'time':>8}{'x song':>8}{'score':>14}{'vs ref':>8}"
          f"{'CPU=GPU':>9}")
    for case, name, s, quick in RUNS:
        if a.quick and not quick:
            continue
        y, truth, score, metric = C[case]
        base, key, shown = score(truth, y), f"{case}/{name}", f"{case:<11}{name:<29}"
        for dname, xp, dev in devs:
            if dname == "GPU" and not uses_gpu(s):
                continue
            t0 = time.perf_counter()
            o = U.process(y, sr, s, xp=xp, dev=dev)
            dt = time.perf_counter() - t0
            gain = min(score(truth, o), 100.0) - base
            same = snr(out[key], o) if key in out else None
            out.setdefault(key, o)
            d = None if key not in ref else gain - ref[key]
            rows.append(dict(case=case, algorithm=name, device=dname, seconds=round(dt, 3),
                             x_song=round(dt / (len(x) / sr), 3), score_db=round(gain, 3), metric=metric,
                             vs_ref_db=None if d is None else round(d, 3),
                             cpu_gpu_db=None if same is None else round(same, 1)))
            print(f"{shown}{dname:<7}{dt:7.1f}s{dt / (len(x) / sr):7.2f}x"
                  f"{'   exact' if gain + base >= 100 else f'{gain:+8.2f}'} {metric:<6}"
                  f"{'' if d is None else f'{d:+8.2f}':>7}"
                  f"{'' if same is None else f'{same:7.0f} dB':>9}", flush=True)
            shown = " " * 40  # the case and algorithm once

    if a.json:
        json.dump(dict(song=song_id, mode=mode, env=U.env_info(), cpu=platform.processor() or platform.machine(),
                       threads=threads, results=rows), open(a.json, "w", encoding="utf-8"), indent=1)
    if a.save_reference:
        r = json.load(open(REF, encoding="utf-8")) if os.path.exists(REF) else {}
        if r.get("song") != song_id:
            r = {"song": song_id}
        cpu = {f"{w['case']}/{w['algorithm']}": w["score_db"] for w in rows if w["device"] == "CPU"}
        if not cpu:
            sys.exit("the reference holds CPU scores: run without --gpu")
        r[mode] = cpu
        json.dump(r, open(REF, "w", encoding="utf-8"), indent=1)
        print(f"saved the reference scores ({mode}) to {os.path.relpath(REF, HERE)}")


if __name__ == "__main__":
    main()
