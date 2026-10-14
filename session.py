# SPDX-FileCopyrightText: 2026 thatrandomalt
#
# SPDX-License-Identifier: MIT

"""GUI session logic without tkinter: waveform envelopes, manual-detection results, background
workers, and region previews as two calls of the real process()."""
import contextlib
import math
import threading
from dataclasses import dataclass

import numpy as np

import sharpunclip as U
from betterdeclipper.auto import LIMITER_MAX_GAIN_DB

try:
    import sounddevice as sd  # optional: loop audition
    SD_ERR = None
except Exception as e:  # ImportError, or OSError when libportaudio2 is missing (Linux)
    sd, SD_ERR = None, str(e)
SD_HINT = "Playback needs: pip install sounddevice (Linux: also libportaudio2)"

GPU_LOCK = threading.Lock()  # held by the heavy worker per job and by a render throughout: one GPU user


class Cancelled(Exception):
    """Raised from a cancelled job's progress callback."""


# Setting groups: what a cached result depends on (each group includes the one before)
PEAK_KEYS = {"peak_ignore", "peak_time", "reset_amp", "reset_time", "diff_clips", "trans_amp", "trans_time",
             "input_gain", "process_rate"}
FLAG_KEYS = PEAK_KEYS | {"clip_above", "not_clip_below", "not_clip_amp", "dynamic", "soft", "remove_glitch"}
A_KEYS = FLAG_KEYS | {"declip", "engine", "nmf_quality", "auto_detect", "nmf_knee", "max_restore",
                      "speed_quality", "lookahead", "linked_stereo", "gpu"}
UC_KEYS = A_KEYS | {"unlimit", "unlimit_amount", "uncompress", "uc_threshold", "uc_ratio", "uc_attack",
                    "uc_release", "uc_makeup", "uc_link"}
B_KEYS = UC_KEYS | {"uc_amount", "output_gain", "limiter", "limiter_peak"}
IGNORED = {"threads"}  # no effect on the output
G_KEYS = {"input_gain", "process_rate"}  # whole-song analyses of y0: the unlimit curve, NMF auto detection
BLOCK = 4096  # Stats.block_counts block length
HIST_DB = 0.02  # Song.ceilings' level histogram bin, dB
LIM_CAP = 10 ** (LIMITER_MAX_GAIN_DB / 20)  # NMF auto detection's cap under limiter ceilings
NINES = (0.3, 4.3, 2048)  # hist9: range of u = -log10(1 - level / peak), and its bins
OV = (2, 4, 4, 8, 8, 16, 16)  # SPADE frames per frame length (hop = w / OV[speed_quality])
RES = 32  # Song.flags counts flags per RES samples: every SPADE frame length and hop is a multiple


def keyof(cfg, keys):
    return tuple(sorted((k, cfg[k]) for k in keys))


def _runs(m):
    """(starts, ends) of the True runs of m."""
    d = np.diff(m.astype(np.int8), prepend=np.int8(0), append=np.int8(0))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _reduce(mn, mx, step):
    i = np.arange(0, len(mn), step)
    return np.minimum.reduceat(mn, i, axis=0), np.maximum.reduceat(mx, i, axis=0)


class Pyramid:
    """Min/max envelope of x (n, ch): level k holds float32 blocks of 64 * 4**k samples."""

    def __init__(self, x):
        self.n = len(x)
        mn, mx = (a.astype(np.float32) for a in _reduce(x, x, 64))
        self.mn, self.mx = [mn], [mx]
        while len(mn) > 1:
            mn, mx = _reduce(mn, mx, 4)
            self.mn.append(mn); self.mx.append(mx)

    def update(self, x, i0, i1):
        """Recompute the blocks covering x[i0:i1] and their parents."""
        b0, b1 = i0 // 64, -(-min(i1, self.n) // 64)
        if b1 <= b0:
            return
        seg = x[b0 * 64:b1 * 64]
        self.mn[0][b0:b1], self.mx[0][b0:b1] = _reduce(seg, seg, 64)
        for k in range(1, len(self.mn)):
            b0, b1 = b0 // 4, -(-b1 // 4)
            self.mn[k][b0:b1], self.mx[k][b0:b1] = _reduce(self.mn[k - 1][b0 * 4:b1 * 4],
                                                           self.mx[k - 1][b0 * 4:b1 * 4], 4)

    def minmax(self, x, t0, t1, cols):
        """(mn, mx), each float32 (cols, ch), over the columns of [t0, t1). A column covers at least
        its own samples (exactly them when its edges fall on block edges, or below 64 samples a
        column, where x is read directly); one narrower than a sample holds the sample under it."""
        e = np.linspace(t0, t1, cols + 1).astype(np.int64)
        spp = (t1 - t0) / cols
        if spp < 64:
            seg = x[:max(e[-1], e[-2] + 1)]
            return (np.minimum.reduceat(seg, e[:-1], axis=0).astype(np.float32),
                    np.maximum.reduceat(seg, e[:-1], axis=0).astype(np.float32))
        k = 0  # the level just finer than a column
        while k + 1 < len(self.mn) and 64 * 4 ** (k + 1) <= spp:
            k += 1
        B, mn, mx = 64 * 4 ** k, self.mn[k], self.mx[k]
        b, last = e[:-1] // B, (e[1:] - 1) // B  # first and last block of each column
        end = last[-1] + 1
        return (np.minimum(np.minimum.reduceat(mn[:end], b, axis=0), mn[last]),
                np.maximum(np.maximum.reduceat(mx[:end], b, axis=0), mx[last]))


@dataclass
class Stats:
    frac: list                # flagged fraction per channel
    runs: int
    longest_s: float
    longest_at: int           # first sample of the longest run
    block_counts: np.ndarray  # int32 flags per BLOCK samples, summed over channels
    little_ctx: dict          # {512: f, 1024: f, 2048: f}, see little_context
    dropped: int              # single samples dropped by Remove glitches
    warn: tuple | None        # (fraction, centre sample) of the worst 1 s window, when over 70 % flagged


def _flagset(M, dropped):
    """What the display reads of flags M bool (n, ch): (M, dropped, M's envelope pyramid, per channel the
    flag counts summed up to every RES samples, for risk_cols)."""
    n = len(M)
    cs = [np.r_[0, np.cumsum(np.add.reduceat(M[:, c], np.arange(0, n, RES), dtype=np.int32))]
          for c in range(M.shape[1])]
    return M, dropped, Pyramid(M), cs


def _stats(M, dropped, sr):
    n, ch = M.shape
    counts = np.zeros(-(-n // BLOCK), np.int32)
    frac, runs, longest, at, ctx = [], 0, 0, 0, dict.fromkeys((512, 1024, 2048), 0.0)
    for c in range(ch):
        m = M[:, c]
        k = int(np.count_nonzero(m))
        frac.append(k / max(n, 1))
        if not k:
            continue
        counts += np.add.reduceat(m, np.arange(0, n, BLOCK), dtype=np.int32)
        s, e = _runs(m)
        runs += len(s)
        j = int(np.argmax(e - s))
        if e[j] - s[j] > longest:
            longest, at = int(e[j] - s[j]), int(s[j])
        for w, v in little_context(m, ctx).items():
            ctx[w] += v * k
    total = sum(frac) * n
    L = max(1, round(sr / BLOCK))  # the worst 1 s window
    cs = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    win = cs[L:] - cs[:-L] if len(counts) > L else cs[-1:]
    j = int(np.argmax(win))
    pct = win[j] / (min(L * BLOCK, n) * ch) if n else 0.0
    return Stats(frac, runs, longest / sr, at, counts, {w: v / max(total, 1) for w, v in ctx.items()},
                 sum(len(d) for d in dropped),
                 (float(pct), min(n - 1, j * BLOCK + L * BLOCK // 2)) if pct > 0.7 else None)


class Song:
    """One loaded file: x float64 (n, ch) as read, its envelope pyramid, and detection as process()
    does it, for the latest settings (one cached result per kind; the workers compute them)."""

    def __init__(self, x, sr, path):
        self.x, self.sr, self.path, self.pyr = x, sr, path, Pyramid(x)
        self._memo, self._locks, self._peak = {}, {}, None

    def _cached(self, name, cfg, keys, fn, extra=()):
        """fn()'s result for these settings (and extra), kept until others ask. The light and heavy
        workers share it: a second caller waits for the first one's result."""
        k = keyof(cfg, keys) + tuple(extra)
        with self._locks.setdefault(name, threading.Lock()):
            m = self._memo.get(name)
            if m is None or m[0] != k:
                m = self._memo[name] = k, fn()
            return m[1]

    def peek(self, name, cfg, keys, extra=()):
        """The cached result of `name` for these settings, or None (nothing is computed)."""
        m = self._memo.get(name)
        return m[1] if m is not None and m[0] == keyof(cfg, keys) + tuple(extra) else None

    def peak(self):
        """max |x| over the song."""
        if self._peak is None:
            self._peak = max(self.x.max(), -self.x.min()) if self.x.size else 0.0
        return self._peak

    def unlimit_raw(self, cfg, tick=None):
        """The 100 % unlimit gain of the whole song (over its median, as process() computes it),
        float32 (n,). Needs PyTorch."""
        import unlimit
        y0 = self.x * U._db2lin(cfg["input_gain"])
        return unlimit.raw_gain(y0, rate_p(cfg, self.sr), tick).astype(np.float32)

    def y0(self, cfg, c):
        """|y0| of channel c: the level detection sees."""
        return np.abs(self.x[:, c] * U._db2lin(cfg["input_gain"]))

    def _y(self, cfg):
        """y0 (n, ch) as process() makes it."""
        return self.x * U._db2lin(cfg["input_gain"])

    def ceilings(self, cfg):
        """NMF auto detection's ceilings of y0, as the render finds them: dict(lsb, refs, fingerprint,
        hist), hist = (lo, per channel [counts of +y0, of -y0] per HIST_DB dB from lo, 9 dB under the
        highest ceiling, to 0.5 dB over the peak), or None when no side has a level."""
        def f():
            from betterdeclipper.auto import analyze_ceilings, fingerprint
            from betterdeclipper.detect import estimate_lsb
            y = self._y(cfg)
            lsb = estimate_lsb(y)
            refs = analyze_ceilings(y, lsb, rate_p(cfg, self.sr))
            th = [r["theta"] for row in refs for r in row if r["theta"] is not None]
            hist = None
            if th:
                lo = 20 * np.log10(max(th)) - 9
                nb = int((20 * np.log10(max(np.abs(y).max(), *th)) + 0.5 - lo) / HIST_DB) + 1
                t = 10 ** (lo / 20)
                bins = lambda v: np.bincount(np.minimum(((20 * np.log10(v) - lo) / HIST_DB).astype(np.int64), nb - 1),
                                             minlength=nb)
                hist = lo, [[bins(v[v >= t]), bins(-v[v <= -t])] for v in y.T]
            return dict(lsb=lsb, refs=refs, fingerprint=fingerprint(y), hist=hist)
        return self._cached("ceil", cfg, G_KEYS, f)

    def analysis(self, cfg, sh=None):
        """betterdeclipper's analysis of y0 (declip_nmf's nmf_analysis hint): the ceilings and the
        shoulder search's result sh (None: not searched)."""
        c, (n, ch) = self.ceilings(cfg), self.x.shape
        return dict(refs=c["refs"], shoulder=sh, source=dict(sr=int(rate_p(cfg, self.sr)), channels=int(ch),
                                                              samples=int(n), fingerprint=c["fingerprint"]))

    def shoulder(self, cfg, dev, step=None):
        """NMF auto detection's soft-shoulder search on torch device dev, as the render runs it: the
        engine's fast pass on the loudest windows. step(i, n) runs before each of its n windows (raising
        cancels)."""
        def f():
            import torch
            from betterdeclipper import auto
            from betterdeclipper.engine import fast_pass
            torch.set_flush_denormal(True)  # as declip_nmf: the render's search runs with it
            y, sp, c, i = self._y(cfg), rate_p(cfg, self.sr), self.ceilings(cfg), [0]
            fast, n = fast_pass(sp, dev), len(auto._loud_windows(y, c["refs"], sp))

            def rf(yy, cons):
                if step:
                    step(i[0], n)
                i[0] += 1
                return fast(yy, cons)
            return auto.search_shoulder(y, c["lsb"], c["refs"], rf, sp)
        return self._cached("shoulder", cfg, G_KEYS, f, (dev,))

    def shoulder_ready(self, cfg, dev):
        return self.peek("shoulder", cfg, G_KEYS, (dev,))

    def auto_flags(self, cfg, sh=None):
        """NMF auto detection of the whole song as declip_nmf does it with this song's analysis and the
        shoulder sh: dict(flags (as flags() gives them), stats, knees (level or None per channel and
        polarity), prov, scale: the constraints' threshold_scale, which the lambda reference is divided
        by). Without sh, a knee of 0 (as found) is taken as 1, ceilings only: prov (not cached: a set
        knee of 1 gives the same flags, not provisional)."""
        prov = sh is None and not float(cfg["nmf_knee"])
        cfg = dict(cfg, nmf_knee=1.0) if prov else cfg

        def f():
            from betterdeclipper.auto import auto_constraints
            y, sp = self._y(cfg), rate_p(cfg, self.sr)
            mode, an = U._nmf_analysis(cfg, y, sp, self.analysis(cfg, sh))
            m_hi, m_lo, th_hi, th_lo, rep = auto_constraints(y, self.ceilings(cfg)["lsb"], None, mode, sp, an)
            M = m_hi | m_lo
            th = [np.abs(t[np.isfinite(t)]) for t in (th_hi, th_lo)]
            return dict(flags=_flagset(M, []), stats=_stats(M, [], self.sr), knees=rep["knees"],
                        scale=max([1e-3] + [float(t.max()) for t in th if t.size]))  # = threshold_scale
        k = float(cfg["nmf_knee"])
        return dict(self._cached("auto", cfg, G_KEYS | {"nmf_knee"}, f, (None if k else sh["knee"],)), prov=prov)

    def z0(self, cfg, dev):
        """The NMF models' lambda reference of the whole y0 before normalisation, {window ms: max |STFT|}
        on torch device dev: lam_refs = z0 / the constraints' threshold_scale."""
        def f():
            from betterdeclipper.engine import PRESETS, lam_refs_for
            return lam_refs_for(self._y(cfg), rate_p(cfg, self.sr), 1.0, [m for p in PRESETS.values() for m in p],
                                dev)
        return self._cached("z0", cfg, G_KEYS, f, (dev,))

    def nmf_scale(self, cfg):
        """threshold_scale of manual detection's NMF constraints (the lambda reference's divisor): max |y0|
        over the flagged samples, at least 1e-3."""
        def f():
            M = self.flags(cfg)[0]
            return max([1e-3] + [float(self.y0(cfg, c)[M[:, c]].max()) for c in range(M.shape[1]) if M[:, c].any()])
        return self._cached("scale", cfg, FLAG_KEYS, f)

    def steps(self, cfg):
        """Per channel: the measured-peak staircase (starts, values)."""
        sp = rate_p(cfg, self.sr)
        return self._cached("steps", cfg, PEAK_KEYS,
                            lambda: [U.peak_steps(self.y0(cfg, c), sp, cfg) for c in range(self.x.shape[1])])

    def gaps(self, cfg):
        """Per channel: (starts, ends) of the gaps between sections, or None without Different clips."""
        def f():
            if not cfg["diff_clips"]:
                return None
            win = max(1, int(round(float(rate_p(cfg, self.sr)) * max(0.0, float(cfg["trans_time"])))))
            return [_runs(U.mask_min_run(self.y0(cfg, c) < U._db2lin(cfg["trans_amp"]), win))
                    for c in range(self.x.shape[1])]
        return self._cached("gaps", cfg, PEAK_KEYS, f)

    def flags(self, cfg):
        """(M, dropped, pyr, cs): M bool (n, ch), column c = _clip_mask of channel c; per channel, the
        single samples Remove glitches dropped; M's envelope pyramid; per channel, the flag counts
        summed up to every RES samples (for risk_cols)."""
        def f():
            n, ch = self.x.shape
            M, dropped, raw_opt = np.empty((n, ch), bool), [], dict(cfg, remove_glitch=False)
            for c, st in enumerate(self.steps(cfg)):
                raw = U.flag_mask(self.y0(cfg, c), U.expand_steps(*st, n), raw_opt)
                M[:, c] = U.mask_min_run(raw, 2) if cfg["remove_glitch"] else raw
                dropped.append(np.flatnonzero(raw & ~M[:, c]))
            return _flagset(M, dropped)
        return self._cached("flags", cfg, FLAG_KEYS, f)

    def masks(self, cfg):
        M = self.flags(cfg)[0]
        return [M[:, c] for c in range(M.shape[1])]

    def stats(self, cfg):
        return self._cached("stats", cfg, FLAG_KEYS, lambda: _stats(*self.flags(cfg)[:2], self.sr))

    def hist9(self, cfg):
        """Per channel: counts of u = -log10(1 - |y0| / peaks(t)) in NINES bins, plus a last bin for
        the samples at or above the measured peak."""
        def f():
            lo, hi, bins = NINES
            out = []
            for c, st in enumerate(self.steps(cfg)):
                r = self.y0(cfg, c) / U.expand_steps(*st, len(self.x))
                top = int(np.count_nonzero(r >= 1))
                r = r[(r >= 1 - 10 ** -lo) & (r < 1)]
                b = ((-np.log10(1 - r) - lo) * (bins / (hi - lo))).astype(np.int64)
                out.append(np.r_[np.bincount(np.clip(b, 0, bins - 1), minlength=bins), top])
            return out
        return self._cached("hist9", cfg, PEAK_KEYS, f)


def hist_level(counts, r):
    """Samples at or above r x the measured peak, from hist9 counts (to one bin; samples below
    50 % of the peak are not counted)."""
    lo, hi, bins = NINES
    if r >= 1:
        return int(counts[bins])
    j = math.floor((-math.log10(1 - r) - lo) / (hi - lo) * bins)
    return int(counts[min(max(j, 0), bins):].sum())


def little_context(mask, ws):
    """{w: share of the flagged samples whose centred w-sample frame is more than half flagged}."""
    i = np.flatnonzero(mask)
    if not i.size:
        return dict.fromkeys(ws, 0.0)
    cs = np.r_[0, np.cumsum(mask, dtype=np.int32)]
    out = {}
    for w in ws:
        a, b = np.maximum(i - w // 2, 0), np.minimum(i + w // 2, len(mask))
        out[w] = float(np.mean(2 * (cs[b] - cs[a]) > b - a))
    return out


def risk_cols(cs, w, a, e):
    """bool per column (sample edges e): the column overlaps a SPADE render frame [k*a - w, k*a)
    more than half flagged. cs: a channel's flag counts summed up to every RES samples
    (Song.flags); w and a are multiples of RES."""
    e0, e1, nb = int(e[0]), int(e[-1]), len(cs) - 1
    ends = np.arange(e0 // a + 1, (e1 + w - 1) // a + 1) * a  # the frames overlapping [e0, e1)
    starts = ends - w
    r = 2 * (cs[np.clip(ends // RES, 0, nb)] - cs[np.clip(starts // RES, 0, nb)]) > w
    c0 = np.searchsorted(e[1:], starts[r], "right")
    c1 = np.searchsorted(e[:-1], ends[r], "left")
    d = np.bincount(c0, minlength=len(e)) - np.bincount(c1, minlength=len(e))
    return np.cumsum(d[:len(e) - 1]) > 0


def hotspots(counts, block, sr, length_s, k=5):
    """Up to k non-overlapping windows (i0, i1) of length_s seconds holding the most of `counts`
    (one value >= 0 per `block` samples), best first, each moved to centre on what it holds."""
    L = max(1, int(round(length_s * sr / block)))
    cs = np.r_[0, np.cumsum(counts, dtype=np.float64)]
    win = cs[L:] - cs[:-L] if len(counts) > L else cs[-1:]
    free, out = np.ones(len(win), bool), []
    for _ in range(k):
        j = int(np.argmax(np.where(free, win, -1)))
        if not free[j] or win[j] <= 0:
            break
        w = np.asarray(counts[j:j + L], np.float64)
        c = int(min(max(round(j + w @ np.arange(len(w)) / w.sum() + 0.5 - L / 2), 0), len(win) - 1))
        j = c if free[c] else j
        out.append((j * block, j * block + int(round(length_s * sr))))
        free[max(0, j - L + 1):j + L] = False
    return out


class Worker(threading.Thread):
    """A daemon thread that runs jobs one at a time, lowest prio first, then in submission order.
    A new job of a kind replaces the queued one and cancels the running one. fn(cancel, *args) gets
    a threading.Event that is set on cancel; its value is posted to q as ("job", kind, gen, value),
    an exception as ("job_err", kind, gen, message). A cancelled job posts nothing. Each job holds
    `lock` (the heavy worker's GPU_LOCK) while it runs; while paused, queued jobs wait. A long job
    lets urgent ones run inside it (run_pending)."""

    def __init__(self, q, lock=None):
        super().__init__(daemon=True)
        self.q, self.cv, self.jobs, self.stack, self.seq = q, threading.Condition(), {}, [], 0
        self.lock, self.paused = lock or contextlib.nullcontext(), False
        self.start()

    @property
    def running(self):
        """(kind, cancel event) of the job running (the innermost, see run_pending), or None."""
        return self.stack[-1] if self.stack else None

    def pause(self, on):
        with self.cv:
            self.paused = on
            self.cv.notify()

    def submit(self, kind, gen, fn, *args, prio=0):
        with self.cv:
            self.seq += 1
            self.jobs[kind] = prio, self.seq, gen, fn, args
            self._stop(kind)
            self.cv.notify()

    def cancel(self, kind=None):
        with self.cv:
            for k in [k for k in self.jobs if kind in (None, k)]:
                del self.jobs[k]
            self._stop(kind)

    def _stop(self, kind):
        for k, ev in self.stack:
            if kind in (None, k):
                ev.set()

    def run_pending(self, max_prio):
        """Run the queued jobs of priority <= max_prio now, on the calling job's thread (which holds the
        lock): a long job lets urgent ones through between its steps. Not while paused."""
        while True:
            with self.cv:
                ready = [k for k, j in self.jobs.items() if j[0] <= max_prio]
                if self.paused or not ready:
                    return
                job = self._take(min(ready, key=lambda k: self.jobs[k][:2]))
            self._exec(*job)

    def _take(self, kind):  # under cv: the job of this kind is now running
        _, _, gen, fn, args = self.jobs.pop(kind)
        ev = threading.Event()
        self.stack.append((kind, ev))
        return ev, kind, gen, fn, args

    def _exec(self, ev, kind, gen, fn, args):
        try:  # a job cancelled while it waited for the lock does not start
            msg = None if ev.is_set() else ("job", kind, gen, fn(ev, *args))
        except Exception as e:
            msg = "job_err", kind, gen, f"{type(e).__name__}: {e}"
        with self.cv:
            self.stack.pop()
            if msg and not ev.is_set():
                self.q.put(msg)

    def run(self):
        while True:
            with self.cv:
                while not self.jobs or self.paused:
                    self.cv.wait()
                job = self._take(min(self.jobs, key=lambda k: self.jobs[k][:2]))
            with self.lock:
                self._exec(*job)


class Estimates:
    """Seconds of work per audio second, per kind of job: rough seeds, then an EMA (alpha 0.5) of
    what previews and renders took. Kinds: ("SPADE", quality, lookahead, gpu), ("NMF", quality, gpu),
    ("unlimit", gpu) and ("knee", gpu) per second searched; see est_kind."""
    SPADE = {False: (0.9, 0.9, 1.5, 1.5, 2.6, 2.6, 4.0), True: (0.12, 0.12, 0.15, 0.15, 0.23, 0.23, 0.35)}
    LOOKAHEAD = (1.0, 1.6, 2.5)
    NMF = {False: {"fast": 2.3, "normal": 6.7, "high": 10.0, "best": 20.0, "normal+spade": 13.0},
           True: {"fast": 0.15, "normal": 0.5, "high": 0.8, "best": 1.5, "normal+spade": 0.75}}
    UNLIMIT = {False: 0.45, True: 0.05}  # plus loading the model on the CPU the first time: 1.7 s
    KNEE = {False: 1.5, True: 0.12}

    def __init__(self):
        self.r, self.measured = {}, set()

    def rate(self, kind):
        if kind not in self.r:
            self.r[kind] = (self.SPADE[kind[3]][kind[1]] * self.LOOKAHEAD[kind[2]] if kind[0] == "SPADE" else
                            self.NMF[kind[2]][kind[1]] if kind[0] == "NMF" else
                            self.KNEE[kind[1]] if kind[0] == "knee" else self.UNLIMIT[kind[1]])
        return self.r[kind]

    def update(self, kind, audio_s, work_s):
        if audio_s > 0:
            self.r[kind] = 0.5 * self.rate(kind) + 0.5 * work_s / audio_s
            self.measured.add(kind)

    def seconds(self, kind, audio_s):
        return audio_s * self.rate(kind) + (1.7 if kind == ("unlimit", False) and kind not in self.measured else 0.0)


def est_kind(cfg, fast, gpu):
    """The Estimates kind of declipping with cfg (None when Declip is off); gpu: on a GPU."""
    if not cfg["declip"]:
        return None
    if cfg["engine"] == "SPADE":
        q = max(0, min(6, int(cfg["speed_quality"])))
        return "SPADE", min(q, 3) if fast else q, max(0, min(2, int(cfg["lookahead"]))), gpu
    return "NMF", "fast" if fast else cfg["nmf_quality"], gpu


def rate_p(cfg, sr):
    pr = float(cfg["process_rate"] or 0)
    return pr if pr > 0 and abs(pr - sr) > 0.5 else sr


def stage_cfg_a(cfg, fast):
    """Stage A: declip (with the input gain) only."""
    opt = dict(cfg, unlimit=False, uncompress=False, output_gain=0.0, limiter=False)
    if fast:
        opt.update(nmf_quality="fast", speed_quality=min(int(cfg["speed_quality"]), 3))
    return opt


def stage_cfg_b(cfg):
    """Stage B: everything after declip, on stage A's output."""
    return dict(cfg, declip=False, input_gain=0.0)


def preview_span(song, cfg, r0, r1, fast):
    """[S, E) around the region [r0, r1) (file samples) that stage A processes.
    SPADE: S on the frame grid, so every frame touching the region sees the render's samples."""
    n, sr = len(song.x), song.sr
    # the inverse's state forgets its start 9x per release time / ratio: 5 of those leave < 1e-3 dB.
    # ponytail: capped at 10 s so stage A never declips most of the song; beyond it the region's
    # start is approximate (ratio x release > 2 s)
    warm = (min(math.ceil(5 * max(1.0, float(cfg["uc_ratio"])) * float(cfg["uc_release"]) / 1000 * sr), 10 * sr)
            if cfg["uncompress"] else 0)
    if not cfg["declip"]:
        return max(0, r0 - warm), r1
    # declipped once for any Release and Ratio (they only redo uc_gain), at the cost of 10 s more stage A
    warm = 10 * sr if cfg["uncompress"] else 0
    if cfg["engine"] == "SPADE":
        opt = stage_cfg_a(cfg, fast)
        w = 512 << max(0, min(2, int(opt["lookahead"])))
        a = w // OV[max(0, min(6, int(opt["speed_quality"])))]
        return max(0, r0 - warm - w) // a * a, min(n, r1 + w)
    from betterdeclipper.engine import nice_len
    sp = rate_p(cfg, sr)
    ctx = int((0.75 if fast else 1.5) * sp)
    S, E = max(0, r0 - warm - ctx), min(n, r1 + ctx)
    need = -(-427 * (nice_len(0.093 * sp) // 4) // song.x.shape[1])  # NMF rank reaches its 128 cap
    if not fast and E - S < need:
        S = max(0, min((S + E) // 2 - need // 2, n - need))
        E = min(n, S + need)
    return S, E


def hints_a(song, cfg, S, E, dev=None, sh=None):
    """Stage A's file-global inputs, so that x[S:E] is declipped as in the render: the manual-detection
    masks of the whole song (sliced) and SPADE's level normalisation; NMF: auto detection's analysis
    (with the shoulder sh) and the whole song's lambda reference (dev: the SPADE backend's device)."""
    h, pk = {}, song.peak() * U._db2lin(cfg["input_gain"])
    manual = cfg["engine"] == "SPADE" or not cfg["auto_detect"]
    if cfg["declip"] and manual:
        h["masks"] = [m[S:E] for m in song.masks(cfg)]
    if cfg["declip"] and cfg["engine"] == "SPADE" and pk > 0:
        h["norm"] = 1.0 / pk  # = 1 / max|y0|: scaling by the gain commutes with max |.|
    if cfg["declip"] and cfg["engine"] == "NMF":
        if manual:
            scale = song.nmf_scale(cfg)
        else:
            h["nmf_analysis"] = song.analysis(cfg, sh)
            scale = song.auto_flags(cfg, sh)["scale"]
        h["lam_refs"] = {w: z / scale for w, z in song.z0(cfg, U._nmf_device(dev)).items()}
    return h


def stage_a(song, opt, S, E, xp, dev, hints, progress=None):
    """Declip x[S:E] with opt = stage_cfg_a(...); hints carry the file-global masks, norm, analysis."""
    return U.process(song.x[S:E], song.sr, opt, xp, dev, progress, hints=dict(hints or {}, start=S))


def stage_b(opt, D, sr, hints, progress=None):
    """Unlimit, uncompress and output stages of stage A's output, opt = stage_cfg_b(...)."""
    return U.process(D, sr, opt, progress=progress, hints=hints)


def limit(P, sr, opt):
    """Stage B's last step, opt's output limiter, on P = stage B's output without it: process() itself
    with every other stage off, so limit(P) is the one-call result exactly."""
    if not opt["limiter"]:
        return P
    return U.process(P, sr, dict(opt, declip=False, unlimit=False, uncompress=False, input_gain=0.0, output_gain=0.0,
                                 process_rate=0.0))


def build_buffers(A, B, sr, gin_gout, monitor=None, match=False, side=False, hold=None):
    """Loop buffers for audition, float32 (n, 2): {1: Orig A, 2: Proc B, 3: Diff B - A x 10**(gin_gout/20)}
    (Proc and Diff None without B); mono doubled, past 2 channels the first two. side: (L-R)/2 on both
    outputs. match: Proc and Diff scaled by RMS(A)/RMS(B). monitor: the trim in dB for all three, or None
    for Auto: 0.3 dB under the louder peak of Orig and Proc, at most 0 dB (and at most hold, the Auto trim
    kept from before). 5 ms raised-cosine fades at both ends: the loop seam never clicks.
    Returns (buffers, trim dB, match dB)."""
    two = lambda v: (np.repeat(v, 2, 1) if v.shape[1] == 1 else v[:, :2]).astype(np.float32)
    bufs, gain, mdb = {1: two(A), 2: None, 3: None}, {1: 1.0, 2: 1.0, 3: 1.0}, 0.0
    if B is not None:
        bufs[2], bufs[3] = two(B), two(B - A * 10 ** (gin_gout / 20))
        ea, eb = float(np.vdot(bufs[1], bufs[1])), float(np.vdot(bufs[2], bufs[2]))
        if match and ea > 0 and eb > 0:
            mdb = 10 * np.log10(ea / eb)
            gain[2] = gain[3] = 10 ** (mdb / 20)
    live = {k: v for k, v in bufs.items() if v is not None}
    if side:
        for v in live.values():
            v[:, 0] -= v[:, 1]
            v[:, 0] *= 0.5
            v[:, 1] = v[:, 0]
    if monitor is None:
        pk = max(max(v.max(), -v.min()) * gain[k] if v.size else 0.0 for k, v in live.items() if k < 3)
        monitor = min(0.0, -(20 * np.log10(max(pk, 1e-9)) + 0.3))
        if hold is not None:  # kept unless the new peak would clip
            monitor = min(monitor, hold)
    nf = min(round(0.005 * sr), len(bufs[1]) // 2)
    w = (0.5 - 0.5 * np.cos(np.pi * np.arange(nf) / max(nf, 1)))[:, None].astype(np.float32)
    for k, v in live.items():
        v *= np.float32(10 ** (monitor / 20) * gain[k])
        v[:nf] *= w
        v[len(v) - nf:] *= w[::-1]
    return bufs, float(monitor), float(mdb)


def _take(b, pos, n, loop):
    """n frames of b from pos, wrapping when loop, else silent past its end; and the next pos."""
    if loop:
        return b[(pos + np.arange(n)) % len(b)], (pos + n) % len(b)
    out = np.zeros((n, b.shape[1]), np.float32)
    k = max(0, min(n, len(b) - pos))
    out[:k] = b[pos:pos + k]
    return out, pos + n


class Player:
    """Audition through sounddevice: plays bufs[src] (Orig while that is None) from pos, looping (or
    once to the end, loop False), times gain. Every change of source, buffers, position or play/stop
    crossfades 10 ms (equal power) from what was heard, so you keep hearing the same beat. The
    stream (file rate, stereo) is opened on the first play and kept; it is stopped while idle."""

    def __init__(self, sr):
        self.sr, self.stream, self.bufs, self.src, self.pos, self.loop, self.gain = sr, None, None, 1, 0, True, 1.0
        self.playing, self.xf, self.nxf, self.lock = False, None, max(1, round(0.01 * sr)), threading.Lock()

    @property
    def available(self):
        return sd is not None

    def _cur(self):
        if not (self.playing and self.bufs):
            return None
        b = self.bufs.get(self.src)
        return self.bufs.get(1) if b is None else b

    def _switch(self, **kw):
        """Change the state (under the lock), crossfading from the buffer heard at its position (a fade
        not yet played stays: its buffer is the one heard)."""
        with self.lock:
            if self.xf is None:
                self.xf = self._cur(), self.pos, self.gain, self.loop, self.nxf
            for k, v in kw.items():
                setattr(self, k, v)

    def play(self, bufs, pos=0, loop=True, gain=1.0):
        """Start (or restart from pos); raises sd.PortAudioError when the device fails."""
        if not self.available:
            return
        if self.stream is None:
            s = sd.OutputStream(samplerate=self.sr, channels=2, dtype="float32", blocksize=2048, latency="high",
                                callback=self._cb)
            self.stream = s
        self._switch(bufs=bufs, pos=pos, loop=loop, gain=gain, playing=True)
        if not self.stream.active:
            try:
                self.stream.start()
            except Exception:  # the next play retries with a new stream
                self.close()
                raise

    def stop(self):
        self._switch(playing=False, bufs=None)  # the fade-out keeps its own buffer

    def idle(self):
        """Stop the stream once a stop has faded out and played (the device no longer kept busy)."""
        if self.stream is not None and not self.playing and self.xf is None and self.stream.active:
            self.stream.abort()

    def set_bufs(self, bufs, pos=None, gain=None):
        self._switch(bufs=bufs, pos=self.pos if pos is None else pos, gain=self.gain if gain is None else gain)

    def set_src(self, src):
        if src != self.src:
            self._switch(src=src)

    def seek(self, pos):
        self._switch(pos=pos)

    def position(self):
        """The sample of the buffer being heard (output latency allowed for), or None when stopped."""
        b = self._cur()
        if b is None:
            return None
        p = self.pos - (self.stream.latency * self.sr if self.stream is not None else 0)
        return p % len(b) if self.loop else min(max(p, 0), len(b))

    def close(self):
        s, self.stream, self.playing, self.xf = self.stream, None, False, None
        if s is not None:
            try:
                s.abort(); s.close()
            except Exception:
                pass

    def _cb(self, out, frames, t, status):  # the audio thread
        with self.lock:
            b = self._cur()
            if b is None:
                out.fill(0)
            else:
                blk, self.pos = _take(b, self.pos, frames, self.loop)
                out[:] = blk * self.gain
                if not self.loop and self.pos >= len(b):  # the end of the song
                    self.playing, self.xf = False, None
            if self.xf:
                old, opos, og, oloop, left = self.xf
                k = min(frames, left)
                ph = (np.arange(k) + (self.nxf - left)) / self.nxf * (np.pi / 2)
                prev = 0.0 if old is None else _take(old, opos, k, oloop)[0] * og * np.cos(ph)[:, None]
                out[:k] = prev + out[:k] * np.sin(ph)[:, None]
                self.xf = (old, opos + k, og, oloop, left - k) if left > k else None
