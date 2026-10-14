# SPDX-FileCopyrightText: 2026 thatrandomalt
#
# SPDX-License-Identifier: MIT

"""Waveform editor: overview, ruler, lanes, gain lane and readout, drawn as numpy RGB rasters put
into Tk photo images (only the changed bands after the first put), draggable detection and level
lines, the level histogram panel and the uncompress transfer panel."""
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import ttk
from types import SimpleNamespace

import numpy as np

from session import (BLOCK, HIST_DB, LIM_CAP, NINES, OV, PEAK_KEYS, Pyramid, _runs, keyof, rate_p,
                     risk_cols)

BG, PANEL, LINE, FG, MUTED = "#1b1d22", "#262a31", "#3a3f48", "#e8e8e8", "#8b919c"
ACCENT, WAVE_IN = "#36d399", "#5a616e"
CANVAS, GUIDE, CONTOUR, OUT_STALE = "#14161a", "#c9ced6", "#9aa1ad", "#1f7a5a"
THRESH, FLAG, FLAG_HOT, FLAG_DIM, SOFT, CAP = "#ffd166", "#ff6b6b", "#ff3b3b", "#7a3a3a", "#f59e0b", "#b48cff"
CEIL = "#56c2e6"  # NMF auto detection's ceilings
UC, UNL, LIMIT = "#7aa2f7", "#c9ced6", "#f472b6"  # uncompress, unlimit gain, output limiter
REGION, SECTION, HATCH, WARN, WARN_BG = "#172927", "#171a1f", "#1f232a", "#f5b041", "#3a2a12"
PLAYHEAD = "#5ee3ad"  # audio
GUT = 34  # lane gutter px
FONT, TAB_FONT, TAB_FG = ("Segoe UI", 8), ("Segoe UI", 8, "bold"), "#0d1310"
STEPS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800)


def _rgb(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)


def _mix(a, b, f):
    """Colour a, f of the way to b."""
    return "#%02x%02x%02x" % tuple(np.round(_rgb(a) + (_rgb(b).astype(float) - _rgb(a)) * f).astype(int))


def _tfmt(t, d=3):
    m, s = divmod(round(t, d), 60)
    return f"{int(m)}:{s:0{3 + d if d else 2}.{d}f}"


def _stair(starts, values, e):
    """(min, max) per column of the staircase values[k] from starts[k] (starts[0] = 0), over
    column edges e."""
    i0 = np.searchsorted(starts, e[:-1], "right") - 1
    i1 = np.searchsorted(starts, np.maximum(e[1:] - 1, e[:-1]), "right") - 1
    v = values[:i1[-1] + 1]
    return (np.minimum(np.minimum.reduceat(v, i0), v[i1]),
            np.maximum(np.maximum.reduceat(v, i0), v[i1]))


def _segments(ys, ok):
    """Flat point lists (x = column) of a staircase through ys, one per run of ok columns."""
    out = []
    for a, b in zip(*_runs(ok)):
        y = ys[a:b]
        k = np.r_[0, np.flatnonzero(np.diff(y)) + 1]
        x = np.r_[k, b - a] + a
        p = np.empty((len(k), 4))
        p[:, 0], p[:, 2], p[:, 1], p[:, 3] = x[:-1], x[1:], y[k], y[k]
        out.append(p.ravel())
    return out


def _bounds(m, k, lim=1 << 16):
    """[start, end) of the True run of m through k (up to lim each way)."""
    a, b = m[max(0, k - lim):k + 1][::-1], m[k:k + lim]
    return k + 1 - (len(a) if a.all() else int(np.argmin(a))), k + (len(b) if b.all() else int(np.argmin(b)))


def _run(m, k):
    s, e = _bounds(m, k)
    return e - s


@dataclass
class Handle:
    """A draggable level line, + and -: in column j of lane c, polarity pol (1, -1), it sits at
    pol * ref(c, e, pol)[j] * lev(value) (e: column edges in samples), or where draw(c, e, pol) says
    (nan: not there); val inverts lev, so a drag to level v gives val(|v| / ref)."""
    key: str
    value: float
    colour: str
    dash: tuple | None
    width: int
    ref: Callable
    lev: Callable
    val: Callable
    step: float
    fine: float
    lo: float
    hi: float
    snaps: tuple = ()
    tab: str = ""
    help: str = ""
    draw: Callable | None = None


_DRAG = " Drag; Shift = fine; Ctrl = snap; double-click = preset; right-click = more."
REL = dict(clip_above=(("dynamic", "Dynamic threshold"), ("soft", "Soft-clip detection"),
                       ("remove_glitch", "Remove glitches")),
           trans_amp=(("diff_clips", "Measure each song separately"),))  # a handle's menu toggles
REL["not_clip_below"] = REL["clip_above"]
MENU = dict(tearoff=0, background=PANEL, foreground=FG, activebackground=LINE, activeforeground=FG)


_CAP_HELP = "Max restore: rebuilt peaks stay under this line; drag above the lane for no cap." + _DRAG


def _auto_handles(app, cfg, wv):
    """NMF auto detection: the knee (once a ceiling is found) and the cap, at each side's ceiling."""
    refs = wv.det["ceil"]["refs"]
    th = lambda c, pol: refs[c][pol < 0]["theta"] or np.nan
    ref = lambda c, e, pol=1: np.full(len(e) - 1, th(c, pol))
    out, k, cap = [], cfg["nmf_knee"], cfg["max_restore"]
    if any(r["kind"] != "none" for row in refs for r in row):
        sh = app._sh(cfg)
        shown = [wv._auto_lv(c, p)[1] / th(c, 1 - 2 * p) for c in range(len(refs)) for p in (0, 1)]
        tab = ("ceilings only" if k >= 1 else f"knee {k:.2f}" if k else "provisional" if sh is None
               else f"k{sh['knee']:.2f} found" if sh["knee"] is not None else "no shoulder")
        out.append(Handle("nmf_knee", k or float(np.nanmax(shown + [0.5])), THRESH, (4, 2), 2, ref, lambda v: v,
                          lambda r: 1.0 if r >= 0.995 else r, 0.01, 0.001, 0.5, 1.0, tab=tab,
                          help="NMF knee: samples above this line are rebuilt (amber up to the ceiling); 1 = the "
                               "ceilings only. Drag; Shift = fine; double-click = as found; right-click = more.",
                          draw=lambda c, e, pol=1: np.full(len(e) - 1, wv._auto_lv(c, pol < 0)[1])))
    if app.applies("max_restore"):
        def draw(c, e, pol=1):  # under limiter ceilings at most LIM_CAP x
            g = cap if cap > 0 else np.inf
            g = min(g, LIM_CAP) if refs[c][pol < 0]["kind"] == "limiter" else g
            return ref(c, e, pol) * (g if np.isfinite(g) else np.nan)
        out.append(Handle("max_restore", cap, CAP, (6, 3), 1, ref, lambda v: v, lambda r: r, 0.1, 0.01, 1.0, 20.0,
                          (2.0, 2.5, 3.0), f"×{cap:g}" if cap > 0 else "no cap", _CAP_HELP, draw))
    return out


def setting_handle(app, key):
    """A handle without a lane line (the transfer panel's, the gain lane's), for the keys and the wheel."""
    st, fine = CurvePanel.STEP.get(key, (0.5, 0.05))
    return Handle(key, float(app.cfg[key]), UC, None, 1, None, None, None, st, fine, *app.ranges[key],
                  help=CurvePanel.HELP.get(key, "Unlimit Amount: drag the gain lane up or down")
                  + "; Up/Down = step, Return = type, BackSpace = preset.")


def build_handles(app, cfg, page):
    """The live handles of the page, in hit-test priority order. Only settings app.applies() reads
    get one, so a greyed field never has a live handle."""
    det = app.wv.det
    one = lambda c, e, pol=1: np.ones(len(e) - 1)
    dbv = lambda r: 20 * np.log10(max(r, 1e-12))
    if cfg is None:
        return []
    if page == "Uncompress" and app.applies("uc_threshold"):  # on the stage input, which the input approximates
        T, M = cfg["uc_threshold"], cfg["uc_makeup"]
        return [Handle("uc_threshold", T, UC, (8, 3), 1, one, lambda v: 10 ** ((v + M) / 20), lambda r: dbv(r) - M,
                       0.5, 0.05, -60.0, 24.0, tab=f"≈ T {T:g} dB",
                       help="Uncompress threshold: louder levels (tinted in dB view) get turned up." + _DRAG)]
    if page == "Output" and app.applies("limiter_peak"):
        v = cfg["limiter_peak"]
        return [Handle("limiter_peak", v, LIMIT, (4, 2), 1, one, lambda u: 10 ** (u / 20), dbv, 0.1, 0.01, -24.0, 6.0,
                       tab=f"limit {v:g} dB", help="Limiter ceiling: the output is cut here (pink)." + _DRAG)]
    if page != "Declip":
        return []
    if app.applies("nmf_knee"):
        return _auto_handles(app, cfg, app.wv) if det.get("ceil") is not None else []
    if not app.applies("clip_above") or det.get("steps") is None:
        return []
    pk = lambda c, e, pol=1: _stair(*det["steps"][c], e)[1]
    pct = dict(ref=pk, lev=lambda v: v / 100, val=lambda r: 100 * r, step=0.1, fine=0.01, lo=1.0, hi=100.0)
    db = dict(ref=lambda c, e, pol=1: np.ones(len(e) - 1), lev=lambda v: 10 ** (v / 20),
              val=lambda r: 20 * np.log10(max(r, 1e-12)), step=1.0, fine=0.1, lo=-120.0, hi=0.0)
    h, low, cap = cfg["clip_above"], cfg["not_clip_below"], cfg["max_restore"]
    ncb = app.applies("not_clip_below")
    ghost = ncb and low <= h  # Dynamic off: the lower line wins
    out = [Handle("clip_above", h, _mix(THRESH, CANVAS, 0.55) if ghost else THRESH, None, 1 if ghost else 2,
                  snaps=(99.9, 78.0, 75.0),
                  tab="ignored (lower line wins)" if ghost else f"{h:g}% · {20 * np.log10(max(h, 1e-9) / 100):.2f} dB",
                  help="Clip above: samples above this line (red) are rebuilt." + _DRAG, **pct)]
    if app.applies("max_restore"):
        out.append(Handle("max_restore", cap, CAP, (6, 3), 1, pk, lambda v: v, lambda r: r, 0.1, 0.01, 1.0, 20.0,
                          (2.0, 2.5, 3.0), f"×{cap:g}" if cap > 0 else "no cap", _CAP_HELP,
                          lambda c, e, pol=1: pk(c, e) * (cap if cap > 0 else np.nan)))
    if ncb:
        out.append(Handle("not_clip_below", low, THRESH, (6, 3), 2 if ghost else 1, tab=f"{low:g}%",
                          help="Not-clip below: with Dynamic off, the lower line of the two decides." + _DRAG,
                          **pct))
    if app.applies("not_clip_amp"):
        v = cfg["not_clip_amp"]
        out.append(Handle("not_clip_amp", v, MUTED, (2, 3), 1, tab=f"{v:g} dB",
                          help="Not-clip level: nothing quieter than this is rebuilt." + _DRAG, **db))
    if app.applies("trans_amp"):
        v = cfg["trans_amp"]
        out.append(Handle("trans_amp", v, MUTED, (4, 4), 1, tab=f"gap {v:g} dB",
                          help="Gap level: quieter stretches (Gap length or longer) split the songs." + _DRAG, **db))
    return out


class Scale:
    """Lane amplitude axis. lin: lo..hi from bottom to top; db: |v| from floor_db at the centre to
    top_db at the edges, + up and - down."""

    def __init__(self, pad=8):
        self.mode, self.lo, self.hi, self.floor_db, self.top_db, self.pad = "lin", -1.0, 1.0, -48.0, 0.0, pad
        self.auto = True  # lo/hi follow the peak until the user zooms the amplitude

    def key(self):
        return self.mode, self.lo, self.hi, self.floor_db, self.top_db

    def _db(self):
        return self.floor_db, max(self.top_db, self.floor_db + 6)

    def y(self, v, H):
        v = np.asarray(v, dtype=np.float64)
        if self.mode == "lin":
            return self.pad + (self.hi - v) / (self.hi - self.lo) * (H - 2 * self.pad)
        lo, hi = self._db()
        f = np.clip((20 * np.log10(np.maximum(np.abs(v), 1e-12)) - lo) / (hi - lo), 0, 1)
        return H / 2 - np.sign(v) * f * (H / 2 - self.pad)

    def v(self, y, H):
        y = np.asarray(y, dtype=np.float64)
        if self.mode == "lin":
            return self.hi - (y - self.pad) / (H - 2 * self.pad) * (self.hi - self.lo)
        lo, hi = self._db()
        d = H / 2 - y
        return np.where(d < 0, -1.0, 1.0) * 10 ** ((lo + np.abs(d) / (H / 2 - self.pad) * (hi - lo)) / 20)

    def inside(self, v):
        """Which levels v the lane shows (not past its edges, nor under the dB floor)."""
        v = np.asarray(v, dtype=np.float64)
        if self.mode == "lin":
            return (v >= self.lo) & (v <= self.hi)
        lo, hi = self._db()
        d = 20 * np.log10(np.maximum(np.abs(v), 1e-12))
        return (d >= lo) & (d <= hi + 1e-6)


def _fill(mn, mx, H, s):
    """bool (H, cols): the rows each column's [mn, mx] covers, reaching to the next column's
    range so a line stays connected when zoomed in."""
    lo = np.minimum(mn, np.r_[mx[1:], mx[-1:]])
    hi = np.maximum(mx, np.r_[mn[1:], mn[-1:]])
    top = np.floor(s.y(hi, H))
    bot = np.maximum(np.floor(s.y(lo, H)), top)
    rows = np.arange(H)[:, None]
    return (rows >= top) & (rows <= bot)


def _edge(m):
    """The top and bottom pixel of each column's run in m."""
    i = m.copy()
    i[1:] &= m[:-1]; i[:-1] &= m[1:]
    return m & ~i


class WaveView(ttk.Frame):
    """Overview (whole song, view box, heat strip, hotspots), time ruler, one lane per channel
    (input grey, render output green up to `done` with the input's contour over it, manual
    detection overlays) and a readout line. Left-drag on the lanes or the ruler makes a region."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app, self.song, self.out, self.pyr_out, self.done, self.stale = app, None, None, None, 0, False
        self.pv = None  # region preview: (B, first sample, stale, peak)
        self.gain, self.t0, self.t1, self.scale = 1.0, 0.0, 1.0, Scale()
        self._parts, self._img, self._base, self._ovbase, self._pan0, self._ovgrab = set(), {}, None, None, None, None
        self._deco = None
        self.det, self._detv, self.region, self.cursor, self._rg, self._ovreg = {}, 0, None, None, None, None
        # handles: live list, tab boxes, hovered / keyboard-selected key, the drag in progress, the
        # pointer it waits to apply, extra fit while dragging the cap; frame bracket, hovered frame width
        self.handles, self._tabs, self.hover, self.sel, self.drag = [], [], None, None, None
        self._hpt, self._hjob, self.grow, self.brk, self.cand_w, self._ptr = None, None, 0.0, None, None, None
        self._riskc, self._help, self._tabx = None, None, {}  # _tabx: lane -> left edge of its value tabs
        self._tabfont, self._tabw = tkfont.Font(self, font=TAB_FONT), {}  # tab text -> width
        self._ls = tkfont.Font(self, font=FONT).metrics("linespace")  # a canvas label's height
        kw = dict(background=CANVAS, highlightthickness=0, borderwidth=0)
        self.ov = tk.Canvas(self, height=50, **kw)
        self.ruler = tk.Canvas(self, height=20, **kw)
        self.lanes = tk.Canvas(self, height=160, **dict(kw, highlightthickness=1, highlightcolor=ACCENT,
                                                        highlightbackground=CANVAS))  # focus ring
        self.glane = tk.Canvas(self, height=64, **kw)  # gain lane: packed while Unlimit or Uncompress is on
        self._gcurves, self._gd = [], None  # its curves drawn; its drag
        self.readout = ttk.Label(self, text="", style="Muted.TLabel")
        self.ov.pack(fill="x")
        self.ruler.pack(fill="x", pady=(1, 0))
        self.readout.pack(side="bottom", fill="x")  # packed before the lanes: they shrink first
        self.lanes.pack(fill="both", expand=True, pady=(1, 0))
        self.photo = {cv: tk.PhotoImage(master=self) for cv in (self.ov, self.lanes, self.glane)}
        self.glane.bind("<ButtonPress-1>", self._g_press)
        self.glane.bind("<B1-Motion>", self._g_move)
        self.glane.bind("<ButtonRelease-1>", self._g_release)
        self.glane.bind("<Double-Button-1>", self._g_dbl)
        self.glane.bind("<Button-3>", self._g_menu)
        self.glane.bind("<Motion>", self._g_motion)
        self.glane.bind("<Leave>", lambda e: self._gd is None and self.readout.config(text=""))
        for cv in (self.ov, self.ruler, self.lanes, self.glane):
            cv.bind("<Configure>", lambda e: self.redraw())
            cv.bind("<MouseWheel>", lambda e, cv=cv: self._wheel(e, cv, e.delta / 120))
            cv.bind("<Button-4>", lambda e, cv=cv: self._wheel(e, cv, 1))
            cv.bind("<Button-5>", lambda e, cv=cv: self._wheel(e, cv, -1))
            cv.bind("<Shift-MouseWheel>", lambda e, cv=cv: self._swheel(e, cv, -e.delta / 120))
            cv.bind("<Shift-Button-4>", lambda e, cv=cv: self._swheel(e, cv, -1))
            cv.bind("<Shift-Button-5>", lambda e, cv=cv: self._swheel(e, cv, 1))
        self.ruler.bind("<ButtonPress-1>", lambda e: self._r_press(e, True))
        self.ruler.bind("<B1-Motion>", self._r_move)
        self.ruler.bind("<ButtonRelease-1>", self._r_release)
        self.ruler.bind("<Shift-ButtonPress-1>", self._r_extend)
        self.lanes.bind("<ButtonPress-1>", self._press)
        self.lanes.bind("<Shift-ButtonPress-1>", lambda e: self._press(e, True))
        self.lanes.bind("<B1-Motion>", self._move)
        self.lanes.bind("<ButtonRelease-1>", self.release)
        self.lanes.bind("<Button-3>", self._menu)
        self.lanes.bind("<Control-MouseWheel>", lambda e: self._amp(e, e.delta / 120))
        self.lanes.bind("<Control-Button-4>", lambda e: self._amp(e, 1))
        self.lanes.bind("<Control-Button-5>", lambda e: self._amp(e, -1))
        self.lanes.bind("<Motion>", self._motion)
        self.lanes.bind("<Leave>", self._leave)
        self.lanes.bind("<Double-Button-1>", self._dbl)
        self.lanes.bind("<ButtonPress-2>", self._pan_start)
        self.lanes.bind("<B2-Motion>", self._pan)
        self.ov.bind("<Motion>", self._ov_motion)
        self.ov.bind("<ButtonPress-1>", self._ov_press)
        self.ov.bind("<B1-Motion>", self._ov_move)
        self.ov.bind("<Shift-ButtonPress-1>", lambda e: self._ov_region(e, True))
        self.ov.bind("<Shift-B1-Motion>", self._ov_region)
        self.ov.bind("<ButtonRelease-1>", self._ov_release)
        self.ov.bind("<Double-Button-1>", lambda e: self.fit())

    # ---- state ----
    def set_song(self, song):
        self.song = song if song is not None and len(song.x) else None
        self.out = self.pyr_out = self._base = self._ovbase = self.region = self.cursor = self.brk = self.pv = None
        self.done = 0
        self.det = dict(steps=None, gaps=None, pkey=None, flags=None, fkey=None, stats=None, hist=None, hot=[],
                        ceil=None, af=None)
        self.fit()
        self.redraw()

    def set_det(self, **kw):
        """Detection results for the current song. Manual: steps, gaps (Song.steps/gaps) and pkey (the
        PEAK_KEYS they are for), hist (Song.hist9); NMF auto: ceil (Song.ceilings), af (Song.auto_flags);
        both: flags (as Song.flags gives them) and fkey (App.fkey of their settings), stats, hot
        (hotspots)."""
        self.det.update(kw)
        if "steps" in kw:
            self._detv += 1  # the lanes' cached layer holds section bands and gap hatch
        self.redraw()

    def set_output(self, out, done, stale=None):
        """Render output, final up to sample `done`; stale: made with other settings (None: as
        before). The same array may come back with more final samples (only those are re-read); a
        new array, or a finished one, is read whole."""
        if out is not None and (self.song is None or len(out) != len(self.song.x)):
            out = None  # from a song no longer loaded
        if out is None:
            self.pyr_out, done = None, 0
        elif out is not self.out or done >= len(out):
            self.pyr_out = Pyramid(out)
        elif done > self.done:
            self.pyr_out.update(out, self.done, done)
        self.out, self.done = out, done
        if stale is not None:
            self.stale = stale
        self.redraw(("ov", "lanes"))

    def set_preview(self, B, S=0, stale=False, P=None):
        """Region preview B (n, ch) of the samples from S, or None; stale: made with other settings; P:
        B before the output limiter (the Output page draws what it cuts). It is drawn in its columns
        over any render output."""
        pk = lambda a: float(np.abs(a).max()) if a.size else 0.0
        self.pv = None if B is None else (B, int(S), stale, pk(B), B if P is None else P, pk(B if P is None else P))
        self.redraw(("ov", "lanes", "gain"))

    def _lim(self):
        """The limiter's level while the Output page shows what it cuts, else None."""
        cfg = self.app.cfg
        on = self.app.page == "Output" and self.app.applies("limiter_peak")
        return 10 ** (cfg["limiter_peak"] / 20) if on else None

    def set_region(self, r0, r1=None):
        """Region [r0, r1) in samples, at least one (also mid-drag), or None."""
        if r0 is None:
            self.region = None
        else:
            n = len(self.song.x)
            r0 = int(min(r0, n - 1))
            self.region = r0, int(min(max(r1, r0 + 1), n))
        self.redraw()

    def set_playhead(self, t, page=False):
        """The audio playhead at sample t (None: none); page: when it leaves the view, the next page."""
        if self.song is None:
            t = None
        elif t is not None and page and not self.t0 <= t < self.t1:
            self._view(t, self.t1 - self.t0)
        for cv, x in ((self.lanes, None if t is None or not self.t0 <= t <= self.t1 else self._x(t)),
                      (self.ov, None if t is None else self._ovx(t))):
            item = cv.find_withtag("ph")
            if x is None:
                cv.delete("ph")
            elif item:
                cv.coords(item[0], x, 0, x, cv.winfo_height())
                cv.tag_raise("ph")
            else:
                cv.create_line(x, 0, x, cv.winfo_height(), fill=PLAYHEAD, tags="ph")

    def set_mode(self, mode):
        self.scale.mode = mode
        self.redraw(("lanes",))

    def fit(self):
        """Whole song; amplitude follows the peak; dB floor -48."""
        self.scale.auto, self.scale.floor_db = True, -48.0
        if self.song is not None:
            self._view(0.0, len(self.song.x))

    def zoom_region(self):
        """The region with one region length either side."""
        r0, r1 = self.region
        self._view(r0 - (r1 - r0), 3 * (r1 - r0))

    def jump(self, r0, r1):
        """Region [r0, r1), the view around it, and the app told (a user action)."""
        self.set_region(r0, r1)
        self.zoom_region()
        self._region_done()

    def _view(self, t0, span):
        n = len(self.song.x)
        span = min(n, max(span, (self.lanes.winfo_width() - GUT) / 16))  # down to 16 px per sample
        self.t0 = min(max(0.0, t0), n - span)
        self.t1 = self.t0 + span
        self.redraw()

    def _show(self):
        """Declip overlays: "page" (lines, bands, tips) on the Declip page, "dim" (tips at 35 %) on
        the others, None with Declip off."""
        if self.app.cfg is None or not self.app.cfg["declip"]:
            return None
        return "page" if self.app.page == "Declip" else "dim"

    def _auto_lv(self, c, p):
        """NMF auto detection, lane c, polarity p (0 +, 1 -): (ceiling theta, knee level), nan where
        none. The knees are the flags' when those are for these settings (or the knee is as found),
        else knee x theta (a drag)."""
        d, k = self.det, float(self.app.cfg["nmf_knee"])
        r, af = d["ceil"]["refs"][c][p], d["af"]
        th = np.nan if r["theta"] is None else r["theta"]
        if af is not None and (not k or d["fkey"] == self.app.fkey()):
            kn = af["knees"][c][p]
            return th, np.nan if kn is None else float(kn)
        return th, th * k if k else np.nan

    # ---- navigation ----
    def _t(self, x):
        return self.t0 + (x - GUT) / max(1, self.lanes.winfo_width() - GUT) * (self.t1 - self.t0)

    def _ts(self, x):
        return int(min(max(self._t(x), 0), len(self.song.x)))

    def _x(self, t):
        return GUT + (t - self.t0) / (self.t1 - self.t0) * (self.lanes.winfo_width() - GUT)

    def _e(self):
        """Column edges (samples) of the lanes."""
        return np.linspace(self.t0, self.t1, max(1, self.lanes.winfo_width() - GUT) + 1).astype(np.int64)

    def _wheel(self, e, cv, notches):
        """Zoom about the pointer; over a handle, one drag step; over the frame label, SPADE quality."""
        if self.song is None:
            return
        if cv is self.lanes:
            hit = self._hit(e.x, e.y)
            if hit:
                return self.nudge(hit[0], np.sign(notches))
            if self._brk_at(e.x, e.y, "label"):
                self.app._set("speed_quality", int(self.app.cfg["speed_quality"]) + int(np.sign(notches)))
                return self.app._commit()
        f = 1.25 ** -notches
        a = self._t(e.x) if cv is not self.ov else min(max(self._ovt(e.x), self.t0), self.t1)
        self._view(a - (a - self.t0) * f, (self.t1 - self.t0) * f)

    def _swheel(self, e, cv, notches):
        """Shift+wheel: scroll; over a handle, one fine step."""
        hit = cv is self.lanes and self.song is not None and self._hit(e.x, e.y)
        if hit:
            self.nudge(hit[0], -np.sign(notches), True)
        else:
            self._scroll(notches)

    def _scroll(self, notches):
        if self.song is not None:
            self._view(self.t0 + 0.1 * notches * (self.t1 - self.t0), self.t1 - self.t0)

    def zoom(self, f):
        """Time zoom by f about the view centre."""
        if self.song is not None:
            a = (self.t0 + self.t1) / 2
            self._view(a - (a - self.t0) * f, (self.t1 - self.t0) * f)

    def home(self, end):
        if self.song is not None:
            self._view(len(self.song.x) if end else 0.0, self.t1 - self.t0)

    def shift_region(self, k):
        """Move the region by k region lengths; the view follows when it leaves it."""
        if self.song is None or not self.region:
            return
        r0, r1 = self.region
        r0 = min(max(r0 + k * (r1 - r0), 0), len(self.song.x) - (r1 - r0))
        self.set_region(r0, r0 + r1 - self.region[0])
        if not (self.t0 <= self.region[0] and self.region[1] <= self.t1):
            self.zoom_region()
        self._region_done()

    def next_hot(self, k):
        """The next (k = 1) or previous (-1) hotspot: region there, view around it."""
        hot = self.det["hot"] if self.song is not None and self._show() else []
        if hot:
            i = next((i for i, r in enumerate(hot) if r == self.region), -1 if k > 0 else 0)
            self.jump(*hot[(i + k) % len(hot)])

    def clear_region(self):
        self.region = None
        self.redraw()
        self.app.region_changed()  # the user chose none: no automatic one either

    def escape(self):
        """Esc: clear the region, else deselect the handle."""
        if self.region is not None:
            self.clear_region()
        elif self.sel is not None:
            self.sel = None
            self._sel_shown()

    def _nl(self, H):
        """Lanes shown: mono and stereo always; more channels up to 8, at least 60 px each."""
        return min(self.song.x.shape[1], 8, max(2, H // 60))

    def _lane(self, y):
        H = self.lanes.winfo_height()
        nl = self._nl(H)
        hl = max(1, H // nl)
        return min(int(y) // hl, nl - 1), hl

    def _amp(self, e, notches):
        if self.song is None:
            return
        c, hl = self._lane(e.y)
        s = self.scale
        if s.mode == "lin":
            v, f = float(s.v(e.y - c * hl, hl)), 1.25 ** -notches
            s.lo, s.hi, s.auto = v - (v - s.lo) * f, v + (s.hi - v) * f, False
        else:
            s.floor_db = min(-6.0, max(-120.0, s.floor_db + 6 * np.sign(notches)))
        self.redraw(("lanes",))

    def _pan_start(self, e):
        self._pan0 = e.x, e.y, self.t0, self.scale.lo, self.scale.hi

    def _pan(self, e):
        if self.song is None or self._pan0 is None:
            return
        x0, y0, t0, lo, hi = self._pan0
        s, (_, hl) = self.scale, self._lane(e.y)
        if s.mode == "lin" and not s.auto:
            dv = (e.y - y0) / (hl - 2 * s.pad) * (hi - lo)
            s.lo, s.hi = lo + dv, hi + dv
        span = self.t1 - self.t0
        self._view(t0 - (e.x - x0) / max(1, self.lanes.winfo_width() - GUT) * span, span)

    def _ovx(self, t):
        return GUT + t / len(self.song.x) * (self.ov.winfo_width() - GUT)

    def _ovt(self, x):
        return (x - GUT) / max(1, self.ov.winfo_width() - GUT) * len(self.song.x)

    def _ov_motion(self, e):
        if self.song is not None:
            x0, x1 = self._ovx(self.t0), self._ovx(self.t1)
            near = x1 - x0 > 8 and min(abs(e.x - x0), abs(e.x - x1)) <= 4
            self.ov.config(cursor="sb_h_double_arrow" if near else "")

    def _ov_press(self, e):
        self._ovgrab = None
        if self.song is None:
            return
        if e.y <= 9 and self._show():  # a hotspot triangle
            for r in self.det["hot"]:
                if abs(e.x - self._ovx((r[0] + r[1]) / 2)) <= 5:
                    self.jump(*r)
                    return
        x0, x1 = self._ovx(self.t0), self._ovx(self.t1)
        if x1 - x0 > 8 and abs(e.x - x0) <= 4:
            self._ovgrab = "l"
        elif x1 - x0 > 8 and abs(e.x - x1) <= 4:
            self._ovgrab = "r"
        else:  # grab the box, or centre it here first
            if not x0 <= e.x <= x1:
                self._view(self._ovt(e.x) - (self.t1 - self.t0) / 2, self.t1 - self.t0)
            self._ovgrab = self._ovt(e.x) - self.t0

    def _ov_move(self, e):
        if self.song is None or self._ovgrab is None:
            return
        t, span = self._ovt(e.x), self.t1 - self.t0
        if self._ovgrab == "l":
            t = max(0.0, min(t, self.t1 - 1))
            self._view(t, self.t1 - t)
        elif self._ovgrab == "r":
            self._view(self.t0, t - self.t0)
        else:
            self._view(t - self._ovgrab, span)

    def _ov_region(self, e, start=False):
        """Shift+drag on the overview: a region."""
        if self.song is None:
            return
        t = int(min(max(self._ovt(e.x), 0), len(self.song.x)))
        if start:
            self._ovreg = t
        elif self._ovreg is not None:
            self.set_region(min(self._ovreg, t), max(self._ovreg, t))

    def _ov_release(self, e):
        if self._ovreg is not None and self.region is not None:
            self._region_done()
        self._ovreg = None

    # ---- region and edit cursor ----
    def _r_press(self, e, move_ok):
        """A drag makes a region, or resizes it from an edge (+-4 px), or moves it (ruler); a
        click (under 3 px) sets the edit cursor."""
        self._rg = None
        if self.song is None or e.x < GUT:
            return
        t, grab = self._ts(e.x), "new"
        if self.region:
            x0, x1 = self._x(self.region[0]), self._x(self.region[1])
            if abs(e.x - x0) <= 4:
                grab = "l"
            elif abs(e.x - x1) <= 4:
                grab = "r"
            elif move_ok and x0 < e.x < x1:
                grab = t - self.region[0]
        self._rg = [grab, e.x, t, False]

    def _r_move(self, e):
        if not self._rg:
            return
        grab, x0, t0, moved = self._rg
        if not moved and abs(e.x - x0) < 3:
            return
        self._rg[3] = True
        t, n = self._ts(e.x), len(self.song.x)
        r0, r1 = self.region or (t0, t0)
        if grab == "new":
            r0, r1 = min(t0, t), max(t0, t)
        elif grab == "l":
            r0 = min(t, r1 - 1)
        elif grab == "r":
            r1 = max(t, r0 + 1)
        else:
            r0 = min(max(t - grab, 0), n - (r1 - r0))
            r1 += r0 - self.region[0]
        self.set_region(r0, r1)
        r0, r1 = self.region
        self.readout.config(text=f"{(r1 - r0) / self.song.sr:.1f} s" + self.app.est_text(r0, r1))

    def _r_release(self, e):
        rg, self._rg = self._rg, None
        if not rg:
            return
        if rg[3]:
            self._region_done()
        elif rg[0] == "new":
            self.cursor = rg[2]
            self.redraw(("lanes",))
            self.app.cursor_set(rg[2])

    def _r_extend(self, e):
        """Shift+click: stretch the region (or from the edit cursor) to here."""
        if self.song is None or e.x < GUT:
            return
        t = self._ts(e.x)
        if self.region:
            self.region = min(self.region[0], t), max(self.region[1], t)
        elif self.cursor is not None:
            self.region = min(self.cursor, t), max(self.cursor, t)
        else:
            return
        self._region_done()

    def _dbl(self, e):
        """Double-click: a handle goes back to the preset's value, the gutter fits, a lane sets the
        region to 1 s either side."""
        if self.song is None:
            return
        hit = self._hit(e.x, e.y)
        if hit:
            return self.reset(hit[0])
        if self._lk_at(e.x, e.y):  # each click switches it
            return self._press(e)
        if e.x < GUT:
            self.fit()
            return
        t, sr = self._ts(e.x), self.song.sr
        self.region = max(0, t - sr), min(len(self.song.x), t + sr)
        self._region_done()

    def _region_done(self):
        """A region set by the user: at least 0.25 s."""
        n, m = len(self.song.x), int(0.25 * self.song.sr)
        r0, r1 = self.region
        if r1 - r0 < m:
            r0 = max(0, min(r0, n - m))
            r1 = min(n, r0 + m)
        self.region = r0, r1
        self.redraw()
        self.app.region_changed()

    # ---- the pointer on the lanes: handles, frame bracket, linked-stereo glyph, region ----
    def _press(self, e, shift=False):
        self.lanes.focus_set()
        if self.hide_help() or self.song is None:
            return
        hit = self._hit(e.x, e.y)
        if hit:
            self._h_press(e, *hit)
        elif self._brk_at(e.x, e.y, "edge"):
            self.drag, self.app.dragging = {"brk": True}, "lookahead"
        elif self._lk_at(e.x, e.y):
            v = self.app.T["linked_stereo"]
            v.set(1 - v.get())
            self.app._commit()
        elif shift:
            self._r_extend(e)
        else:
            self._r_press(e, False)

    def _move(self, e):
        if self.drag is None:
            self._r_move(e)
        elif "brk" in self.drag:
            self._brk_move(e.x)
        else:  # one apply in flight: the latest pointer wins
            self._hpt = e.x, e.y, e.state
            if self._hjob is None:
                self._hjob = self.after_idle(self._h_apply)

    def release(self, e=None):
        if self.drag is None:
            return self._r_release(e)
        if self._hjob is not None:
            self.after_cancel(self._hjob)
            self._h_apply()
        self.drag = None
        if self.grow:
            self.grow = 0.0
            self.redraw()
        self.app.drag_end()

    def _col(self, x):
        """Column edges of the lanes' column at x."""
        e = self._e()
        j = min(max(int(x) - GUT, 0), len(e) - 2)
        return e[j:j + 2]

    def _ref(self, h, c, x, pol=1):
        """Handle h's reference level in column x of lane c, polarity pol."""
        return float(np.asarray(h.ref(c, self._col(x), pol), float)[0])

    def value_at(self, h, c, x, y):
        """The value of h that puts its line at y (px in lane c) in column x."""
        v = float(self.scale.v(y, self._lane(0)[1]))
        return float(h.val(abs(v) / max(self._ref(h, c, x, 1 if v >= 0 else -1), 1e-12)))

    def _hit(self, x, y):
        """(handle, lane) under the pointer: on its tab, or within 5 px of its line (the selected
        handle first, then catalogue order). A line within 6 px of the centre is not grabbable there."""
        if self.song is None or self.drag:
            return None
        for x0, y0, x1, y1, h, c in self._tabs:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return h, c
        if x < GUT or not self.handles:
            return None
        (c, hl), s = self._lane(y), self.scale
        mid, near = float(s.y(0.0, hl)), []
        for h in self.handles:
            for pol in (1, -1):
                v = pol * float(self._levels(h, c, self._col(x), pol)[0])
                yy = float(s.y(v, hl))
                if s.inside(v) and abs(y - c * hl - yy) <= 5 and abs(yy - mid) >= 6:
                    near.append(h)
        if near:
            return next((h for h in near if h.key == self.sel), near[0]), c

    def _grow(self, h):
        """Dragging the cap widens the fit (until release) so its line, or 3x, shows."""
        if h.key == "max_restore":
            ed = self._e()
            ref = np.nanmax([np.max(h.ref(k, ed, pol)) for k in range(self._nl(self.lanes.winfo_height()))
                             for pol in (1, -1)])
            self.grow = 1.15 * max(h.value, 3.0) * float(ref)
            self._fit_scale()

    def _h_press(self, e, h, c):
        self.app.dragging = h.key
        hl = self._lane(0)[1]
        self._grow(h)
        y = e.y - c * hl
        off = h.value - self.value_at(h, c, e.x, y)
        pol = 1 if float(self.scale.v(y, hl)) >= 0 else -1
        ly = float(self.scale.y(pol * self._levels(h, c, self._col(e.x), pol)[0], hl))
        if h.key == "nmf_knee" or h.key == "max_restore" and (h.value <= 0 or not abs(ly - y) <= 5):
            off = 0.0  # absolute: the fit grew under the pointer (or no cap), or the sides' knees differ
        self.drag = dict(h=h, c=c, y=e.y, v=h.value, off=off, pol=pol)
        self.redraw(("lanes",))

    def _h_apply(self):
        """The latest pointer of a handle drag: absolute (keeping the grab offset), Shift = relative
        fine, Ctrl = a snap; a plain drag also snaps within 3 px and 3 steps of one. Fine drags
        continue from the unsnapped value. The cap 12 px above its lane = no cap."""
        self._hjob, d = None, self.drag
        if not d or "h" not in d or self._hpt is None:
            return
        x, y, state = self._hpt
        h, c, hl = d["h"], d["c"], self._lane(0)[1]
        if state & 1:
            v = d["v"] - (y - d["y"]) * h.fine * d["pol"]
        else:
            v = self.value_at(h, c, x, y - c * hl) + d["off"]
        v = min(max(v, h.lo), h.hi)
        d["y"], d["v"] = y, v
        if h.snaps:
            ref = self._ref(h, c, x, d["pol"])
            py = lambda u: float(self.scale.y(d["pol"] * ref * h.lev(u), hl))
            sn = min(h.snaps, key=lambda u: abs(py(u) - py(v)))
            if state & 4 or (not state & 1 and abs(py(sn) - py(v)) <= 3 and abs(sn - v) <= 3 * h.step):
                v = sn
        if h.key == "max_restore" and y - c * hl < -12:
            v = 0.0
        self.app._set(h.key, v, h.fine if state & 1 else h.step)

    def drag_to(self, key, value=None, c=0, y=None, state=0):
        """Drag handle `key` from its line in lane c to `value` (or to y, px in the lanes) as the
        pointer would; release() ends the drag (selfcheck)."""
        h = self._live(key)
        hl, ed = self._lane(0)[1], self._e()
        self._grow(h)
        lv = self._levels(h, c, ed)
        ok = np.flatnonzero(self.scale.inside(lv))
        j = int(ok[len(ok) // 2]) if ok.size else 0
        E = SimpleNamespace
        self._h_press(E(x=GUT + j, y=c * hl + float(self.scale.y(lv[j], hl))), h, c)
        if y is None:
            y = c * hl + float(self.scale.y(self._ref(h, c, GUT + j) * h.lev(value), hl))
        self._move(E(x=GUT + j, y=y, state=state))
        self.after_cancel(self._hjob)
        self._h_apply()

    def _live(self, key):
        return next((h for h in self.handles if h.key == key), None)

    def _levels(self, h, c, e, pol=1):
        """Handle h's level (magnitude, nan: none) per column of lane c (column edges e), polarity pol."""
        return np.asarray(h.draw(c, e, pol) if h.draw else h.ref(c, e, pol) * h.lev(h.value), float)

    def nudge(self, h, k, fine=False):
        """Handle h moved by k drag steps (fine steps); one undo step each."""
        st = h.fine if fine else h.step
        try:
            v = float(self.app.V[h.key].get())
        except ValueError:
            v = h.value
        if h.key == "nmf_knee" and v <= 0:  # as found: from the knee shown
            v = h.value
        if h.key == "max_restore" and v <= 0:  # no cap: up starts at the first snap, never at 1 (nothing rebuilt)
            if k <= 0:
                return
            v, k = h.snaps[0], 0
        self.app._set(h.key, min(max(v + k * st, h.lo), h.hi), st)
        self.app._commit()

    def reset(self, h):
        """Handle h back to the preset's value."""
        self.app._set(h.key, self.app.base[h.key])
        self.app._commit()

    def _keys(self):
        """The page's handles for Tab: the lanes', then the transfer panel's and the gain lane's."""
        keys = [h.key for h in self.handles]
        if self.app.page == "Uncompress" and self.app.applies("uc_threshold"):
            keys += [k for k in ("uc_threshold", "uc_ratio", "uc_amount") if k not in keys]
        g = self._gkey()
        return keys + [g] if g and g not in keys else keys

    def sel_handle(self):
        """The keyboard-selected handle, or None."""
        h = self._live(self.sel)
        return h if h is not None or self.sel not in self._keys() else setting_handle(self.app, self.sel)

    def select(self, k):
        """Tab: the next (k = 1) or previous (-1) handle of the page."""
        keys = self._keys()
        if keys:
            i = keys.index(self.sel) if self.sel in keys else (-1 if k > 0 else 0)
            self.sel = keys[(i + k) % len(keys)]
            self.readout.config(text=self.sel_handle().help)
            self._sel_shown()

    def _sel_shown(self):
        self.redraw(("lanes", "gain"))
        self.app._show_stages()  # the transfer panel's dots

    def _set_hover(self, key):
        """The hovered handle's line one px wider."""
        if key != self.hover:
            for k, dw in ((self.hover, 0), (key, 1)):
                h = self._live(k)
                if h:
                    self.lanes.itemconfig("h_" + k, width=h.width + dw)
            self.hover = key

    def _bracket(self, x, y):
        """(lane, start, length, hop, read-only) of the frame to show at the pointer, or None: the
        SPADE render frame whose centre is nearest the flagged run under the pointer (within 2
        frames), else the pointer; with NMF, its analysis window there (read-only)."""
        cfg = self.app.cfg
        if self.song is None or x < GUT or self._show() != "page":
            return None
        c, sr = self._lane(y)[0], self.song.sr
        t, spade = self._ts(x), cfg["engine"] == "SPADE"
        if spade:
            w = 512 << min(2, max(0, int(cfg["lookahead"])))
            a = w // OV[min(6, max(0, int(cfg["speed_quality"])))]
        else:
            try:
                from betterdeclipper.engine import nice_len
            except ImportError:  # no PyTorch: no NMF
                return None
            w = nice_len(0.093 * rate_p(cfg, sr))
            a = w // 4
        mid, M = t, self.det["flags"]
        if M is not None:
            lo = max(0, t - 2 * w)
            i = np.flatnonzero(M[0][lo:t + 2 * w, c])
            if i.size:
                s, e = _bounds(M[0][:, c], lo + int(i[np.argmin(np.abs(i + lo - t))]))
                mid = (s + e) / 2
        f0 = round((mid + w / 2) / a) * a - w if spade else int(mid - w / 2)
        if self._x(f0 + w) - self._x(f0) < 12:
            return None
        return c, int(f0), w, a, not spade

    def _brk_at(self, x, y, part):
        """On the SPADE frame bracket: its right "edge", its "label" row, or its "body"."""
        b = self.brk
        if not b or b[4] or self._lane(y)[0] != b[0]:
            return False
        x0, x1 = self._x(b[1]), self._x(b[1] + b[2])
        if part == "edge":
            return abs(x - x1) <= 4
        return x0 <= x <= x1 and (part == "body" or y - b[0] * self._lane(0)[1] <= 22)

    def _brk_move(self, x):
        """Right-edge drag: the frame length nearest the pointer (512/1024/2048) becomes the Frame."""
        c, f0, w, a, ro = self.brk
        k = int(np.argmin([abs(self._t(x) - f0 - (512 << k)) for k in range(3)]))
        self.app._set("lookahead", k)
        self.brk = c, f0, 512 << k, (512 << k) // (w // a), ro
        self._draw_brk()

    def _draw_brk(self):
        cv = self.lanes
        cv.delete("brk")
        if self.brk is None or self.song is None:
            return
        c, f0, w, a, ro = self.brk
        hl, sr, W = self._lane(0)[1], self.song.sr, cv.winfo_width()
        y0, x0, x1 = c * hl, self._x(f0), self._x(f0 + w)
        col, M = MUTED if ro else THRESH, self.det["flags"]
        cv.create_rectangle(x0, y0 + 1, x1, y0 + hl - 2, outline=col, dash=(4, 3), tags="brk")
        if ro:
            sp = rate_p(self.app.cfg, sr)
            text = (f"NMF window {w} smp = {1000 * w / sr:.0f} ms, hop {a}"
                    + (f" (as processed: {1000 * w / sp:.0f} ms)" if sp != sr else "") + " · Lookahead is SPADE-only")
        else:
            for k in range(1, w // a):
                x = self._x(f0 + k * a)
                cv.create_line(x, y0 + 1, x, y0 + 6, fill=col, tags="brk")
            f = np.count_nonzero(M[0][max(0, f0):max(0, f0 + w), c]) / w if M is not None else 0.0
            text = (f"frame {w} smp · {1000 * w / sr:.1f} ms · hop {a} (×{w // a}) · "
                    f"this frame {100 * f:.0f} % flagged")
            col = SOFT if f > 0.5 else col
        t = cv.create_text(max(x0, GUT) + 3, y0 + 8, text=text, anchor="nw", fill=col, font=FONT, tags="brk")
        bb, lim = cv.bbox(t), self._tabx.get(c, W) - 4
        if bb[2] > lim:
            cv.move(t, max(GUT + 3 - bb[0], lim - bb[2]), 0)
        cv.tag_lower(cv.create_rectangle(*cv.bbox(t), fill=CANVAS, width=0, tags="brk"), t)

    def _lk(self):
        """Centre of the linked-stereo glyph (SPADE, stereo, Declip page), or None."""
        if (self.song is None or self.song.x.shape[1] != 2 or self._show() != "page"
                or not self.app.applies("linked_stereo")):
            return None
        return GUT // 2, self._lane(0)[1]

    def _lk_at(self, x, y):
        p = self._lk()
        return p is not None and abs(x - p[0]) <= 8 and abs(y - p[1]) <= 10

    def _motion(self, e):
        self._ptr = e.x, e.y
        if self.song is None:
            return
        hit = self._hit(e.x, e.y)
        self._set_hover(hit[0].key if hit else None)
        b = self._bracket(e.x, e.y)
        if b != self.brk:
            self.brk = b
            self._draw_brk()
        cur, text, ceil = "", None, self._ceil_at(e.x, e.y, hit and hit[0].key == "nmf_knee")
        if hit:  # the knee on a ceiling (ceilings only, no shoulder): the ceiling's details
            cur, text = "sb_v_double_arrow", ceil if hit[0].key == "nmf_knee" and ceil else hit[0].help
        elif ceil:
            text = ceil
        elif self._brk_at(e.x, e.y, "edge"):
            cur = "sb_h_double_arrow"
            text = "Frame: drag the edge to 512, 1024 or 2048 samples. Wheel on its label: SPADE quality."
        elif self._lk_at(e.x, e.y):
            cur = "hand2"
            text = (f"Linked stereo is {'on' if self.app.cfg['linked_stereo'] else 'off'}: one SPADE solve "
                    "for both channels. Click to switch.")
        elif self.region and min(abs(e.x - self._x(t)) for t in self.region) <= 4:
            cur = "sb_h_double_arrow"
        self.lanes.config(cursor=cur)
        self.readout.config(text=self._where(e) if text is None else text)

    def _leave(self, e):
        self._ptr = None
        if self.drag is None:
            self._set_hover(None)
            if self.brk is not None:
                self.brk = None
                self._draw_brk()
            self.readout.config(text="")

    def _ceil_at(self, x, y, knee=False):
        """NMF auto detection: what the ceiling within 5 px of the pointer is (knee: the knee is on it),
        or None."""
        d = self.det
        if x < GUT or self._show() != "page" or d["ceil"] is None:
            return None
        c, hl = self._lane(y)
        for p, r in enumerate(d["ceil"]["refs"][c]):
            v = (1 - 2 * p) * (r["theta"] or 0.0)
            if r["kind"] != "none" and self.scale.inside(v) and abs(float(self.scale.y(v, hl)) - y + c * hl) <= 5:
                return (f"{self._label(c)}{'+-'[p]} {r['kind']} ceiling {20 * np.log10(r['theta']):.2f} dB "
                        f"(θ {r['theta']:.5f}, σ {r['sigma']:.5f}) · flat {100 * r.get('flat', 0):.0f} % · "
                        f"long {100 * r.get('long', 0):.0f} %" + (" · shared" if r.get("shared") else "")
                        + (". The knee is on it: drag it, or right-click." if knee else
                           ". Found by the analysis; right-click for the knee."))
        return None

    def _where(self, e):
        """Readout: time, channel, level, % of local peak (or of the ceiling) and the flagged run at the
        pointer."""
        if e.x < GUT:
            return ""
        c, hl = self._lane(e.y)
        v = float(self.scale.v(e.y - c * hl, hl))
        text = f"{_tfmt(self._t(e.x) / self.song.sr)} · {self._label(c)} · {20 * np.log10(max(abs(v), 1e-12)):.2f} dBFS"
        d, n = self.det, len(self.song.x)
        if not self._show():
            return text
        i0 = int(min(max(self._t(e.x), 0), n - 1))
        i1 = max(i0 + 1, int(min(self._t(e.x + 1), n)))
        if d["steps"] is not None:
            st, pk = d["steps"][c]
            text += f" · {100 * abs(v) / pk[np.searchsorted(st, i0, 'right') - 1]:.1f} % of local peak"
        elif d["ceil"] is not None:
            p = int(v < 0)
            r = d["ceil"]["refs"][c][p]
            if r["kind"] != "none":
                text += f" · {100 * abs(v) / r['theta']:.1f} % of {self._label(c)}{'+-'[p]} ceiling"
        if d["flags"] is not None:
            m = d["flags"][0][:, c]
            k = i0 + int(np.argmax(m[i0:i1]))
            if m[k]:
                text += f" · flagged (run {_run(m, k):,} smp)"
        return text

    def _menu(self, e):
        """Right-click: the menu of a handle, of the frame bracket, or of the region."""
        if self.song is None or self.drag:
            return
        app, m = self.app, tk.Menu(self, **MENU)
        hit = self._hit(e.x, e.y)
        if hit and hit[0].key != "nmf_knee":
            h = hit[0]
            m.add_command(label="Type value…", command=lambda: app.type_value(h.key))
            m.add_command(label="Reset to preset", command=lambda: self.reset(h))
            if h.key == "max_restore":
                m.add_command(label="No cap", command=lambda: (app._set(h.key, 0), app._commit()))
            for k, label in REL.get(h.key, ()):
                m.add_checkbutton(label=label, variable=app.T[k], command=app._commit)
        elif hit or self._ceil_at(e.x, e.y):  # the knee, or a ceiling
            knee = lambda v: (app._set("nmf_knee", v), app._commit())
            m.add_command(label="Type knee…", command=lambda: app.type_value("nmf_knee"))
            m.add_command(label="Knee as found", command=lambda: knee(0))
            m.add_command(label="Ceilings only (knee 1)", command=lambda: knee(1))
            m.add_command(label="Use as manual threshold", command=app.seed_manual)
        elif self._brk_at(e.x, e.y, "body"):
            for k in range(3):
                m.add_radiobutton(label=f"Frame {512 << k}", variable=app.V["lookahead"], value=str(k),
                                  command=app._commit)
            q = tk.Menu(m, **MENU)
            for k in range(7):
                q.add_radiobutton(label=str(k), variable=app.V["speed_quality"], value=str(k), command=app._commit)
            m.add_cascade(label="SPADE quality", menu=q)
        else:
            on = "normal" if self.region else "disabled"
            m.add_command(label="Preview region", command=self.app.preview_now, state=on)
            m.add_command(label="Clear region", command=self.clear_region, state=on)
            m.add_command(label="Zoom to region", command=self.zoom_region, state=on)
            m.add_command(label="Fit", command=self.fit)
        m.tk_popup(e.x_root, e.y_root)

    def toggle_help(self, rows):
        """F1: the keyboard map, rows of (keys, action), over the lanes. Any key or click closes it."""
        if self.hide_help():
            return
        f = self._help = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=LINE, padx=14, pady=10)
        half = -(-len(rows) // 2)
        for i, (k, t) in enumerate(rows):
            r, c = i % half, 2 * (i // half)
            tk.Label(f, text=k, bg=PANEL, fg=FG, font=("Segoe UI", 9, "bold")).grid(
                row=r, column=c, sticky="w", padx=(24 if c else 0, 10))
            tk.Label(f, text=t, bg=PANEL, fg=FG, font=("Segoe UI", 9)).grid(row=r, column=c + 1, sticky="w")
        for w in (f, *f.winfo_children()):
            w.bind("<Button-1>", lambda e: self.hide_help())
        f.place(in_=self.lanes, relx=0.5, rely=0.5, anchor="center")

    def hide_help(self):
        """Close the keyboard map; True if it was open."""
        if self._help is None:
            return False
        self._help.destroy()
        self._help = None
        return True

    def _label(self, c):
        ch = self.song.x.shape[1]
        return "M" if ch == 1 else "LR"[c] if ch == 2 else str(c + 1)

    # ---- drawing ----
    def redraw(self, parts=("ov", "ruler", "lanes", "gain")):
        """Coalesced: draws once when Tk is idle."""
        if not self._parts:
            self.after_idle(self._draw)
        self._parts |= set(parts)

    def _draw(self):
        parts, self._parts = self._parts or {"ov", "ruler", "lanes", "gain"}, set()
        W, H = self.lanes.winfo_width(), self.lanes.winfo_height()
        try:
            g = float(self.app.V["input_gain"].get())
            if g == g:  # typing is not limited to the field's range: keep 10**(g/20) finite
                self.gain = 10 ** (min(max(g, -240.0), 240.0) / 20)
        except ValueError:
            pass
        cfg = self.app.cfg
        gl = self.song is not None and cfg is not None and bool(cfg["unlimit"] or cfg["uncompress"])
        if gl != bool(self.glane.winfo_manager()):  # the gain lane takes its height from the lanes
            if gl:
                self.glane.pack(side="bottom", fill="x", pady=(1, 0), before=self.lanes)
            else:
                self.glane.pack_forget()
        if self.song is None or W - GUT < 20 or H < 20:
            for cv in (self.ov, self.ruler, self.lanes, self.glane):
                cv.delete("all")
            self._img.clear()
            self.handles, self._tabs, self.brk, self._gcurves = [], [], None, []
            self.lanes.create_text(W / 2, H / 2, text="No file loaded", fill=MUTED, tags="deco")
            self.lanes.create_text(W / 2, H / 2 + 16, text="Ctrl+O: open a clipped song.", fill=MUTED, font=FONT,
                                   tags="deco")
            return
        self._fit_scale()
        if "lanes" in parts:
            self._draw_lanes(W, H)
        if "ruler" in parts:
            self._draw_ruler(self.ruler.winfo_width())
        if "ov" in parts:
            self._draw_ov(self.ov.winfo_width(), self.ov.winfo_height())
        if "gain" in parts:
            self._draw_gain(self.glane.winfo_width(), self.glane.winfo_height())

    def _fit_scale(self):
        """Amplitude fit: the peak (or what a cap drag needs), unless the user zoomed it."""
        pk, s = max(self._pk(), self.grow), self.scale
        if s.auto:
            s.lo, s.hi = -pk, pk
        s.top_db = 20 * np.log10(pk)

    def _pk(self):
        p, q = self.song.pyr, self.pyr_out
        pk = max(-p.mn[-1].min(), p.mx[-1].max()) * self.gain
        if q is not None:
            pk = max(pk, -q.mn[-1].min(), q.mx[-1].max())
        if self.pv is not None:
            pk = max(pk, self.pv[5])
        return max(float(pk), 1e-9)

    def _outputs(self, e, render, cut=False):
        """Runs of columns (column edges e) that show output, as (c0, c1, mn, mx, stale) with mn and
        mx over all columns: the render up to `done` (render() gives its envelope), and the region
        preview, which wins in its columns (cut: before the limiter)."""
        k = int(np.count_nonzero(e[:-1] < self.done)) if self.pyr_out is not None else 0
        runs = []
        if k:
            mn, mx = render()
            runs.append((0, k, mn, mx, self.stale))
        if self.pv is not None:
            B, S, stale = self.pv[4 if cut else 0], self.pv[1], self.pv[2]
            c0, c1, mn, mx = self._colmm(B, S, e)
            if c1 > c0:
                runs = [(a, b, *r[2:]) for r in runs for a, b in ((r[0], min(r[1], c0)), (c1, r[1])) if b > a]
                runs.append((c0, c1, mn, mx, stale))
        return runs

    def _colmm(self, B, S, e):
        """(c0, c1, mn, mx): the columns [c0, c1) of edges e that B (n, ch), samples from S, covers and
        its min and max in each (over all columns, 0 outside). A column narrower than a sample holds
        the sample under it."""
        c0, c1 = self._cols(e, (S, S + len(B)))
        mn, mx = (np.zeros((len(e) - 1, B.shape[1])) for _ in range(2))
        if c1 > c0:
            i = np.clip(e[c0:c1] - S, 0, len(B) - 1)
            mn[c0:c1], mx[c0:c1] = np.minimum.reduceat(B, i, axis=0), np.maximum.reduceat(B, i, axis=0)
        return c0, c1, mn, mx

    def _cols(self, e, r):
        """Column range [c0, c1) overlapping the samples [r0, r1), for column edges e."""
        return np.searchsorted(e[1:], r[0], "right"), np.searchsorted(e[:-1], r[1], "left")

    def _bands(self, rgb, bg, c, e, hl, cfg):
        """Section bands (Different clips), and hatch over the floor band and the gaps, on the
        background bg (cleared under the hatch)."""
        if cfg["diff_clips"]:
            rgb[bg & ((np.searchsorted(self.det["steps"][c][0], e[:-1], "right") - 1) % 2 == 1)] = _rgb(SECTION)
        rows, cols = np.arange(hl)[:, None], np.arange(len(e) - 1)
        h = np.zeros((hl, len(cols)), bool)
        if self.app.applies("not_clip_amp"):
            fl = 10 ** (cfg["not_clip_amp"] / 20)
            h |= (rows >= self.scale.y(fl, hl)) & (rows <= self.scale.y(-fl, hl))
        if self.app.applies("trans_amp"):
            g0, g1 = self._gaps(c, cfg)
            if len(g0):
                j = np.minimum(np.searchsorted(g1, e[:-1], "right"), len(g0) - 1)
                h |= ((g1[j] > e[:-1]) & (g0[j] < e[1:]))[None, :]
        h &= ((rows + cols) % 4 == 0) & bg
        rgb[h] = _rgb(HATCH)
        bg &= ~h

    def _gaps(self, c, cfg):
        """(starts, ends) of lane c's gaps between songs; while the results are for other settings
        (a Gap level drag), from 64-sample block maxima."""
        if self.det["pkey"] == keyof(cfg, PEAK_KEYS) and self.det["gaps"] is not None:
            return self.det["gaps"][c]
        p = self.song.pyr
        s, e = _runs(np.maximum(p.mx[0][:, c], -p.mn[0][:, c]) * self.gain < 10 ** (cfg["trans_amp"] / 20))
        k = (e - s) * 64 >= cfg["trans_time"] * rate_p(cfg, self.song.sr)
        return s[k] * 64, np.minimum(e[k] * 64, len(self.song.x))

    def _tips(self, nl, hl, e, fill, mn, mx):
        """Bool masks over the lanes, or None: rebuilt (above the threshold, or NMF auto detection's
        ceiling, in a flagged column, plus a 4 px mark past the column's peak so it shows above the
        line), flagged only by Soft (or between the knee and the ceiling), and over the threshold but
        dropped by Remove glitches. mn, mx: the drawn envelope. While the flags are for other settings
        (a drag, or new ones on their way), a column counts as flagged when its envelope reaches the
        threshold (or knee)."""
        d, cfg, s = self.det, self.app.cfg, self.scale
        auto = d["ceil"] is not None
        if d["flags"] is None or not auto and d["steps"] is None:
            return None
        M, dropped, fpyr, _ = d["flags"]
        exact = d["fkey"] == self.app.fkey()
        flagged = fpyr.minmax(M, self.t0, self.t1, len(e) - 1)[1] > 0 if exact else None
        h = (cfg["clip_above"] if cfg["dynamic"] else min(cfg["clip_above"], cfg["not_clip_below"])) / 100
        floor = 10 ** (cfg["not_clip_amp"] / 20)
        rows = np.arange(hl)[:, None]
        above = lambda vp, vn: (rows <= s.y(vp, hl)) | (rows >= s.y(-vn, hl))  # nan: no row
        out = [np.zeros_like(fill) for _ in range(3)]
        for c in range(nl):
            sl = slice(c * hl, (c + 1) * hl)
            if auto:  # red above the ceiling, amber from the knee
                (tp, ap), (tn, an) = (self._auto_lv(c, p) for p in (0, 1))
                tp, tn = (t if r["kind"] != "none" else np.nan for t, r in zip((tp, tn), d["ceil"]["refs"][c]))
                fp, fn = ap, an
            else:
                pk = _stair(*d["steps"][c], e)[0]
                tp = tn = np.maximum(pk * h, floor)
                ap = an = np.maximum(0.92 * pk, floor) if cfg["soft"] else np.nan
                fp = fn = np.maximum(pk * (min(h, 0.92) if cfg["soft"] else h), floor)
            fc = flagged[:, c] if exact else (mx[:, c] >= fp) | (mn[:, c] <= -fn)
            top = fill[sl] & above(tp, tn)
            yt, yb = s.y(mx[:, c], hl), s.y(mn[:, c], hl)
            out[0][sl] = ((top & fc) | (fc & (mx[:, c] >= tp) & (rows >= yt - 4) & (rows < yt))
                          | (fc & (mn[:, c] <= -tn) & (rows > yb) & (rows <= yb + 4)))
            out[1][sl] = fill[sl] & above(ap, an) & ~top & fc
            if exact and not auto:
                out[2][sl] = top & ~fc & (np.diff(np.searchsorted(dropped[c], e)) > 0)
        return out

    def _risk(self, nl, e):
        """Per lane, the columns overlapping a SPADE frame more than half flagged, for the Frame in
        use or the one hovered in the inspector (cand_w); None when not SPADE or no flags yet."""
        cfg, d = self.app.cfg, self.det
        if d["flags"] is None or cfg["engine"] != "SPADE":
            return None
        w = self.cand_w or 512 << min(2, max(0, int(cfg["lookahead"])))
        a = w // OV[min(6, max(0, int(cfg["speed_quality"])))]
        key = e[0], e[-1], len(e), w, a, id(d["flags"]), nl
        if self._riskc is None or self._riskc[0] != key:
            self._riskc = key, [risk_cols(cs, w, a, e) for cs in d["flags"][3][:nl]]
        return self._riskc[1]

    def _lanes_rgb(self, W, H):
        x, s, d, cfg = self.song.x, self.scale, self.det, self.app.cfg
        nl = self._nl(H)
        hl, wc = H // nl, W - GUT
        e = np.linspace(self.t0, self.t1, wc + 1).astype(np.int64)
        show = self._show()
        deco = show == "page" and d["steps"] is not None  # section bands and hatch
        key = wc, nl, hl, self.t0, self.t1, self.gain, s.key()
        if self._base is None or self._base[0] != key:  # the waveform
            mn, mx = self.song.pyr.minmax(x, self.t0, self.t1, wc)
            rgb = np.empty((nl * hl, wc, 3), np.uint8)
            rgb[:] = _rgb(CANVAS)
            fill = np.concatenate([_fill(mn[:, c] * self.gain, mx[:, c] * self.gain, hl, s) for c in range(nl)])
            rgb[fill] = _rgb(WAVE_IN)
            edge = np.concatenate([_edge(fill[c * hl:(c + 1) * hl]) for c in range(nl)])
            self._base, self._deco = (key, rgb, fill, edge, mn * self.gain, mx * self.gain), None
        _, rgb, fill, edge, mn, mx = self._base
        dkey = deco and (self._detv, cfg["diff_clips"], cfg["not_clip_amp"], self.app.applies("not_clip_amp"),
                         keyof(cfg, PEAK_KEYS) if self.app.applies("trans_amp") else None)
        if self._deco is None or self._deco[0] != dkey:  # + section bands and hatch (a Floor drag redoes only these)
            rgb, bg = rgb.copy(), ~fill
            for c in range(nl if deco else 0):
                sl = slice(c * hl, (c + 1) * hl)
                self._bands(rgb[sl], bg[sl], c, e, hl, cfg)
            self._deco = dkey, rgb, bg
        _, rgb, bg = self._deco
        rgb = rgb.copy()
        if self.region:
            c0, c1 = self._cols(e, self.region)
            rgb[:, c0:c1][bg[:, c0:c1]] = _rgb(REGION)
        if show == "page" and d["ceil"] is not None:  # NMF auto: the soft region, smeared ceilings' +-3 sigma
            rows = np.arange(hl)[:, None]
            for c in range(nl):
                for p, r in enumerate(d["ceil"]["refs"][c]):
                    th, kn = self._auto_lv(c, p)
                    sg, sm = 1 - 2 * p, r["kind"] == "smeared"
                    for a, b, col, f in ((kn, th, SOFT, 0.15), (th - 3 * r["sigma"], th + 3 * r["sigma"], CEIL, 0.2 * sm)):
                        ya, yb = sorted((float(s.y(sg * a, hl)), float(s.y(sg * b, hl))))
                        if f and ya < yb:
                            m = bg[c * hl:(c + 1) * hl] & (rows >= ya) & (rows <= yb)
                            lane = rgb[c * hl:(c + 1) * hl]
                            lane[m] = np.round(lane[m] + (_rgb(col) - lane[m].astype(float)) * f).astype(np.uint8)
        rows = np.arange(hl)[:, None]
        beyond = lambda v: (rows < np.floor(s.y(v, hl))) | (rows > np.ceil(s.y(-v, hl)))  # past +-v
        if self.app.page == "Uncompress" and s.mode == "db" and self.app.applies("uc_threshold"):
            m = np.tile(beyond(10 ** ((cfg["uc_threshold"] + cfg["uc_makeup"]) / 20)), (nl, 1)) & fill
            rgb[m] = _rgb(_mix(WAVE_IN, UC, 0.5))  # the levels over the uncompress threshold
        tips = self._tips(nl, hl, e, fill, mn, mx) if show else None
        tipc = [_rgb(_mix(WAVE_IN, col, 0.35) if show == "dim" else col) for col in (FLAG, SOFT, FLAG_DIM)]
        if tips is not None:
            for m, col in zip(tips, tipc):
                rgb[m] = col
        lim = self._lim()
        for c0, c1, omn, omx, stale in self._outputs(e, lambda: self.pyr_out.minmax(self.out, self.t0, self.t1, wc),
                                                     lim is not None):
            for c in range(nl):
                sl = slice(c * hl, (c + 1) * hl)
                lane, ed = rgb[sl, c0:c1], edge[sl, c0:c1]
                of = _fill(omn[c0:c1, c], omx[c0:c1, c], hl, s)
                lane[of] = _rgb(OUT_STALE if stale else ACCENT)
                if lim is not None:  # what the limiter cuts
                    lane[of & beyond(lim)] = _rgb(LIMIT)
                lane[ed] = _rgb(CONTOUR)
                if tips is not None:
                    lane[ed & tips[0][sl, c0:c1]] = tipc[0]
        risk = self._risk(nl, e) if show == "page" else None
        for c, rc in enumerate(risk or ()):  # frames with little context
            rgb[(c + 1) * hl - 3:(c + 1) * hl, rc] = _rgb(FLAG_HOT)
        return rgb

    def _lines(self, cv, nl, hl, e, W):
        """Declip page: manual detection's measured peak and Soft line, NMF auto detection's ceilings;
        and the page's handles; in each lane, + and -, each with a value tab at the right (a handle's
        tab grabs it too)."""
        cfg, s, self._tabs, d = self.app.cfg, self.scale, [], self.det
        none = np.full(len(e) - 1, np.nan)
        page = self._show() == "page"
        for c in range(nl):
            y0 = c * hl
            spec = []  # (levels +, levels -, colour, dash, width, tab, handle)
            if page and d["steps"] is not None:
                pk = _stair(*d["steps"][c], e)[1]
                spec.append((pk, pk, GUIDE, (1, 3), 1, "peak", None))
                if self.app.applies("soft") and cfg["soft"]:
                    spec.append((pk * 0.92, pk * 0.92, SOFT, (2, 2), 1, "92%", None))
            for p, r in enumerate(d["ceil"]["refs"][c] if page and d["ceil"] is not None else ()):
                if r["kind"] != "none":
                    v = np.full(len(e) - 1, r["theta"])
                    tab = f"{self._label(c)}{'+-'[p]} {r['kind']} {20 * np.log10(r['theta']):.2f} dB"
                    if r["kind"] == "limiter":
                        tab += (" · pairs only" if self._auto_lv(c, p)[1] >= 0.99 * r["theta"] else "") + \
                               f" · cap +{20 * np.log10(LIM_CAP):.0f} dB"
                    spec.append(((v, none) if p == 0 else (none, v)) + (CEIL, None, 1, tab, None))
            spec += [(self._levels(h, c, e), self._levels(h, c, e, -1), h.colour, h.dash,
                      h.width + (h.key == self.hover), h.tab, h) for h in self.handles]
            tabs = []
            for vp, vn, col, dash, w, tab, h in spec:
                placed = False
                for sg, v in ((1, vp), (-1, vn)):
                    ok = s.inside(sg * v)
                    ys = y0 + s.y(sg * v, hl)
                    for p in _segments(ys, ok):
                        p[0::2] += GUT
                        cv.create_line(p.tolist(), fill=col, dash=dash, width=w,
                                       tags=("deco", "h_" + h.key) if h else "deco")
                    if not placed and ok.any():
                        tabs.append((ys[np.flatnonzero(ok)[-1]], col, tab, h))
                        placed = True
                if not placed and h is not None and h.key == "max_restore":  # no cap, or over the lane
                    tabs.append((y0, col, tab + " ^", h))
            last = y0 - 99
            for y, col, tab, h in sorted(tabs, key=lambda t: t[0]):  # pushed apart, inside the lane
                y = min(max(y, last + 15, y0 + 8), y0 + hl - 8)
                if tab not in self._tabw:
                    self._tabw[tab] = max(46, self._tabfont.measure(tab) + 8)
                last, w = y, self._tabw[tab]
                self._tabx[c] = min(self._tabx.get(c, W), W - 2 - w)  # the frame bracket's label stays left of it
                sel = h is not None and h.key == self.sel  # the keyboard-selected handle: outlined
                cv.create_rectangle(W - 2 - w, y - 7, W - 2, y + 7, fill=col, outline=ACCENT, width=2 if sel else 0,
                                    tags="deco")
                cv.create_text(W - 2 - w / 2, y, text=tab, fill=TAB_FG, font=TAB_FONT, tags="deco")
                if h is not None:
                    self._tabs.append((W - 2 - w, y - 7, W - 2, y + 7, h, c))

    def _draw_lanes(self, W, H):
        cv, s, n, ch = self.lanes, self.scale, len(self.song.x), self.song.x.shape[1]
        self._put(cv, self._lanes_rgb(W, H))
        cv.delete("deco")
        nl = self._nl(H)
        hl = H // nl
        line = lambda y, **kw: cv.create_line(GUT, y, W, y, tags="deco", **kw)
        text = lambda x, y, t, **kw: cv.create_text(x, y, text=t, font=FONT, tags="deco", **kw)
        self.handles, self._tabs, self._tabx = build_handles(self.app, self.app.cfg, self.app.page), [], {}
        lk = self._lk()
        for c in range(nl):
            y0 = c * hl
            if c:
                cv.create_line(0, y0, W, y0, fill=LINE, tags="deco")
            text(4, y0 + 3, self._label(c), anchor="nw", fill=FG)
            line(y0 + float(s.y(0.0, hl)), fill=LINE)
            if s.mode == "lin":
                ticks = [(v, f"{v:g}") for v in (0, 1, -1, 0.5, -0.5)]
                guide = max(s.hi, -s.lo) > 1
            else:
                ticks = [(sg * 10 ** (d / 20), f"{d:g}") for d in (0, -12, -24, -6, -48)
                         if s.floor_db < d <= s.top_db for sg in (1, -1)]
                guide = s.top_db > 0
            shown = [lk[1]] if lk else []
            for v, t in ticks:  # in order of preference, 12 px apart
                y = y0 + float(s.y(v, hl))
                if y0 + 6 <= y <= y0 + hl - 6 and all(abs(y - u) >= 12 for u in shown):
                    shown.append(y)
                    text(GUT - 4, y, t, anchor="e", fill=MUTED)
            if guide:  # 0 dBFS
                for v in (1.0, -1.0):
                    y = y0 + float(s.y(v, hl))
                    if y0 <= y <= y0 + hl:
                        line(y, fill=GUIDE, dash=(4, 4))
                y = y0 + float(s.y(1.0, hl))
                if y0 + 12 <= y <= y0 + hl:
                    text(GUT + 6, y - 3, "0 dBFS", fill=GUIDE, anchor="sw")
        if lk:  # two joined rings between the L and R labels
            col = ACCENT if self.app.cfg["linked_stereo"] else MUTED
            cv.create_rectangle(lk[0] - 6, lk[1] - 9, lk[0] + 6, lk[1] + 9, fill=CANVAS, width=0, tags="deco")
            for dy in (-4, 4):
                cv.create_oval(lk[0] - 4, lk[1] + dy - 4, lk[0] + 4, lk[1] + dy + 4, outline=col, width=1.5,
                               tags="deco")
        if self.handles or self._show() == "page" and (self.det["steps"] is not None or self.det["ceil"] is not None):
            self._lines(cv, nl, hl, np.linspace(self.t0, self.t1, W - GUT + 1).astype(np.int64), W)
        marks = [(t, ACCENT) for t in self.region or ()] + [(self.cursor, MUTED)]
        if 0 < self.done < n:
            marks.append((self.done, FG))
        for t, col in marks:
            if t is not None and self.t0 <= t <= self.t1:
                x = self._x(t)
                cv.create_line(x, 0, x, nl * hl, fill=col, tags="deco")
        if ch > nl:
            text(W - 6, H - 3, f"showing {nl} of {ch} channels", anchor="se", fill=MUTED)
        if self._ptr and not (self.drag and "brk" in self.drag):  # the settings may have changed
            self.brk = self._bracket(*self._ptr)
        self._draw_brk()

    def _draw_ruler(self, W):
        cv, sr, wc = self.ruler, self.song.sr, W - GUT
        cv.delete("all")
        a, b = self.t0 / sr, self.t1 / sr
        if self._show() == "page" and self.det["steps"] is not None:
            st = self.det["steps"][0][0]  # the first channel's sections or peak_time blocks
            if self.app.cfg["diff_clips"]:
                e = np.linspace(self.t0, self.t1, wc + 1).astype(np.int64)
                for x0, x1 in zip(*_runs((np.searchsorted(st, e[:-1], "right") - 1) % 2 == 1)):
                    cv.create_rectangle(GUT + x0, 0, GUT + x1, 20, fill=SECTION, width=0)
            else:
                v = st[(st > self.t0) & (st < self.t1)]
                if 4 * len(v) <= wc:
                    for t in v:
                        cv.create_line(self._x(t), 16, self._x(t), 20, fill=LINE)
        if self.region:
            x0, x1 = (min(max(self._x(t), GUT), W) for t in self.region)
            cv.create_rectangle(x0, 16, max(x1, x0 + 1), 20, fill=ACCENT, width=0)
        step = next((st for st in STEPS if st * wc / (b - a) >= 70), STEPS[-1])
        d = max(0, -int(np.floor(np.log10(step) + 1e-9)))
        for k in range(int(np.ceil(a / step)), int(b / step) + 1):
            x = GUT + (k * step - a) / (b - a) * wc
            cv.create_line(x, 13, x, 20, fill=MUTED)
            cv.create_text(x + 3, 9, text=_tfmt(k * step, d), anchor="w", fill=MUTED, font=FONT)

    def _draw_ov(self, W, H):
        cv, x, n, wc = self.ov, self.song.x, len(self.song.x), W - GUT
        Hw = H - 7  # the heat strip takes the last 6 rows
        s = Scale(pad=3)
        pk = self._pk()
        s.lo, s.hi = -pk, pk
        key = (wc, H, self.gain, pk)
        if self._ovbase is None or self._ovbase[0] != key:
            mn, mx = self.song.pyr.minmax(x, 0, n, wc)
            rgb = np.empty((H, wc, 3), np.uint8)
            rgb[:] = _rgb(CANVAS)
            f = _fill(mn.min(1) * self.gain, mx.max(1) * self.gain, Hw, s)
            rgb[:Hw][f] = _rgb(WAVE_IN)
            self._ovbase = key, rgb, ~f
        _, rgb, bg = self._ovbase
        rgb = rgb.copy()
        e = np.linspace(0, n, wc + 1).astype(np.int64)
        if self.region:
            c0, c1 = self._cols(e, self.region)
            rgb[:Hw, c0:c1][bg[:, c0:c1]] = _rgb(REGION)
        for c0, c1, omn, omx, stale in self._outputs(e, lambda: self.pyr_out.minmax(self.out, 0, n, wc)):
            rgb[:Hw, c0:c1][_fill(omn[c0:c1].min(1), omx[c0:c1].max(1), Hw, s)] = _rgb(OUT_STALE if stale else ACCENT)
        show = self._show()
        st = self.det["stats"] if show else None
        if st is not None:  # share of rebuilt samples per column
            cs = np.r_[0, np.cumsum(st.block_counts, dtype=np.int64)]
            b0 = e[:-1] // BLOCK
            b1 = np.minimum(np.maximum(-(-e[1:] // BLOCK), b0 + 1), len(st.block_counts))
            share = (cs[b1] - cs[b0]) / ((b1 - b0) * BLOCK * x.shape[1])
            a = np.minimum(1, np.sqrt(share / 0.1))[:, None]
            rgb[Hw + 1:] = np.round(_rgb(CANVAS) + (_rgb(FLAG).astype(float) - _rgb(CANVAS)) * a).astype(np.uint8)
        self._put(cv, rgb)
        cv.delete("deco")
        x0, x1 = self._ovx(self.t0), self._ovx(self.t1)
        cv.create_rectangle(x0, 0, max(x1 - 1, x0 + 1), Hw, outline=FG, tags="deco")
        if 0 < self.done < n:
            cv.create_line(self._ovx(self.done), 0, self._ovx(self.done), H, fill=FG, tags="deco")
        for r in self.det["hot"] if show else ():
            xc = self._ovx((r[0] + r[1]) / 2)
            cv.create_polygon(xc - 4, 0, xc + 4, 0, xc, 7, fill=FLAG, tags="deco")

    def _draw_gain(self, W, H):
        """The gain lane, in dB: the unlimit gain of the whole song and the uncompress gain of the region's
        preview (R dashed unless Stereo-linked), faint at 100 % and solid at their Amount, which the
        Unlimit or Uncompress page drags up and down."""
        cv, app, cfg, wc = self.glane, self.app, self.app.cfg, W - GUT
        cv.delete("deco")
        self._gcurves = []
        if not cv.winfo_manager() or wc < 20 or H < 16:
            return
        e = np.linspace(self.t0, self.t1, wc + 1).astype(np.int64)
        notes = []
        if cfg["unlimit"]:
            G = app.unl(cfg)
            if G is None:
                notes.append(app.unl_text(cfg))
            else:  # the min and max of G per column: of 20 log10 G, as it is monotonic
                mn, mx = (20 * np.log10(v[:, 0].astype(np.float64))
                          for v in G[2].minmax(G[1][:, None], self.t0, self.t1, wc))
                self._gcurves.append(("unlimit_amount", UNL, 0, wc, mn, mx, cfg["unlimit_amount"] / 100, False))
        if cfg["uncompress"]:
            p = app.prev
            if p is None or p["ucg"] is None:
                notes.append("uncompress gain: not kept from the render" if self.region and app.full_b(cfg)
                             else "uncompress gain is shown in the region")
            else:
                c0, c1, mn, mx = self._colmm(p["ucg"], p["r"][0], e)
                for c in range(1 if cfg["uc_link"] else min(2, mn.shape[1])):
                    self._gcurves.append(("uc_amount", UC, c0, c1, mn[c0:c1, c], mx[c0:c1, c], cfg["uc_amount"] / 100,
                                          c == 1))
        v = [u for *_, mn, mx, a, _ in self._gcurves if len(mn)
             for u in (mn.min(), mx.max(), a * mn.min(), a * mx.max())]
        s = Scale(pad=4)  # -6..+12 dB, wider when needed
        s.lo, s.hi = min(-6.0, np.floor(min(v, default=0.0))), max(12.0, np.ceil(max(v, default=0.0)))
        rgb = np.empty((H, wc, 3), np.uint8)
        rgb[:] = _rgb(CANVAS)
        if self.region:
            c0, c1 = self._cols(e, self.region)
            rgb[:, c0:c1] = _rgb(REGION)
        dash = (np.arange(wc) // 4) % 2 == 0
        for faint in (True, False):
            for key, col, c0, c1, mn, mx, a, dashed in self._gcurves:
                if c1 > c0:
                    f = 1.0 if faint else a
                    m = _fill(f * mn, f * mx, H, s) & (dash[c0:c1] if dashed else True)
                    rgb[:, c0:c1][m] = _rgb(_mix(CANVAS, col, 0.4) if faint else col)
        self._put(cv, rgb)
        y = lambda d: float(s.y(d, H))
        cv.create_line(GUT, y(0), W, y(0), fill=LINE, tags="deco")
        for d, anchor in ((s.hi, "ne"), (0.0, "e"), (s.lo, "se")):
            if anchor != "e" or min(y(0), H - y(0)) >= 1.5 * self._ls:
                cv.create_text(GUT - 4, {"ne": 1, "se": H - 1}.get(anchor, y(0)), text=f"{d:+g}" if d else "0 dB",
                               anchor=anchor, fill=MUTED, font=FONT, tags="deco")
        names, gk = dict(unlimit_amount="unlimit", uc_amount="uncompress"), self._gkey()
        for i, (key, col, *_, a, _) in enumerate(dict((c[0], c) for c in self._gcurves).values()):
            t = cv.create_text(GUT + 4, 1 + self._ls * i, text=f"{names[key]} {100 * a:g} %", anchor="nw", fill=col,
                               font=FONT, tags="deco")
            if key == self.sel == gk:  # the keyboard-selected handle: outlined
                x0, y0, x1, y1 = cv.bbox(t)
                cv.create_rectangle(x0 - 2, y0, x1 + 2, y1, outline=ACCENT, width=2, tags="deco")
        if notes:
            cv.create_text(GUT + wc / 2, H / 2, text="   ·   ".join(notes), fill=MUTED, font=FONT, tags="deco")

    def _gkey(self):
        """The setting the gain lane drags on this page (its curve shown), or None."""
        k = {"Unlimit": "unlimit_amount", "Uncompress": "uc_amount"}.get(self.app.page)
        return k if any(c[0] == k for c in self._gcurves) and self.app.applies(k) else None

    def _g_press(self, e):
        k = self._gkey()
        if k is not None and not self.app.dragging and self.song is not None:
            self.app.dragging = k
            self._gd = [k, e.y, float(self.app.cfg[k])]

    def _g_move(self, e):
        """Up or down: Amount by 0.5 % per px (Shift: 0.05)."""
        if self._gd:
            k, y, v = self._gd
            st = 0.05 if e.state & 1 else 0.5
            lo, hi = self.app.ranges[k]
            self._gd = [k, e.y, min(max(v - (e.y - y) * st, lo), hi)]
            self.app._set(k, self._gd[2], st)

    def _g_release(self, e):
        if self._gd:
            self._gd = None
            self.app.drag_end()

    def _g_dbl(self, e):
        k = self._gkey()
        if k is not None:
            self.app._set(k, self.app.base[k])
            self.app._commit()

    def _g_menu(self, e):
        k = self._gkey()
        if k is not None and self._gd is None:
            m = tk.Menu(self, **MENU)
            m.add_command(label="Type value…", command=lambda: self.app.type_value(k))
            m.add_command(label="Reset to preset", command=lambda: self._g_dbl(e))
            m.tk_popup(e.x_root, e.y_root)

    def _g_motion(self, e):
        """Readout: the gains at the pointer, and how to drag the page's curve."""
        if self.song is None or self._gd is not None:
            return
        k, j = self._gkey(), e.x - GUT
        self.glane.config(cursor="sb_v_double_arrow" if k else "")
        text = [_tfmt(self._t(e.x) / self.song.sr)] if j >= 0 else []
        for key, col, c0, c1, mn, mx, a, dashed in self._gcurves:
            if c0 <= j < c1:
                name = "unlimit" if key == "unlimit_amount" else "uncompress" + (" R" if dashed else "")
                text.append(f"{name} {a * mx[j - c0]:+.2f} dB (100 %: {mx[j - c0]:+.2f})")
        if k:
            text.append("drag up or down: Amount (Shift = fine); double-click = preset")
        self.readout.config(text=" · ".join(text))

    def _put(self, cv, rgb):
        """Show rgb at the gutter's right edge: a full put when the size changed or most of it
        did, else one put per band of changed rows."""
        old, photo = self._img.get(cv), self.photo[cv]
        self._img[cv] = rgb
        if not cv.find_withtag("img"):
            cv.create_image(GUT, 0, image=photo, anchor="nw", tags="img")
        h, w = rgb.shape[:2]
        ppm = lambda a: b"P6 %d %d 255 " % (a.shape[1], a.shape[0]) + a.tobytes()
        if old is None or old.shape != rgb.shape:
            photo.configure(data=ppm(rgb), format="PPM")
            return
        ne = old != rgb
        diff = ne[..., 0] | ne[..., 1] | ne[..., 2]  # 4x faster than .any(2)
        rows = np.flatnonzero(diff.any(1))
        if not rows.size:
            return
        bands = []
        for r in np.split(rows, np.flatnonzero(np.diff(rows) > 8) + 1):
            c = np.flatnonzero(diff[r[0]:r[-1] + 1].any(0))
            bands.append((r[0], r[-1] + 1, c[0], c[-1] + 1))
        if sum((r1 - r0) * (c1 - c0) for r0, r1, c0, c1 in bands) > 0.6 * h * w:
            photo.configure(data=ppm(rgb), format="PPM")
            return
        for r0, r1, c0, c1 in bands:
            self.tk.call(photo, "put", ppm(rgb[r0:r1, c0:c1]), "-format", "ppm", "-to", c0, r0)


class HistPanel(tk.Canvas):
    """Manual detection: the nines histogram, counts (log scale, channels summed) of the level relative
    to the measured peak, on u = -log10(1 - |y0| / peaks(t)); the bin at the right holds the samples at
    or above the peak. Bars red where samples are rebuilt, amber in the Soft band. NMF auto detection:
    the level in dBFS (show_auto)."""
    W, H, X0, X1, PK, Y0, Y1 = 280, 120, 6, 244, 252, 16, 100  # plot x, the peak bin from PK, plot y
    TICKS = (50, 80, 90, 95, 99, 99.9, 99.99)

    def __init__(self, parent, app):
        super().__init__(parent, width=self.W, height=self.H, background=CANVAS, highlightthickness=0,
                         borderwidth=0)
        self.photo = tk.PhotoImage(master=self, width=self.W, height=self.H)
        self.create_image(0, 0, image=self.photo, anchor="nw")
        self.app, self.cols, self.marks, self._key = app, None, [], None  # marks: (x, setting) of the markers
        self.dbr, self.title = None, ""  # auto: (lo dB, hi dB, the highest ceiling) of the x axis
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda e: self.itemconfig("title", text=self.title))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-Button-1>", self._dbl)

    def _press(self, e):
        """Click or drag: the nearest marker goes to the pointer."""
        if self.cols is not None and self.marks:
            self._key = self.app.dragging = min(self.marks, key=lambda m: abs(m[0] - e.x))[1]
            self._drag(e)

    def _drag(self, e):
        if not self._key:
            return
        if self.dbr:  # the knee, relative to the highest ceiling; 0.995 or more: 1 (ceilings only)
            v = 10 ** (self._db(e.x) / 20) / self.dbr[2]
            self.app._set(self._key, 1.0 if v >= 0.995 else min(max(v, 0.5), 1.0), 0.001 if e.state & 1 else 0.01)
        else:
            r = 100.0 if e.x >= self.PK - 4 else 100 * float(self._r(e.x))
            self.app._set(self._key, min(max(r, 1.0), 100.0), 0.01 if e.state & 1 else 0.1)

    def _dbl(self, e):
        """Double-click: the knee as found (auto); a click otherwise."""
        if self.dbr and self.marks:
            self.app._set("nmf_knee", 0)
            self.app._commit()
        else:
            self._press(e)

    def _release(self, e):
        if self._key:
            self._key = None
            self.app.drag_end()

    def _db(self, x):
        lo, hi, _ = self.dbr
        return lo + (x - self.X0) / (self.X1 - self.X0) * (hi - lo)

    def _xdb(self, db):
        lo, hi, _ = self.dbr
        return self.X0 + (db - lo) / (hi - lo) * (self.X1 - self.X0)

    def _x(self, r):
        lo, hi, _ = NINES
        return self.X0 + (-np.log10(1 - r) - lo) / (hi - lo) * (self.X1 - self.X0)

    def _r(self, x):
        lo, hi, _ = NINES
        return 1 - 10 ** -(lo + (x - self.X0) / (self.X1 - self.X0) * (hi - lo))

    def show_text(self, msg):
        self.cols, self.marks, self.dbr = None, [], None
        self.delete("deco")
        self.photo.blank()
        self.create_text(self.W / 2, self.H / 2, text=msg, fill=MUTED, font=FONT, tags="deco")

    def _bars(self, col, colour, extra=0):
        """The bars of col (one count per plot column, log heights over max(col, extra)) in colour
        (per column); returns the raster and the height function."""
        top = np.log10(max(col.max(), extra, 1) + 1)
        hgt = lambda c: np.round(np.log10(np.asarray(c) + 1) / top * (self.Y1 - self.Y0))
        rows = np.arange(self.H)[:, None]
        rgb = np.empty((self.H, self.W, 3), np.uint8)
        rgb[:] = _rgb(CANVAS)
        bar = (rows >= self.Y1 - hgt(col)) & (rows < self.Y1)
        rgb[:, self.X0:self.X1][bar] = np.broadcast_to(colour, (self.H, len(col), 3))[bar]
        return rgb, hgt

    def _photo(self, rgb):
        rgb[self.Y1, self.X0:self.W - 6] = _rgb(LINE)
        self.photo.configure(data=b"P6 %d %d 255 " % (self.W, self.H) + rgb.tobytes(), format="PPM")
        self.delete("deco")

    def show_auto(self, red, amber, grey, lo, ceils, knee, tmax):
        """NMF auto detection: Song.ceilings' level histogram summed over the sides, per HIST_DB dB
        from lo, split into the counts each side rebuilds above its ceiling (red), between its knee and
        its ceiling (amber) and not (grey): each bar is coloured in those shares. ceils: the ceilings'
        levels; knee: the knee marker's level; tmax: the highest ceiling (the marker's reference)."""
        nc, n = self.X1 - self.X0, len(red)
        self.dbr, self.title = (lo, lo + n * HIST_DB, tmax), "level dBFS (all sides)"
        cr, ca, cg = (np.add.reduceat(v, (np.arange(nc) * n / nc).astype(int)) for v in (red, amber, grey))
        col = cr + ca + cg
        self.cols = col, 0, 0.0
        rgb, hgt = self._bars(col, np.broadcast_to(_rgb(WAVE_IN), (nc, 3)))
        h, rows = hgt(col), np.arange(self.H)[:, None]
        top = self.Y1 - h
        fr = np.round(h * cr / np.maximum(col, 1))  # the top of each bar: red, then amber
        fa = np.round(h * (cr + ca) / np.maximum(col, 1))
        plot = rgb[:, self.X0:self.X1]
        plot[(rows >= top + fr) & (rows < top + fa)] = _rgb(SOFT)
        plot[(rows >= top) & (rows < top + fr)] = _rgb(FLAG)
        self._photo(rgb)
        for t in (-9, -6, -3, 0):
            x = self._xdb(20 * np.log10(tmax) + t)
            if x < self.X0 - 1:  # under the axis' start (a side without a ceiling peaks higher)
                continue
            self.create_line(x, self.Y1, x, self.Y1 + 3, fill=MUTED, tags="deco")
            self.create_text(x, self.Y1 + 11, text=f"{20 * np.log10(tmax) + t:.1f}", fill=MUTED, font=FONT,
                             anchor="w" if x < 20 else "center", tags="deco")  # the first one starts at its tick
        for th in ceils:
            x = self._xdb(20 * np.log10(th))
            self.create_line(x, self.Y0 - 2, x, self.Y0 + 5, fill=CEIL, width=2, tags="deco")
        self.marks = []
        if knee == knee:
            x = min(max(self._xdb(20 * np.log10(knee)), self.X0), self.X1)
            self.create_line(x, self.Y0 - 2, x, self.Y1, fill=THRESH, width=2, dash=(4, 2), tags="deco")
            self.marks.append((x, "nmf_knee"))
        self.create_text(4, 2, text=self.title, anchor="nw", fill=MUTED, font=FONT, tags=("deco", "title"))

    def show_manual(self, counts, h, low, soft, ignore):
        """counts: Song.hist9 summed over channels; h: Clip above (0-1); low: Not-clip below when it
        is used, else None; soft: Soft-clip detection; ignore: Peak ignore."""
        bins, nc = NINES[2], self.X1 - self.X0
        col = np.add.reduceat(counts[:bins], (np.arange(nc) * bins / nc).astype(int))
        self.cols, self.dbr, self.title = (col, int(counts[bins]), ignore), None, "level vs local peak"
        eff = min(h, low) if low is not None else h
        r = self._r(self.X0 + np.arange(nc) + 0.5)
        colour = np.where((r >= eff)[:, None], _rgb(FLAG),
                          np.where((soft & (r >= 0.92))[:, None], _rgb(SOFT), _rgb(WAVE_IN)))
        rgb, hgt = self._bars(col, colour, counts[bins])
        rows = np.arange(self.H)[:, None]
        hatch = ((rows >= self.Y1 - hgt(counts[bins])) & (rows < self.Y1)
                 & ((rows + np.arange(self.W - 6 - self.PK)) % 3 > 0))
        rgb[:, self.PK:self.W - 6][hatch] = _rgb(FLAG)
        self._photo(rgb)
        for p in self.TICKS:
            x = float(self._x(p / 100))
            self.create_line(x, self.Y1, x, self.Y1 + 3, fill=MUTED, tags="deco")
            self.create_text(x, self.Y1 + 11, text=f"{p:g}", fill=MUTED, font=FONT, tags="deco")
        self.create_text((self.PK + self.W - 6) / 2, self.Y1 + 11, text="≥pk", fill=MUTED, font=FONT, tags="deco")
        self.marks = []
        for v, dash, key in ((h, None, "clip_above"), (low, (4, 2), "not_clip_below")):
            if v is not None:
                x = max(self.X0, float(self._x(v))) if v < 1 else self.PK - 2
                self.create_line(x, self.Y0 - 2, x, self.Y1, fill=THRESH, width=2 if dash is None else 1, dash=dash,
                                 tags="deco")
                self.marks.append((x, key))
        self.create_text(4, 2, text="level vs local peak", anchor="nw", fill=MUTED, font=FONT, tags=("deco", "title"))

    def _motion(self, e):
        if self.cols is None:
            return
        col, top, ignore = self.cols
        j = e.x - self.X0
        if 0 <= j < len(col) and self.dbr:
            text = f"{self._db(e.x + 0.5):.2f} dBFS: {int(col[j]):,} samples"
        elif 0 <= j < len(col):
            r = 100 * self._r(e.x + 0.5)
            text = f"{r:.{max(1, int(-np.log10(100 - r)) + 2)}f} %: {int(col[j]):,} samples"
        elif e.x >= self.PK and not self.dbr:
            text = f"≥ peak: {top:,} samples, ignored by Peak ignore {100 * ignore:g} %"
        else:
            text = self.title
        self.itemconfig("title", text=text)


class CurvePanel(tk.Canvas):
    """Uncompress, in dB, x the stage input (the level heard) and y the level out: the compressor
    (original to heard, MUTED), its full inverse (heard to original, dashed) and the inverse at Amount
    (solid), over a histogram of the region's stage-input levels. Drag the knee dot sideways
    (Threshold), the ratio dot (Ratio) or the solid curve (Amount) up and down."""
    W, H, X0, X1, Y0, Y1 = 272, 180, 30, 266, 16, 160  # plot x, plot y
    IN, OUT = (-48.0, 6.0), (-48.0, 18.0)  # dB at X0..X1, Y1..Y0
    STEP = dict(uc_threshold=(0.5, 0.05), uc_ratio=(0.1, 0.01), uc_amount=(1.0, 0.1))  # drag step, fine step
    HELP = dict(uc_threshold="Threshold: drag the knee sideways", uc_ratio="Ratio: drag up or down",
                uc_amount="Amount: drag the solid curve up or down")
    TITLE = "transfer dB: heard (in) → out"

    def __init__(self, parent, app):
        super().__init__(parent, width=self.W, height=self.H, background=CANVAS, highlightthickness=0,
                         borderwidth=0)
        self.app, self.p, self.lv, self._d = app, None, None, None  # p: (T, R, M, a); lv: sorted levels
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda e: self._d or (self.config(cursor=""), self.itemconfig("title", text=self.TITLE)))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-Button-1>", lambda e: self._reset(self._hit(e.x, e.y)))
        self.bind("<Button-3>", self._menu)
        for ev, d in (("MouseWheel", 0), ("Button-4", 1), ("Button-5", -1)):  # Button-4/5: X11
            self.bind(f"<{ev}>", lambda e, d=d: self._wheel(e, False, d))
            self.bind(f"<Shift-{ev}>", lambda e, d=d: self._wheel(e, True, d))

    def _x(self, d):
        return self.X0 + (np.asarray(d) - self.IN[0]) / (self.IN[1] - self.IN[0]) * (self.X1 - self.X0)

    def _y(self, d):
        return self.Y1 - (np.asarray(d) - self.OUT[0]) / (self.OUT[1] - self.OUT[0]) * (self.Y1 - self.Y0)

    def _xd(self, x):
        return self.IN[0] + (x - self.X0) / (self.X1 - self.X0) * (self.IN[1] - self.IN[0])

    def _yd(self, y):
        return self.OUT[0] + (self.Y1 - y) / (self.Y1 - self.Y0) * (self.OUT[1] - self.OUT[0])

    def inv(self, y):
        """The full inverse: the original level of heard level y."""
        T, R, M = self.p[:3]
        y = np.asarray(y, dtype=np.float64)
        return np.where(y < T + M, y - M, T + R * (y - M - T))

    def _xr(self):
        """The ratio dot's level in: +6 dB, or where the full inverse leaves the top; None: no room."""
        T, R, M = self.p[:3]
        xr = min(self.IN[1], T + M + (self.OUT[1] - T) / R)
        return xr if xr > max(T + M, self.IN[0]) + 1 else None

    def pos(self, key):
        """(x, y) px of key's handle (the amount's: halfway from the knee to the ratio dot), or None."""
        T, R, M, a = self.p
        if key == "uc_threshold":
            ok = self.IN[0] <= T + M <= self.IN[1] and self.OUT[0] <= T <= self.OUT[1]
            return (float(self._x(T + M)), float(self._y(T))) if ok else None
        xr = self._xr()
        if xr is None:
            return None
        u = xr if key == "uc_ratio" else (max(T + M, self.IN[0]) + xr) / 2
        return float(self._x(u)), float(self._y(u + (1 if key == "uc_ratio" else a) * (float(self.inv(u)) - u)))

    def value_at(self, key, x, y):
        """The value of key that puts its handle at (x, y): the threshold from x, the ratio and the
        amount (%) from y on the curve at x."""
        T, R, M, a = self.p
        u = self._xd(x)
        if key == "uc_threshold":
            return u - M
        if key == "uc_ratio":
            return (self._yd(y) - T) / max(u - M - T, 1e-6)
        d = float(self.inv(u)) - u
        return 100 * (self._yd(y) - u) / d if abs(d) > 1e-6 else 100 * a

    def show(self, T, R, M, a, lv):
        """Threshold T, ratio R, makeup M (dB), amount a (0-1); lv: the region's stage-input levels in dB,
        sorted, or None."""
        self.p, self.lv = (float(T), max(float(R), 1.0), float(M), float(a)), lv
        self.delete("all")
        X0, X1, Y0, Y1 = self.X0, self.X1, self.Y0, self.Y1
        for d in (-48, -36, -24, -12, 0):
            x = float(self._x(d))
            self.create_line(x, Y1, x, Y1 + 3, fill=MUTED)
            self.create_text(x, Y1 + 11, text=str(d), fill=MUTED, font=FONT)
        for d in (-36, -24, -12, 0, 12):
            y = float(self._y(d))
            self.create_line(X0 - 3, y, X0, y, fill=MUTED)
            self.create_text(X0 - 5, y, text=f"{d:+d}" if d else "0", anchor="e", fill=MUTED, font=FONT)
        self.create_line(X0, Y1, X1, Y1, fill=LINE)
        self.create_line(X0, Y0, X0, Y1, fill=LINE)
        if lv is not None and len(lv):  # the region's levels, a log count per px along the bottom
            cnt = np.diff(np.searchsorted(lv, self._xd(np.arange(X0, X1 + 1))))
            h = np.log10(cnt + 1) / np.log10(cnt.max() + 1) * 30 if cnt.max() else 0 * cnt
            pts = np.c_[np.arange(X0, X1) + 0.5, Y1 - h].ravel().tolist()
            self.create_polygon([X0, Y1] + pts + [X1, Y1], fill=_mix(CANVAS, UC, 0.3), outline="")
        T, R, M, a = self.p
        xs = np.unique(np.r_[np.linspace(*self.IN, 109), np.clip([T, T + M], *self.IN)])  # the knees exactly

        def line(ys, **kw):  # rising curves: cut where they leave the top
            k = int(np.argmax(ys > self.OUT[1])) if ys[-1] > self.OUT[1] else len(ys)
            x, y = xs[:k], ys[:k]
            if 0 < k < len(ys):
                x = np.r_[x, x[-1] + (self.OUT[1] - y[-1]) / (ys[k] - y[-1]) * (xs[k] - x[-1])]
                y = np.r_[y, self.OUT[1]]
            if len(x) > 1:
                self.create_line(np.c_[self._x(x), np.minimum(self._y(y), Y1)].ravel().tolist(), **kw)
        full = self.inv(xs)
        line(xs, fill=LINE, dash=(2, 2))  # identity
        line(np.where(xs < T, xs + M, T + (xs - T) / R + M), fill=MUTED, width=1.5)  # the compressor
        line(full, fill=UC, dash=(8, 3))
        line(xs + a * (full - xs), fill=UC, width=2)
        sel = self.app.wv.sel
        for key in ("uc_threshold", "uc_ratio", "uc_amount") if self.app.applies("uc_threshold") else ():
            p = self.pos(key)
            if p is not None and (key != "uc_amount" or sel == key):  # the selected handle: outlined
                self.create_oval(p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4, fill=UC, outline=ACCENT if sel == key else FG,
                                 width=2 if sel == key else 1)
        self.create_text(4, 2, text=self.TITLE, anchor="nw", fill=MUTED, font=FONT, tags="title")

    def _hit(self, x, y):
        """The handle under the pointer: a dot within 6 px, else the solid curve within 5 px past the knee."""
        if self.p is None or not self.app.applies("uc_threshold"):
            return None
        for key in ("uc_threshold", "uc_ratio"):
            p = self.pos(key)
            if p is not None and abs(x - p[0]) <= 6 and abs(y - p[1]) <= 6:
                return key
        T, R, M, a = self.p
        u = self._xd(x)
        on = self.X0 <= x <= self.X1 and u > T + M + 0.5
        return "uc_amount" if on and abs(y - float(self._y(u + a * (float(self.inv(u)) - u)))) <= 5 else None

    def _press(self, e):
        k = self._hit(e.x, e.y)
        if k and not self.app.dragging:
            self.app.dragging = k
            v = float(self.app.cfg[k])  # absolute from the grab point; the ratio and amount at the level pressed
            self._d = dict(k=k, x=e.x, y=e.y, x0=e.x, v=v, off=v - self.value_at(k, e.x, e.y))

    def _drag(self, e):
        d = self._d
        if not d:
            return
        k = d["k"]
        st, fine = self.STEP[k]
        if e.state & 1:  # relative fine
            v = d["v"] + ((e.x - d["x"]) if k == "uc_threshold" else (d["y"] - e.y)) * fine
        else:
            v = self.value_at(k, e.x if k == "uc_threshold" else d["x0"], e.y) + d["off"]
        lo, hi = self.app.ranges[k]
        d.update(x=e.x, y=e.y, v=min(max(v, lo), hi))
        self.app._set(k, d["v"], fine if e.state & 1 else st)

    def _release(self, e):
        if self._d:
            self._d = None
            self.app.drag_end()

    def _reset(self, k):
        """Handle k (double-click or menu) back to the preset's value."""
        if k:
            self.app._set(k, self.app.base[k])
            self.app._commit()

    def _menu(self, e):
        k = self._hit(e.x, e.y)
        if k and not self._d:
            m = tk.Menu(self, **MENU)
            m.add_command(label="Type value…", command=lambda: self.app.type_value(k))
            m.add_command(label="Reset to preset", command=lambda: self._reset(k))
            m.tk_popup(e.x_root, e.y_root)

    def _wheel(self, e, fine, d=0):
        """Over a handle: one drag step (Shift: fine) up or down."""
        k = self._hit(e.x, e.y)
        if k and not self._d:
            self.app.wv.nudge(setting_handle(self.app, k), d or int(np.sign(e.delta)), fine)

    def _motion(self, e):
        if self.p is None or self._d:
            return
        k = self._hit(e.x, e.y)
        self.config(cursor="" if not k else "sb_h_double_arrow" if k == "uc_threshold" else "sb_v_double_arrow")
        if k:
            text = self.HELP[k] + "; Shift = fine"
        elif self.X0 <= e.x <= self.X1:
            u, a = self._xd(e.x), self.p[3]
            X = float(self.inv(u))
            text = f"in {u:.1f} → {X:+.1f} dB (100 %), {u + a * (X - u):+.1f} dB ({100 * a:.0f} %)"
        else:
            text = self.TITLE
        self.itemconfig("title", text=text)
