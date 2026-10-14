# SPDX-FileCopyrightText: 2026 thatrandomalt
#
# SPDX-License-Identifier: MIT

import sys, os, copy, json, threading, queue, time, traceback
from collections import OrderedDict
import numpy as np
import soundfile as sf
from sharpunclip import (process, DEFAULTS, PRESETS, NMF_QUALITIES, settings_from_cfg, spade_backend, backend_name,
                         _write_log)
import session
from session import Pyramid, Song

# Inspector pages, one per stage chip: (title, on/off setting or None, columns, items).
# An item is (key, label, min, max, step) for a number, (key, label) for a checkbox,
# (key, label, choices) for a drop-down or (key, label, choices, "radio") for radio buttons
# whose values are "0", "1", ...
PAGES = (
    ("Declip", "declip", 1, (("engine", "Engine", ("NMF", "SPADE")),
                             ("nmf_quality", "NMF quality", NMF_QUALITIES),
                             ("auto_detect", "Auto detect"),
                             ("nmf_knee", "NMF knee", 0, 1, 0.01),
                             ("clip_above", "Clip above %", 1, 100, 0.1),
                             ("max_restore", "Max restore ×", 0, 20, 0.5),
                             ("speed_quality", "SPADE quality 0-6", 0, 6, 1),
                             ("lookahead", "Frame", ("512", "1024", "2048"), "radio"),
                             ("linked_stereo", "Linked stereo"))),
    ("Unlimit", "unlimit", 1, (("unlimit_amount", "Amount %", 0, 300, 10),)),
    ("Uncompress", "uncompress", 1, (("uc_threshold", "Threshold dB", -60, 24, 0.5),
                                     ("uc_ratio", "Ratio", 1, 20, 0.5),
                                     ("uc_attack", "Attack ms", 0.1, 500, 1),
                                     ("uc_release", "Release ms", 1, 5000, 10),
                                     ("uc_makeup", "Makeup dB", -24, 24, 0.5),
                                     ("uc_amount", "Amount %", 0, 100, 10),
                                     ("uc_link", "Stereo-linked"))),
    ("Output", None, 1, (("input_gain", "Input gain dB", -24, 24, 0.5),
                         ("output_gain", "Output gain dB", -24, 24, 0.5),
                         ("limiter", "Hard-clip the output"),
                         ("limiter_peak", "Ceiling dB", -24, 6, 0.5))),
)
# Advanced window: groups of (title, on/off setting that greys it, subsections); a subsection is
# (title, columns, items) like PAGES.
ADVANCED = (
    ("Declip", "declip", (
        ("Detection", 1, (("not_clip_below", "Not-clip below %", 1, 100, 0.1),
                          ("not_clip_amp", "Not-clip level dB", -120, 0, 1),
                          ("dynamic", "Dynamic threshold"),
                          ("remove_glitch", "Remove glitches"),
                          ("soft", "Soft-clip detection"))),
        ("Peak measurement", 1, (("peak_ignore", "Peak ignore", 0, 0.5, 0.001),
                                 ("peak_time", "Peak time s", 0.01, 10, 0.1),
                                 ("reset_amp", "Reset level dB", -120, 0, 1),
                                 ("reset_time", "Reset time s", 0.01, 10, 0.1))),
        ("Different clips", 1, (("diff_clips", "Measure each song separately"),
                                ("trans_amp", "Gap level dB", -120, 0, 1),
                                ("trans_time", "Gap length s", 0.01, 10, 0.1))),
        ("SPADE frames", 1, (("lookahead", "Lookahead 0-2", 0, 2, 1),)),
    )),
    ("Output", None, (
        ("Levels", 1, (("input_gain", "Input gain dB", -24, 24, 0.5),
                       ("output_gain", "Output gain dB", -24, 24, 0.5))),
        ("Limiter", 1, (("limiter", "Hard-clip the output"),
                        ("limiter_peak", "Ceiling dB", -24, 6, 0.5))),
        ("Performance", 1, (("threads", "CPU threads", 1, 256, 1),)),
    )),
)
INTS = {"threads", "speed_quality", "lookahead", "process_rate"}
INIT = dict(DEFAULTS, threads=os.cpu_count() or 1)  # initial field values
# Settings used by only one engine or detection mode; greyed out otherwise. g(key) = current value.
_spade = lambda g: g("engine") == "SPADE"
_manual = lambda g: g("engine") == "SPADE" or not g("auto_detect")
_sections = lambda g: _manual(g) and g("diff_clips")
APPLIES = dict({k: _spade for k in ("speed_quality", "linked_stereo", "lookahead")},
               nmf_quality=lambda g: g("engine") == "NMF", auto_detect=lambda g: g("engine") == "NMF",
               nmf_knee=lambda g: g("engine") == "NMF" and g("auto_detect"),
               peak_time=lambda g: _manual(g) and not g("diff_clips"), trans_amp=_sections, trans_time=_sections,
               reset_time=lambda g: False,  # no effect: a block quieter than Reset level gets it as its peak anyway
               not_clip_below=lambda g: _manual(g) and not g("dynamic"),  # Dynamic flags above Clip above only
               limiter_peak=lambda g: g("limiter"),
               **{k: _manual for k in ("clip_above", "not_clip_amp", "dynamic",
                                       "remove_glitch", "soft", "peak_ignore", "reset_amp", "diff_clips")})
LIGHT_KEYS = session.FLAG_KEYS | {"declip", "engine", "auto_detect"}  # changes that redo manual detection
DET_KINDS = ("steps", "flags", "stats", "hist9")  # the light worker's manual-detection jobs
AUTO_KEYS = session.G_KEYS | {"declip", "engine", "auto_detect", "nmf_knee", "gpu"}  # ... NMF auto detection
AUTO_KINDS = ("ceil", "aflags")  # and its jobs
KEYS = (("Space", "Play / stop (loops the region)"), ("1  2  3,  X", "Hear Orig / Proc / Diff,  swap Orig and Proc"),
        ("S, M", "Side (L−R) / level match on or off"), ("P", "Preview at render quality"),
        ("Esc", "Stop audio or preview, clear region, deselect"),
        ("[  ]", "Region back / forward by its length"),
        ("N, Shift+N", "Next / previous most-clipped spot"), ("V", "Lin / dB"),
        ("F, Z", "Whole song / zoom to the region"), ("+  −", "Zoom in / out"),
        ("Left, Right", "Scroll (no line selected)"), ("Home, End", "Start / end of the song"),
        ("Tab, Shift+Tab", "Select a line on the page"),
        ("Up, Down", "Nudge it (Shift: fine)"), ("Return", "Type its value"), ("BackSpace", "Back to the preset"),
        ("Ctrl+1..4", "Declip / Unlimit / Uncompress / Output page"), ("Ctrl+Z, Ctrl+Y", "Undo, redo"),
        ("Ctrl+O", "Open a song"), ("Ctrl+S, Ctrl+Shift+O", "Save / load settings"),
        ("Ctrl+E", "Advanced settings"), ("Ctrl+Return", "Render"), ("F1, ?", "This list"))  # the F1 overlay

def _backend(cfg):
    """(array module, device) for cfg's GPU setting, with PyTorch's CPU threads set: renders and
    previews run alike."""
    xp, dev = spade_backend(cfg.get("gpu"))
    try:
        import torch
        torch.set_num_threads(max(1, int(cfg.get("threads", 1))))
    except Exception:
        pass
    return xp, dev


def compute(x, sr, cfg, progress=None, hints=None, info=None):
    xp, dev = _backend(cfg)
    return process(x, sr, settings_from_cfg(cfg), xp=xp, dev=dev, progress=progress, info=info, hints=hints)


def _tk():
    """tkinter and the waveform view; imported only when a window is made (selfcheck runs headless)."""
    global tk, ttk, filedialog, WaveView, HistPanel, CurvePanel, BG, PANEL, LINE, FG, MUTED, ACCENT, FLAG, WARN
    global WARN_BG, SOFT, THRESH, _tfmt
    import tkinter as tk
    from tkinter import ttk, filedialog
    from waveview import (WaveView, HistPanel, CurvePanel, BG, PANEL, LINE, FG, MUTED, ACCENT, FLAG, WARN, WARN_BG,
                          SOFT, THRESH, _tfmt)


def mmss(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def _secs(s):
    return f"{s:.1f} s" if s < 10 else f"{s:.0f} s" if s < 60 else mmss(s)


def _covers(k, key):
    """Stage-A result k serves key: the same settings and fast, and its span holds key's (the slice is as
    exact: SPADE spans share the frame grid)."""
    return k[2:] == key[2:] and k[0] <= key[0] and key[1] <= k[1]


class _Cancelled(Exception):
    pass


class App:
    def __init__(self, root):
        self.root = root
        self.V, self.T, self.C, self.labels, self.ranges = {}, {}, {}, {}, {}
        self.grey, self.bound, self.stage_of = [], {}, {}  # (key, widget) greyed by _sync; key -> widgets; key -> toggle
        self._dirty, self._pending = set(), None
        self.base, self.base_name = dict(DEFAULTS), None  # what "modified" compares with
        self.x, self.cfg, self._err = None, None, False  # cfg: the last valid settings
        # the setting being dragged (no jobs until release) and what changed meanwhile; undo snapshots
        self.dragging, self._drag_dirty, self.committed, self.undo_stack, self.redo_stack = None, set(), None, [], []
        self._det_cfg = None  # the settings of the latest manual-detection jobs
        self._mode, self._shp = "?", None  # the detection shown (see _det_mode); knee search progress
        self.sr, self.path = 0, None
        self.q = queue.Queue()
        self.light, self.gen, self.region_user_set = session.Worker(self.q), 0, False
        # region previews: the heavy worker (shares the GPU with renders), stage-A results (LRU), the
        # unlimit and uncompress gains, the B shown, the B-key of the render drawn and its finished output
        self.heavy, self.est, self.lru = session.Worker(self.q, session.GPU_LOCK), session.Estimates(), OrderedDict()
        self.G = self.ucg = self.prev = self.rkey = self.full = self._pjob = self._stopped = self._wait = None
        self._jobs, self._gp, self.sgen, self.n_a, self.rendering = {}, 0.0, 0, 0, False  # _jobs: kind -> key
        self.est_lab = {}  # "~1:50 render" next to the quality fields
        # the Unlimit, Uncompress and Output pages' text lines; the region's stage-input levels (key, sorted dB);
        # what the limiter cuts of the region's preview (share, up to dB) and of the render (share)
        self.stage_txt, self._lv, self.cut, self.full_cut = {}, None, None, None
        # audition: the player, what it plays ("loop" the region, "song" from the cursor), the key of the
        # loop buffers, the Auto trim kept and what it was measured on, the Mon mode, the playhead (or
        # stream idle) timer
        self.player, self._ctx, self._abuf, self._hold, self._hold_key = session.Player(48000), None, None, None, None
        self.mon_i, self._mon_db, self._mdb, self._ph, self.full_pk, self._song0 = 0, None, None, None, 0.0, 0
        self.cancel = threading.Event()
        self._style()
        self._build()
        self.root.after(50, self._poll)

    def _style(self):
        r = self.root
        r.title("SharpUnClip")
        r.configure(background=BG)
        s = ttk.Style(r)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=PANEL, bordercolor=LINE,
                    lightcolor=BG, darkcolor=BG, troughcolor=PANEL, arrowcolor=FG,
                    insertcolor=FG, selectbackground=LINE, selectforeground=FG)
        s.configure("TButton", background=PANEL, padding=(10, 4))
        s.map("TButton", background=[("disabled", BG), ("active", LINE)],
              foreground=[("disabled", MUTED)])
        s.configure("Accent.TButton", background=ACCENT, foreground="#0d1310")
        s.map("Accent.TButton", background=[("disabled", PANEL), ("active", "#5ee3ad")])
        s.configure("Muted.TLabel", foreground=MUTED)
        s.map("Muted.TLabel", foreground=[("disabled", "#555b66")], background=[("disabled", BG)])
        s.map("TCheckbutton", foreground=[("disabled", "#555b66")])
        s.configure("Head.TLabel", foreground=MUTED, font=("Segoe UI", 9, "bold"))
        s.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("disabled", "#555b66"), ("readonly", FG)],
              selectbackground=[("readonly", PANEL)], selectforeground=[("readonly", FG)])
        s.map("TCheckbutton", background=[("active", BG)],
              indicatorbackground=[("selected", ACCENT), ("!selected", PANEL)])
        s.configure("TCheckbutton", indicatorbackground=PANEL, indicatorforeground=BG)
        s.configure("TRadiobutton", indicatorbackground=PANEL, indicatorforeground=BG)
        s.map("TRadiobutton", background=[("active", BG)], foreground=[("disabled", "#555b66")],
              indicatorbackground=[("selected", ACCENT), ("!selected", PANEL)])
        s.configure("Warn.TLabel", foreground=WARN)
        s.configure("Chip.TCheckbutton", background=PANEL)
        s.map("Chip.TCheckbutton", background=[("active", PANEL)])
        s.map("Toolbutton", background=[("selected", LINE), ("active", PANEL)], foreground=[("disabled", "#555b66")])
        s.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=PANEL,
                    bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT, thickness=8)
        r.option_add("*TCombobox*Listbox.background", PANEL)
        r.option_add("*TCombobox*Listbox.foreground", FG)
        r.option_add("*TCombobox*Listbox.selectBackground", LINE)

    def _var(self, key, kind):
        """The one Tk variable of a setting (kind "V" number, "T" on/off, "C" choice). Every widget of
        the setting binds it, and every write schedules _apply_changes."""
        d = getattr(self, kind)
        if key not in d:
            d[key] = tk.IntVar(value=int(INIT[key])) if kind == "T" else tk.StringVar(value=str(INIT[key]))
            d[key].trace_add("write", lambda *a, k=key: self._changed(k))
        return d[key]

    def _field(self, key, w):
        self.bound.setdefault(key, []).append(w)
        self.grey.append((key, w))

    def _section(self, parent, title, toggle, show_toggle, cols, items):
        """A titled box of settings. With an on/off toggle, its fields grey out when off."""
        head = None
        if toggle and show_toggle:
            head = ttk.Checkbutton(parent, text=title, variable=self._var(toggle, "T"), command=self._commit,
                                   takefocus=0)  # Space stays Play
            self.bound.setdefault(toggle, []).append(head)
        box = ttk.LabelFrame(parent, text=title, labelwidget=head, padding=(10, 4, 10, 8))
        for i, it in enumerate(items):
            r, c = i // cols, 2 * (i % cols)
            key, label = it[:2]
            self.stage_of[key] = toggle
            if len(it) == 2:  # checkbox
                cb = ttk.Checkbutton(box, text=label, variable=self._var(key, "T"), command=self._commit, takefocus=0)
                cb.grid(row=r, column=c, columnspan=2, sticky="w", pady=2)
                self._field(key, cb)
                continue
            lab = ttk.Label(box, text=label, style="Muted.TLabel")
            radio = len(it) == 4  # its label sits by the buttons, over the little-context row
            lab.grid(row=r, column=c, sticky="nw" if radio else "w", padx=(0, 6), pady=(4, 2) if radio else 2)
            self.grey.append((key, lab))
            if radio:  # the Frame ones show each width's little context under them
                w = ttk.Frame(box)
                for v, text in enumerate(it[2]):
                    rb = ttk.Radiobutton(w, text=text, value=str(v), variable=self._var(key, "V"),
                                         command=self._commit, takefocus=0)
                    rb.grid(row=0, column=v, sticky="w", padx=(0, 6))
                    self._field(key, rb)
                    if key == "lookahead":  # hovering one shows its risk strip in the lanes
                        rb.bind("<Enter>", lambda e, v=v: self._frame_hover(v))
                        rb.bind("<Leave>", lambda e: self._frame_hover(None))
                if key == "lookahead":
                    self.ctx = [ttk.Label(w, text="", style="Muted.TLabel") for _ in it[2]]
                    for v, lab in enumerate(self.ctx):
                        lab.grid(row=1, column=v, sticky="w")
            elif len(it) == 3:  # drop-down
                w = ttk.Combobox(box, textvariable=self._var(key, "C"), values=it[2],
                                 width=max(8, max(map(len, it[2]))), state="readonly")
                w.bind("<<ComboboxSelected>>", lambda e: self._commit())
                self._field(key, w)
            else:
                lo, hi, step = it[2:]
                self.labels[key], self.ranges[key] = label, (lo, hi)
                w = self._spin(box, key, from_=lo, to=hi, increment=step)
            est = key in ("nmf_quality", "speed_quality")  # followed by the render time estimate
            w.grid(row=r, column=c + 1, columnspan=2 if radio else 1, sticky="w", padx=(0, 6 if est else 14), pady=2)
            if est:
                self.est_lab[key] = ttk.Label(box, text="", style="Muted.TLabel")
                self.est_lab[key].grid(row=r, column=c + 2, sticky="w")
        return box

    def _spin(self, parent, key, width=8, **kw):
        """A number field: arrows, Return and leaving it commit (one undo step)."""
        w = ttk.Spinbox(parent, textvariable=self._var(key, "V"), width=width, command=self._commit, **kw)
        for ev in ("<Return>", "<KP_Enter>", "<FocusOut>"):
            w.bind(ev, lambda e: self._commit(), add="+")
        self._field(key, w)
        return w

    def _get(self, key):
        return self.C[key].get() if key in self.C else self.T[key].get()

    def applies(self, key):
        """The setting's stage is on and the engine/detection in use reads it."""
        t = self.stage_of.get(key)
        return (t is None or bool(self.T[t].get())) and APPLIES.get(key, lambda g: True)(self._get)

    def _sync(self):
        """Grey out the settings of stages that are off and of the engine/detection not in use."""
        for key, w in self.grey:
            use = self.applies(key)
            if isinstance(w, ttk.Combobox):
                w.state(["!disabled", "readonly"] if use else ["disabled"])
            else:
                w.state(["!disabled"] if use else ["disabled"])

    def _set(self, key, v, step=None):
        """Write a number to its field's variable, clamped to the field's range and rounded to step."""
        lo, hi = self.ranges[key]
        v = min(max(float(v), lo), hi)
        if step:
            v = min(max(round(v / step) * step, lo), hi)
        s = str(int(round(v))) if key in INTS else f"{round(v, 4):g}"
        if self.V[key].get() != s:
            self.V[key].set(s)

    # ---- commits, undo and redo; drags ----
    def _snap(self):
        """Every setting's raw field value (an invalid entry comes back too)."""
        return dict({k: v.get() for d in (self.V, self.T, self.C) for k, v in d.items()}, gpu=self.gpu.get())

    def _commit(self):
        """A finished change (release, checkbox, choice, Return, ...): one undo step."""
        cur = self._snap()
        if self.committed is not None and cur != self.committed:
            self.undo_stack = (self.undo_stack + [self.committed])[-200:]
            self.redo_stack = []
        self.committed = cur
        self._undo_buttons()

    def _undo_buttons(self):
        self.undo_btn.state(["!disabled" if self.undo_stack else "disabled"])
        self.redo_btn.state(["!disabled" if self.redo_stack else "disabled"])

    def undo(self, redo=False):
        if self.dragging:
            return
        self._commit()  # a change not committed yet is undone too
        src, dst = (self.redo_stack, self.undo_stack) if redo else (self.undo_stack, self.redo_stack)
        if src:
            dst.append(self.committed)
            self.committed = snap = src.pop()
            for k, v in snap.items():
                var = self.gpu if k == "gpu" else {**self.V, **self.T, **self.C}[k]
                if var.get() != v:
                    var.set(v)
            self._undo_buttons()

    def redo(self):
        self.undo(True)

    def drag_end(self):
        """A drag released: its changes are applied (light jobs too) and committed."""
        self.dragging, self._dirty = None, self._dirty | self._drag_dirty
        self._drag_dirty = set()
        if self._pending is not None:
            self.root.after_cancel(self._pending)
        self._apply_changes()
        self._commit()

    def type_value(self, key):
        """Focus the setting's field with its text selected (Advanced fields show that window)."""
        w = next((w for w in self.bound.get(key, ()) if isinstance(w, ttk.Spinbox)), None)
        if w is not None:
            if w.winfo_toplevel() == self.adv:
                self.adv.deiconify(); self.adv.lift()
            w.focus_set()
            w.selection_range(0, "end")
            w.icursor("end")

    def _frame_hover(self, v):
        """A Frame radio button hovered (v) or left (None): the lanes' risk strip for that width."""
        self.wv.cand_w = None if v is None else 512 << v
        st, text = self.wv.det.get("stats"), ""
        if v is not None and st is not None and self.applies("lookahead"):
            text = (f"Frame {512 << v}: {100 * st.little_ctx[512 << v]:.1f}% of rebuilt samples have little "
                    "context (red strip under the lanes)")
        self.wv.readout.config(text=text)
        self.wv.redraw(("lanes",))

    # ---- keyboard ----
    def _focus(self):
        try:
            return self.root.focus_get()
        except KeyError:  # e.g. a combobox's drop-down list
            return None

    def _typing(self):
        """A text field has the focus (a read-only drop-down has no text: its keys stay shortcuts)."""
        f = self._focus()
        if isinstance(f, ttk.Combobox) and f.instate(["readonly"]):
            return False
        return isinstance(f, (tk.Entry, tk.Spinbox, ttk.Entry))

    def _key(self, e):
        """Unmodified keys; they do nothing while a text field has the focus (Esc leaves it; a read-only
        drop-down keeps Up and Down)."""
        wv, ks = self.wv, e.keysym
        if wv.hide_help():
            return "break"
        if ks == "F1" or (ks == "question" and not self._typing()):
            wv.toggle_help(KEYS)
            return "break"
        if e.state & 4:  # Ctrl shortcuts are bound on their own
            return None
        f = self._focus()
        if self._typing() or isinstance(f, ttk.Combobox) and ks in ("Escape", "Up", "Down"):
            if ks == "Escape":
                wv.lanes.focus_set()
                return "break"
            return None
        if ks in ("Tab", "ISO_Left_Tab") and f not in (None, self.root, self.wv.lanes) and f.winfo_viewable():
            return None  # moving between buttons and boxes (the Advanced window's too, while it is shown)
        if ks == "space" and isinstance(f, (ttk.Button, ttk.Checkbutton, ttk.Radiobutton)):
            return None  # Space has pressed the focused button
        shift = bool(e.state & 1)
        h = wv.sel_handle()
        act = {"space": self.toggle_play, "1": lambda: self._source(1), "2": lambda: self._source(2),
               "3": lambda: self._source(3), "x": lambda: self._source(2 if self.src.get() == 1 else 1),
               "s": lambda: self._toggle(self.side), "m": lambda: self._toggle(self.match),
               "Escape": self._escape, "p": lambda: self.preview_now(True), "bracketleft": lambda: wv.shift_region(-1),
               "bracketright": lambda: wv.shift_region(1), "n": lambda: wv.next_hot(-1 if shift else 1),
               "v": lambda: self._mode("lin" if self.lin_db.get() == "db" else "db"), "f": wv.fit,
               "z": lambda: wv.region and wv.zoom_region(),
               "plus": lambda: wv.zoom(0.8), "equal": lambda: wv.zoom(0.8), "KP_Add": lambda: wv.zoom(0.8),
               "minus": lambda: wv.zoom(1.25), "KP_Subtract": lambda: wv.zoom(1.25),
               "Left": lambda: h or wv._scroll(-1), "Right": lambda: h or wv._scroll(1),
               "Home": lambda: wv.home(False), "End": lambda: wv.home(True),
               "Tab": lambda: wv.select(-1 if shift else 1), "ISO_Left_Tab": lambda: wv.select(-1),
               "Up": lambda: h and wv.nudge(h, 1, shift), "Down": lambda: h and wv.nudge(h, -1, shift),
               "Return": lambda: h and self.type_value(h.key), "BackSpace": lambda: h and wv.reset(h),
               }.get(ks.lower() if len(ks) == 1 else ks)
        if act is None:
            return None
        act()
        return "break"

    def _mode(self, mode):
        self.lin_db.set(mode)
        self.wv.set_mode(mode)

    def _changed(self, key):
        self._dirty.add(key)
        if self._pending is None:  # a bulk apply_cfg collapses into one _apply_changes
            self._pending = self.root.after_idle(self._apply_changes)

    def _apply_changes(self):
        self._pending, dirty = None, self._dirty
        if dirty & (self.T.keys() | self.C.keys()):
            self._sync()
        try:
            cfg = self.collect()
        except ValueError as e:  # e.g. "" or "-" while typing: wait for a valid value
            self.mod.config(text="invalid value")
            self.wv.readout.config(text=str(e))
            self._err = True
            return
        if self._err:
            self.wv.readout.config(text="")
            self._err = False
        self._dirty, self.cfg = set(), cfg
        self._summaries(cfg)
        self.mod.config(text="modified" if settings_from_cfg(cfg) != self.base
                        else f"from {self.base_name}" if self.base_name else "")
        self._detect(dirty)  # during a drag: only the display
        a = dirty & session.A_KEYS
        self._preview(submit=not a)  # stage B only: at once, also while dragging
        if a:  # stage A: at once for boxes, choices and Frame, else once typing pauses
            self._preview_soon(0 if a & (self.T.keys() | self.C.keys() | {"gpu", "lookahead"}) else 300)

    def _summaries(self, cfg):
        g = lambda k: f"{cfg[k]:g}"
        if not cfg["declip"]:
            d = "off"
        elif cfg["engine"] == "NMF":
            d = (f"NMF auto · {cfg['nmf_quality']}" if cfg["auto_detect"] else
                 f"NMF manual {g('clip_above')}% · {cfg['nmf_quality']}")
        else:
            d = f"SPADE {g('clip_above')}% · q{cfg['speed_quality']} · {512 << min(2, max(0, cfg['lookahead']))}"
        text = dict(Declip=d,
                    Unlimit=("needs PyTorch" if not self.torch else
                             f"{g('unlimit_amount')} %" if cfg["unlimit"] else "off"),
                    Uncompress=(f"{g('uc_threshold')} dB · {g('uc_ratio')}:1 · {g('uc_amount')} %"
                                if cfg["uncompress"] else "off"),
                    Output=f"in {g('input_gain')} / out {g('output_gain')} dB · "
                           + (f"limit {g('limiter_peak')} dB" if cfg["limiter"] else "no limiter"))
        for page, t in text.items():
            self.chips[page][0].config(text=t)

    def _select_page(self, name):
        self.pages[self.page].pack_forget()
        self.pages[name].pack(fill="x")
        self.page = name
        for page, (summary, line) in self.chips.items():
            line.config(bg=ACCENT if page == name else PANEL)
        self.wv.redraw()
        self._show_stages()

    def _build(self):
        pad = dict(padx=14)
        # packed first so a small window shrinks the waveform, not the buttons
        foot = ttk.Frame(self.root)
        foot.pack(side="bottom", fill="x", pady=(4, 14), **pad)
        self.bar = ttk.Progressbar(foot, maximum=1.0)
        self.bar.pack(fill="x")
        row = ttk.Frame(foot); row.pack(fill="x", pady=(8, 0))
        self.cancel_btn = ttk.Button(row, text="Cancel", command=self.cancel.set, state="disabled", takefocus=0)
        self.cancel_btn.pack(side="right")
        self.run_btn = ttk.Button(row, text="Render", style="Accent.TButton", command=self.run, takefocus=0)
        self.run_btn.pack(side="right", padx=6)
        # packed after the buttons so a long message can't push them off
        self.status = ttk.Label(row, text="Open a clipped file to start.", style="Muted.TLabel")
        self.status.pack(side="left", fill="x", expand=True)
        # audition and region preview bar. Packed right to left from the badge, so a narrow window
        # truncates the audio note, then the region text, and never a control
        self.gpu_dev = spade_backend(True)[1]
        aud = ttk.Frame(self.root); aud.pack(side="bottom", fill="x", pady=(0, 6), **pad)
        self.badge = ttk.Label(aud, text="", style="Muted.TLabel")
        self.badge.pack(side="right")
        self.dot = tk.Label(aud, text="", bg=BG, fg=MUTED)
        self.dot.pack(side="right")
        self.auto = tk.BooleanVar(value=True)  # view state, not settings: preview when it takes 10 s or less
        self.fast = tk.BooleanVar(value=self.gpu_dev not in ("opencl", "cuda"))  # NMF fast, SPADE quality <= 3
        for text, var in (("Fast", self.fast), ("Auto", self.auto)):
            ttk.Checkbutton(aud, text=text, variable=var, takefocus=0, command=self._preview).pack(side="right",
                                                                                                padx=(10, 0))
        self.pbtn = ttk.Button(aud, text="Preview", width=15, takefocus=0, command=self.preview_now)
        self.pbtn.pack(side="right", padx=(12, 0))
        # audition (view state too): Play, what to hear, Side, Match and the monitor trim
        self.src, self.side, self.match = tk.IntVar(value=1), tk.BooleanVar(), tk.BooleanVar()
        self.play_btn = ttk.Button(aud, text="▶", width=3, takefocus=0, command=self.toggle_play)
        self.play_btn.pack(side="left")
        self.audio_w = [self.play_btn]
        for v, text in enumerate(("1 Orig", "2 Proc", "3 Diff"), 1):
            w = ttk.Radiobutton(aud, text=text, value=v, variable=self.src, style="Toolbutton", takefocus=0,
                                command=lambda: self._source(self.src.get()))
            w.pack(side="left", padx=(8 if v == 1 else 0, 0))
            self.audio_w.append(w)
        for text, var in (("Side", self.side), ("Match", self.match)):
            w = ttk.Checkbutton(aud, text=text, variable=var, takefocus=0, command=self._audition)
            w.pack(side="left", padx=(10, 0))
            self.audio_w.append(w)
        self.mon_btn = ttk.Button(aud, text="Mon Auto", width=16, takefocus=0, command=self._cycle_mon)
        self.mon_btn.pack(side="left", padx=(10, 0))
        self.audio_w.append(self.mon_btn)
        self.region_lab = ttk.Label(aud, text="", style="Muted.TLabel")
        self.region_lab.pack(side="right")
        self.anote = ttk.Label(aud, text="", style="Muted.TLabel")  # match gain, B missing, or no sounddevice
        self.anote.pack(side="right", padx=(10, 12))
        if not self.player.available:
            for w in self.audio_w:
                w.state(["disabled"])

        files = ttk.Frame(self.root); files.pack(fill="x", pady=(12, 6), **pad)
        for row, (label, var, cmd) in enumerate((("Input", "inp", self.browse_in),
                                                 ("Output", "outp", self.browse_out))):
            ttk.Label(files, text=label, style="Muted.TLabel", width=7).grid(row=row, column=0, sticky="w")
            setattr(self, var, tk.StringVar())
            ttk.Entry(files, textvariable=getattr(self, var)).grid(row=row, column=1, sticky="we", pady=2)
            ttk.Button(files, text="Browse", command=cmd, takefocus=0).grid(row=row, column=2, padx=(6, 0))
        files.columnconfigure(1, weight=1)

        bar = ttk.Frame(self.root); bar.pack(fill="x", pady=(0, 6), **pad)
        ttk.Label(bar, text="Preset", style="Muted.TLabel").pack(side="left")
        self.preset = tk.StringVar()
        pc = ttk.Combobox(bar, textvariable=self.preset, values=list(PRESETS), width=22, state="readonly")
        pc.pack(side="left", padx=6)
        pc.bind("<<ComboboxSelected>>", lambda e: self.apply_preset(self.preset.get()))
        self.mod = ttk.Label(bar, text="", style="Muted.TLabel", width=-18)  # "modified" / "from <file>"
        self.mod.pack(side="left")
        ttk.Button(bar, text="Save", command=self.save_preset, takefocus=0).pack(side="left", padx=2)
        ttk.Button(bar, text="Load", command=self.load_preset, takefocus=0).pack(side="left", padx=2)
        self.undo_btn = ttk.Button(bar, text="Undo", state="disabled", takefocus=0, command=self.undo)
        self.redo_btn = ttk.Button(bar, text="Redo", state="disabled", takefocus=0, command=self.redo)
        self.undo_btn.pack(side="left", padx=(8, 2)); self.redo_btn.pack(side="left", padx=2)
        self.adv_btn = ttk.Button(bar, text="Advanced…", command=self.toggle_adv, takefocus=0)
        self.adv_btn.pack(side="right")
        self.gpu = tk.BooleanVar(value=self.gpu_dev in ("opencl", "cuda"))
        self.gpu.trace_add("write", lambda *a: self._changed("gpu"))
        ttk.Checkbutton(bar, text="GPU", variable=self.gpu, command=self._commit, takefocus=0).pack(side="right",
                                                                                                    padx=10)
        self.log = tk.BooleanVar(value=False)  # writes <output>.log
        ttk.Checkbutton(bar, text="Log", variable=self.log, takefocus=0).pack(side="right")
        self.labels["process_rate"] = "Process rate"
        self._spin(bar, "process_rate", width=7, values=("0", "11025", "16000", "22050", "24000", "32000", "44100",
                                                         "48000", "88200", "96000")).pack(side="right", padx=(0, 10))
        ttk.Label(bar, text="Process at Hz", style="Muted.TLabel").pack(side="right", padx=(0, 6))

        # stage chips (checkbox = stage on/off, click = show its page) over the inspector and the editor
        strip = ttk.Frame(self.root); strip.pack(fill="x", pady=(0, 8), **pad)
        self.body = ttk.Frame(self.root); self.body.pack(fill="both", expand=True, pady=(0, 6), **pad)
        self.insp = ttk.Frame(self.body, width=300)
        self.insp.pack(side="right", fill="y", padx=(10, 0)); self.insp.pack_propagate(False)
        try:
            import torch  # unlimit and the NMF engine need it
            self.torch, self.cuda = True, torch.cuda.is_available()
        except ImportError:
            self.torch = self.cuda = False
        self.pages, self.chips = {}, {}
        for i, (title, toggle, cols, items) in enumerate(PAGES):
            self.pages[title] = page = ttk.Frame(self.insp)
            box = self._section(page, title, toggle, True, cols, items)
            box.pack(fill="x")
            if toggle == "unlimit" and not self.torch:
                ttk.Label(box, text="needs PyTorch", style="Muted.TLabel").grid(row=1, column=0, columnspan=2,
                                                                                 sticky="w")
            if title == "Uncompress":  # the transfer curves
                self.curve = CurvePanel(page, self)
                self.curve.pack(anchor="w", pady=(8, 2))
            if title != "Declip":  # one line each, so the page's height never changes
                self.stage_txt[title] = [ttk.Label(page, text=" ", style="Muted.TLabel")
                                         for _ in range(2 if title == "Output" else 3)]
                for j, w in enumerate(self.stage_txt[title]):
                    w.pack(anchor="w", pady=(6 if j == 0 and title != "Uncompress" else 0, 0))
            if toggle == "declip":  # manual detection: nines histogram and stats
                self.hist = HistPanel(page, self)
                self.hist.pack(anchor="w", pady=(8, 4))
                self.stat = [ttk.Label(page, text=" ", style="Warn.TLabel" if i == 4 else "Muted.TLabel")
                             for i in range(5)]  # one line each, so the page's height never changes
                for w in self.stat:
                    w.pack(anchor="w")
                self.stat[1].configure(cursor="hand2")
                self.stat[1].bind("<Button-1>", lambda e: self._jump_longest())
                row = ttk.Frame(page)  # NMF auto detection only: a fixed height, shown or not
                self.seed_btn = ttk.Button(row, text="Use as manual threshold", takefocus=0, command=self.seed_manual)
                row.configure(height=self.seed_btn.winfo_reqheight())
                row.pack(fill="x", pady=(4, 0)); row.pack_propagate(False)
            if i:
                ttk.Label(strip, text="→", style="Muted.TLabel").pack(side="left", padx=6)
            chip = tk.Frame(strip, bg=PANEL, highlightthickness=1, highlightbackground=LINE, cursor="hand2")
            chip.pack(side="left", fill="x", expand=True)
            line = tk.Frame(chip, height=2, bg=PANEL)
            line.pack(side="bottom", fill="x")
            row = tk.Frame(chip, bg=PANEL)
            row.pack(fill="x", padx=8, pady=(5, 3))
            if toggle:
                cb = ttk.Checkbutton(row, variable=self._var(toggle, "T"), style="Chip.TCheckbutton", takefocus=0,
                                     command=self._commit)
                cb.pack(side="left")
                self.bound[toggle].append(cb)
            head = tk.Label(row, text=title.upper(), bg=PANEL, fg=FG, font=("Segoe UI", 9, "bold"))
            summary = tk.Label(row, text="", bg=PANEL, fg=MUTED)
            head.pack(side="left"); summary.pack(side="left", padx=(8, 0))
            for w in (chip, row, head, summary):
                w.bind("<Button-1>", lambda e, p=title: self._select_page(p))
            self.chips[title] = summary, line

        self.adv = tk.Toplevel(self.root); self.adv.withdraw()
        self.adv.title("SharpUnClip - Advanced"); self.adv.configure(background=BG)
        self.adv.protocol("WM_DELETE_WINDOW", self.adv.withdraw)
        for gi, (gtitle, toggle, subs) in enumerate(ADVANCED):
            grp = ttk.LabelFrame(self.adv, text=gtitle, padding=(8, 4, 8, 4))
            grp.grid(row=gi, column=0, sticky="we", padx=14, pady=(12 if gi == 0 else 0, 12))
            per_row = 2 if len(subs) > 3 else len(subs)
            for si, (title, cols, items) in enumerate(subs):
                self._section(grp, title, toggle, False, cols, items).grid(
                    row=si // per_row, column=si % per_row, sticky="nsew", padx=(0, 8), pady=(0, 6))
        self.adv.resizable(False, False)

        self.wave = ttk.Frame(self.body); self.wave.pack(fill="both", expand=True)
        head = ttk.Frame(self.wave); head.pack(fill="x")
        # the buttons first, so long texts on the left are cut rather than push them off
        ttk.Button(head, text="Fit", takefocus=0, command=lambda: self.wv.fit()).pack(side="right", padx=(8, 0))
        self.lin_db = tk.StringVar(value="lin")  # view only, not a setting
        for val, text in (("db", "dB"), ("lin", "Lin")):
            ttk.Radiobutton(head, text=text, value=val, variable=self.lin_db, style="Toolbutton", takefocus=0,
                            command=lambda: self.wv.set_mode(self.lin_db.get())).pack(side="right")
        self.info = ttk.Label(head, text="", style="Muted.TLabel")
        self.info.pack(side="left")
        self.rebuilt = ttk.Label(head, text="", style="Muted.TLabel")
        self.rebuilt.pack(side="left", padx=(12, 0))
        self.tag = ttk.Label(head, text="", style="Muted.TLabel")  # processing rate, output made with other settings
        self.tag.pack(side="left", padx=(12, 0))
        self.warn = tk.Label(head, text="", bg=WARN_BG, fg=WARN, cursor="hand2", padx=4)  # packed when needed
        self.warn.bind("<Button-1>", lambda e: self._jump_to(self.wv.det["stats"].warn[1]))
        self.wv = WaveView(self.wave, self)
        self.wv.pack(fill="both", expand=True, pady=(4, 0))
        self.page = "Declip"
        self._select_page("Declip")
        for r in (self.root, self.adv):  # the Advanced window's widgets do not reach the root's bindings
            r.bind("<Key>", self._key)
            for k in "oO":  # both cases (Caps Lock); Shift tells them apart
                r.bind(f"<Control-{k}>", lambda e: self.load_preset() if e.state & 1 else self.browse_in())
            for k in "sS":
                r.bind(f"<Control-{k}>", lambda e: self.save_preset())
            for k in "zZ":  # Ctrl+Shift+Z: redo
                r.bind(f"<Control-{k}>", lambda e: self._typing() or self.undo(bool(e.state & 1)))
            for k in "yY":
                r.bind(f"<Control-{k}>", lambda e: self._typing() or self.redo())
            for k in "eE":
                r.bind(f"<Control-{k}>", lambda e: self.toggle_adv())
            r.bind("<Control-Return>", lambda e: self.run())
            for i, title in enumerate(self.pages, 1):
                r.bind(f"<Control-Key-{i}>", lambda e, p=title: self._select_page(p))

        self.apply_preset("Accurate")
        self.root.update_idletasks()
        # the lanes ask for the tallest page's height + 60, so the minimum size below (-60) only
        # ever shrinks the lanes, never a page
        lanes = self.wv.lanes
        need = max(p.winfo_reqheight() for p in self.pages.values())
        lanes.configure(width=720, height=max(200, need - (self.wave.winfo_reqheight() - lanes.winfo_reqheight()) + 60))
        r = self.root
        r.update_idletasks()  # minimum size keeps every control visible (on a small screen: the footer)
        r.minsize(min(r.winfo_reqwidth(), r.winfo_screenwidth()),
                  min(r.winfo_reqheight() - 60, r.winfo_screenheight() - 80))
        if not self.player.available:  # set after the minimum size: it truncates rather than widen it
            self.anote.config(text=session.SD_HINT)
        r.geometry(f"{min(1360, r.winfo_screenwidth())}x{min(860, r.winfo_screenheight() - 80)}")

    def toggle_adv(self):
        if self.adv.winfo_ismapped():
            self.adv.withdraw()
        else:
            self.adv.deiconify(); self.adv.lift()

    # ---- files / presets ----
    def browse_in(self):
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.flac *.ogg *.aiff *.mp3"), ("All", "*.*")])
        if p:
            self.inp.set(p)
            self.outp.set(self.default_out(p))
            if self.load_input(p):
                self.wv.lanes.focus_set()  # Space plays at once

    def default_out(self, inp):
        """<input name>_unclipped.wav, in the output folder already chosen (else the input's)."""
        name = os.path.splitext(os.path.basename(inp))[0] + "_unclipped.wav"
        folder = os.path.dirname(self.outp.get()) or os.path.dirname(inp)
        return os.path.normpath(os.path.join(folder, name))

    def browse_out(self):
        cur = self.outp.get() or (self.default_out(self.inp.get()) if self.inp.get() else "")
        p = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")],
                                         initialdir=os.path.dirname(cur) or None,
                                         initialfile=os.path.basename(cur))  # pick a folder, keep the name
        if p:
            self.outp.set(os.path.normpath(p))

    def load_input(self, p):
        try:
            self.x, self.sr = sf.read(p, always_2d=True)
        except Exception as e:
            self.status.config(text=f"can't read input: {e}")
            return False
        self.path = p
        self._stop_audio()
        if self.player.sr != self.sr:  # the stream runs at the file's rate
            self.player.close()
            self.player = session.Player(self.sr)
        self.wv.set_song(Song(self.x, self.sr, p))
        self.gen += 1  # results for the previous song are dropped
        self.region_user_set = False
        self.sgen += 1  # previews too
        self.heavy.cancel(); self.light.cancel("uc")
        self.lru.clear()
        self.G = self.ucg = self.prev = self.rkey = self.full = self._stopped = self._shp = self._lv = None
        self._jobs, self._gp, self._mode = {}, 0.0, "?"
        n, ch = self.x.shape
        self.info.config(text=f"{os.path.basename(p)}   ·   {self.sr / 1000:g} kHz   ·   "
                              f"{'mono' if ch == 1 else 'stereo' if ch == 2 else f'{ch} ch'}   ·   {mmss(n / self.sr)}")
        self.bar["value"] = 0
        self.status.config(text="Ready.")
        self._detect(LIGHT_KEYS)
        self._preview_soon()
        return True

    def apply_preset(self, name):
        self.preset.set(name)
        self.base, self.base_name = dict(DEFAULTS, **PRESETS[name]), None
        self.apply_cfg(PRESETS[name])
        self._commit()

    def apply_cfg(self, cfg):
        for k, val in cfg.items():  # unknown keys are ignored
            k = k.removeprefix("pa_")  # older preset files
            if k in self.V:
                self.V[k].set(str(int(float(val))) if k in INTS else str(val))
            elif k in self.C:
                self.C[k].set(str(val))
            elif k in self.T:
                self.T[k].set(1 if float(val) else 0)  # old presets stored unlimit as a %
        if "gpu" in cfg:
            self.gpu.set(bool(cfg["gpu"]))
        self._sync()

    def collect(self):
        cfg = {}
        for k, v in self.V.items():
            try:
                cfg[k] = int(float(v.get())) if k in INTS else float(v.get())
            except ValueError:
                raise ValueError(f"{self.labels[k]} is {v.get()!r}, not a number") from None
        cfg.update({k: v.get() for k, v in self.T.items()})
        cfg.update({k: v.get() for k, v in self.C.items()})
        cfg["gpu"] = self.gpu.get()
        return cfg

    def save_preset(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON preset", "*.json")])
        if p:
            with open(p, "w") as f:
                json.dump(self.collect(), f, indent=2)
            self.status.config(text="saved " + p)

    def load_preset(self, p=None):
        p = p or filedialog.askopenfilename(filetypes=[("JSON preset", "*.json"), ("All", "*.*")])
        if p:
            with open(p) as f:
                cfg = json.load(f)
            self.apply_cfg(cfg)
            try:  # as applied: a partial or non-canonical file still reads "from <file>"
                self.base = settings_from_cfg(self.collect())
            except ValueError:
                self.base = settings_from_cfg(cfg)
            self.base_name = os.path.splitext(os.path.basename(p))[0]
            self.status.config(text="loaded " + p)
            self._commit()

    # ---- detection, region ----
    def _det_mode(self, cfg=None):
        """The detection the settings use: "manual", "auto" (NMF auto detection) or None (Declip off)."""
        cfg = cfg or self.cfg
        return None if not cfg["declip"] else "auto" if cfg["engine"] == "NMF" and cfg["auto_detect"] else "manual"

    def _ndev(self, cfg):
        """The torch device of the NMF engine (and its analysis) with cfg's GPU setting."""
        return "cuda" if cfg["gpu"] and self.cuda else "cpu"

    def _sh(self, cfg):
        """The shoulder search's result for cfg (its song, levels and device), or None: not searched yet."""
        return self.wv.song.shoulder_ready(cfg, self._ndev(cfg)) if self.wv.song is not None else None

    def _prov(self, cfg):
        """NMF auto detection with the knee as found but no shoulder search yet: ceilings only for now."""
        return self._det_mode(cfg) == "auto" and not cfg["nmf_knee"] and self._sh(cfg) is None

    def _eff(self, cfg):
        """The settings previews run with: knee 1 (ceilings only) while _prov."""
        return dict(cfg, nmf_knee=1.0) if self._prov(cfg) else cfg

    def fkey(self, cfg=None):
        """The key of the flags these settings ask for: manual detection's FLAG_KEYS; NMF auto detection's
        levels, knee in effect and the shoulder's knee it uses."""
        cfg = cfg or self.cfg
        if self._det_mode(cfg) != "auto":
            return session.keyof(cfg, session.FLAG_KEYS)
        sh, k = self._sh(cfg), cfg["nmf_knee"]
        return "auto", session.keyof(cfg, session.G_KEYS), k or (1.0 if sh is None else 0.0), \
            None if k or sh is None else sh["knee"]

    def _detect(self, dirty):
        """Detection: manual detection's lines follow the settings at once; the flags, stats and
        histogram counts (of NMF auto detection too, and its ceilings) come from the light worker, the
        shoulder search from the heavy one."""
        song = self.wv.song
        if self.dragging:  # the jobs wait for the release
            self._drag_dirty |= dirty
        elif song is not None and self.cfg is not None:
            mode = self._det_mode()
            if mode != self._mode:  # the other detection's results go, with any still on their way
                self._mode, self.gen, dirty = mode, self.gen + 1, dirty | LIGHT_KEYS | AUTO_KEYS
                for kind in DET_KINDS + AUTO_KINDS:
                    self.light.cancel(kind)
                self.wv.set_det(steps=None, gaps=None, pkey=None, flags=None, fkey=None, stats=None, hist=None,
                                hot=[], ceil=None, af=None)
            if mode is None and self.wv.region is None:
                self._auto_region()
            elif mode == "manual" and dirty & LIGHT_KEYS:
                self.gen, self._det_cfg = self.gen + 1, self.cfg
                for kind, fn in zip(DET_KINDS, (lambda c: (song.steps(c), song.gaps(c)), song.flags, song.stats,
                                                song.hist9)):
                    self.light.submit(kind, self.gen, lambda ev, c, fn=fn: fn(c), self.cfg)
            elif mode == "auto" and dirty & AUTO_KEYS:
                self._auto_jobs()
            self._need_sh()
        self._show_stats()
        self._show_hist()
        self.wv.redraw(("lanes", "gain") if self.dragging else ("ov", "ruler", "lanes", "gain"))

    def _auto_jobs(self):
        """NMF auto detection: the ceilings, then the flags with the shoulder in use (light worker)."""
        song, cfg = self.wv.song, self.cfg
        self.gen += 1
        self.light.submit("ceil", self.gen, lambda ev, c: song.ceilings(c), cfg)
        self.light.submit("aflags", self.gen, lambda ev, c, sh, k: (k, song.auto_flags(c, sh)), cfg, self._sh(cfg),
                          self.fkey(cfg))

    def _need_sh(self):
        """NMF auto detection with the knee as found: the shoulder search (heavy, lowest priority), unless
        it is done or on its way. Another song, level, rate or device, or a set knee, cancels it."""
        cfg, song = self.cfg, self.wv.song
        if cfg is None or song is None or not self.torch or self.rendering:
            return
        key = session.keyof(cfg, session.G_KEYS) + (self._ndev(cfg),)
        job = self._jobs.get("sh")
        if job is not None and (job != key or cfg["nmf_knee"]):
            self.heavy.cancel("sh")
            del self._jobs["sh"]
            self._shp = None
        if self._det_mode(cfg) == "auto" and not cfg["nmf_knee"] and "sh" not in self._jobs and self._sh(cfg) is None:
            self._jobs["sh"], self._shp = key, None
            self.heavy.submit("sh", (self.sgen, key), self._job_sh, song, cfg, key, prio=3)

    def _job_sh(self, ev, song, cfg, key):  # the heavy worker
        spent, last, nw = [0.0], [None], [0]  # time in the windows (not in jobs run between them)

        def step(i, n):  # before each window: urgent jobs first, then cancel or carry on
            if last[0] is not None:
                spent[0] += time.time() - last[0]
            self.heavy.run_pending(2)
            if ev.is_set():
                raise session.Cancelled
            self.q.put(("job_p", "sh", key, (i, n, spent[0] / i * (n - i) if i else None)))
            last[0], nw[0] = time.time(), n
        sh = song.shoulder(cfg, key[-1], step)
        return key, sh, spent[0] + (time.time() - last[0] if last[0] else 0.0), nw[0]

    def _searched(self, cfg, n):
        """Seconds of audio the shoulder search runs its fast pass on, in n windows."""
        sp = session.rate_p(cfg, self.sr)
        return len(self.x) / sp if len(self.x) <= 75 * sp else 10.0 * n

    def _knee_text(self, cfg):
        """The knee in effect, or the search's progress."""
        k, sh = cfg["nmf_knee"], self._sh(cfg)
        if k >= 1:
            return "ceilings only (knee 1.00)"
        if k:
            return f"knee set {k:.2f}"
        if sh is None:
            if not self.torch:
                return "knee search needs PyTorch"
            if self._shp is None:
                return "knee search waiting · or set a knee"
            i, n, left = self._shp
            if left is None:
                left = self.est.seconds(("knee", self._ndev(cfg) == "cuda"), self._searched(cfg, n))
            return f"knee search {i + 1}/{n} · ~{mmss(left)} · or set a knee"
        acc, lo = sh.get("acc"), sh.get("acc_min") or 0.0
        if sh["knee"] is not None:
            return f"knee {sh['knee']:.2f} found (accel {acc:.2f} ≥ {lo:.2f})"
        refs = self.wv.det["ceil"]["refs"] if self.wv.det["ceil"] is not None else []
        if any(r["kind"] == "limiter" for row in refs for r in row):
            return "limiter ceilings: knee 0.80 (default)"
        return "no shoulder" + (f" ({acc:.2f} < {lo:.2f})" if acc == acc and acc is not None else "") + ": ceilings only"

    def seed_manual(self):
        """Use as manual threshold: Auto detect off, Clip above at the lowest knee of the sides with a
        ceiling (as a fraction of it); one undo step. For a clipped song the measured peak sits at about
        the ceiling."""
        d = self.wv.det
        if d["ceil"] is None or d["af"] is None:
            return
        ks = [k / r["theta"] for kr, row in zip(d["af"]["knees"], d["ceil"]["refs"]) for k, r in zip(kr, row)
              if k is not None and r["kind"] != "none"]
        if not ks:
            return
        v = min(99.9, round(100 * min(ks), 1))
        self.T["auto_detect"].set(0)
        self._set("clip_above", v)
        self._commit()
        if self.wv.region is not None:  # the region and view stay on what was looked at (and this message)
            self.region_user_set = True
        self.wv.readout.config(text=f"Manual detection seeded from the found ceiling: Clip above {v:.1f} %")

    def _on_job(self, msg, kind, gen, v):
        if kind in ("A", "G", "uc", "sh"):
            return self._on_preview_job(msg, kind, gen, v)
        if gen != self.gen:
            return  # for older settings or another song
        wv = self.wv
        if msg == "job_err":
            wv.readout.config(text=f"Detection failed: {v}")
        elif kind == "steps":
            wv.set_det(steps=v[0], gaps=v[1], pkey=session.keyof(self._det_cfg, session.PEAK_KEYS))
        elif kind == "flags":
            wv.set_det(flags=v, fkey=session.keyof(self._det_cfg, session.FLAG_KEYS))
        elif kind in ("stats", "aflags"):
            st = v[1]["stats"] if kind == "aflags" else v
            if kind == "aflags":
                wv.set_det(flags=v[1]["flags"], fkey=v[0], af=v[1])
            wv.set_det(stats=st, hot=session.hotspots(st.block_counts, session.BLOCK, wv.song.sr, self._rlen()))
            self._show_stats()
            self._show_hist()
            self._auto_region()
        elif kind == "ceil":
            wv.set_det(ceil=v)
            self._show_stats()
            self._show_hist()
        else:
            wv.set_det(hist=v)
            self._show_hist()

    def _show_stats(self):
        st, cfg, wv = self.wv.det.get("stats"), self.cfg, self.wv
        lines, warn, ctx = [""] * 5, None, [""] * 3
        wv.readout.config(foreground="")
        mode = cfg and wv.song is not None and self._det_mode()
        d = wv.det
        if st is not None and mode:
            sr, frac, approx = wv.song.sr, st.frac, ""
            hist = d["hist"] if mode == "manual" else d["ceil"] and d["ceil"]["hist"]
            if d["fkey"] != self.fkey() and hist is not None:
                # a drag, or new flags on their way: estimated from the histogram
                if mode == "manual":
                    h = (cfg["clip_above"] if cfg["dynamic"] else min(cfg["clip_above"], cfg["not_clip_below"])) / 100
                    h = min(h, 0.92) if cfg["soft"] else h
                    frac, approx = [session.hist_level(hc, h) / len(wv.song.x) for hc in hist], "~"
                elif cfg["nmf_knee"]:  # from each side's knee level
                    def side(c, p):
                        k = wv._auto_lv(c, p)[1]
                        return hist[1][c][p][max(0, int((20 * np.log10(k) - hist[0]) / session.HIST_DB)):].sum() \
                            if k == k else 0
                    frac, approx = [(side(c, 0) + side(c, 1)) / len(wv.song.x) for c in range(len(hist[1]))], "~"
                if self.dragging and approx:  # README: over about 70 % no declipper can rebuild it
                    big = np.mean(frac) > 0.7
                    wv.readout.config(text=f"~{100 * np.mean(frac):.2f}% rebuilt"
                                      + (": over 70% cannot be rebuilt well" if big else ""),
                                      foreground=FLAG if big else "")
            lines[0] = f"{approx}{100 * np.mean(frac):.2f}% rebuilt" + (
                " (" + " / ".join(f"{wv._label(c)} {100 * f:.2f}" for c, f in enumerate(frac)) + ")"
                if len(frac) > 1 else "")
            lines[1] = (f"{st.runs:,} runs · longest {1000 * st.longest_s:.1f} ms @{_tfmt(st.longest_at / sr, 1)} ›"
                        if st.runs else "no clipped runs")
            if mode == "manual" and cfg["engine"] == "SPADE":
                lc = st.little_ctx[512 << min(2, max(0, cfg["lookahead"]))]
                lines[2] = f"{100 * lc:.1f}% of rebuilt samples have little context"
                if lc > 0.05:
                    lines[4] = "Long clipped runs: try Frame 2048 or NMF"
                ctx = [f"{100 * st.little_ctx[w]:.1f}%" for w in (512, 1024, 2048)]
            if mode == "manual" and cfg["remove_glitch"]:
                lines[3] = f"{st.dropped:,} single samples dropped (Remove glitches)"
            warn = st.warn
        seed = False
        if mode == "auto" and d["ceil"] is not None:  # the ceilings per side, the knee
            refs = d["ceil"]["refs"]
            side = lambda c, p, r: (f"{wv._label(c)}{'+-'[p]} " + (
                f"{r['kind']} {20 * np.log10(r['theta']):.2f}" if r["kind"] != "none" else
                "no ceiling" + (f" (peak {20 * np.log10(r['theta']):.1f})" if r["theta"] else "")))
            for i, row in enumerate(refs[:2]):
                lines[2 + i] = " · ".join(side(i, p, r) for p, r in enumerate(row)) + (
                    " · …" if i and len(refs) > 2 else "") + " dB"
            lines[4] = self._knee_text(cfg)
            seed = d["af"] is not None and any(r["kind"] != "none" for row in refs for r in row)
        elif mode == "auto":
            lines[4] = "finding the ceilings…"
        self.stat[4].configure(style="Muted.TLabel" if mode == "auto" else "Warn.TLabel")
        if seed:
            self.seed_btn.pack(side="left")
        else:
            self.seed_btn.pack_forget()
        for w, t in zip(self.stat + self.ctx, lines + ctx):
            w.config(text=t)
        prov = mode == "auto" and d["af"] is not None and d["af"]["prov"]
        self.rebuilt.config(text=lines[0].split(" (")[0] + (" · provisional" if prov else ""))
        if warn:  # README: over about 70 % no declipper can rebuild it
            self.warn.config(text=f"! {100 * warn[0]:.0f}% @{mmss(warn[1] / wv.song.sr)}")
            self.warn.pack(side="left", padx=(8, 0), after=self.rebuilt)
        else:
            self.warn.pack_forget()

    def _show_hist(self):
        cfg, d, wv = self.cfg, self.wv.det, self.wv
        if cfg is None:
            return
        mode = self._det_mode()
        if mode is None:
            self.hist.show_text("Declip is off")
        elif wv.song is None:
            self.hist.show_text("No file loaded")
        elif mode == "auto" and d["ceil"] is not None and d["ceil"]["hist"] is not None:
            lv = [(r["kind"] != "none", *wv._auto_lv(c, p)) for c, row in enumerate(d["ceil"]["refs"])
                  for p, r in enumerate(row)]  # (a ceiling, theta, knee level)
            ceils = [th for cl, th, _ in lv if cl]
            if not ceils:
                return self.hist.show_text("No ceiling found")
            knee = cfg["nmf_knee"] or max([k / th for cl, th, k in lv if cl and k == k], default=1.0)
            lo, hs = d["ceil"]["hist"]
            hs = [h for row in hs for h in row]  # per side, as lv
            db = lo + (np.arange(len(hs[0])) + 1) * session.HIST_DB  # bin tops: the bin of the level itself counts
            red, amber = np.zeros(len(db)), np.zeros(len(db))
            for (cl, th, k), h in zip(lv, hs):  # each side: red from its ceiling, amber from its knee
                if k == k:
                    r = db > 20 * np.log10(th) if cl else np.zeros(len(db), bool)
                    red += h * r
                    amber += h * (~r & (db > 20 * np.log10(k)))
            self.hist.show_auto(red, amber, np.sum(hs, 0) - red - amber, lo, ceils, knee * max(ceils), max(ceils))
        elif mode == "auto":
            self.hist.show_text("finding the ceilings…" if d["ceil"] is None else "No level found")
        elif d["hist"] is None:
            self.hist.show_text("measuring…")
        else:
            self.hist.show_manual(np.sum(d["hist"], 0), cfg["clip_above"] / 100,
                                  cfg["not_clip_below"] / 100 if self.applies("not_clip_below") else None,
                                  bool(cfg["soft"]), cfg["peak_ignore"])

    def _rlen(self):
        """Default region length, s: 4 with a GPU backend, else 2."""
        return 4.0 if self.gpu.get() and self.gpu_dev in ("opencl", "cuda") else 2.0

    def _auto_region(self):
        """Until the user sets a region: the region on the most-clipped spot (else the loudest part),
        and the view around it."""
        wv, song = self.wv, self.wv.song
        if (song is None or self.region_user_set or self.dragging or wv.drag or wv._rg
                or wv._ovreg is not None):
            return  # never under a drag: the view would move under the pointer
        det = self.cfg is not None and self._det_mode()
        st = wv.det["stats"] if det else None
        if st is not None and wv.det["hot"]:
            r = wv.det["hot"][0]
            msg = (f"Region set to the most-clipped spot ({_tfmt((r[0] + r[1]) / 2 / song.sr, 1)}). "
                   "Fit shows the whole song.")
        elif det and st is None:
            return  # the flags are still coming
        else:
            hot = session.hotspots(np.maximum(song.pyr.mx[0], -song.pyr.mn[0]).max(1), 64, song.sr, self._rlen(), 1)
            if not hot:
                return
            r = hot[0]
            msg = "No clipping found; region on the loudest part." if det else "Region on the loudest part."
        r = r[0], min(r[1], len(song.x))
        if wv.region != r:
            wv.set_region(*r)
            wv.zoom_region()
            wv.readout.config(text=msg)
            self._preview_soon()

    def region_changed(self):
        """The user set the region: it stays where they put it, and its preview starts at once."""
        self.region_user_set = True
        self._preview_soon()

    def _jump_to(self, i):
        """A region of the default length centred on sample i, and the view around it."""
        song = self.wv.song
        n, L = len(song.x), int(self._rlen() * song.sr)
        r0 = int(min(max(0, i - L // 2), max(0, n - L)))
        self.wv.jump(r0, r0 + L)

    def _jump_longest(self):
        st = self.wv.det.get("stats")
        if st is not None and st.runs and self.wv._show():
            self._jump_to(st.longest_at + int(st.longest_s * self.wv.song.sr) // 2)

    # ---- region previews ----
    def _preview_soon(self, ms=0):
        if self._pjob is not None:
            self.root.after_cancel(self._pjob)
        self._pjob = self.root.after(ms, self._preview)

    def preview_now(self, exact=False):
        """The Preview button, or P (exact: at render quality): the region's preview whatever it costs."""
        self._stopped = None
        self._preview(force=True, exact=exact)

    def _escape(self):
        """Esc: stop the audio, else cancel the preview being computed, else clear the region, else
        deselect the line."""
        if self.player.playing:
            self._stop_audio()
        elif "A" in self._jobs:
            self.heavy.cancel("A")
            self._stopped = self._jobs.pop("A")
            self._preview()
        else:
            self.wv.escape()

    def _fast(self, cfg):
        """Fast preview applies: NMF at fast quality with less context, or SPADE above quality 3."""
        return self.fast.get() and cfg["declip"] and (cfg["engine"] == "NMF" or int(cfg["speed_quality"]) > 3)

    def _ekind(self, cfg, fast):
        gpu = bool(cfg["gpu"]) and (self.gpu_dev in ("opencl", "cuda") if cfg["engine"] == "SPADE" else self.cuda)
        return session.est_kind(cfg, fast, gpu)

    def _akey(self, cfg, fast):
        """Stage A's cache key for the region: (S, E) of its span, the settings it reads, fast."""
        S, E = session.preview_span(self.wv.song, cfg, *self.wv.region, fast)
        return S, E, session.keyof(cfg, session.A_KEYS), fast

    def _estimate(self, cfg, akey):
        """Seconds until B: stage A of the span, plus what is left of the unlimit curve if it is needed."""
        song, k = self.wv.song, self._ekind(cfg, akey[3])
        s = self.est.seconds(k, (akey[1] - akey[0]) / song.sr) if k else 0.0
        if cfg["unlimit"] and cfg["unlimit_amount"] > 0 and (self.G is None
                                                            or self.G[0] != session.keyof(cfg, session.G_KEYS)):
            s += self.est.seconds(("unlimit", self.cuda), len(song.x) / song.sr) * (1 - self._gp)
        return s

    def est_text(self, r0, r1):
        """' · preview ~1.2 s' for a region being dragged, or ''."""
        cfg, song = self.cfg, self.wv.song
        if cfg is None or song is None or r1 <= r0 or not self.torch and cfg["declip"] and cfg["engine"] == "NMF":
            return ""
        fast = self._fast(cfg)
        return f" · preview ~{_secs(self._estimate(cfg, (*session.preview_span(song, cfg, r0, r1, fast), 0, fast)))}"

    def _badge(self, text, colour=None, wait=None):
        """The preview's state, at the right of the preview bar (never in the footer status)."""
        self.badge.config(text=text)
        self.dot.config(text="●" if text else "", fg=colour or MUTED)
        self._wait = wait

    def _fresh(self):
        """The badge of the B shown, which matches the settings and the region."""
        cfg, p = self.cfg, self.prev
        pk = f" · peak {20 * np.log10(max(self.wv.pv[3], 1e-9)):+.1f} dBFS"
        if p["prov"]:  # NMF auto detection before the shoulder search: redone when it lands
            s = self._shp
            return self._badge("provisional (ceilings only)" + (f" · knee search {s[0] + 1}/{s[1]}" if s else "") + pk,
                               SOFT, "prov")
        if p["fast"]:
            return self._badge("fast preview ≠ render" + pk, SOFT)
        what = "exact" if not cfg["declip"] else "≈ NMF preview" if cfg["engine"] == "NMF" else "SPADE exact"
        self._badge(f"B fresh · {what} · {p['secs']:.1f} s" + pk, ACCENT)

    def _marks(self):
        """Output made with other settings drawn dimmed, with a note in the header; the region text."""
        cfg, wv, sr = self.cfg, self.wv, self.sr
        bkey = None if cfg is None else session.keyof(cfg, session.B_KEYS)
        if self.prev is not None and self.prev["r"] != wv.region:
            self.prev = None  # another region: its own B follows
        p = self.prev
        pst = p is not None and p["bkey"] != (None if cfg is None else session.keyof(self._eff(cfg), session.B_KEYS))
        rst = wv.out is not None and self.rkey != bkey
        if wv.stale != rst:
            wv.stale = rst
            wv.redraw(("ov", "lanes"))
        if p is None:
            if wv.pv is not None:
                wv.set_preview(None)
        elif wv.pv is None or wv.pv[0] is not p["B"] or wv.pv[2] != pst:
            wv.set_preview(p["B"], p["r"][0], pst, p["P"])
        pr = 0.0 if cfg is None or wv.song is None else float(cfg["process_rate"] or 0)
        self.tag.config(text="   ·   ".join(t for t, on in (
            (f"processing as {pr / 1000:g} kHz ({pr / max(sr, 1):.2f}× speed)", pr > 0 and abs(pr - sr) > 0.5),
            ("older settings", rst), ("changes pending", pst)) if on))
        r = wv.region
        self.region_lab.config(text="" if r is None or wv.song is None else
                               f"{_tfmt(r[0] / sr)} – {_tfmt(r[1] / sr)}   ({(r[1] - r[0]) / sr:.1f} s)")
        self._audition()
        self._show_stages()

    def full_b(self, cfg):
        """B is the finished render: it was made with cfg's B-settings."""
        return self.full is not None and cfg is not None and self.rkey == session.keyof(cfg, session.B_KEYS)

    def unl(self, cfg):
        """The unlimit curve for cfg's levels, (key, G, its pyramid, (peak, p99.9) dB, seconds), or None."""
        return self.G if self.G is not None and self.G[0] == session.keyof(cfg, session.G_KEYS) else None

    def unl_text(self, cfg):
        if not self.torch:
            return "gain curve: needs PyTorch"
        if self.wv.song is None or cfg is None:
            return ""
        u = self.unl(cfg)
        if u is not None:
            return f"gain curve: ready ({'CUDA' if self.cuda else 'CPU'} {u[4]:.1f} s)"
        if self._jobs.get("G") == session.keyof(cfg, session.G_KEYS):
            return f"gain curve: computing {100 * self._gp:.0f} %"
        return "gain curve: after the render" if self.rendering else "gain curve: computed when Unlimit is on"

    def _levels(self, p):
        """The region's stage-input levels in dB, sorted, as the compressor's detector sees them (cached),
        or None."""
        if p is None or p["lv"] is None:
            return None
        k, u, g, a, link, ref = p["lv"]
        if self._lv is None or self._lv[0] != k:
            u = u if g is None else ref * g[:, None] ** a + (u - ref)  # as process() combines them
            u = np.abs(u).max(1) if link else np.abs(u).ravel()
            self._lv = k, np.sort(20 * np.log10(np.maximum(u, 1e-12)))
        return self._lv[1]

    def _show_stages(self):
        """The Unlimit, Uncompress and Output pages: the unlimit curve's state and peaks, the transfer
        panel and its numbers, the output's peak and what the limiter cuts (the region's preview before
        the limiter, else the current render)."""
        cfg, p, t = self.cfg, self.prev, self.stage_txt
        if cfg is None:
            return
        u, a = self.unl(cfg), cfg["unlimit_amount"] / 100
        t["Unlimit"][0].config(text=self.unl_text(cfg))
        t["Unlimit"][1].config(text="" if u is None else
                               f"peaks up to {a * u[3][0]:+.1f} dB · p99.9 {a * u[3][1]:+.1f} dB · body 0 dB")
        t["Unlimit"][2].config(text="Drag the gain lane up or down to set Amount." if u is not None and cfg["unlimit"]
                               else "")
        if self.page == "Uncompress":  # the panel and the levels only while shown
            T, R, M, ua = cfg["uc_threshold"], cfg["uc_ratio"], cfg["uc_makeup"], cfg["uc_amount"] / 100
            lv = self._levels(p)
            self.curve.show(T, R, M, ua, lv)
            t["Uncompress"][0].config(text=f"T {T:.1f} dB · {R:.1f}:1" + (f" · makeup {M:+.1f} dB" if M else ""))
            if p is not None and p["ucg"] is not None and p["ucg"].size:  # the region's, else at 0 dBFS
                g = float(p["ucg"].max())
                t["Uncompress"][1].config(text=f"restores up to {g:+.1f} dB (100 %) / {ua * g:+.1f} dB "
                                               f"({100 * ua:.0f} %)")
            else:
                g = float(self.curve.inv(0.0))
                t["Uncompress"][1].config(text=f"restores {g:+.1f} dB at 0 dBFS (100 %) / {ua * g:+.1f} dB "
                                               f"({100 * ua:.0f} %)")
            t["Uncompress"][2].config(text="" if lv is None or not len(lv) else
                                      f"{100 * (1 - np.searchsorted(lv, T + M, 'right') / len(lv)):.0f} % of region "
                                      "samples above threshold")
        self.cut, lines = None if p is None else p["cut"], ["", ""]
        db = lambda v: f"{20 * np.log10(max(v, 1e-9)):+.1f} dBFS"
        if p is not None:  # after the limiter, as the badge and the render's: before it, the ceiling + "up to"
            lines[0] = f"output peak {db(self.wv.pv[3])} (region)"  # wv.pv shows p
            if self.cut:
                lines[1] = f"limiter cuts {100 * self.cut[0]:.2f} % of region samples" + (
                    f" (up to {self.cut[1]:.1f} dB)" if self.cut[0] else "")
        elif self.full_b(cfg):
            lines[0] = f"output peak {db(self.full_pk)} (song)"
            if cfg["limiter"]:
                lines[1] = f"limiter cuts {100 * self.full_cut:.2f} % of the song"
        for w, text in zip(t["Output"], lines):
            w.config(text=text)

    def _est_labels(self, cfg):
        """The render time estimate by the quality setting of the engine in use."""
        song = self.wv.song
        for key, lab in self.est_lab.items():
            on = song is not None and cfg["declip"] and (cfg["engine"] == "NMF") == (key == "nmf_quality")
            lab.config(text=f"~{mmss(self.est.seconds(self._ekind(cfg, False), len(song.x) / song.sr))} render"
                       if on else "")

    def _preview(self, force=False, exact=False, submit=True):
        """Bring the region's B up to date with the settings: the finished render while its settings
        match; else stage A (cached, or a heavy job when its estimate passes the 10 s gate, or forced)
        and stage B. Stage A never starts while a drag is held or a render runs."""
        if self._pjob is not None:
            self.root.after_cancel(self._pjob)
            self._pjob = None
        self._marks()
        song, cfg, r = self.wv.song, self.cfg, self.wv.region
        if song is None or cfg is None:
            return self._badge("")
        self._est_labels(cfg)
        ul = cfg["unlimit"] and cfg["unlimit_amount"] > 0
        if self.rendering:
            return self._badge("paused while rendering")
        if not self.torch and (ul or cfg["declip"] and cfg["engine"] == "NMF"):
            return self._badge(("Unlimit" if ul else "The NMF engine") + " needs PyTorch", FLAG)
        if cfg["unlimit"] and self.torch:  # also at Amount 0: the gain lane
            self._need_g(cfg)
        if r is None:
            self._keep_a()
            self.pbtn.config(text="Preview")
            return self._badge("")
        if self.full_b(cfg):
            self._keep_a()
            self.prev = None
            self._marks()
            return self._badge("B = full render", ACCENT)
        cfg = self._eff(cfg)  # provisional: ceilings only until the shoulder search lands
        bkey = session.keyof(cfg, session.B_KEYS)
        fast = not exact and self._fast(cfg)
        keys = [self._akey(cfg, False)] + ([self._akey(cfg, True)] if fast else [])  # render quality first
        akey = keys[-1]
        est = self._estimate(cfg, akey)
        self.pbtn.config(text=f"Preview ~{_secs(est)}")
        p = self.prev  # fresh; one without its uncompress gain (Amount 0) only while a gain job is on its way
        if p is not None and p["bkey"] == bkey and (fast or not p["fast"]) and not (p["uk"] and "uc" not in self._jobs):
            self._keep_a(keys[:1] if p["fast"] else ())
            return self._fresh()
        hit = next(((c, k) for k in keys for c in reversed(self.lru) if _covers(c, k)), None)
        if hit is None and not cfg["declip"]:  # stage A is only the input gain: instant
            k = keys[0]
            opt = settings_from_cfg(session.stage_cfg_a(cfg, False))
            self.lru[k] = session.stage_a(song, opt, k[0], k[1], np, None, {}), 0.0
            hit = k, k
        if hit is not None:
            self._keep_a(keys[:keys.index(hit[1])])
            self.lru.move_to_end(hit[0])
            return self._stage_b(cfg, *hit)
        self._keep_a(keys)
        if "A" in self._jobs:
            return  # on its way: the badge shows its progress
        held = self.dragging or self.wv._rg or self.wv._ovreg is not None  # a handle or the region
        if not submit or held or not (force or self.auto.get() and est <= 10 and akey != self._stopped):
            return self._badge("stale · P")
        self._stopped, self._jobs["A"] = None, akey
        self.n_a += 1
        self.heavy.submit("A", (self.sgen, akey), self._job_a, song, cfg, akey, self._ekind(cfg, akey[3]),
                          self._sh(cfg))
        self._badge("computing 0 % · Esc", THRESH)

    def _keep_a(self, keys=()):
        """Cancel the stage-A job in flight unless it serves one of keys."""
        job = self._jobs.get("A")
        if job is not None and not any(_covers(job, k) for k in keys):
            self.heavy.cancel("A")
            del self._jobs["A"]

    def _stage_b(self, cfg, c, akey):
        """B from the stage-A result c, sliced to akey's span: unlimit, uncompress and output; instant once
        their gains are here. Kept for the region: B, P (B before the limiter), what the limiter cuts, the
        uncompress gain at 100 % (or the key of the one it lacks) and what the stage input's levels need."""
        song, (r0, r1) = self.wv.song, self.wv.region
        S, E, _, fast = akey
        D, secs = self.lru[c]
        D, h, ucg, uk, lv = D[S - c[0]:E - c[0]], {}, None, None, None
        if cfg["unlimit"] and cfg["unlimit_amount"] > 0:
            if self.unl(cfg) is None:
                return self._badge(f"waiting for the unlimit curve {100 * self._gp:.0f} %", THRESH, "G")
            h["unlimit_g"] = self.G[1][S:E]
            h["unlimit_ref"] = song.x[S:E] * 10 ** (cfg["input_gain"] / 20)  # stage A's input: the gain applies to it
        if cfg["uncompress"]:  # its gain also at Amount 0, for the gain lane
            uk = c, akey, session.keyof(cfg, session.UC_KEYS - session.A_KEYS)
            if self.ucg is None or self.ucg[0] != uk:  # one in flight; the latest settings follow when it lands
                if "uc" not in self._jobs:
                    self._jobs["uc"] = uk
                    self.light.submit("uc", (self.sgen, uk), self._job_uc, cfg, uk, D, h.get("unlimit_g"),
                                      h.get("unlimit_ref"), song.sr)
                if cfg["uc_amount"] > 0:
                    return self._badge("computing the uncompress gain", THRESH)
            else:
                ucg = h["uc_gain"] = self.ucg[1]
            g, ref = h.get("unlimit_g"), h.get("unlimit_ref")  # the stage input's levels, for the Uncompress page
            lv = ((c, akey, (r0, r1), session.keyof(cfg, {"unlimit", "unlimit_amount", "uc_link", "input_gain"})),
                  D[r0 - S:r1 - S], None if g is None else g[r0 - S:r1 - S], cfg["unlimit_amount"] / 100,
                  cfg["uc_link"], None if ref is None else ref[r0 - S:r1 - S])
        opt = session.stage_cfg_b(cfg)
        P = session.stage_b(dict(opt, limiter=False), D, song.sr, h)
        B, P = session.limit(P, song.sr, opt)[r0 - S:r1 - S], P[r0 - S:r1 - S]
        cut = None  # what the limiter cuts: (share of the region's samples, up to dB)
        if cfg["limiter"] and P.size:
            lim, a = 10 ** (cfg["limiter_peak"] / 20), np.abs(P)
            cut = float(np.mean(a > lim)), 20 * np.log10(max(float(a.max()), 1e-9) / lim)
        self.prev = dict(B=B, P=P, ucg=None if ucg is None else ucg[r0 - S:r1 - S], uk=uk if ucg is None else None,
                         lv=lv, cut=cut, r=(r0, r1), bkey=session.keyof(cfg, session.B_KEYS), fast=fast, secs=secs,
                         a=c, prov=self._prov(self.cfg))
        self._marks()
        self._fresh()

    def _need_g(self, cfg):
        """The unlimit curve for these settings: a heavy job, unless it is here or on its way."""
        gk = session.keyof(cfg, session.G_KEYS)
        if (self.G is None or self.G[0] != gk) and self._jobs.get("G") != gk:
            self.G, self._gp, self._jobs["G"] = None, 0.0, gk
            self.heavy.submit("G", (self.sgen, gk), self._job_g, self.wv.song, cfg, gk, prio=2)

    def _ticker(self, ev, kind, key):
        """A job's progress callback: stops the job once it is cancelled; posts whole percents."""
        last = [-1]

        def tick(f, *_):
            if ev.is_set():
                raise session.Cancelled
            if int(100 * f) != last[0]:
                last[0] = int(100 * f)
                self.q.put(("job_p", kind, key, f))
        return tick

    def _job_a(self, ev, song, cfg, akey, ek, sh):  # the heavy worker: no Tk calls
        S, E, _, fast = akey
        t0, opt = time.time(), session.stage_cfg_a(cfg, fast)
        xp, dev = _backend(opt)
        D = session.stage_a(song, settings_from_cfg(opt), S, E, xp, dev, session.hints_a(song, cfg, S, E, dev, sh),
                            self._ticker(ev, "A", akey))
        return akey, D, time.time() - t0, ek

    def _job_g(self, ev, song, cfg, gk):  # the heavy worker; also the gain lane's pyramid and the peaks in dB
        t0 = time.time()
        G = song.unlimit_raw(cfg, self._ticker(ev, "G", gk))
        secs = time.time() - t0
        return gk, G, secs, Pyramid(G[:, None]), tuple(20 * np.log10([float(G.max()), float(np.quantile(G, 0.999))]))

    def _job_uc(self, ev, cfg, uk, D, g, ref, sr):  # the light worker: the uncompress gain of stage B's input
        h = dict(uc_gain=None) if g is None else dict(uc_gain=None, unlimit_g=g, unlimit_ref=ref)
        session.stage_b(session.stage_cfg_b(dict(cfg, uc_amount=100.0)), D, sr, h,  # the gain at any Amount
                        self._ticker(ev, "uc", uk))
        return uk, h["uc_gain"]

    def _on_preview_job(self, msg, kind, gen, v):
        if msg == "job_p":  # gen: the job's key
            if self._jobs.get(kind) == gen and kind == "A":
                self._badge(f"computing {100 * v:.0f} % · Esc", THRESH)
            elif self._jobs.get(kind) == gen and kind == "G":
                self._gp = v
                if self._wait == "G":
                    self._badge(f"waiting for the unlimit curve {100 * v:.0f} %", THRESH, "G")
                self.stage_txt["Unlimit"][0].config(text=self.unl_text(self.cfg))
                self.wv.redraw(("gain",))
            elif self._jobs.get(kind) == gen and kind == "sh":
                self._shp = v
                self._show_stats()
                if self._wait == "prov":
                    self._fresh()
            return
        if gen[0] != self.sgen:
            return  # another song
        mine = self._jobs.get(kind) == gen[1]  # else it finished as a newer one started
        if mine:
            del self._jobs[kind]
        if msg == "job_err":
            return self._badge("error: " + v, FLAG) if mine else None
        song = self.wv.song
        if kind == "A":
            akey, D, secs, ek = v
            self.lru[akey] = D, secs
            while len(self.lru) > 6:
                self.lru.popitem(last=False)
            if ek:
                self.est.update(ek, (akey[1] - akey[0]) / song.sr, secs)
        elif kind == "G":
            gk, G, secs, pyr, db = v
            self.est.update(("unlimit", self.cuda), len(song.x) / song.sr, secs)
            if self.cfg is not None and gk == session.keyof(self.cfg, session.G_KEYS):
                self.G = gk, G, pyr, db, secs
                self.wv.redraw(("gain",))
        elif kind == "sh":  # the song keeps it: the flags and the preview follow the knee found
            key, _, secs, n = v
            self._shp = None
            if n:
                self.est.update(("knee", key[-1] == "cuda"), self._searched(self.cfg, n), secs)
            if self._det_mode() == "auto":
                self._auto_jobs()
            self._show_stats()
        else:
            self.ucg = v
        if self._pjob is None:  # else the debounce brings it on
            self._preview()

    # ---- audition ----
    MON = (None, 0.0, -6.0, -12.0)  # the Mon button: Auto, then fixed trims in dB
    LOOP_S = 30  # a longer region loops its first 30 s: whole-song buffers would stall the UI

    def toggle_play(self):
        """Space or Play: loop the region (without one, the song from the edit cursor), or stop."""
        pl, wv = self.player, self.wv
        if not pl.available:
            return wv.readout.config(text=session.SD_HINT)
        if pl.playing:
            return self._stop_audio()
        if wv.song is None:
            return
        try:
            if wv.region is not None:
                self._loop(start=True)
            else:
                self._play_song(wv.cursor or 0)
        except Exception as e:  # e.g. sd.PortAudioError; the next Play tries again
            return self._badge(f"Audio: {e}", FLAG)
        self.play_btn.config(text="■")
        self._audio_text()
        if self._ph is not None:
            self.root.after_cancel(self._ph)
        self._playhead()

    def _stop_audio(self):
        self.player.stop()
        self._ctx = self._abuf = None
        self.play_btn.config(text="▶")
        self._audio_text()

    def _b_now(self):
        """The region's B to hear: (B, its input + output gain in dB, its identity, its stage A's): the
        preview shown (also while a newer one is computed), else the render's slice; or Nones."""
        r, p = self.wv.region, self.prev
        if p is not None and p["r"] == r:
            c = dict(p["bkey"])
            return p["B"], c["input_gain"] + c["output_gain"], id(p["B"]), p["a"]
        if self.full is not None and r is not None:
            c = self.rcfg
            return self.full[r[0]:r[1]], c["input_gain"] + c["output_gain"], id(self.full), "full"
        return None, 0.0, None, None

    def _audition(self):
        """While playing: the region loops (from the song when one was just made), with new buffers
        whenever the region, its B, Side, Match or Mon change; a cleared region stops it."""
        if self.player.playing and (self._ctx == "loop" or self._ctx == "song" and self.wv.region is not None):
            self._loop(start=self._ctx == "song")
        self._audio_text()

    def _loop(self, start=False):
        """The region's Orig, Proc (the B shown) and Diff buffers, crossfaded in at the same position (a
        new region from its start). Auto trim: measured again for a new region or stage-A result (or
        Side, Match), else kept unless the new peak would clip."""
        wv, r = self.wv, self.wv.region
        if r is None:
            return self._stop_audio()
        B, gg, bid, aid = self._b_now()
        mon, side, match = self.MON[self.mon_i], self.side.get(), self.match.get()
        key, ev = (r, bid, mon, side, match), (r, aid, side, match)
        if key == self._abuf and not start:
            return
        n = min(r[1] - r[0], int(self.LOOP_S * wv.song.sr))
        bufs, self._mon_db, self._mdb = session.build_buffers(
            wv.song.x[r[0]:r[0] + n], None if B is None else B[:n], wv.song.sr, gg, mon, match, side,
            self._hold if mon is None and ev == self._hold_key else None)
        if mon is None:
            self._hold, self._hold_key = self._mon_db, ev
        if start:
            self.player.play(bufs)
        else:
            self.player.set_bufs(bufs, None if self._abuf[0] == r else 0)
        self._ctx, self._abuf = "loop", key

    def _play_song(self, c):
        """No region: the song from sample c to its end, as it is or (2 Proc) the render while it is
        current; Auto trim from both songs' peaks."""
        song, cfg = self.wv.song, self.cfg
        c = int(min(max(c, 0), len(song.x) - 1))
        cur = self.full_b(cfg)
        mon = self.MON[self.mon_i]
        if mon is None:
            mon = min(0.0, -(20 * np.log10(max(song.peak(), self.full_pk if cur else 0.0, 1e-9)) + 0.3))
        self.player.play({1: song.x[c:, :2], 2: self.full[c:, :2] if cur else None, 3: None}, 0, False,
                         10 ** (mon / 20))
        self._ctx, self._abuf, self._song0, self._mon_db = "song", None, c, mon

    def _source(self, v):
        """1, 2, 3: hear Orig, Proc or Diff, at the same position."""
        if self.player.available:
            self.src.set(v)
            self.player.set_src(v)
            self._audio_text()

    def _toggle(self, var):
        if self.player.available:
            var.set(not var.get())
            self._audition()

    def _cycle_mon(self):
        """Mon: Auto, 0, -6, -12 dB."""
        self.mon_i = (self.mon_i + 1) % len(self.MON)
        if self.player.playing and self._ctx == "song":
            self._play_song(self._song0 + self.player.pos)
        self._audition()

    def _audio_text(self):
        """The Mon button shows the trim applied; the note the match gain, or what plays instead."""
        if not self.player.available:
            return
        mon, d, r = self.MON[self.mon_i], self._mon_db, self.wv.region
        self.mon_btn.config(text=f"Mon {mon:g} dB" if mon is not None else
                            "Mon Auto" if d is None else f"Mon Auto {d + 0.0:.1f} dB")
        note = ""
        if self._ctx == "song":  # whole-song buffers: no Diff, Side or Match (they need a region)
            note = "no region: the song from the cursor" + (
                " · Orig plays (no current render)" if self.src.get() == 2 and self.player.bufs[2] is None else
                " · Orig plays (Diff needs a region)" if self.src.get() == 3 else "") + (
                " · Side, Match need a region" if self.side.get() or self.match.get() else "")
        elif self.src.get() > 1 and r is not None and self._b_now()[0] is None:
            note = "B not ready: Orig"
        elif self._ctx == "loop" and r is not None and len(self.player.bufs[1]) < r[1] - r[0]:
            note = f"loops the first {self.LOOP_S} s"
        elif self.match.get() and self._ctx == "loop" and self._mdb is not None:
            note = f"B {self._mdb + 0.0:.1f} dB matched"
        self.anote.config(text=note)

    def cursor_set(self, t):
        """A click on the lanes: while playing, the loop goes on from there (inside the region), or the
        song from there."""
        r, pl = self.wv.region, self.player
        if pl.playing and self._ctx == "loop" and r is not None and 0 <= t - r[0] < len(pl.bufs[1]):
            pl.seek(t - r[0])
        elif pl.playing and self._ctx == "song":
            self._play_song(t)

    def _playhead(self):
        """Every 33 ms while playing: the playhead, paging the view along a song; after a stop, the
        stream is stopped once the fade-out has played."""
        pl, wv = self.player, self.wv
        p, self._ph = pl.position(), None
        if p is None:  # stopped, or the song's end
            wv.set_playhead(None)
            if self._ctx is not None:
                self._stop_audio()
            if pl.stream is not None:  # a Play before it cancels it
                self._ph = self.root.after(int(1000 * pl.stream.latency) + 100, pl.idle)
            return
        t = self._song0 + p if self._ctx == "song" else (wv.region or (0,))[0] + p
        wv.set_playhead(t, page=self._ctx == "song")
        self._ph = self.root.after(33, self._playhead)

    # ---- render ----
    def run(self):
        if self.run_btn.instate(["disabled"]):  # a render is running (Ctrl+Return gets here too)
            return
        inp = self.inp.get()
        if not inp or not os.path.isfile(inp):
            self.status.config(text="Pick a valid input file.")
            return
        if inp != self.path and not self.load_input(inp):
            return
        try:
            cfg = self.collect()
        except ValueError as e:
            self.status.config(text=str(e))
            return
        outp = self.outp.get() or os.path.splitext(inp)[0] + "_unclipped.wav"
        self.outp.set(outp)
        self.wv.set_output(None, 0)
        self.heavy.cancel(); self.heavy.pause(True)  # the render has the GPU; previews resume after it
        for kind in ("A", "G", "sh"):
            self._jobs.pop(kind, None)
        self._shp, hints = None, self._render_hints(cfg)
        self.rendering, self.rcfg, self.full, self._rgen = True, cfg, None, self.sgen
        self.rkey = session.keyof(cfg, session.B_KEYS)  # the drawn output is stale once the settings differ
        self._preview()
        self.cancel.clear()
        self.run_btn.config(state="disabled"); self.cancel_btn.config(state="normal")
        self.t0, self.frac = time.time(), 0.0
        self.note = "   ·   " + backend_name(*spade_backend(cfg["gpu"]), settings_from_cfg(cfg))
        if cfg["process_rate"] > 0 and cfg["process_rate"] != self.sr:
            self.note += f"   ·   at {cfg['process_rate']} Hz ({cfg['process_rate'] / self.sr:.2f}× speed)"
        self.status.config(text="Starting" + self.note)
        threading.Thread(target=self._work, args=(self.x, self.sr, outp, cfg, self.log.get(), hints),
                         daemon=True).start()

    def _render_hints(self, cfg):
        """The GUI's analysis for the render (NMF auto detection), when it is the one the render would
        make: this song's ceilings at these levels, with the shoulder searched on the render's device, or
        a knee set (no search). Else None: the render analyses as the CLI does."""
        song = self.wv.song
        if song is None or self._det_mode(cfg) != "auto" or song.peek("ceil", cfg, session.G_KEYS) is None:
            return None
        sh = self._sh(cfg)
        return None if sh is None and not cfg["nmf_knee"] else {"nmf_analysis": copy.deepcopy(song.analysis(cfg, sh))}

    def _work(self, x, sr, outp, cfg, log_on, hints=None):  # a worker thread: no Tk calls
        def progress(frac, done, out):
            if self.cancel.is_set():
                raise _Cancelled
            self.q.put(("p", frac, done, out))
        t0 = time.time()
        log = [f"SharpUnClip GUI {time.strftime('%Y-%m-%d %H:%M:%S')}",
               f"input={self.path} output={outp} sr={sr} samples={len(x)} channels={x.shape[1]}",
               f"backend={backend_name(*spade_backend(cfg.get('gpu')), settings_from_cfg(cfg))}",
               "settings " + " ".join(f"{k}={v}" for k, v in sorted(cfg.items()))]
        an = (hints or {}).get("nmf_analysis")
        if an is not None:
            k = cfg["nmf_knee"] or an["shoulder"]["knee"]
            log.append(f"analysis from GUI kinds={','.join(r['kind'] for row in an['refs'] for r in row)} "
                       f"knee={'none' if k is None else f'{k:.2f}'}")
        info = {}
        try:
            with session.GPU_LOCK:  # a preview job still running stops at its next tick
                out = compute(x, sr, cfg, progress, hints, info)
            sf.write(outp, out, sr, subtype="FLOAT")
            log.append(f"done peak={np.abs(out).max():.3f} elapsed={time.time() - t0:.1f}s")
            self.q.put(("done", out, outp, info))
        except _Cancelled:
            log.append(f"cancelled after {time.time() - t0:.1f}s")
            self.q.put(("cancelled",))
        except Exception as e:
            log.append("error:\n" + traceback.format_exc())
            self.q.put(("error", str(e)))
        if log_on:
            try:
                _write_log(os.path.splitext(outp)[0] + ".log", log + [""])
            except OSError:
                pass

    def _poll(self):
        msgs = []
        try:
            while True:
                msgs.append(self.q.get_nowait())
        except queue.Empty:
            pass
        render = [m for m in msgs if not m[0].startswith("job")]
        for m in msgs:
            if m[0] == "p":
                self.frac = m[1]
                if m[3] is not None:
                    self.wv.set_output(m[3], m[2])
            elif m[0].startswith("job"):
                self._on_job(*m)
            else:
                self._finished(m)
        if render and render[-1][0] == "p":
            self._progress_text()
        self.root.after(80, self._poll)

    def _progress_text(self):
        el = time.time() - self.t0
        dur = len(self.x) / self.sr
        parts = [f"Rendering {self.frac:4.0%}", f"{mmss(el)} elapsed"]
        if self.frac > 0.02:
            parts += [f"{mmss(el * (1 - self.frac) / self.frac)} left",
                      f"{self.frac * dur / max(el, 1e-6):.1f}× realtime"]
        self.bar["value"] = self.frac
        self.status.config(text="   ·   ".join(parts) + self.note)

    def _finished(self, m):
        self.run_btn.config(state="normal"); self.cancel_btn.config(state="disabled")
        el = time.time() - self.t0
        if m[0] == "done":
            self.wv.set_output(m[1], len(m[1]))
            self.bar["value"] = 1.0
            pk = float(np.abs(m[1]).max())
            self.status.config(text=f"Done in {mmss(el)} ({len(self.x) / self.sr / max(el, 1e-6):.1f}× realtime)"
                                    f"   ·   peak {pk:.2f} ({20 * np.log10(max(pk, 1e-9)):+.1f} dBFS)"
                                    f"   ·   {os.path.basename(m[2])}{self.note}")
            if self._rgen == self.sgen:  # B = this render for any region while the settings match
                self.full, self.full_pk = m[1], pk
                c = self.rcfg
                self.full_cut = m[3].get("limited", 0.0)
                if c["declip"] and not (c["unlimit"] or c["uncompress"]):
                    self.est.update(self._ekind(c, False), len(m[1]) / self.sr, el)
        elif m[0] == "cancelled":
            self.status.config(text="Cancelled. Nothing was written.")
        else:
            self.status.config(text="Error: " + m[1])
        self.rendering = False
        self.heavy.pause(False)
        if self.cfg is not None and self.wv.song is not None and self._det_mode() == "auto":
            self._auto_jobs()  # a shoulder search the render cut short may have finished
            self._need_sh()
        self._preview()


def _selfcheck():
    sr = 48000
    t = np.arange(2 * sr) / sr
    rng = np.random.default_rng(1)
    k = 6 * np.exp(-t * 3) * np.sin(2 * np.pi * 55 * t)
    b = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 660 * t) + 0.03 * rng.standard_normal(t.size)
    src = 1.4 * (k + b) / np.max(np.abs(k + b))
    cl = np.clip(src, -0.9, 0.9)
    snr = lambda r: 20 * np.log10(np.linalg.norm(src) / np.linalg.norm(r - src))
    for name, cfg in PRESETS.items():
        seen = []
        out = compute(cl, sr, cfg, lambda f, d, o: seen.append((f, d)))
        assert out.shape == cl.shape, (name, out.shape)
        assert not np.any(np.isnan(out)), name
        assert np.max(np.abs(out)) > 0.95, f"{name}: peaks not restored"
        assert seen and seen[-1] == (1.0, len(cl)), f"{name}: progress {seen[-1:]}"
        print(f"{name:20s} SNR {snr(cl):.2f} -> {snr(out):.2f} dB  peak {np.max(np.abs(out)):.3f}")

    def envelope(x, cols):  # the old waveform's (top, bottom) per pixel column
        x = np.asarray(x)
        x = x[:, None] if x.ndim == 1 else x
        edges = np.linspace(0, len(x), max(1, min(cols, len(x))) + 1).astype(int)[:-1]
        return np.maximum.reduceat(x.max(1), edges), np.minimum.reduceat(x.min(1), edges)
    f32 = lambda a: a.astype(np.float32)
    for n, cols in ((4096 * 50, 50), (256 * 300, 300), (40 * 500, 500), (123457, 777), (5000, 3000)):
        xr = rng.standard_normal((n, 2))
        pyr = Pyramid(xr)
        mn, mx = pyr.minmax(xr, 0, n, cols)
        for c in range(2):
            top, bot = envelope(xr[:, c], cols)
            if n % cols == 0:  # column edges on block edges (or under 64 samples a column)
                assert np.array_equal(mx[:, c], f32(top)) and np.array_equal(mn[:, c], f32(bot)), (n, cols)
            assert np.all(mx[:, c] >= f32(top)) and np.all(mn[:, c] <= f32(bot)), (n, cols)
        t0, t1 = 0.37 * n, 0.81 * n  # a zoomed view: still contains every column's samples
        e = np.linspace(t0, t1, 301).astype(int)
        mn, mx = pyr.minmax(xr, t0, t1, 300)
        assert np.all(mx >= f32(np.maximum.reduceat(xr[:e[-1]], e[:-1]))), (n, cols)
        assert np.all(mn <= f32(np.minimum.reduceat(xr[:e[-1]], e[:-1]))), (n, cols)
        i0, i1 = sorted(rng.integers(0, n, 2))
        xr[i0:i1] = 3 * rng.standard_normal((i1 - i0, 2))
        pyr.update(xr, i0, i1)
        ref = Pyramid(xr)
        assert all(np.array_equal(a, b) for a, b in zip(pyr.mn + pyr.mx, ref.mn + ref.mx)), (n, i0, i1)
    xr = rng.standard_normal((100, 1))  # 16 px per sample: each column holds the sample under it
    mn, mx = Pyramid(xr).minmax(xr, 0, 100, 1600)
    assert np.array_equal(mn, mx) and np.array_equal(mn[::16], f32(xr))
    # (j) Song.masks is _clip_mask per channel for every manual preset; (k) the nines histogram's
    # % rebuilt is within 0.5 points of it (Remove glitches off)
    from sharpunclip import _clip_mask
    xs = np.stack([cl, 0.6 * cl + 0.01 * rng.standard_normal(len(cl))], 1)
    song = Song(xs, sr, None)
    manual = [name for name, p in PRESETS.items() if p["engine"] == "SPADE" or not p["auto_detect"]]
    assert len(manual) == 4
    for name in manual:
        for cfg in (PRESETS[name], dict(PRESETS[name], process_rate=24000.0, input_gain=-3.0)):
            y0 = xs * 10 ** (cfg["input_gain"] / 20)
            for c, m in enumerate(song.masks(cfg)):
                assert np.array_equal(m, _clip_mask(y0[:, c], session.rate_p(cfg, sr), cfg)), (name, c)
        cfg = dict(PRESETS[name], remove_glitch=False)
        h = (cfg["clip_above"] if cfg["dynamic"] else min(cfg["clip_above"], cfg["not_clip_below"])) / 100
        est = sum(session.hist_level(hc, min(h, 0.92) if cfg["soft"] else h) for hc in song.hist9(cfg)) / xs.size
        assert abs(est - np.mean(song.masks(cfg))) < 0.005, (name, est)
    q = 0.05 * np.sin(2 * np.pi * 220 * np.arange(4 * sr) / sr)  # (l) a lone clipped burst at 1.2 s
    i = int(1.2 * sr)
    q[i:i + 2400] = np.clip(3 * np.sin(2 * np.pi * 60 * np.arange(2400) / sr), -0.9, 0.9)
    hot = session.hotspots(Song(np.stack([q, q], 1), sr, None).stats(PRESETS["Accurate SPADE"]).block_counts,
                           session.BLOCK, sr, 2.0)
    assert hot and hot[0][0] <= i and i + 2400 <= hot[0][1] and abs(sum(hot[0]) / 2 - i - 1200) < 0.1 * sr, hot
    # audition offline, the callback called directly: (t) a Player starts without a stream; (u) a switch of
    # source crossfades: no step over the signal's own largest (a 1-sample fade, a hard switch, jumps);
    # (v) the loop seam fades in and out; (w) Diff = B - A x g, mono doubled, Side, Match and Auto trim
    assert session.Player(sr).stream is None
    sine = lambda n, f=440: 0.5 * np.sin(2 * np.pi * f * np.arange(n) / sr + 1.0)[:, None] * [1.0, 0.8]
    A = sine(sr).astype(np.float32)
    own = np.abs(np.diff(A, axis=0)).max()

    def hear(p, srcs):  # one 2048-frame callback per entry, after switching to that source (None: no switch)
        out, got = np.zeros((2048, 2), np.float32), []
        for s in srcs:
            if s:
                p.set_src(s)
            p._cb(out, 2048, None, None)
            got.append(out.copy())
        return np.concatenate(got)
    for nxf in (None, 1):
        pl = session.Player(sr)
        pl.bufs, pl.playing, pl.nxf = {1: A, 2: None, 3: np.zeros_like(A)}, True, nxf or pl.nxf
        step = np.abs(np.diff(hear(pl, (None, 3, None, 1, None)), axis=0)).max()
        assert (step <= own + 1e-7) == (nxf is None), (nxf, step, own)
    pl = session.Player(sr)  # two changes within one block: the fade is still from what was heard (1)
    pl.bufs, pl.playing = {1: A, 2: sine(sr, 660).astype(np.float32), 3: np.zeros_like(A)}, True
    heard = hear(pl, (None,)); pl.set_src(2)
    assert np.abs(np.diff(np.concatenate([heard, hear(pl, (3, None))]), axis=0)).max() <= own + 1e-7
    n, nf = int(0.3 * sr) + 50, round(0.005 * sr)  # not whole periods: the seam would jump 0.7
    Aw = sine(n)
    a1 = session.build_buffers(Aw, None, sr, 0.0, 0.0)[0][1]
    w = (0.5 - 0.5 * np.cos(np.pi * np.arange(nf) / nf))[:, None]
    assert np.array_equal(a1[nf:-nf], Aw[nf:-nf].astype(np.float32)) and not a1[0].any() and not a1[-1].any()
    assert np.allclose(a1[:nf], Aw[:nf] * w, atol=1e-7) and np.allclose(a1[-nf:], Aw[-nf:] * w[::-1], atol=1e-7)
    pl = session.Player(sr)
    pl.bufs, pl.pos, pl.playing = {1: a1}, n - 1000, True
    assert np.abs(np.diff(hear(pl, (None,)), axis=0)).max() <= own + 1e-7 < 0.1 * abs(Aw[-1, 0] - Aw[0, 0])
    Bw, gg = 1.3 * sine(n, 660) + 0.7 * Aw, -1.5
    bufs = session.build_buffers(Aw, Bw, sr, gg, 0.0)[0]
    assert np.allclose(bufs[3][nf:-nf], (Bw - Aw * 10 ** (gg / 20))[nf:-nf], atol=1e-6)
    bufs, g, mdb = session.build_buffers(Aw[:, :1], 2 * Bw[:, :1], sr, gg, None)  # B over 0 dBFS: Auto trims
    assert all(np.array_equal(v[:, 0], v[:, 1]) for v in bufs.values()) and g < -5 and mdb == 0
    assert abs(max(np.abs(bufs[1]).max(), np.abs(bufs[2]).max()) - 10 ** (-0.3 / 20)) < 1e-6
    assert session.build_buffers(Aw[:, :1], 2 * Bw[:, :1], sr, gg, None, hold=g - 2)[1] == g - 2
    assert session.build_buffers(Aw[:, :1], 2 * Bw[:, :1], sr, gg, None, hold=g + 2)[1] == g  # it would clip
    bufs, g, mdb = session.build_buffers(Aw, Bw, sr, gg, 0.0, match=True)
    mid = lambda v: np.sqrt(np.mean(np.square(v[nf:-nf], dtype=np.float64)))
    assert abs(mid(bufs[2]) / mid(bufs[1]) - 1) < 0.01 and mdb < -1, mdb
    bufs = session.build_buffers(np.c_[Aw, Aw[:, :1]], np.c_[Bw, Bw[:, :1]], sr, gg, 0.0, side=True)[0]
    assert np.allclose(bufs[1][nf:-nf], ((Aw[:, :1] - Aw[:, 1:]) / 2)[nf:-nf], atol=1e-7), "Side, first two of 3"
    try:
        import tkinter as tk_
        root = tk_.Tk(); root.withdraw()
    except Exception:  # no display
        print("selfcheck OK (no display: GUI fields not checked)")
        return
    _tk()
    from types import SimpleNamespace as E

    class _Stream:  # stands in for sounddevice's: the selfcheck never opens an audio stream
        latency, active = 0.0, False

        def __init__(self, **kw):
            self.kw = kw

        def start(self):
            self.active = True

        def abort(self):
            self.active = False
        close = abort
    real_sd, session.sd = session.sd, E(OutputStream=_Stream)
    app = App(root)
    app.auto.set(False)  # region previews only where (q)-(s) ask for them
    import waveview
    hp, z = app.hist, np.zeros(app.hist.X1 - app.hist.X0)  # NMF auto: a bar half rebuilt is half red; axis labels fit
    red, grey = z.copy(), z.copy()
    red[100], grey[100] = 500, 500
    hp.show_auto(red, z, grey, -9.0, [1.0], 0.9, 1.0)
    col = [hp.photo.get(hp.X0 + 100, y) for y in range(hp.Y0, hp.Y1)]
    nr, ng = col.count(tuple(waveview._rgb(waveview.FLAG))), col.count(tuple(waveview._rgb(waveview.WAVE_IN)))
    assert nr and abs(nr - ng) <= 1 and col.index(tuple(waveview._rgb(waveview.FLAG))) < col.index(
        tuple(waveview._rgb(waveview.WAVE_IN))), (nr, ng)
    assert all(hp.bbox(i)[0] >= 0 for i in hp.find_withtag("deco") if hp.type(i) == "text")
    for name in PRESETS:
        app.apply_cfg(PRESETS[name])
        cfg = app.collect()  # raises if any box is empty or not a number
        assert set(DEFAULTS) <= set(cfg), (name, set(DEFAULTS) - set(cfg))
        assert settings_from_cfg(cfg) == dict(DEFAULTS, **PRESETS[name]), name
    import tempfile
    allv = {**app.V, **app.T, **app.C}
    assert len(allv) == len(app.V) + len(app.T) + len(app.C)  # one variable per setting
    for key, ws in app.bound.items():  # (d) every widget of a setting is bound to its one variable
        for w in ws:
            opt = "textvariable" if isinstance(w, (ttk.Spinbox, ttk.Combobox)) else "variable"
            assert str(w.cget(opt)) == str(allv[key]), (key, w)
    for key in ("lookahead", "input_gain", "output_gain", "limiter", "limiter_peak", "declip", "unlimit",
                "uncompress"):
        assert len({str(w) for w in app.bound[key]}) >= 2, key  # page/chip and Advanced share it
    keys = set(app.collect())
    assert set(DEFAULTS) <= set(allv) and keys == set(DEFAULTS) | {"gpu", "threads"}, keys  # (e)
    assert session.B_KEYS | session.IGNORED == keys  # (f) every setting is in a cache group or ignored
    ncb = [w for k, w in app.grey if k == "not_clip_below"]  # (g)
    for dyn in (1, 0):
        app.apply_preset("Accurate SPADE"); app.T["dynamic"].set(dyn); root.update()
        assert all(w.instate(["disabled"]) == bool(dyn) for w in ncb), dyn
    for name in PRESETS:  # (h)
        app.apply_preset(name); root.update()
        assert app.mod.cget("text") == "", name
    app._set("clip_above", 93); root.update()
    assert app.mod.cget("text") == "modified"
    app._set("clip_above", 1e9); assert app.V["clip_above"].get() == "100"
    app.V["clip_above"].set("99.9."); root.update()
    assert app.mod.cget("text") == "invalid value" and "Clip above" in app.wv.readout.cget("text")
    app.apply_preset("SharpUnClip"); root.update()
    assert app.mod.cget("text") == "" and app.chips["Declip"][0].cget("text") == "SPADE 78% · q3 · 1024"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "mine.json")
        with open(p, "w") as f:
            json.dump(dict(PRESETS["Unmaster"], clip_above=80.0), f)
        app.load_preset(p); root.update()
        assert app.mod.cget("text") == "from mine" and app.preset.get() == "SharpUnClip"
        app.T["soft"].set(1); root.update()
        assert app.mod.cget("text") == "modified"
    app.apply_preset("Accurate"); root.update()
    for page in PAGES:  # chips select pages; Ctrl+1..4 do the same
        app._select_page(page[0])
        assert app.pages[page[0]].winfo_manager() and sum(bool(p.winfo_manager()) for p in app.pages.values()) == 1
    # (i) at the minimum size the 60 px shrink comes out of the body, which still fits the tallest page
    need = max(p.winfo_reqheight() for p in app.pages.values())
    assert root.minsize()[1] == min(root.winfo_reqheight() - 60, root.winfo_screenheight() - 80)
    assert need <= app.body.winfo_reqheight() - 60
    app.outp.set("C:/Users/x/Desktop/old_unclipped.wav")  # new input keeps the output folder
    assert app.default_out("F:/rips/song.flac") == os.path.normpath("C:/Users/x/Desktop/song_unclipped.wav")
    app.outp.set("")
    assert app.default_out("F:/rips/song.flac") == os.path.normpath("F:/rips/song_unclipped.wav")
    wv = app.wv
    wv.redraw(); wv._draw()  # withdrawn root: the W < 20 early return
    x2, full = np.stack([cl, 0.5 * cl], 1), np.stack([out, 0.5 * out], 1)
    wv.set_song(Song(x2, sr, None))
    o2 = x2.copy()  # a render's array, final up to done
    o2[:len(o2) // 4] = full[:len(o2) // 4]
    wv.set_output(o2, len(o2) // 4)
    wv._draw()
    rgb = wv._lanes_rgb(waveview.GUT + 400, 200)
    green = (rgb == waveview._rgb(waveview.ACCENT)).all(2).any(0)
    assert rgb.shape == (200, 400, 3) and green[:99].all() and not green[101:].any()  # output up to done only
    o2[:len(o2) // 2] = full[:len(o2) // 2]
    wv.set_output(o2, len(o2) // 2)  # same array, more of it final: only that part is re-read
    ref = Pyramid(o2)
    assert all(np.array_equal(a, b) for a, b in zip(wv.pyr_out.mn + wv.pyr_out.mx, ref.mn + ref.mx))
    wv._put(wv.lanes, rgb)
    rgb2 = rgb.copy(); rgb2[10:20, 30:40] = (1, 2, 3)
    wv._put(wv.lanes, rgb2)  # one band
    assert wv.photo[wv.lanes].get(35, 15) == (1, 2, 3) and wv.photo[wv.lanes].get(50, 15) == tuple(rgb[15, 50])
    for gdb in ("10000", "-inf", "nan", "0"):  # typed far outside the field's range: still drawable
        app.V["input_gain"].set(gdb); wv._draw()
        assert np.isfinite(wv.gain) and wv.gain > 0 and np.isfinite(wv._pk()), gdb
    for ch in (1, 2, 3, 10):  # lanes: more than 2 channels only at 60 px or more each
        wv.song = Song(np.zeros((100, ch)), sr, None)
        for H in (100, 130, 300, 700):
            nl = wv._nl(H)
            assert nl == min(ch, 2) or (nl <= min(ch, 8) and H // nl >= 60), (ch, H, nl)
    wv.set_song(None)
    for mode in ("lin", "db"):
        sc = waveview.Scale()
        sc.mode, sc.lo, sc.hi, sc.top_db = mode, -1.5, 1.5, 3.5
        v = np.array([1.2, 0.5, 0.05, -0.3, -1.4])
        assert np.allclose(sc.v(sc.y(v, 200), 200), v), mode
    # the burst through the App: the light worker's results place the region on it, and the
    # overlays, histogram, stats and region edits draw and behave
    def settle():  # until the workers are idle (a paused one waits) and their results are shown
        end = time.time() + 60
        root.update()
        while (app.light.jobs or app.light.running or app.heavy.running or app.heavy.jobs and not app.heavy.paused
               or app._pjob or not app.q.empty()) and time.time() < end:
            root.update(); time.sleep(0.01)
        root.update()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "burst.wav")
        sf.write(p, np.stack([q, q], 1), sr, subtype="FLOAT")
        app.apply_preset("Accurate SPADE"); app.gpu.set(False); root.update()
        assert app.load_input(p)
        settle()
    r = wv.region
    assert r and r[0] <= i and i + 2400 <= r[1] and not app.region_user_set, r
    assert wv.readout.cget("text").startswith("Region set to the most-clipped spot")
    assert app.rebuilt.cget("text").endswith("% rebuilt") and "runs" in app.stat[1].cget("text")
    W = waveview.GUT + 600
    wv.lanes.winfo_width = wv.ruler.winfo_width = wv.ov.winfo_width = lambda: W  # never laid out: withdrawn
    wv.lanes.winfo_height, wv.ov.winfo_height = (lambda: 300), (lambda: 50)
    thresh = lambda: [k for k in wv.lanes.find_withtag("deco") if wv.lanes.itemcget(k, "fill") == waveview.THRESH]
    for page, tip in (("Declip", waveview.FLAG), ("Output", waveview._mix(waveview.WAVE_IN, waveview.FLAG, 0.35))):
        app._select_page(page); wv._draw()
        assert (wv._img[wv.lanes] == waveview._rgb(tip)).all(2).any(), page  # rebuilt tips (35 % on other pages)
        assert bool(thresh()) == (page == "Declip"), page  # lines only on the Declip page
    app._select_page("Declip")
    app.hist._motion(E(x=200, y=50))
    assert "samples" in app.hist.itemcget("title", "text")
    G = waveview.GUT
    wv._r_press(E(x=G + 100, y=40), False); wv._r_move(E(x=G + 400, y=40)); wv._r_release(E(x=G + 400, y=40))
    assert app.region_user_set and wv.region == (wv._ts(G + 100), wv._ts(G + 400))
    wv._r_press(E(x=G + 200, y=40), False); wv._r_release(E(x=G + 201, y=40))  # a click: the edit cursor
    assert wv.cursor == wv._ts(G + 200) and wv.region == (wv._ts(G + 100), wv._ts(G + 400))
    r = wv.region  # the ruler drags it along
    wv._r_press(E(x=G + 250, y=18), True); wv._r_move(E(x=G + 300, y=18)); wv._r_release(E(x=G + 300, y=18))
    assert wv.region[1] - wv.region[0] == r[1] - r[0] and wv.region[0] == r[0] + wv._ts(G + 300) - wv._ts(G + 250)
    x0, x1 = (round(wv._x(t)) for t in wv.region)  # the right edge dragged onto the left: 0.25 s at least
    wv._r_press(E(x=x1, y=40), False); wv._r_move(E(x=x0 - 20, y=40))
    assert wv.region[1] - wv.region[0] == 1, wv.region  # never empty, also mid-drag
    wv._r_release(E(x=x0 - 20, y=40))
    assert wv.region[1] - wv.region[0] == sr // 4, wv.region
    app._jump_longest()  # centred on the longest flagged run
    assert wv.region[0] <= i + 1200 <= wv.region[1] and wv.region[1] - wv.region[0] == int(app._rlen() * sr)
    app.T["diff_clips"].set(0); app.T["dynamic"].set(0); app.T["soft"].set(1); settle()  # blocks, both lines
    assert wv.det["gaps"] is None and app.hist.cols is not None
    for mode in ("db", "lin"):
        wv.set_mode(mode); wv._draw()
    # handles: (m) value_at(y(value)) round-trips within one step, in both views
    app.apply_preset("Accurate SPADE"); app.T["dynamic"].set(0); app._commit(); settle()
    wv.fit(); wv._draw()
    assert [h.key for h in wv.handles] == ["clip_above", "max_restore", "not_clip_below", "not_clip_amp",
                                           "trans_amp"], [h.key for h in wv.handles]
    hl = wv._lane(0)[1]
    for mode in ("db", "lin"):
        wv.set_mode(mode); wv.grow = 4.0; wv._fit_scale()  # the cap's line in view too
        for h in wv.handles:
            for c in range(2):
                y = float(wv.scale.y(wv._ref(h, c, G + 300) * h.lev(h.value), hl))
                assert abs(wv.value_at(h, c, G + 300, y) - h.value) <= h.step, (mode, h.key, c)
    wv.grow = 0.0
    # (n) a drag changes only its setting; undo restores it, redo re-applies it
    for key, v in (("clip_above", 95.0), ("max_restore", 3.0), ("not_clip_below", 90.0), ("not_clip_amp", -20.0),
                   ("trans_amp", -50.0)):
        wv._draw()
        before = settings_from_cfg(app.collect())
        wv.drag_to(key, v); wv.release(); root.update()
        after = settings_from_cfg(app.collect())
        assert {k for k in after if after[k] != before[k]} == {key} and after[key] == v, (key, after[key])
        app.undo(); root.update()
        assert settings_from_cfg(app.collect()) == before, key
        app.redo(); root.update()
        assert settings_from_cfg(app.collect()) == after, key
        app.undo(); root.update()
    # (o) the cap snaps to 2.5 within 3 px, to the nearest snap with Ctrl, and 12 px over its lane is no cap
    for v, kw, want in ((2.2, dict(state=4), "2"), (2.47, {}, "2.5"), (None, dict(y=-13), "0")):
        wv._draw()
        wv.drag_to("max_restore", v, **kw); wv.release(); root.update()
        assert app.V["max_restore"].get() == want and wv.grow == 0, (v, app.V["max_restore"].get())
    # (o2) a Shift (fine) drag moves a line off its snap; in dB view a plain drag reaches 90 % (no magnet)
    app.apply_preset("Accurate SPADE"); root.update()
    wv._draw(); wv.drag_to("clip_above", 90.0, state=1); wv.release(); root.update()
    assert float(app.V["clip_above"].get()) < 99.9, app.V["clip_above"].get()
    app.apply_preset("Accurate SPADE"); root.update()
    wv.set_mode("db"); wv._draw(); wv.drag_to("clip_above", 90.0); wv.release(); root.update()
    assert abs(float(app.V["clip_above"].get()) - 90.0) <= 0.1, app.V["clip_above"].get()
    wv.set_mode("lin")
    # (p) the frame bracket's edge dragged to 2048 samples: Frame 2048 in the shared variable
    app.apply_preset("Accurate SPADE"); root.update()
    wv._view(i - 4000, 16000); wv._draw()  # 512 samples = 19 px
    wv._motion(E(x=G + 300, y=40))
    c, f0, w, a, ro = wv.brk
    assert w == 512 and not ro and wv._risk(2, wv._e()) is not None
    wv._press(E(x=wv._x(f0 + w), y=40))
    wv._move(E(x=wv._x(f0 + 2048), y=40, state=0)); wv.release(); root.update()
    spin = [w for w in app.bound["lookahead"] if isinstance(w, ttk.Spinbox)]
    assert app.V["lookahead"].get() == "2" and len(spin) == 1 and spin[0].get() == "2"
    wv._press(E(x=G // 2, y=hl)); wv.release(E(x=G // 2, y=hl)); root.update()  # the linked-stereo glyph
    assert app.T["linked_stereo"].get() == 0
    key = lambda k, state=0: app._key(E(keysym=k, state=state))
    assert wv.sel is None; key("Tab"); key("Up"); root.update()  # drags select nothing; Tab selects, Up nudges
    assert wv.sel == "clip_above" and app.V["clip_above"].get() == "100", (wv.sel, app.V["clip_above"].get())
    key("BackSpace"); root.update()
    assert app.V["clip_above"].get() == "99.9"
    key("F1"); assert wv._help is not None
    key("x"); assert wv._help is None  # any key closes it
    app.hist._press(E(x=app.hist._x(0.99), y=50, state=0)); app.hist._release(None); root.update()  # marker
    assert app.V["clip_above"].get() == "99" and app.dragging is None
    # (p2) no cap never steps to cap 1 (nothing rebuilt); the off-screen cap's tab drags absolute
    app._set("max_restore", 0); root.update(); wv._draw()
    h = wv._live("max_restore")
    wv.nudge(h, -1); assert app.V["max_restore"].get() == "0"
    wv.nudge(h, 1); assert app.V["max_restore"].get() == "2"
    app._set("max_restore", 3); root.update(); wv._draw()
    x0, y0, x1, y1, h, c = next(t for t in wv._tabs if t[4].key == "max_restore" and t[5] == 0)
    wv._press(E(x=x0 + 4, y=y0 + 7)); wv._move(E(x=x0 + 4, y=-11, state=0)); wv.after_cancel(wv._hjob); wv._h_apply()
    assert abs(float(app.V["max_restore"].get()) - wv.value_at(h, 0, x0 + 4, -11)) <= h.step, app.V["max_restore"].get()
    wv.release(); root.update()
    # (p3) a Floor drag keeps the cached waveform; Ctrl+Return while rendering is ignored; Advanced has the keys
    wv._draw(); b, d = wv._base, wv._deco
    wv.drag_to("not_clip_amp", -50.0); root.update(); wv._draw()
    assert wv._base is b and wv._deco is not d
    wv.release(); root.update()
    app.run_btn.state(["disabled"]); app.status.config(text="x"); app.run()
    assert app.status.cget("text") == "x"
    app.run_btn.state(["!disabled"])
    assert all(app.adv.bind(k) for k in ("<Key>", "<Control-z>", "<Control-Return>", "<Control-e>"))
    assert all(str(w.cget("takefocus")) == "0" for ws in app.bound.values() for w in ws
               if isinstance(w, (ttk.Checkbutton, ttk.Radiobutton)))  # a click leaves Space to Play
    # region previews: (q) Accurate SPADE, from sample 0 and later, equals compute()'s output slice;
    # (r) an Output gain drag gives a new B without a stage-A job; (s) a render cancels the running
    # preview and pauses the heavy worker until it ends
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "burst.wav")
        sf.write(p, np.stack([q, q], 1), sr, subtype="FLOAT")
        app.inp.set(p); app.outp.set(os.path.join(d, "out.wav"))
        app.apply_preset("Accurate SPADE"); app.gpu.set(False); app.fast.set(False); app.auto.set(True)
        assert app.load_input(p)
        settle()
        ref = compute(app.x, sr, app.collect())
        for r in ((0, int(1.22 * sr)), (int(1.23 * sr), 2 * sr)):  # each holds part of the burst
            wv.jump(*r); settle()
            assert wv.region == r and wv.pv is not None and not wv.pv[2], (r, app.badge.cget("text"))
            assert wv.pv[1] == r[0] and np.array_equal(wv.pv[0], ref[r[0]:r[1]]), r
        assert app.badge.cget("text").startswith("B fresh · SPADE exact"), app.badge.cget("text")
        # audition through the App: Space loops the region's Orig, Proc (the preview) and Diff = B - A x g;
        # 1/2/3, X, S, M and Mon switch them, crossfaded; the playhead follows
        pl = app.player
        key("space")
        assert pl.playing and app._ctx == "loop" and pl.stream.kw == dict(
            samplerate=sr, channels=2, dtype="float32", blocksize=2048, latency="high", callback=pl._cb)
        b, tr = pl.bufs, 10 ** (app._mon_db / 20)
        g = 10 ** ((app.cfg["input_gain"] + app.cfg["output_gain"]) / 20)
        assert np.allclose(b[2][nf:-nf], wv.pv[0][nf:-nf] * tr, atol=1e-6)
        assert np.allclose(b[3], b[2] - b[1] * g, atol=1e-6) and np.abs(b[3]).max() > 0.01  # (w) via the App
        pl._cb(np.zeros((2048, 2), np.float32), 2048, None, None); app._playhead()
        assert wv.lanes.find_withtag("ph") and wv.ov.find_withtag("ph") and app.play_btn.cget("text") == "■"
        for k, v in (("2", 2), ("x", 1), ("x", 2), ("3", 3), ("1", 1)):
            key(k)
            assert app.src.get() == pl.src == v and pl.xf is not None, k
        key("s")
        assert app.side.get() and pl.bufs is not b and np.array_equal(pl.bufs[1][:, 0], pl.bufs[1][:, 1])
        key("m")
        assert app.match.get() and app.anote.cget("text").endswith("dB matched"), app.anote.cget("text")
        key("s"); key("m"); app._cycle_mon()
        assert app.mon_btn.cget("text") == "Mon 0 dB" and app._mon_db == 0
        for _ in range(3):
            app._cycle_mon()
        assert app.mon_btn.cget("text").startswith("Mon Auto ")
        app.LOOP_S = 0.5  # a longer region loops its start; a one-sample one (mid-drag) plays too
        key("s"); key("s")
        assert len(pl.bufs[1]) == sr // 2 and app.anote.cget("text") == "loops the first 0.5 s"
        wv.region = r[0], r[0] + 1
        app._audition(); pl._cb(np.zeros((2048, 2), np.float32), 2048, None, None)
        assert len(pl.bufs[1]) == 1 and pl.playing and pl.position() == 0
        wv.region = r; del app.LOOP_S
        key("s"); key("s")
        assert len(pl.bufs[1]) == r[1] - r[0]
        n_a, b, hold = app.n_a, pl.bufs, app._mon_db  # (r) while playing: new buffers, the Auto trim kept
        app.dragging = "output_gain"; app._set("output_gain", -6.0); root.update()
        assert app.n_a == n_a and not wv.pv[2] and np.array_equal(wv.pv[0], ref[r[0]:r[1]] * 10 ** (-6 / 20))
        assert pl.bufs is not b and app._mon_db == hold
        app.drag_end(); settle()
        assert app.n_a == n_a and not wv.pv[2]
        # (cc) the limiter (stage B, instant): B is the render's slice; what it cuts is read from B before it,
        # drawn pink on the Output page, with the ceiling's line
        app.T["limiter"].set(1); app._set("limiter_peak", -9.0); settle()
        cfg, lim = app.collect(), 10 ** (-9 / 20)
        P = compute(app.x, sr, dict(cfg, limiter=False))[r[0]:r[1]]
        assert app.n_a == n_a and np.array_equal(wv.pv[0], compute(app.x, sr, cfg)[r[0]:r[1]])
        assert np.array_equal(wv.pv[4], P) and app.cut[0] == np.mean(np.abs(P) > lim) > 0, app.cut
        assert f"limiter cuts {100 * app.cut[0]:.2f} % of region samples" in app.stage_txt["Output"][1].cget("text")
        pk = 20 * np.log10(np.abs(wv.pv[0]).max())  # after the limiter, as the badge; before it, the ceiling + up to
        assert app.stage_txt["Output"][0].cget("text") == f"output peak {pk:+.1f} dBFS (region)"
        assert f"(up to {20 * np.log10(np.abs(P).max() / lim):.1f} dB)" in app.stage_txt["Output"][1].cget("text")
        assert app.badge.cget("text").endswith(f" · peak {pk:+.1f} dBFS")
        app._select_page("Output"); wv._draw()
        pink = lambda: (wv._img[wv.lanes] == waveview._rgb(waveview.LIMIT)).all(2).any()
        assert [h.key for h in wv.handles] == ["limiter_peak"] and pink()
        app.T["limiter"].set(0); root.update(); wv._draw()
        assert not wv.handles and not pink()
        assert all(w.instate(["disabled"]) for k, w in app.grey if k == "limiter_peak")
        app._select_page("Declip"); settle()
        key("Escape"); app._playhead()  # Esc stops the audio first (and lets go of the buffers)
        assert not pl.playing and wv.region == r and not wv.lanes.find_withtag("ph") and pl.xf is not None
        idle = app._ph  # the stream stop after the fade; a Play before it cancels it
        assert idle in root.tk.call("after", "info")
        key("space")
        assert pl.playing and idle not in root.tk.call("after", "info")
        key("Escape"); app._playhead()
        assert not pl.playing and pl.bufs is None and pl.xf is not None
        pl._cb(np.zeros((2048, 2), np.float32), 2048, None, None); pl.idle()  # faded out: the stream stops
        assert pl.xf is None and not pl.stream.active
        with session.GPU_LOCK:  # the next preview job waits for the GPU, so it is running when the render starts
            app._set("speed_quality", 4)
            end = time.time() + 10
            while not app.heavy.running and time.time() < end:
                root.update(); time.sleep(0.01)
            kind, ev = app.heavy.running
            app.run()
            assert kind == "A" and ev.is_set() and app.heavy.paused, (kind, ev.is_set())
            assert app.badge.cget("text") == "paused while rendering"
            app.cancel.set()
        end = time.time() + 60
        while app.rendering and time.time() < end:
            root.update(); time.sleep(0.01)
        assert not app.heavy.paused and app.status.cget("text").startswith("Cancelled"), app.status.cget("text")
        # stage B's uncompress gain (warmed up) and unlimit curve (whole song, float32); after a render
        # B is the render while the settings match, and the render is drawn as older settings after a change
        for name in ("Uncompress", "Unmaster SPADE") if app.torch else ("Uncompress",):
            app.apply_preset(name); settle()  # the region stays
            ref = compute(app.x, sr, app.collect())[r[0]:r[1]]
            k = np.abs(ref) > 1e-4
            assert not wv.pv[2] and np.abs(20 * np.log10(np.abs(wv.pv[0][k] / ref[k]))).max() < 1e-3, name
        # with Declip on too, Release and Ratio only redo the uncompress gain, and Uncompress off slices
        # the stage A that is cached
        app.apply_preset("Accurate SPADE"); app.gpu.set(False); app.T["uncompress"].set(1)
        r2 = int(2.5 * sr), 3 * sr
        wv.jump(*r2); root.update(); app.preview_now(); settle()  # the gate may hold it on a slow CPU
        n_a, app.dragging = app.n_a, "uc_release"
        for v in (300.0, 600.0):
            app._set("uc_release", v); root.update()
        app.drag_end(); app._set("uc_ratio", 4.0); settle()
        ref = compute(app.x, sr, app.collect())[r2[0]:r2[1]]
        k = np.abs(ref) > 1e-4
        assert app.n_a == n_a and not wv.pv[2] and np.abs(20 * np.log10(np.abs(wv.pv[0][k] / ref[k]))).max() < 1e-3
        app.T["uncompress"].set(0); settle()
        assert app.n_a == n_a and np.array_equal(wv.pv[0], compute(app.x, sr, app.collect())[r2[0]:r2[1]])
        assert app.badge.cget("text").endswith(f" · peak {20 * np.log10(np.abs(wv.pv[0]).max()):+.1f} dBFS")
        with session.GPU_LOCK:  # back to the cached settings: the job for the others is cancelled
            v0 = app.V["clip_above"].get()
            app._set("clip_above", float(v0) - 1)
            end = time.time() + 10
            while not app.heavy.running and time.time() < end:
                root.update(); time.sleep(0.01)
            ev = app.heavy.running[1]
            app.V["clip_above"].set(v0); root.update()
            assert "A" not in app._jobs and ev.is_set() and app.badge.cget("text").startswith("B fresh")
        wv.jump(*r); app.apply_preset("Uncompress"); settle()
        app.dragging = "uc_threshold"  # one uncompress-gain job at a time; the last value follows it
        for v in (-14.0, -16.0, -18.0):
            app._set("uc_threshold", v); root.update()
        app.drag_end(); settle()
        ref = compute(app.x, sr, app.collect())[r[0]:r[1]]
        k = np.abs(ref) > 1e-4
        assert not wv.pv[2] and np.abs(20 * np.log10(np.abs(wv.pv[0][k] / ref[k]))).max() < 1e-3
        # the Uncompress page: the threshold line round-trips in both views and tints what is over it in dB
        # view; (bb) the transfer panel's knee, ratio and amount handles round-trip within one step, and
        # dragging them writes the settings
        app._select_page("Uncompress"); wv._draw()
        h = wv._live("uc_threshold")
        assert [h.key for h in wv.handles] == ["uc_threshold"] and app.stage_txt["Uncompress"][2].cget("text")
        for mode in ("db", "lin"):
            wv.set_mode(mode); wv._draw()
            y = float(wv.scale.y(wv._ref(h, 0, G + 300) * h.lev(h.value), hl))
            assert abs(wv.value_at(h, 0, G + 300, y) - h.value) <= h.step, mode
            tint = (wv._img[wv.lanes] == waveview._rgb(waveview._mix(waveview.WAVE_IN, waveview.UC, 0.5))).all(2).any()
            assert tint == (mode == "db"), mode
        cp = app.curve
        for T, R, M, a in ((-12, 2, 0, 50), (-24, 4, 3, 100), (-6, 1.5, -2, 20), (-30, 10, 0, 70)):
            for kk, v in (("uc_threshold", T), ("uc_ratio", R), ("uc_makeup", M), ("uc_amount", a)):
                app._set(kk, v)
            root.update()
            for kk, v in (("uc_threshold", T), ("uc_ratio", R), ("uc_amount", a)):
                assert abs(cp.value_at(kk, *cp.pos(kk)) - v) <= cp.STEP[kk][0], (kk, T, R, M, a)
        for kk, want in (("uc_threshold", -36.0), ("uc_ratio", 6.0), ("uc_amount", 40.0)):
            x0, y0 = cp.pos(kk)
            T, R, M, a = cp.p
            u = cp._xd(x0)
            cp._press(E(x=x0, y=y0))
            if kk == "uc_threshold":
                cp._drag(E(x=float(cp._x(want + M)), y=y0, state=0))
            else:  # the curve through the pressed level in at the wanted ratio or amount
                out = T + want * (u - M - T) if kk == "uc_ratio" else u + want / 100 * (float(cp.inv(u)) - u)
                cp._drag(E(x=x0, y=float(cp._y(out)), state=0))
            cp._release(None); root.update()
            assert abs(float(app.V[kk].get()) - want) <= cp.STEP[kk][0] and app.dragging is None, (kk, app.V[kk].get())
        # Tab: the lane's line, then the panel's dots (the selected one outlined); Up and the wheel step them
        wv.sel, r0 = None, float(app.V["uc_ratio"].get())
        key("Tab"); key("Tab"); key("Up"); root.update()
        assert wv.sel == "uc_ratio" and abs(float(app.V["uc_ratio"].get()) - r0 - 0.1) < 1e-6, app.V["uc_ratio"].get()
        key("Tab"); root.update()
        assert wv.sel == "uc_amount" and any(cp.type(i) == "oval" and cp.itemcget(i, "outline") == waveview.ACCENT
                                             for i in cp.find_all())
        x0, y0 = cp.pos("uc_amount")
        a0 = float(app.V["uc_amount"].get())
        cp._wheel(E(x=x0, y=y0, delta=-120), True); root.update()
        assert abs(float(app.V["uc_amount"].get()) - a0 + 0.1) < 1e-6, app.V["uc_amount"].get()
        assert all(cp.bind(f"<{m}Button-{i}>") for m in ("", "Shift-") for i in (4, 5))  # X11's wheel
        cp._wheel(E(x=x0, y=y0, delta=0), True, 1); root.update()
        assert abs(float(app.V["uc_amount"].get()) - a0) < 1e-6, app.V["uc_amount"].get()
        # (dd) the gain lane (an unlimit curve put in): the curve drawn at Amount a is a x its 100 % dB curve,
        # per column exactly; dragging the lane up sets Amount by 0.5 % per px (Shift: 0.05)
        real_G, n = app.G, len(app.x)
        Gs = (10 ** (6 * np.sin(np.linspace(0, 40, n)) ** 2 / 20)).astype(np.float32)
        app.G = session.keyof(app.cfg, session.G_KEYS), Gs, Pyramid(Gs[:, None]), (6.0, 6.0), 0.0
        app.T["unlimit"].set(1); app._set("unlimit_amount", 40); app._select_page("Unlimit"); settle()
        wv.glane.winfo_width, wv.glane.winfo_height = (lambda: W), (lambda: 64)
        wv._view(0, 128 * (W - G)); wv._draw()  # 128 samples a column: on 64-sample blocks
        e = wv._e()
        _, _, c0, c1, mn, mx, a, _ = next(c for c in wv._gcurves if c[0] == "unlimit_amount")
        g64 = Gs[:e[-1]].astype(np.float64) ** a
        assert (c0, c1, a) == (0, W - G, 0.4) and wv.glane.winfo_manager()
        assert np.allclose(a * mx, 20 * np.log10(np.maximum.reduceat(g64, e[:-1])), atol=1e-4)
        assert np.allclose(a * mn, 20 * np.log10(np.minimum.reduceat(g64, e[:-1])), atol=1e-4)
        assert (wv._img[wv.glane] == waveview._rgb(waveview.UNL)).all(2).any()
        assert app.stage_txt["Unlimit"][1].cget("text") == "peaks up to +2.4 dB · p99.9 +2.4 dB · body 0 dB"
        wv._g_press(E(x=G + 100, y=40)); wv._g_move(E(x=G + 100, y=20, state=0)); wv._g_release(None); root.update()
        assert app.V["unlimit_amount"].get() == "50" and app.dragging is None, app.V["unlimit_amount"].get()
        wv._g_press(E(x=G + 100, y=40)); wv._g_move(E(x=G + 100, y=30, state=1)); wv._g_release(None); root.update()
        assert app.V["unlimit_amount"].get() == "50.5", app.V["unlimit_amount"].get()
        wv.sel = None; key("Tab"); key("Up"); root.update()  # by the keys too, its label outlined
        assert wv.sel == "unlimit_amount" and app.V["unlimit_amount"].get() == "51", app.V["unlimit_amount"].get()
        assert any(wv.glane.type(i) == "rectangle" and wv.glane.itemcget(i, "outline") == waveview.ACCENT
                   for i in wv.glane.find_withtag("deco"))
        app._select_page("Uncompress"); wv._draw()
        assert wv._gkey() == "uc_amount" and {c[0] for c in wv._gcurves} == {"unlimit_amount", "uc_amount"}
        app.G = real_G
        # at Amount 0 the uncompress gain still comes, also after B, and the latest settings' gain follows one
        # still in flight: the lane shows it and drags Amount up from 0
        app.apply_preset("Uncompress"); app._set("uc_amount", 0); settle()
        app.dragging = "uc_threshold"
        for v in (-14.0, -16.0):
            app._set("uc_threshold", v); root.update()
        app.drag_end(); settle(); wv.zoom_region(); wv._draw()
        assert app.prev["ucg"] is not None and app.ucg[0][2] == session.keyof(app.cfg, session.UC_KEYS - session.A_KEYS)
        assert wv._gkey() == "uc_amount" and "restores up to" in app.stage_txt["Uncompress"][1].cget("text")
        wv._g_press(E(x=G + 100, y=40)); wv._g_move(E(x=G + 100, y=20, state=0)); wv._g_release(None); root.update()
        assert app.V["uc_amount"].get() == "10", app.V["uc_amount"].get()
        app.apply_preset("Uncompress"); app.T["limiter"].set(1); app._set("limiter_peak", -3.0)
        app._select_page("Declip"); settle()
        app.run()
        end = time.time() + 60
        while app.rendering and time.time() < end:
            root.update(); time.sleep(0.01)
        settle()
        assert app.badge.cget("text") == "B = full render" and wv.pv is None, app.badge.cget("text")
        # what the limiter cut of the song is counted before it; the gain lane says why it has no curve
        full_cut = np.mean(np.abs(compute(app.x, sr, dict(app.collect(), limiter=False))) > 10 ** (-3 / 20))
        assert app.full_cut == full_cut > 0, (app.full_cut, full_cut)
        assert app.stage_txt["Output"][1].cget("text") == f"limiter cuts {100 * full_cut:.2f} % of the song"
        wv._draw()
        assert any(wv.glane.itemcget(i, "text") == "uncompress gain: not kept from the render"
                   for i in wv.glane.find_withtag("deco") if wv.glane.type(i) == "text")
        cfg0 = app.cfg  # Process at Hz: the header says so
        app.cfg = dict(cfg0, process_rate=sr // 2); app._marks()
        assert app.tag.cget("text").startswith("processing as 24 kHz (0.50× speed)"), app.tag.cget("text")
        app.cfg = cfg0; app._marks()
        app.T["limiter"].set(0); settle()
        app._set("output_gain", -1.0); settle()
        assert app.tag.cget("text") == "older settings" and wv.stale and wv.pv is not None and not wv.pv[2]
        # no region: Space plays the song from the edit cursor (the render is stale: Orig only); the view
        # pages along with the playhead
        wv.clear_region(); root.update()
        c, out = int(0.5 * sr), np.zeros((2048, 2), np.float32)
        wv.cursor = c
        key("space")
        assert app._ctx == "song" and pl.bufs[2] is None and pl.loop is False
        for _ in range(2):
            pl._cb(out, 2048, None, None)
        assert np.allclose(out, app.x[c + 2048:c + 4096] * pl.gain, atol=1e-6)
        wv._view(0, sr // 4); app._playhead()
        assert wv.t0 == c + 4096, wv.t0
        key("Escape"); app.heavy.cancel()
    # NMF auto detection (PyTorch): (y) Song.ceilings finds what the render finds; the preview is provisional
    # (ceilings only) until the shoulder search lands, which it lets through, then redone; (z) the knee
    # handle writes nmf_knee, 1 gives the hard flags, not provisional; Use as manual threshold; (aa) the
    # render reuses the GUI's analysis (exactly) and logs it
    if not app.torch:
        print("NMF auto detection checks skipped (no PyTorch)")
    else:
        from betterdeclipper.engine import declip
        from betterdeclipper.auto import auto_constraints
        c2 = dict(PRESETS["Accurate"], input_gain=-3.0, process_rate=24000.0)
        refs = declip(xs * 10 ** (-3 / 20), 24000.0, preset="fast", mode="hard", device="cpu", verbose=False)[1]
        assert Song(xs, sr, None).ceilings(c2)["refs"] == refs["analysis"]["refs"]  # (y)
        badges, real_badge = [], app._badge
        app._badge = lambda text, *a: (badges.append(text), real_badge(text, *a))[1]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "kick.wav")
            sf.write(p, np.stack([cl, 0.7 * cl], 1), sr, subtype="FLOAT")
            app.inp.set(p); app.outp.set(os.path.join(d, "out.wav"))
            app.apply_preset("Accurate"); app.C["nmf_quality"].set("fast")
            app.gpu.set(False); app.fast.set(True); app.auto.set(True); root.update()
            with session.GPU_LOCK:  # the search waits until the region's preview is queued: it runs that first
                assert app.load_input(p) and app._prov(app.cfg) and "sh" in app._jobs
                end = time.time() + 30
                while "A" not in app._jobs and time.time() < end:
                    root.update(); time.sleep(0.01)
            settle()
            assert any(t.startswith("provisional (ceilings only)") for t in badges), badges
            assert not app._prov(app.cfg) and not wv.det["af"]["prov"] and not app.prev["prov"], badges[-3:]
            assert app.badge.cget("text").startswith("fast preview"), app.badge.cget("text")
            assert app.stat[4].cget("text").startswith("no shoulder") and app.seed_btn.winfo_manager()
            assert wv.det["stats"].runs and app.rebuilt.cget("text").endswith("% rebuilt")
            wv._draw()
            assert [h.key for h in wv.handles] == ["nmf_knee", "max_restore"], [h.key for h in wv.handles]
            wv.drag_to("nmf_knee", 0.9); wv.release(); settle()  # (z)
            k = float(app.V["nmf_knee"].get())
            assert abs(k - 0.9) <= 0.011 and app.stat[4].cget("text") == f"knee set {k:.2f}", k
            r0 = wv.region
            app.seed_manual(); settle()  # the manual flags land: the region, the view and the message stay
            assert app.T["auto_detect"].get() == 0 and app.V["clip_above"].get() == f"{round(100 * k, 1):g}"
            assert wv.readout.cget("text").startswith("Manual detection seeded") and wv.region == r0
            app.undo(); settle()
            wv._draw(); wv.drag_to("nmf_knee", 1.05); wv.release(); settle()
            assert app.V["nmf_knee"].get() == "1" and app.stat[4].cget("text") == "ceilings only (knee 1.00)"
            song, y = wv.song, wv.song.x
            m_hi, m_lo, *_ = auto_constraints(y, song.ceilings(app.cfg)["lsb"], None, "hard", sr, song.analysis(app.cfg))
            assert np.array_equal(wv.det["flags"][0], m_hi | m_lo) and not wv.det["af"]["prov"]
            assert "provisional" not in app.badge.cget("text"), app.badge.cget("text")
            a0, a1 = (song.auto_flags(dict(app.cfg, nmf_knee=v), None) for v in (0.0, 1.0))  # the same flags
            assert a0["prov"] and not a1["prov"] and a0["flags"] is a1["flags"]
            th, hl = wv.det["ceil"]["refs"][0][0]["theta"], wv._lane(0)[1]  # the knee on the ceiling: its details
            wv._draw(); wv._motion(E(x=G + 300, y=float(wv.scale.y(th, hl))))
            assert wv.hover == "nmf_knee" and wv.readout.cget("text").startswith("L+ clip ceiling")
            app.log.set(True); app.run()  # (aa)
            end = time.time() + 120
            while app.rendering and time.time() < end:
                root.update(); time.sleep(0.01)
            log = open(os.path.join(d, "out.log")).read()
            assert "analysis from GUI kinds=clip,clip,clip,clip knee=1.00" in log, log
            assert np.array_equal(app.full, compute(app.x, sr, app.collect()))
            app.log.set(False); settle()
        app._badge = real_badge
    # (x) without sounddevice the audition controls are off, with the hint, and previews still run
    session.sd = None
    top = tk_.Toplevel(root); top.withdraw()
    a2 = App(top)
    assert not a2.player.available and all(w.instate(["disabled"]) for w in a2.audio_w)
    assert a2.anote.cget("text") == session.SD_HINT
    a2._key(E(keysym="space", state=0))
    assert not a2.player.playing and a2.player.stream is None and a2.wv.readout.cget("text") == session.SD_HINT
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "burst.wav")
        sf.write(p, np.stack([q, q], 1), sr, subtype="FLOAT")
        a2.apply_preset("SharpUnClip"); a2.gpu.set(False); root.update()
        assert a2.load_input(p)
        end = time.time() + 60
        while a2.wv.pv is None and time.time() < end:  # the region goes to the burst; its preview follows
            root.update(); time.sleep(0.01)
            if a2.wv.region and not a2._jobs and a2.badge.cget("text") == "stale · P":
                a2.preview_now()
    assert a2.wv.pv is not None, a2.badge.cget("text")
    a2.heavy.cancel(); top.destroy()
    session.sd = real_sd
    root.destroy()
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        _tk()
        App(tk.Tk()).root.mainloop()
