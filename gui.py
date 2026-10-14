# SPDX-FileCopyrightText: 2026 AG89
#
# SPDX-License-Identifier: MIT

import sys, os, json, threading, queue, time, traceback
import numpy as np
import soundfile as sf
from sharpunclip import process, DEFAULTS, PRESETS, settings_from_cfg, spade_backend, backend_name, _write_log

# Sections: (title, on/off setting or None, show the on/off box in the title, columns, items).
# An item is (key, label, min, max, step) for a number or (key, label) for a checkbox.
SECTIONS = (
    ("Declip", "declip", True, 1, (("clip_above", "Clip above %", 1, 100, 0.1),
                                   ("max_restore", "Max restore ×", 0, 20, 0.5),
                                   ("speed_quality", "Quality 0-6", 0, 6, 1),
                                   ("linked_stereo", "Linked stereo"))),
    ("Unlimit", "unlimit", True, 1, (("unlimit_amount", "Amount %", 0, 300, 10),)),
    ("Uncompress", "uncompress", True, 1, (("uc_threshold", "Threshold dB", -60, 24, 0.5),
                                           ("uc_ratio", "Ratio", 1, 20, 0.5),
                                           ("uc_attack", "Attack ms", 0.1, 500, 1),
                                           ("uc_release", "Release ms", 1, 5000, 10),
                                           ("uc_makeup", "Makeup dB", -24, 24, 0.5),
                                           ("uc_amount", "Amount %", 0, 100, 10),
                                           ("uc_link", "Stereo-linked"))),
)
# Advanced window: groups of (title, on/off setting that greys it, subsections); a subsection is
# (title, columns, items) like SECTIONS.
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
        ("Frames", 1, (("lookahead", "Lookahead 0-2", 0, 2, 1),)),
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

BG, PANEL, LINE, FG, MUTED = "#1b1d22", "#262a31", "#3a3f48", "#e8e8e8", "#8b919c"
ACCENT, WAVE_IN = "#36d399", "#5a616e"


def compute(x, sr, cfg, progress=None):
    xp, dev = spade_backend(cfg.get("gpu"))
    if dev == "cpu":
        xp.set_num_threads(max(1, int(cfg.get("threads", 1))))
    return process(x, sr, settings_from_cfg(cfg), xp=xp, dev=dev, progress=progress)


def envelope(x, cols):
    """(top, bottom) per pixel column of an (n,) or (n, ch) signal."""
    x = np.asarray(x)
    x = x[:, None] if x.ndim == 1 else x
    edges = np.linspace(0, len(x), max(1, min(cols, len(x))) + 1).astype(int)[:-1]
    return np.maximum.reduceat(x.max(1), edges), np.minimum.reduceat(x.min(1), edges)


def mmss(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


class _Cancelled(Exception):
    pass


class App:
    def __init__(self, root):
        self.root = root
        self.V, self.T, self.labels, self.grey = {}, {}, {}, {}
        self.x = self.out = None
        self.sr, self.done, self.path = 0, 0, None
        self.env_in = self.env_out = None
        self.q = queue.Queue()
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
        s.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", FG)],
              selectbackground=[("readonly", PANEL)], selectforeground=[("readonly", FG)])
        s.map("TCheckbutton", background=[("active", BG)],
              indicatorbackground=[("selected", ACCENT), ("!selected", PANEL)])
        s.configure("TCheckbutton", indicatorbackground=PANEL, indicatorforeground=BG)
        s.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=PANEL,
                    bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT, thickness=8)
        r.option_add("*TCombobox*Listbox.background", PANEL)
        r.option_add("*TCombobox*Listbox.foreground", FG)
        r.option_add("*TCombobox*Listbox.selectBackground", LINE)

    def _section(self, parent, title, toggle, show_toggle, cols, items):
        """A titled box of settings. With an on/off toggle, its fields grey out when off."""
        head = None
        if toggle and show_toggle:
            self.T[toggle] = tk.IntVar(value=int(INIT[toggle]))
            head = ttk.Checkbutton(parent, text=title, variable=self.T[toggle], command=self._sync)
        box = ttk.LabelFrame(parent, text=title, labelwidget=head, padding=(10, 4, 10, 8))
        grey = self.grey.setdefault(toggle, []) if toggle else []
        for i, it in enumerate(items):
            r, c = i // cols, 2 * (i % cols)
            if len(it) == 2:  # checkbox
                key, label = it
                self.T[key] = tk.IntVar(value=int(INIT[key]))
                cb = ttk.Checkbutton(box, text=label, variable=self.T[key])
                cb.grid(row=r, column=c, columnspan=2, sticky="w", pady=2)
                grey.append(cb)
                continue
            key, label, lo, hi, step = it
            lab = ttk.Label(box, text=label, style="Muted.TLabel")
            lab.grid(row=r, column=c, sticky="w", padx=(0, 6), pady=2)
            self.V[key] = tk.StringVar(value=str(INIT[key]))
            self.labels[key] = label
            sp = ttk.Spinbox(box, textvariable=self.V[key], from_=lo, to=hi, increment=step, width=8)
            sp.grid(row=r, column=c + 1, sticky="w", padx=(0, 14), pady=2)
            grey += [lab, sp]
        return box

    def _sync(self):
        """Grey out the settings of every stage that is switched off."""
        for toggle, widgets in self.grey.items():
            on = bool(self.T[toggle].get())
            for w in widgets:
                w.state(["!disabled"] if on else ["disabled"])

    def _build(self):
        pad = dict(padx=14)
        # packed first so a small window shrinks the waveform, not the buttons
        foot = ttk.Frame(self.root)
        foot.pack(side="bottom", fill="x", pady=(4, 14), **pad)
        self.bar = ttk.Progressbar(foot, maximum=1.0)
        self.bar.pack(fill="x")
        row = ttk.Frame(foot); row.pack(fill="x", pady=(8, 0))
        self.cancel_btn = ttk.Button(row, text="Cancel", command=self.cancel.set, state="disabled")
        self.cancel_btn.pack(side="right")
        self.run_btn = ttk.Button(row, text="Render", style="Accent.TButton", command=self.run)
        self.run_btn.pack(side="right", padx=6)
        # packed after the buttons so a long message can't push them off
        self.status = ttk.Label(row, text="Open a clipped file to start.", style="Muted.TLabel")
        self.status.pack(side="left", fill="x", expand=True)

        files = ttk.Frame(self.root); files.pack(fill="x", pady=(12, 6), **pad)
        for row, (label, var, cmd) in enumerate((("Input", "inp", self.browse_in),
                                                 ("Output", "outp", self.browse_out))):
            ttk.Label(files, text=label, style="Muted.TLabel", width=7).grid(row=row, column=0, sticky="w")
            setattr(self, var, tk.StringVar())
            ttk.Entry(files, textvariable=getattr(self, var)).grid(row=row, column=1, sticky="we", pady=2)
            ttk.Button(files, text="Browse", command=cmd).grid(row=row, column=2, padx=(6, 0))
        files.columnconfigure(1, weight=1)

        bar = ttk.Frame(self.root); bar.pack(fill="x", pady=6, **pad)
        ttk.Label(bar, text="Preset", style="Muted.TLabel").pack(side="left")
        self.preset = tk.StringVar()
        pc = ttk.Combobox(bar, textvariable=self.preset, values=list(PRESETS), width=22, state="readonly")
        pc.pack(side="left", padx=6)
        pc.bind("<<ComboboxSelected>>", lambda e: self.apply_cfg(PRESETS[self.preset.get()]))
        ttk.Button(bar, text="Save", command=self.save_preset).pack(side="left", padx=2)
        ttk.Button(bar, text="Load", command=self.load_preset).pack(side="left", padx=2)
        self.gpu = tk.BooleanVar(value=spade_backend(True)[1] in ("opencl", "cuda"))
        ttk.Checkbutton(bar, text="GPU", variable=self.gpu).pack(side="right")
        self.log = tk.BooleanVar(value=False)  # writes <output>.log
        ttk.Checkbutton(bar, text="Log", variable=self.log).pack(side="right", padx=10)
        self.V["process_rate"] = tk.StringVar(value="0")
        self.labels["process_rate"] = "Process rate"
        ttk.Spinbox(bar, textvariable=self.V["process_rate"], width=7,
                    values=("0", "11025", "16000", "22050", "24000", "32000", "44100", "48000",
                            "88200", "96000")).pack(side="right")
        ttk.Label(bar, text="Process at Hz", style="Muted.TLabel").pack(side="right", padx=(0, 6))

        stages = ttk.Frame(self.root); stages.pack(fill="x", pady=(6, 0), **pad)
        for i, spec in enumerate(SECTIONS):
            self._section(stages, *spec).grid(row=0, column=i, sticky="nsew", padx=(0, 8))
            stages.columnconfigure(i, weight=1)
        try:
            import torch  # noqa: F401  (unlimit needs it)
        except ImportError:
            ttk.Label(stages.grid_slaves(row=0, column=1)[0], text="needs PyTorch",
                      style="Muted.TLabel").grid(row=5, column=0, columnspan=2, sticky="w")
        self.adv_btn = ttk.Button(self.root, text="Advanced…", command=self.toggle_adv)
        self.adv_btn.pack(anchor="w", pady=(8, 0), **pad)

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

        self.wave = ttk.Frame(self.root); self.wave.pack(fill="both", expand=True, pady=(10, 6), **pad)
        self.info = ttk.Label(self.wave, text="", style="Muted.TLabel")
        self.info.pack(anchor="w")
        self.cv = tk.Canvas(self.wave, height=120, background="#14161a",
                            highlightthickness=1, highlightbackground=LINE)
        self.cv.pack(fill="both", expand=True, pady=(4, 0))
        self.cv.bind("<Configure>", lambda e: self._dirty())

        self.preset.set("Accurate")
        self.apply_cfg(PRESETS["Accurate"])
        self.root.update_idletasks()  # minimum size keeps every control visible
        self.root.minsize(self.root.winfo_reqwidth(), self.root.winfo_reqheight() - 60)

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
            self.load_input(p)

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
        self.path, self.out, self.done = p, None, 0
        self.env_in = self.env_out = None
        n, ch = self.x.shape
        self.info.config(text=f"{os.path.basename(p)}   ·   {self.sr / 1000:g} kHz   ·   "
                              f"{'mono' if ch == 1 else 'stereo' if ch == 2 else f'{ch} ch'}   ·   {mmss(n / self.sr)}")
        self.bar["value"] = 0
        self.status.config(text="Ready.")
        self._dirty()
        return True

    def apply_cfg(self, cfg):
        for k, val in cfg.items():  # unknown keys are ignored
            k = k.removeprefix("pa_")  # older preset files
            if k in self.V:
                self.V[k].set(str(int(float(val))) if k in INTS else str(val))
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
        cfg["gpu"] = self.gpu.get()
        return cfg

    def save_preset(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON preset", "*.json")])
        if p:
            with open(p, "w") as f:
                json.dump(self.collect(), f, indent=2)
            self.status.config(text="saved " + p)

    def load_preset(self):
        p = filedialog.askopenfilename(filetypes=[("JSON preset", "*.json"), ("All", "*.*")])
        if p:
            with open(p) as f:
                self.apply_cfg(json.load(f))
            self.status.config(text="loaded " + p)

    # ---- render ----
    def run(self):
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
        self.out, self.done, self.env_out = None, 0, None
        self.cancel.clear()
        self.run_btn.config(state="disabled"); self.cancel_btn.config(state="normal")
        self.t0, self.frac = time.time(), 0.0
        self.note = "   ·   " + backend_name(*spade_backend(cfg["gpu"]))
        if cfg["process_rate"] > 0 and cfg["process_rate"] != self.sr:
            self.note += f"   ·   at {cfg['process_rate']} Hz ({cfg['process_rate'] / self.sr:.2f}× speed)"
        self.status.config(text="Starting" + self.note)
        self._dirty()
        threading.Thread(target=self._work, args=(self.x, self.sr, outp, cfg), daemon=True).start()

    def _work(self, x, sr, outp, cfg):
        def progress(frac, done, out):
            if self.cancel.is_set():
                raise _Cancelled
            self.q.put(("p", frac, done, out))
        t0 = time.time()
        log = [f"SharpUnClip GUI {time.strftime('%Y-%m-%d %H:%M:%S')}",
               f"input={self.path} output={outp} sr={sr} samples={len(x)} channels={x.shape[1]}",
               f"backend={backend_name(*spade_backend(cfg.get('gpu')))}",
               "settings " + " ".join(f"{k}={v}" for k, v in sorted(cfg.items()))]
        try:
            out = compute(x, sr, cfg, progress)
            sf.write(outp, out, sr, subtype="FLOAT")
            log.append(f"done peak={np.abs(out).max():.3f} elapsed={time.time() - t0:.1f}s")
            self.q.put(("done", out, outp))
        except _Cancelled:
            log.append(f"cancelled after {time.time() - t0:.1f}s")
            self.q.put(("cancelled",))
        except Exception as e:
            log.append("error:\n" + traceback.format_exc())
            self.q.put(("error", str(e)))
        if self.log.get():
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
        for m in msgs:
            if m[0] == "p":
                self.frac = m[1]
                if m[3] is not None:
                    self.done, self.out, self.env_out = m[2], m[3], None
            else:
                self._finished(m)
        if msgs and msgs[-1][0] == "p":
            self._progress_text()
            self._dirty()
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
            self.out, self.done, self.env_out = m[1], len(m[1]), None
            self.bar["value"] = 1.0
            pk = float(np.abs(m[1]).max())
            self.status.config(text=f"Done in {mmss(el)} ({len(self.x) / self.sr / max(el, 1e-6):.1f}× realtime)"
                                    f"   ·   peak {pk:.2f} ({20 * np.log10(max(pk, 1e-9)):+.1f} dBFS)"
                                    f"   ·   {os.path.basename(m[2])}{self.note}")
        elif m[0] == "cancelled":
            self.status.config(text="Cancelled. Nothing was written.")
        else:
            self.status.config(text="Error: " + m[1])
        self._dirty()

    # ---- waveform ----
    def _dirty(self):
        if not getattr(self, "_pending", False):
            self._pending = True
            self.root.after_idle(self.redraw)

    def redraw(self):
        self._pending = False
        cv = self.cv
        cv.delete("all")
        W, H = cv.winfo_width(), cv.winfo_height()
        if self.x is None or W < 20:
            cv.create_text(W / 2, H / 2, text="No file loaded", fill=MUTED)
            return
        cols = W - 2
        if self.env_in is None or len(self.env_in[0]) != min(cols, len(self.x)):
            self.env_in = envelope(self.x, cols)
            self.env_out = None
        if self.out is not None and self.env_out is None:
            self.env_out = envelope(self.out, cols)
        pk_in = max(float(np.abs(self.x).max()), 1e-9)
        pk = max(pk_in, float(np.abs(self.env_out[0]).max()) if self.env_out else 0,
                 float(np.abs(self.env_out[1]).max()) if self.env_out else 0)
        mid = H / 2
        ys = lambda v: mid - np.asarray(v) / pk * (mid - 8)

        def poly(top, bot, color):
            if len(top) < 2:
                return
            xs = np.arange(len(top)) + 1
            pts = np.concatenate([np.c_[xs, ys(top)], np.c_[xs, ys(bot)][::-1]]).ravel().tolist()
            cv.create_polygon(pts, fill=color, outline=color)
        poly(*self.env_in, WAVE_IN)
        if self.env_out is not None:
            dcol = int(round(self.done / len(self.x) * len(self.env_out[0])))
            poly(self.env_out[0][:dcol], self.env_out[1][:dcol], ACCENT)
            if 0 < self.done < len(self.x):
                cv.create_line(dcol + 1, 0, dcol + 1, H, fill=FG, width=1)
        cv.create_line(0, mid, W, mid, fill=LINE)
        if pk > 1.0:  # 0 dBFS guide lines
            for v in (1.0, -1.0):
                cv.create_line(0, ys(v), W, ys(v), fill="#c9ced6", dash=(4, 4))
            cv.create_text(6, ys(1.0) - 3, text="0 dBFS", fill="#c9ced6", anchor="sw",
                           font=("Segoe UI", 8))


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
    try:
        import tkinter as tk_
        root = tk_.Tk(); root.withdraw()
    except Exception:  # no display
        print("selfcheck OK (no display: GUI fields not checked)")
        return
    global tk, ttk, filedialog
    from tkinter import ttk, filedialog
    tk = tk_
    app = App(root)
    for name in PRESETS:
        app.apply_cfg(PRESETS[name])
        cfg = app.collect()  # raises if any box is empty or not a number
        assert settings_from_cfg(cfg) == dict(DEFAULTS, **PRESETS[name]), name
    app.outp.set("C:/Users/x/Desktop/old_unclipped.wav")  # new input keeps the output folder
    assert app.default_out("F:/rips/song.flac") == os.path.normpath("C:/Users/x/Desktop/song_unclipped.wav")
    app.outp.set("")
    assert app.default_out("F:/rips/song.flac") == os.path.normpath("F:/rips/song_unclipped.wav")
    root.destroy()
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        import tkinter as tk
        from tkinter import ttk, filedialog
        App(tk.Tk()).root.mainloop()
