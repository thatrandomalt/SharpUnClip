# SPDX-FileCopyrightText: 2026 Qupci
# SPDX-FileCopyrightText: 2026 thatrandomalt
#
# SPDX-License-Identifier: MIT

"""BetterDeclipper: offline, high-accuracy audio declipping.

Vendored from https://github.com/Qupci/BetterDeclipper (v0.3.0, commit f9ff7f4): only the
modules the engine needs (no CLI). Changes: a progress tick inside the SPADE stage
(engine.py, methods/spade.py); estimate_lsb samples the whole file and skips exact zeros,
so a silent intro no longer looks like 8-bit audio (detect.py); CUDA graphs are off on ROCm
builds of PyTorch, where graph capture can hang (methods/pnp.py); fast_pass() and
lam_refs_for() (which also accepts a non-integer sample rate) split out of declip(), and
declip(lam_refs=) to reuse a file-global lambda reference (engine.py, for SharpUnClip GUI
previews).

Python API (torch is imported on first use):
    import soundfile as sf
    from betterdeclipper import declip
    y, sr = sf.read("in.wav", always_2d=True)
    x, info = declip(y, sr, preset="normal", mode="auto", progress=lambda step, n, seconds: None)
info["analysis"] holds the automatic analysis.
"""
__version__ = "0.3.0"
__all__ = ["declip", "PRESETS", "__version__"]


def __getattr__(name):  # lazy, so that `import betterdeclipper` stays light
    if name in ("declip", "PRESETS"):
        from . import engine
        return getattr(engine, name)
    raise AttributeError(f"module 'betterdeclipper' has no attribute {name!r}")
