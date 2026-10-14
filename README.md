<!--
SPDX-FileCopyrightText: 2026 thatrandomalt

SPDX-License-Identifier: MIT
-->

# SharpUnClip

**Undo clipping, limiting and compression in loud music.** SharpUnClip rebuilds
clipped peaks, reverses a limiter's gain riding and inverts a known
compressor. It has a GUI and a command line. The default declip engine runs
on NVIDIA GPUs (CUDA) or the CPU; the SPADE engine runs on any GPU vendor
(OpenCL).

> **AI-assisted project.** The code, experiments and documentation were written
> by AI coding agents (mainly Claude by Anthropic, in Claude Code) under the
> direction of the project author, who set the goals, rendered every test file
> and judged the results by ear. All claims were verified by automated tests and
> ground-truth measurements, but please review the code before relying on it.
> Details: [METHODS.md § 9](METHODS.md#9-how-this-project-was-made-ai-assistance).

## What it does

Loud masters are usually compressed, then limited, then clipped. SharpUnClip
undoes those steps in reverse, each one switched on separately:

| Stage | Method | Based on |
|---|---|---|
| **Declip, NMF engine** (default) | Finds the clipping/limiting ceiling automatically, then rebuilds the samples above it with a fusion of three sparse models (an NMF spectrogram model, social sparsity, stereo SPADE); untouched samples stay bit-exact | [BetterDeclipper](https://github.com/Qupci/BetterDeclipper) by Qupci (MIT, included in `betterdeclipper/`), built on Siedenburg et al. 2014, Lee & Seung 2001, Kitić et al. 2015 |
| **Declip, SPADE engine** | Rebuilds clipped samples so the frames' spectra are sparse; untouched samples stay bit-exact, and rebuilt ones are never quieter than the input | SPADE, Kitić, Bertin & Gribonval 2015; conventions from Záviška et al. 2019/2021 |
| **Unlimit** | Divides out a limiter's gain curve, as predicted by a pretrained network | De-limiter, Jeon & Lee 2023 |
| **Uncompress** | Runs a known compressor backwards | Gorlow & Reiss 2013, on the compressor design of Giannoulis, Massberg & Reiss 2012 |

The NMF engine is a third-party engine, included with its method unchanged
(only small fixes and hooks); SharpUnClip adds its own clip detection and
peak cap as an alternative to the engine's analysis. The other methods were adapted and tuned: per-sample clipping
bounds with an upper bound, linked stereo, level independence, a fused GPU
kernel, gain applied at the file's own rate, and partial inversion.
[METHODS.md](METHODS.md) lists every change, and why, with the measurements.

## Results

Every default was chosen by processing real degraded renders and scoring them
against the true unprocessed originals: two songs, exported from their projects
before and after clipping, limiting or compression. The full test method and
all numbers are in [METHODS.md](METHODS.md#5-testing-methodology).

| Preset | For | Improvement vs the degraded file (song 1 / song 2) |
|---|---|---|
| **Accurate** (default) | clipped songs | **+8.2 / +2.0 dB** SNR (NMF engine) |
| **Accurate SPADE** | clipped songs, on AMD/Intel GPUs or without PyTorch | +5.7 / +1.8 dB SNR |
| **Unmaster** | mastered (limited) songs | **+2.4 / +1.7 dB** SI-SDR; iZotope Ozone's Unlimiter, same songs: −0.1 / +1.3 (100%), −3.2 / +1.0 (200%) |
| **Unmaster SPADE** | mastered songs, on AMD/Intel GPUs | +1.0 / +1.5 dB SI-SDR |
| **Uncompress** | compressed songs, settings unknown | **+0.2 to +4.0 dB** SI-SDR on all 11 compressed test renders (2:1 to 8:1) |
| **SharpUnClip** | maximum punch | −3.3 / −0.9 dB: louder than the original, by design |
| Uncompress, known settings | compressed songs | up to **+6.9 dB** SI-SDR (2:1 compressor); +0.6 to +1.5 dB for very fast, high-ratio compression |

Higher is better. SNR and SI-SDR measure how close the output is to the real
original.

## Install

There is nothing to compile. SharpUnClip is plain Python. The SPADE engine's
GPU kernel is compiled automatically by your graphics driver on its first GPU
run (a few seconds); pyopencl caches it, so later runs start immediately.

### 1. Get the code and make a virtual environment

Needs Python 3.9 or newer.

```
git clone <this repository's URL>
cd SharpUnClip
python -m venv .venv
```

Activate the environment:
- **Windows:** `.venv\Scripts\activate`
- **Linux / macOS:** `source .venv/bin/activate`

### 2. Install the dependencies

```
pip install numpy scipy soundfile torch torchaudio   # required (NMF engine, Unlimit)
pip install pyopencl                                 # optional: SPADE engine on any GPU vendor
```

**PyTorch:** a plain `pip install torch` gives a CPU-only build on Windows.
For the NMF engine on an NVIDIA GPU, install the CUDA build with the command
for your system from <https://pytorch.org/get-started/locally/>, for example:

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**AMD and Intel GPUs:** the NMF engine runs on the CPU there. For GPU speed,
pick the **Accurate SPADE** or **Unmaster SPADE** preset, which run through
OpenCL; they are less accurate (see [Results](#results)). A ROCm build of
PyTorch (the ROCm option on <https://pytorch.org/get-started/locally/>, for
supported AMD GPUs) shows the GPU to PyTorch as a CUDA device, so the NMF
engine should use it; this has not been tested here.

**GUI (Tkinter)** ships with Python on Windows and macOS. On Linux, install it
from your package manager, e.g. `sudo apt install python3-tk`. To listen to
the region in the GUI, also `pip install sounddevice` (optional; on Linux it
needs `libportaudio2` too).

### 3. GPU drivers (optional)

The NMF engine needs only the NVIDIA driver plus the CUDA build of PyTorch
above. The SPADE engine's GPU path uses OpenCL, so a normal graphics driver
is enough; no CUDA or ROCm install is needed.
- **Windows:** the standard NVIDIA, AMD or Intel driver includes OpenCL.
- **Linux:** you may need your vendor's OpenCL package. Examples:
  `nvidia-opencl-icd` with the NVIDIA driver, Mesa's `mesa-opencl-icd`
  (Rusticl) or ROCm for AMD, `intel-opencl-icd` for Intel. Names vary by
  distribution.
- **macOS:** Apple's OpenCL is deprecated and untested here. Use the CPU.

### 4. Check it works

```
python test_sharpunclip.py
python -c "import sharpunclip as s; print(s.backend_name(*s.spade_backend(True), s.DEFAULTS))"
```

The test suite runs on every backend it finds. The second command prints the
device a `--gpu` render with the default engine will use, e.g. `NMF on CUDA:
NVIDIA GeForce RTX 4070 Laptop GPU`, or `NMF on CPU`.

- **NMF engine:** CUDA when there is an NVIDIA GPU, else the CPU.
- **SPADE engine fallback order:** OpenCL first, then PyTorch CUDA, then CPU
  (PyTorch if installed, else NumPy).
- **Choosing a GPU:** with two GPUs it picks the stronger one. Set
  `SHARPUNCLIP_OPENCL_DEVICE` to part of a device name (e.g. `gfx` or
  `NVIDIA`) to choose.
- **uv users:** `uv venv` and `uv pip install ...` work the same way, or
  prefix the commands with `uv run`.

## Use

### GUI

```
python gui.py
```

1. Open a song: **Browse** next to *Input*, or Ctrl+O. The output name
   (`<name>_unclipped.wav`) is filled in, and a few seconds of the song, the
   *region*, are selected at its most-clipped spot.
2. Pick a preset. The region is previewed with it (by itself when that takes
   10 s or less, else press P).
3. Press Space to listen: 1 plays the original, 2 the processed region, 3
   their difference. Change settings while it plays.
4. Press **Render** (Ctrl+Return) to process the whole song.

#### The window

| Part | What is there |
|---|---|
| Top | Input and Output files; *Preset*, *modified*, Save, Load, Undo, Redo; *Process at Hz*, Log, GPU, **Advanced…** |
| Stage boxes | Declip → Unlimit → Uncompress → Output, in processing order. The checkbox switches the stage on or off, the text sums up its settings, and a click (or Ctrl+1 to Ctrl+4) shows its page |
| Page (right) | The stage's settings and what they do to this song |
| Waveform (left) | File info and warnings, Lin \| dB and Fit; the whole song in a strip; the time ruler; one lane per channel (up to 8); a gain lane while Unlimit or Uncompress is on; a readout line |
| Audition bar | ▶, *1 Orig*, *2 Proc*, *3 Diff*, Side, Match, Mon; the region's times; Preview, Auto, Fast and the preview's state |
| Bottom | Status, **Render**, **Cancel** and the progress bar |

- A setting can appear in several places (a page, the Advanced window, a line
  on the waveform); they all show the same value. Settings that don't apply to
  the current engine or detection mode are greyed out.
- **Advanced…** (Ctrl+E) opens the detailed settings in their own window.
- **Save / Load** store your settings as a `.json` preset. *modified* next to
  the preset name means the settings differ from it; after Load it reads
  *from <file>* while they match the file.
- **Undo / Redo** (Ctrl+Z, Ctrl+Y) step through finished changes.
- **Log** writes `<output>.log` next to the output on every render: settings,
  device, timing and peak, or the full error if it fails.
- **Render** draws the output (green) over the input (grey) as it goes, with
  time left and speed. **Cancel** stops without writing anything. Output made
  with other settings than the current ones is drawn dimmed, and the header
  says *older settings*.

#### Moving around the waveform

- The mouse wheel zooms around the pointer, down to single samples.
  Shift+wheel scrolls, Ctrl+wheel zooms the level (in dB view it moves the
  floor), and middle-drag pans.
- The strip on top is the whole song. Click or drag it to move the view, drag
  the box edges to zoom, and double-click it (or press **Fit**, or F) to see
  everything again.
- **Lin | dB** (V) switches the level scale. Dashed lines mark 0 dBFS, so you
  can see how far restored peaks go.
- The line under the waveform reads the time, channel and level at the
  pointer, and with Declip on that level against the measured peak (or the
  ceiling found) and whether the sample is rebuilt.
- With *Process at Hz* set, the header says so (`processing as 24 kHz (0.50×
  speed)`). Every time shown stays in the file's own time.

#### Region and preview

- **Region:** drag across the waveform or the time ruler. Drag its edges to
  resize it, drag its green bar on the ruler to move it, Shift+click to stretch
  it, or double-click for 1 s either side; Shift+drag on the strip on top works
  too. A click sets the edit cursor. Until you set one, the region goes to the
  most-clipped spot (else the loudest) and the view zooms there. N and Shift+N
  go to the next and previous clipped spot, [ and ] move the region by its
  length, Z zooms to it. Right-click the waveform to preview, clear or zoom to
  it; Esc clears it.
- **Preview:** the region is processed with the current settings and drawn
  green (dimmed: made with older settings). **Preview** shows the estimated
  time; **Auto** previews by itself when that takes 10 s or less; **Fast**
  (on by default without a GPU) previews NMF at fast quality and SPADE at
  quality 3 at most. P previews at render quality, Esc cancels. The quality
  settings show an estimate of the render time.
- *Output gain*, the limiter and the Uncompress *Amount* update the preview,
  and what you hear, at once, even while you drag them; the Unlimit *Amount*
  and the other Uncompress settings within a moment. *Input gain* redoes the
  declip like the Declip settings (at once with Declip off). Previews wait
  while a render runs; after it, the region plays the render while the
  settings match.
- How close a preview is to the render:
  - **SPADE:** exactly the render's samples, on the CPU, CUDA and OpenCL.
  - **Unlimit:** the same up to float rounding.
  - **Uncompress:** within 0.001 dB. The preview starts up to 10 s before the
    region so the compressor's state can settle (with Declip on, always 10 s,
    so a new *Release* or *Ratio* doesn't declip again). With a very slow
    release (*Ratio* × *Release* over about 2 s) the region's first seconds
    can differ slightly more.
  - **NMF:** close but not exact. The engine models the region and its
    surroundings, while the render models 20 s at a time.

The preview's state, at the right of the audition bar:

| State | Meaning |
|---|---|
| *B fresh · SPADE exact* (or *exact*) | Up to date, and the render's samples |
| *B fresh · ≈ NMF preview* | Up to date, close to the render |
| *fast preview ≠ render* | Made with **Fast**; P makes it at render quality |
| *provisional (ceilings only)* | NMF with *Auto detect* before its soft-shoulder search is done; redone when it is |
| *computing 45 % · Esc* | Running; you keep hearing the previous one |
| *stale · P* | Not up to date, and waiting for P (*Auto* is off, or a preview would take more than 10 s) |
| *B = full render* | The finished render, whose settings match |
| *paused while rendering* | Previews wait for the render |

#### Listening

Playback needs `pip install sounddevice` (on Linux also `libportaudio2`).
Without it the audition controls are greyed out with a note, and everything
else works. Nothing plays until you press Play.

- Space (or ▶) loops the region (its first 30 s when it is longer). Esc stops.
- 1, 2 and 3 switch between the original (Orig), the preview or render (Proc)
  and their difference (Diff); X swaps Orig and Proc. Switching keeps the beat
  and never clicks. Proc changes as each new preview lands.
- **Side** (S) plays only L−R, where stereo artifacts show. **Match** (M) plays
  Proc at Orig's loudness, so louder doesn't sound better.
- **Mon** is the playback level: Auto keeps the louder peak 0.3 dB under full
  scale (restored peaks would clip the sound card), or 0, −6 or −12 dB.
- Without a region, Space plays the song from the edit cursor (Proc is the
  render while it matches the settings; Diff, Side and Match need a region),
  and clicking the waveform while it plays jumps there.

#### What gets rebuilt

With manual detection (SPADE, or NMF with *Auto detect* off), on the Declip
page:
- The waveform shows the measured peak (dotted), the *Clip above* threshold
  (yellow), the *Not-clip below* line when *Dynamic threshold* is off,
  *Soft-clip detection* at 92%, the *Not-clip level* and *Gap level* lines, and
  the *Max restore* cap, each with a tag at the right. The gaps between songs
  are hatched and the songs shaded in turn.
- Red marks the rebuilt samples, amber the ones only *Soft-clip detection*
  catches, dark red the single samples *Remove glitches* drops. The other
  pages show the red faintly.
- Red triangles over the strip on top mark the five most-clipped spots (click
  one to go there), and its bottom edge shows how much is rebuilt where.
- Under the settings: a histogram of each sample's level as a share of the
  measured peak (red = rebuilt; the hatched bar holds the samples at or above
  the peak), then the share rebuilt, the clipped runs (click to go to the
  longest), how many rebuilt samples have little unclipped context in a SPADE
  frame, and the samples *Remove glitches* drops.
- An orange tag next to the file name warns of a second where more than 70% of
  samples are rebuilt (click it to go there).
- SPADE: pointing at the waveform shows the frame solved there and how much of
  it is rebuilt. Drag its right edge to set the *Frame* length, or turn the
  wheel on its label to change *SPADE quality*. A red strip under each lane
  marks frames more than half rebuilt (point at a *Frame* option to see that
  length's strip; the numbers under the options are each length's
  little-context share). The two rings between L and R switch *Linked stereo*.

With NMF and *Auto detect* (the Accurate preset), the waveform shows the
analysis the render uses:
- Blue lines mark the ceiling found for each channel and polarity (clip,
  limiter or smeared; point at one for its details). The yellow dashed line is
  the knee: red marks samples rebuilt above the ceiling, amber the ones between
  the knee and the ceiling. *Max restore* is drawn at its multiple of each
  ceiling.
- Drag the knee, or its marker in the level histogram under the settings, to
  set *NMF knee*; double-click goes back to "as found". Right-click offers
  *Ceilings only* and *Use as manual threshold* (also a button: *Auto detect*
  off, *Clip above* at the knee).
- The soft-shoulder search runs by itself (its progress shows under the
  histogram, with the ceilings and the share rebuilt). Until it is done, the
  marks and the preview are *provisional*: ceilings only. A set knee skips it.
  The render reuses this analysis instead of searching again (its log says
  `analysis from GUI`).

#### Dragging lines

The lines with a tag can be dragged, by the line or its tag: *Clip above*,
*Not-clip below*, *Max restore*, *Not-clip level*, *Gap level* and the NMF
knee on the Declip page, *Threshold* on the Uncompress page and *Ceiling* on
the Output page. The red marks follow at once; the counts update, and a
declip preview starts, when you let go.
- Shift+drag moves finely. Ctrl+drag snaps to a preset's value (a plain drag
  also sticks to one it passes close to).
- The wheel over a line moves it one step (Shift: fine).
- Double-click puts it back to the preset (the NMF knee: back to "as
  found").
- Right-click lets you type a value or switch the related options (for
  example *Dynamic threshold*, *Soft-clip detection* and *Remove glitches* on
  *Clip above*).
- Drag *Max restore* above the lane for no cap. In the histogram, drag the
  marker or click where it should go.
- Keyboard: Tab and Shift+Tab select a line (then the Uncompress page's dots
  and the gain lane), Up and Down nudge it (Shift: fine), Return types its
  value, BackSpace puts it back to the preset, Esc deselects.

#### Unlimit, Uncompress and Output

- While Unlimit or Uncompress is on, a gain lane under the waveform shows their
  gain in dB: the unlimit gain over the whole song (computed once when Unlimit
  is switched on; its state and peaks are on the Unlimit page) and the
  uncompress gain inside the region's preview (L solid, R dashed unless
  stereo-linked), faint at 100% and solid at their Amount. Drag the lane up or
  down on the Unlimit or Uncompress page to set that Amount (0.5% per pixel,
  Shift: 0.05%).
- The Uncompress page draws the transfer curves over the region's levels: the
  compressor (grey), its full inverse (dashed) and the inverse at Amount
  (solid). Drag the knee dot sideways for *Threshold*, the ratio dot for
  *Ratio* and the solid curve for *Amount*; the wheel, double-click and
  right-click work there as on the lines. The waveform gets a blue
  *Threshold* line (approximate, as uncompress works on the declipped
  signal), and in dB view the levels over it are tinted.
- On the Output page, with *Hard-clip the output* on, a pink line marks the
  *Ceiling*; the preview shows in pink what the limiter cuts, and the page
  gives the output peak (after the limiter) and the share of the region cut,
  with how far over the ceiling it went (after a render: the share of the
  song).

#### Keys

F1 (or ?) shows this list in the window. Keys without Ctrl work whenever you
are not typing in a field.

| Key | Does |
|---|---|
| Space | Play / stop (loops the region) |
| 1, 2, 3 · X | Hear Orig / Proc / Diff · swap Orig and Proc |
| S · M | Side (L−R) · level match, on or off |
| P | Preview at render quality |
| Esc | Stop the audio or the preview, clear the region, deselect |
| [ · ] | Region back · forward by its length |
| N · Shift+N | Next · previous most-clipped spot |
| V | Lin / dB |
| F · Z | Whole song · zoom to the region |
| + · − | Zoom in · out |
| Left · Right | Scroll (no line selected) |
| Home · End | Start · end of the song |
| Tab · Shift+Tab | Select a line on the page |
| Up · Down | Nudge it (Shift: fine) |
| Return | Type its value |
| BackSpace | Back to the preset |
| Ctrl+1 … Ctrl+4 | Declip / Unlimit / Uncompress / Output page |
| Ctrl+Z · Ctrl+Y | Undo · redo |
| Ctrl+O | Open a song |
| Ctrl+S · Ctrl+Shift+O | Save · load settings |
| Ctrl+E | Advanced settings |
| Ctrl+Return | Render |

### Command line

```
python sharpunclip.py song.flac -o song_unclipped.wav --gpu
python sharpunclip.py master.flac -o unmastered.wav --gpu --preset Unmaster
python sharpunclip.py song.flac --preset my_settings.json --max-restore 3
python sharpunclip.py comp.wav --no-declip --uncompress 50 \
  --comp-threshold -6 --comp-ratio 2 --comp-attack 20 --comp-release 200
```

- **Presets:** with no `--preset` it uses the Accurate settings. `--preset`
  takes a built-in name or a `.json` saved from the GUI; any other flag
  overrides that one setting.
- **Stages:** `--unlimit` / `--uncompress` switch a stage on, optionally with
  an amount (e.g. `--unlimit 30`); `0` switches it off.
- **Logging:** `--log FILE` appends a run log.
- **Files:** any input libsndfile reads (wav, flac, ogg, aiff, …). Output is
  always 32-bit float WAV, because restored peaks go above full scale.
- `python sharpunclip.py --help` lists every flag.

**Speed** (5½-minute song, on AC power; a laptop in power-saving mode can be
2× slower):

| Preset / engine | RTX 4070 laptop GPU | Laptop CPU, 16 threads |
|---|---|---|
| Accurate (NMF, normal) | 2 min (CUDA) | about 14 min |
| NMF, fast | 26 s (CUDA) | about 4 min |
| Accurate SPADE | 79 s (OpenCL) | about 36 min |

A small integrated AMD GPU runs the SPADE engine at about 5× the song's
length.

## Settings guide

Flag names are for the command line.

### Declip (`--declip` / `--no-declip`)
- **Engine** (`--engine nmf|spade`): **NMF** (default) is the most accurate
  and the least "watery"; it needs PyTorch and uses an NVIDIA GPU if there is
  one. **SPADE** runs on any GPU through OpenCL and gives the punchier
  SharpUnClip sound. Settings that don't apply to the chosen engine are
  greyed out in the GUI.
- **NMF quality** (`--nmf-quality`): *fast* (about 4× faster, 1 dB less
  accurate), *normal* (default), *high* and *best* (slower, no measurable
  gain on the test songs), and *normal+spade*, which adds SharpUnClip's own
  linked SPADE as a third model below 4 kHz: +0.25 to +0.4 dB on clipped
  songs, for one extra SPADE pass (about 40-90 s per song on a GPU, several
  minutes on a CPU).
- **Auto detect** (`--auto-detect` / `--no-auto-detect`, NMF engine): finds
  the clip or limiter ceiling itself, which was the most accurate on clipped
  songs. Off = use *Clip above* and the detection settings below; the
  Unmaster preset does that, because on limited masters the ceiling is too
  rarely touched to detect.
- **NMF knee** (`--nmf-knee`, NMF with Auto detect): where the rebuild starts,
  as a fraction of each detected ceiling (of the peak where none was found).
  0 = as the analysis finds it (default); 0.5-0.99 = everything above this
  fraction, which also skips the slow soft-shoulder search; 1 = the ceilings
  only, with no soft region below them.
- **Clip above %** (`--clip-above`; SPADE, or NMF with Auto detect off):
  anything louder than this percentage of the peak gets rebuilt. 99.9 means
  only the samples actually stuck at the ceiling, the most accurate for
  clipped songs; 75 was best for limited masters with the NMF engine. Lower
  values rebuild more and sound louder and punchier, but the extra is
  invented (78 = the SharpUnClip preset).
- **Max restore ×** (`--max-restore`): how loud a rebuilt peak may get, as a
  multiple of where it was clipped. 0 = no limit, the most accurate with the
  NMF engine. With SPADE, 2.5 was the most accurate (2 for lightly clipped
  songs, 3 for brutally clipped ones); without a limit, its long clipped
  stretches can come back as huge spikes.
- **SPADE quality 0-6** (`--quality`): how much the analysis frames overlap.
  Higher is a little more accurate and slower; 5 takes about 2× the time of 3.
- **Linked stereo** (`--linked-stereo`, SPADE, on by default): rebuilds left
  and right from the same frequency components. Separate rebuilds make the
  stereo image swirl in rebuilt spots, heard as a "watery" sound. Linking
  improved every measure on both test songs. The NMF engine always links the
  channels.

### Unlimit (`--unlimit [AMOUNT]`)
- **Amount %**: 100 = the model's full correction, 200 = double (in dB). 50
  was the most accurate with Unlimit alone; above 100 got worse on both test
  songs, and the same held for iZotope's Unlimiter. The Unmaster preset uses
  15, because its declip stage already undoes part of the limiting (the
  declip's correction is added on top of the unlimited song). Peaks are
  turned up and the song's body keeps its level, so expect peaks over
  0 dBFS. Needs PyTorch.

### Uncompress (`--uncompress [AMOUNT]`)
Enter the settings of the compressor that was used, and it is run backwards.
The model is a standard digital compressor: peak detection, hard knee.
- **Threshold dB / Ratio / Attack ms / Release ms / Makeup dB**
  (`--comp-threshold`, `--comp-ratio`, `--comp-attack`, `--comp-release`,
  `--comp-makeup`) and **Stereo-linked** (`--comp-link`).
  - Threshold and ratio matter most.
  - If you don't know the attack and release, leave the defaults (10 /
    150 ms); on the test renders they did as well as the exact times.
  - Makeup only changes the overall level, so 0 is fine.
- **Amount %**: how much of the restoring gain to apply. Real compressors
  differ slightly from the model, and undoing a high ratio multiplies that
  difference.
  - Slow attack (10 ms or more): use 50–75%.
  - Very fast attack (5 ms or less): use 15–25%.
- **There is no automatic mode.** Guessing threshold and ratio from the
  compressed audio alone was tested, and it can't tell compression apart from
  a normally loud mix.

### Main window
- **Process at Hz** (`--process-rate`) treats the samples as if they had this
  sample rate while processing (e.g. half the file's rate = played at half
  speed, no resampling), then writes the result at the file's own rate. 0 =
  off.
  - Declip: SPADE frames are counted in samples, so they don't change
    (*Lookahead* is the matching control). Everything measured in seconds
    does: the NMF engine's 93 ms windows and the detection times (*Peak
    time*, *Reset time*, *Gap length*).
  - Unlimit: changes the timescale the model sees.
  - Uncompress: your entered compressor times are converted automatically.
- **GPU** (`--gpu`, or `--gpu opencl` / `--gpu cuda` to force one): falls back
  to the CPU; the status line shows the device used.

### Advanced… → Declip
These detection settings apply to the SPADE engine, and to the NMF engine
with *Auto detect* off.
- **Detection**
  - *Not-clip below %* (`--not-clip-below`): used only with *Dynamic
    threshold* off (greyed out otherwise); quieter samples are never touched.
    Keep it just under *Clip above*.
  - *Not-clip level dB* (`--not-clip-amp`): quiet parts below this level are
    never touched.
  - *Dynamic threshold* (`--dynamic`): on = only samples above *Clip above*
    are rebuilt; off = the band down to *Not-clip below* is too.
  - *Remove glitches* (`--remove-glitch`): ignores single stray samples.
  - *Soft-clip detection* (`--soft`): also rebuilds rounded, near-ceiling
    tops. It can touch clean peaks.
- **Peak measurement**
  - *Peak ignore* (`--peak-ignore`): ignores the loudest stray samples (0.001
    = 1 in 1000).
  - *Peak time s* (`--peak-time`): how long a stretch the peak is measured
    over. Only used with *Measure each song separately* off.
  - *Reset level dB* (`--reset-amp`): the measured peak never goes below
    this level. *Reset time s* (`--reset-time`) has no effect and is greyed
    out; it is kept for older presets.
- **Different clips**
  - *Measure each song separately* (`--diff-clips`): for mixes and streams.
    It finds the gaps between songs (*Gap level dB*, *Gap length s*:
    `--trans-amp`, `--trans-time`) and measures each song's clip level on its
    own. A gap uses the level of the song after it.
- **SPADE frames**
  - *Lookahead 0-2* (`--lookahead`): frame length 512 / 1024 / 2048 samples.
    0 was the most accurate. (The NMF engine uses fixed 93 ms windows.) It is
    also the *Frame* row on the Declip page.

### Advanced… → Output
Levels and Limiter are also on the Output page of the main window.
- **Levels:** *Input / Output gain dB* (`--input-gain`, `--output-gain`).
- **Limiter:** *Hard-clip the output* (`--limiter`) at *Ceiling dB*
  (`--limiter-peak`). This cuts restored peaks back off, so use it only if the
  file has to stay under full scale.
- **Performance:** *CPU threads* (`-t`).

### Quick recipes
- **As close to the original as possible:** the **Accurate** preset.
- **No NVIDIA GPU and a long song:** *NMF quality* fast (faster than the
  song plays, on a laptop CPU), or **Accurate SPADE** / **Unmaster SPADE** on
  an AMD/Intel GPU.
- **"No clipping found"** (the CLI says so, and the output equals the input):
  auto detection saw no ceiling, which is normal for limited masters. Use the
  **Unmaster** preset, or switch *Auto detect* off.
- **The song is mastered/limited, not just clipped:** the **Unmaster** preset.
- **The song is compressed but you don't know how:** the **Uncompress**
  preset, a gentle generic inversion. Use it only on compressed songs: it
  also changes audio that was never compressed.
- **You know the compressor's settings:** switch on **Uncompress**, enter
  them, and set Amount to 50% (15–25% if its attack was very fast).
- **More punch than was there:** raise *Max restore* to 3–4, or use the
  **SharpUnClip** preset.
- **Spikes or harsh peaks:** lower *Max restore* to 2.
- **Rebuilt peaks top out around +10 dB:** that is usually where the real
  peaks were. With SPADE, *SPADE quality* 0–1 or *Lookahead* 2 add a few dB
  at the very top, at some cost in accuracy.
- **More than about 70% of samples over the threshold:** no declipper can
  recover that.
- **Known soft clipping** (a cubic soft clipper such as PeakEater's *Cubic*,
  no limiter after it): add `--soft-clip cubic` on the command line. It
  undoes the cubic curve exactly before the declip stage rebuilds the flat
  tops (+0.6 to +1.3 dB on the soft-clipped test render). It is wrong for
  anything else: on hard-clipped songs it cost 6 dB, so it is off by
  default, CLI only, and never guessed.

## Testing

```
python test_sharpunclip.py   # engine + CLI tests, on every backend found (CPU, CUDA, each OpenCL GPU)
python gui.py --selfcheck    # every preset through the real GUI fields, plus the GUI's displays, previews and playback (offline, silent)
python abtest.py --selftest  # the scoring code
```

To measure quality on your own material, render the same project twice, once
clean and once with clipping/limiting/compression, then:

```
python abtest.py --orig clean.wav --degraded clipped.wav --args "--preset Accurate --gpu"
```

It checks the renders line up, then reports SNR, SI-SDR, stereo-side accuracy
and high-frequency spectral error before and after. These are the scores used
to tune SharpUnClip. The test songs themselves are not included: they are
unreleased music, and too large for GitHub.

## Files

| File | Contents |
|---|---|
| `sharpunclip.py` | Pipeline, SPADE engine and command line |
| `betterdeclipper/` | NMF declip engine (BetterDeclipper by Qupci, MIT) |
| `sharpunclip_cl.py` | OpenCL GPU kernel for the SPADE engine (any vendor) |
| `unlimit.py` | De-limiter network (PyTorch, no other dependencies) |
| `gui.py` | Tkinter GUI |
| `waveview.py` | GUI waveform view |
| `session.py` | GUI logic without Tk: waveform envelopes, region previews, playback |
| `abtest.py` | Scoring against a known original |
| `test_sharpunclip.py` | Automated tests |
| `weights/` | Pretrained De-limiter weights |
| `LICENSE`, `LICENSES/` | MIT license (SPDX/REUSE layout) |
| `THIRD_PARTY_NOTICES.md` | Third-party material, its full license texts, papers, training data |
| `METHODS.md` | Methods, modifications to the papers, testing, results, limitations |

## References

- Qupci, *BetterDeclipper*, 2026, <https://github.com/Qupci/BetterDeclipper> (MIT): the NMF declip engine.
- K. Siedenburg, M. Kowalski, M. Dörfler, "Audio declipping with social sparsity," *IEEE ICASSP*, 2014.
- D. D. Lee, H. S. Seung, "Algorithms for non-negative matrix factorization," *NIPS*, 2001.
- A. Beck, M. Teboulle, "A fast iterative shrinkage-thresholding algorithm for linear inverse problems," *SIAM Journal on Imaging Sciences*, 2009.
- S. V. Venkatakrishnan, C. A. Bouman, B. Wohlberg, "Plug-and-play priors for model based reconstruction," *IEEE GlobalSIP*, 2013.
- P. Záviška, P. Rajmic, Z. Průša, V. Veselý, "Revisiting synthesis model in sparse audio declipper," *LVA/ICA*, 2018.
- S. Kitić, N. Bertin, R. Gribonval, "Sparsity and cosparsity for audio declipping: a flexible non-convex approach," *LVA/ICA*, 2015.
- P. Záviška, P. Rajmic, O. Mokrý, Z. Průša, "A proper version of synthesis-based sparse audio declipper," *IEEE ICASSP*, 2019.
- P. Záviška, P. Rajmic, A. Ozerov, L. Rencker, "A survey and an extensive evaluation of popular audio declipping methods," *IEEE Journal of Selected Topics in Signal Processing*, 2021.
- C.-B. Jeon, K. Lee, "Music De-limiter Networks via Sample-wise Gain Inversion," *IEEE WASPAA*, 2023, arXiv:2308.01187, <https://github.com/jeonchangbin49/De-limiter>.
- S. Gorlow, J. D. Reiss, "Model-Based Inversion of Dynamic Range Compression," *IEEE Transactions on Audio, Speech, and Language Processing*, 2013.
- D. Giannoulis, M. Massberg, J. D. Reiss, "Digital Dynamic Range Compressor Design—A Tutorial and Analysis," *Journal of the Audio Engineering Society*, 2012.
- Y. Luo, N. Mesgarani, "Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation," *IEEE/ACM TASLP*, 2019.
- M. Pariente et al., "Asteroid: the PyTorch-based audio source separation toolkit for researchers," *Interspeech*, 2020 (MIT); `unlimit.py` mirrors its module structure.
- J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, "SDR – Half-baked or Well Done?," *IEEE ICASSP*, 2019 (SI-SDR, used for scoring).
- Z. Rafii et al., "MUSDB18-HQ - an uncompressed version of MUSDB18," 2019 (De-limiter's training data).
- J. W. Cooley, J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series," *Mathematics of Computation*, 1965 (FFT in the GPU kernel).

How each one was used and modified: [METHODS.md](METHODS.md#10-references).

**Built with** NumPy, SciPy, soundfile/libsndfile, PyTorch + torchaudio,
PyOpenCL and Tkinter. iZotope Ozone's Unlimiter appears only as a comparison; iZotope and
Ozone are trademarks of their owner, and this project is not affiliated with
them.

## License

SharpUnClip is released under the **MIT License**; see [LICENSE](LICENSE) and
[LICENSES/MIT.txt](LICENSES/MIT.txt). The project follows the
[REUSE](https://reuse.software) specification: every file has an SPDX header
with its copyright holders and license, and binary files have a `.license`
file next to them.

Third-party material keeps its copyright; it is all MIT-licensed. Details and the
original license texts are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md):
- `weights/delimiter.pth` is the pretrained De-limiter model by Chang-Bin Jeon
  and Kyogu Lee (© 2023 jeonchangbin49). It was trained on research music data
  (MUSDB18-HQ), so check those terms before any commercial use of the Unlimit
  stage.
- `unlimit.py` mirrors the network structure of Asteroid (© 2019 Pariente Manuel).
- `betterdeclipper/` is BetterDeclipper's engine (© 2026 Qupci), with small
  changes listed in THIRD_PARTY_NOTICES.md.
- The cited papers describe methods; no text or code from them is included, so
  no license from them applies.
