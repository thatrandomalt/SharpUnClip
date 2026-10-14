<!--
SPDX-FileCopyrightText: 2026 AG89

SPDX-License-Identifier: MIT
-->

# SharpUnClip

**Undo clipping, limiting and compression in loud music.** SharpUnClip rebuilds
clipped peaks, reverses a limiter's gain riding and inverts a known
compressor. It has a GUI and a command line, and runs on any GPU vendor
(OpenCL) or the CPU.

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
| **Declip** | Rebuilds clipped samples so the frames' spectra are sparse; untouched samples stay bit-exact, and rebuilt ones are never quieter than the input | SPADE, Kitić, Bertin & Gribonval 2015; conventions from Záviška et al. 2019/2021 |
| **Unlimit** | Divides out a limiter's gain curve, as predicted by a pretrained network | De-limiter, Jeon & Lee 2023 |
| **Uncompress** | Runs a known compressor backwards | Gorlow & Reiss 2013, on the compressor design of Giannoulis, Massberg & Reiss 2012 |

Each method was adapted and tuned: per-sample clipping bounds with an upper
bound, linked stereo, level independence, a fused GPU kernel, gain applied at
the file's own rate, and partial inversion. [METHODS.md](METHODS.md) lists every
change, and why, with the measurements.

## Results

Every default was chosen by processing real degraded renders and scoring them
against the true unprocessed originals: two songs, exported from their projects
before and after clipping, limiting or compression. The full test method and
all numbers are in [METHODS.md](METHODS.md#5-testing-methodology).

| Preset | For | Improvement vs the degraded file (song 1 / song 2) |
|---|---|---|
| **Accurate** (default) | clipped songs | **+5.7 / +1.8 dB** SNR |
| **Unmaster** | mastered (limited) songs | **+0.9 / +1.6 dB** SI-SDR; iZotope Ozone's Unlimiter, same songs: −0.1 / +1.3 (100%), −3.2 / +1.0 (200%) |
| **SharpUnClip** | maximum punch | −3.3 / −0.9 dB: louder than the original, by design |
| Uncompress, known settings | compressed songs | up to **+6.9 dB** SI-SDR (2:1 compressor); +0.6 to +1.5 dB for very fast, high-ratio compression |

Higher is better. SNR and SI-SDR measure how close the output is to the real
original.

## Install

There is nothing to compile. SharpUnClip is plain Python. Its GPU kernel is
compiled automatically by your graphics driver on the first GPU run (a few
seconds); pyopencl caches it, so later runs start immediately.

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
pip install numpy soundfile    # required
pip install pyopencl           # optional: GPU on AMD, NVIDIA or Intel (recommended)
pip install torch torchaudio   # optional: needed for Unlimit; ~8x faster on CPU
```

**PyTorch:** a plain `pip install torch` gives a CPU-only build on Windows.
That is all Unlimit needs. For the CUDA fallback backend, install the CUDA
build with the command for your system from <https://pytorch.org/get-started/locally/>,
for example:

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**GUI (Tkinter)** ships with Python on Windows and macOS. On Linux, install it
from your package manager, e.g. `sudo apt install python3-tk`.

### 3. GPU drivers (optional)

The GPU path uses OpenCL, so a normal graphics driver is enough; no CUDA or
ROCm install is needed.
- **Windows:** the standard NVIDIA, AMD or Intel driver includes OpenCL.
- **Linux:** you may need your vendor's OpenCL package. Examples:
  `nvidia-opencl-icd` with the NVIDIA driver, Mesa's `mesa-opencl-icd`
  (Rusticl) or ROCm for AMD, `intel-opencl-icd` for Intel. Names vary by
  distribution.
- **macOS:** Apple's OpenCL is deprecated and untested here. Use the CPU.

### 4. Check it works

```
python test_sharpunclip.py
python -c "import sharpunclip as s; print(s.backend_name(*s.spade_backend(True)))"
```

The test suite runs on every backend it finds. The second command prints the
device a `--gpu` render will use, e.g. `OpenCL: NVIDIA GeForce RTX 4070 Laptop
GPU`, or `CPU` when no GPU backend is available.

- **Fallback order:** OpenCL first, then PyTorch CUDA, then CPU (PyTorch if
  installed, else NumPy).
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

1. Pick an input and a preset.
2. Tick the stages you want (Declip, Unlimit, Uncompress).
3. Press **Render**.

While it renders:
- The output waveform (green) draws over the input (grey), with time left and
  speed.
- Dashed lines mark 0 dBFS, so you can see how far restored peaks go.
- **Cancel** stops without writing anything.

Other controls:
- **Advanced…** opens the detailed settings in their own window.
- **Save / Load** store your settings as a `.json` preset.
- **Log** writes `<output>.log` next to the output on every render: settings,
  device, timing and peak, or the full error if it fails.

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

**Speed** (Accurate preset, 5½-minute song):
- **RTX 4070 laptop GPU (OpenCL):** about 75 s.
- **CPU only:** roughly 20× slower.
- **Small integrated AMD GPU:** about 5× slower than realtime.

## Settings guide

Flag names are for the command line.

### Declip (`--declip` / `--no-declip`)
- **Clip above %** (`--clip-above`): anything louder than this percentage of
  the peak gets rebuilt. 99.9 means only the samples actually stuck at the
  ceiling, which is the most accurate. Lower values rebuild more and sound
  louder and punchier, but the extra is invented (78 = the SharpUnClip preset).
- **Max restore ×** (`--max-restore`): how loud a rebuilt peak may get, as a
  multiple of where it was clipped. 0 = no limit. 2.5 was the most accurate (2
  for lightly clipped songs, 3 for brutally clipped ones). Without a limit,
  long clipped stretches can come back as huge spikes.
- **Quality 0-6** (`--quality`): how much the analysis frames overlap. Higher
  is a little more accurate and slower; 5 takes about 2× the time of 3.
- **Linked stereo** (`--linked-stereo`, on by default): rebuilds left and
  right from the same frequency components. Separate rebuilds make the stereo
  image swirl in rebuilt spots, heard as a "watery" sound. Linking improved
  every measure on both test songs.

### Unlimit (`--unlimit [AMOUNT]`)
- **Amount %**: 100 = the model's full correction, 200 = double (in dB). 50
  was the most accurate on real masters; above 100 got worse on both test
  songs, and the same held for iZotope's Unlimiter. Peaks are turned up and
  the song's body keeps its level, so expect peaks over 0 dBFS. Needs PyTorch.

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
  - Declip: unaffected; its frames are counted in samples, and *Lookahead* is
    the matching control.
  - Unlimit: changes the timescale the model sees.
  - Uncompress: your entered compressor times are converted automatically.
- **GPU** (`--gpu`, or `--gpu opencl` / `--gpu cuda` to force one): falls back
  to the CPU; the status line shows the device used.

### Advanced… → Declip
- **Detection**
  - *Not-clip below %* (`--not-clip-below`): quieter samples are never
    touched; keep it just under *Clip above*.
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
    over.
  - *Reset level dB / Reset time s* (`--reset-amp`, `--reset-time`): after
    silence this quiet for this long, measuring starts over.
- **Different clips**
  - *Measure each song separately* (`--diff-clips`): for mixes and streams.
    It finds the gaps between songs (*Gap level dB*, *Gap length s*:
    `--trans-amp`, `--trans-time`) and measures each song's clip level on its
    own.
- **Frames**
  - *Lookahead 0-2* (`--lookahead`): frame length 512 / 1024 / 2048 samples.
    0 was the most accurate.

### Advanced… → Output
- **Levels:** *Input / Output gain dB* (`--input-gain`, `--output-gain`).
- **Limiter:** *Hard-clip the output* (`--limiter`) at *Ceiling dB*
  (`--limiter-peak`). This cuts restored peaks back off, so use it only if the
  file has to stay under full scale.
- **Performance:** *CPU threads* (`-t`).

### Quick recipes
- **As close to the original as possible:** the **Accurate** preset.
- **The song is mastered/limited, not just clipped:** the **Unmaster** preset.
- **You know the compressor's settings:** switch on **Uncompress**, enter
  them, and set Amount to 50% (15–25% if its attack was very fast).
- **More punch than was there:** raise *Max restore* to 3–4, or use the
  **SharpUnClip** preset.
- **Spikes or harsh peaks:** lower *Max restore* to 2.
- **Rebuilt peaks top out around +10 dB:** that is usually where the real
  peaks were. *Quality* 0–1 or *Lookahead* 2 add a few dB at the very top, at
  some cost in accuracy.
- **More than about 70% of samples over the threshold:** no declipper can
  recover that.

## Testing

```
python test_sharpunclip.py   # engine + CLI tests, on every backend found (CPU, CUDA, each OpenCL GPU)
python gui.py --selfcheck    # every preset through the real GUI fields
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
| `sharpunclip.py` | Engine and command line |
| `sharpunclip_cl.py` | OpenCL GPU kernel (any vendor) |
| `unlimit.py` | De-limiter network (PyTorch, no other dependencies) |
| `gui.py` | Tkinter GUI |
| `abtest.py` | Scoring against a known original |
| `test_sharpunclip.py` | Automated tests |
| `weights/` | Pretrained De-limiter weights |
| `LICENSE`, `LICENSES/` | MIT license (SPDX/REUSE layout) |
| `THIRD_PARTY_NOTICES.md` | Third-party material, its full license texts, papers, training data |
| `METHODS.md` | Methods, modifications to the papers, testing, results, limitations |

## References

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

**Built with** NumPy, soundfile/libsndfile, PyTorch + torchaudio, PyOpenCL and
Tkinter. iZotope Ozone's Unlimiter appears only as a comparison; iZotope and
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
- The cited papers describe methods; no text or code from them is included, so
  no license from them applies.
