<!--
SPDX-FileCopyrightText: 2026 thatrandomalt

SPDX-License-Identifier: MIT
-->

# SharpUnClip: methods, modifications and testing

This document describes what SharpUnClip computes, which published methods it
is built on, exactly how those methods were changed, and how every default
was chosen and checked. All numbers below were measured on the material
described in [Testing methodology](#5-testing-methodology).

Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Declip: two engines (NMF, SPADE)](#2-declip-two-engines-nmf-spade)
3. [Unlimit: De-limiter](#3-unlimit-de-limiter)
4. [Uncompress: model-based compressor inversion](#4-uncompress-model-based-compressor-inversion)
5. [Testing methodology](#5-testing-methodology)
6. [Results](#6-results)
7. [Tried and rejected](#7-tried-and-rejected)
8. [Limitations](#8-limitations)
9. [How this project was made (AI assistance)](#9-how-this-project-was-made-ai-assistance)
10. [References](#10-references)

---

## 1. Pipeline overview

A loud master is usually made by compressing, then limiting, then clipping.
SharpUnClip undoes those steps in reverse order, each switched on separately:

```
input --(input gain)--> Declip --> Unlimit --> Uncompress --(output gain, optional hard clip)--> 32-bit float WAV
```

| Stage | Built on | What it needs |
|---|---|---|
| Declip, NMF engine (default) | BetterDeclipper (Qupci 2026): NMF and social-sparsity plug-and-play with stereo A-SPADE, fused | PyTorch, SciPy (detects clipping itself) |
| Declip, SPADE engine | SPADE (Kitić et al. 2015), frame-wise A-SPADE | nothing (detects clipping itself) |
| Unlimit | De-limiter (Jeon & Lee 2023), pretrained network | PyTorch |
| Uncompress | Model-based inversion (Gorlow & Reiss 2013) of the compressor design of Giannoulis, Massberg & Reiss (2012) | the compressor's threshold and ratio |

Output is always 32-bit float, because restored peaks are louder than full scale.

---

## 2. Declip: two engines (NMF, SPADE)

SharpUnClip has two declip engines:
- **NMF** (default): BetterDeclipper's restoration engine, by a third party,
  included in `betterdeclipper/`. It was the most accurate on every clipped
  test song (section 6.1).
- **SPADE:** SharpUnClip's own A-SPADE implementation. It is kept for GPUs
  without CUDA (it runs on any vendor through OpenCL), for machines without
  PyTorch, and for the punchy **SharpUnClip** preset.

### 2.1 NMF engine: what it is

**Origin.** `betterdeclipper/` is the restoration engine of
[BetterDeclipper](https://github.com/Qupci/BetterDeclipper) by Qupci (v0.3.0,
commit `f9ff7f4`, MIT license), included without its command line. The only
code change is a progress callback inside its SPADE stage, so SharpUnClip's
progress bar and Cancel keep working during that stage. With the same
settings it gives bit-identical output to the original (checked on the full
MD song: maximum difference 0).

**What it computes** (summarized from its documentation and code):
1. **Analysis.** It fits the top of the amplitude histogram to find the
   ceiling, and tells three kinds of damage apart by the runs of samples at
   the ceiling: hard clipping (long flat runs), a smeared ceiling from a lossy
   codec, resampling or an oversampled clipper (long rippled runs), and
   limiting (isolated touches). A quick first restoration pass finds the knee
   of any soft saturation below the ceiling.
2. **Consistency.** Samples below the knee are kept exactly. Above it, a
   sample may only grow, keeping its sign.
3. **Three restoration models,** each giving a consistent signal:
   - *NMF plug-and-play:* iterations x ← Wiener(P(x)), where P projects onto
     the consistent signals and the Wiener gain V/(V + λ²) takes its power V
     from a low-rank non-negative matrix factorization (Lee & Seung 2001) of
     the current spectrogram. The spectral templates are shared by both
     channels, and λ is annealed. This is the plug-and-play scheme of
     Venkatakrishnan, Bouman & Wohlberg (2013) with an NMF denoiser: repeating
     sounds such as drum hits and notes are explained by a few templates, and
     clipping distortion is not.
   - *PEW plug-and-play:* the same iteration with the "persistent empirical
     Wiener" social shrinkage of Siedenburg, Kowalski & Dörfler (2014) and
     FISTA momentum (Beck & Teboulle 2009).
   - *Stereo A-SPADE* (Kitić et al. 2015; Záviška et al. 2018), with the k
     largest coefficients chosen jointly over both channels.

   All three work on PCA-rotated channels (a mid/side-like basis), so the
   unclipped samples of one channel inform the other; the minor component is
   regularized harder.
4. **Fusion.** The models' corrections are averaged with fixed weights, in two
   bands split at 4 kHz; above 4 kHz only the NMF model contributes, because
   SPADE adds spiky high-frequency errors in long clipped gaps. An average of
   consistent signals is itself consistent.
5. **Processing.** 93 ms STFT windows with 75% overlap, 20 s chunks with 1.5 s
   context, float32 on PyTorch, with CUDA graphs on NVIDIA GPUs.

*NMF quality* picks the model set: **fast** = NMF model only; **normal** =
NMF 0.65 + SPADE 0.35 below 4 kHz; **high** = NMF + PEW + SPADE; **best** =
high with twice the iterations.

### 2.2 NMF engine: how SharpUnClip uses it

1. **Automatic or manual detection.** *Auto detect* (on by default) uses the
   engine's own analysis. With it off, SharpUnClip's own detection (*Clip
   above* %, per-section peaks and the rest of section 2.5, item 1) picks the
   samples to rebuild, and the same per-sample constraint SPADE uses (sign
   kept, |x̂| ≥ |y|, section 2.5, item 2) becomes the engine's constraint set.
   On the mastered songs this beat the engine's own soft-knee analysis
   (section 6.3), so the **Unmaster** preset uses it.
2. **Max restore** becomes the engine's gain cap: |x̂| ≤ m·|y| with manual
   detection, m × the detected threshold with auto detection. For this engine
   no cap is best (section 6.1). Its models undershoot peaks slightly rather
   than overshoot, unlike SPADE.
3. **Device.** PyTorch's GPU when *GPU* is on and PyTorch sees one: CUDA
   (NVIDIA, or AMD through a ROCm build, which PyTorch presents as CUDA), Intel
   XPU or Apple MPS; otherwise the CPU. The engine is plain float32 PyTorch, so
   the only device-specific code is CUDA graphs (CUDA only, off on ROCm) and
   the memory query that sizes the SPADE stage's batches. On ROCm, MIOpen is
   switched off: it tunes every new convolution shape on first use, which can
   take minutes per shape, and the engine's smoothing convolutions change
   shape between chunks; PyTorch's own GPU convolution runs them instead. Ops
   MPS lacks fall back to the CPU (`PYTORCH_ENABLE_MPS_FALLBACK`).
4. **normal+spade** (an *NMF quality* option, off by default). SharpUnClip's
   own linked SPADE (section 2.4: 512-sample frames, 16× overlap, capped at
   2.5|y|) joins the engine's two models below 4 kHz, weighted NMF 0.52,
   engine SPADE 0.28, ours 0.2; above 4 kHz the NMF model is used alone, as in
   *normal*. All three models get the same constraint set, and the engine's
   consistency projection runs after the fusion. It costs one extra SPADE
   pass (OpenCL on a GPU, otherwise the CPU). Results in section 6.1.

### 2.3 SPADE engine: the published method

SPADE (SParse Audio DEclipper, Kitić, Bertin & Gribonval 2015) treats
declipping as a sparse inverse problem. Samples are split into *reliable*
ones (kept exactly) and *clipped* ones, which only need to be consistent with
the clipping: on the positive rail the true value is at least the clipping
threshold θ, on the negative rail at most −θ. Among all signals consistent
with that, SPADE looks for one whose DFT is sparse. The analysis variant
(A-SPADE) solves this with an ADMM loop that alternates hard-thresholding of
the DFT to its *k* largest coefficients with projection onto the consistency
set. *k* starts at 1 and grows by 1 every iteration until the residual drops
below ε. The signal is processed in overlapping frames that are recovered
independently and overlap-added.

The synthesis variant was later corrected by Záviška, Rajmic, Mokrý & Průša
(2019). Their reference implementations and survey (Záviška et al. 2021) set
the conventions used here: rectangular analysis frames, a DFT with redundancy
2 (zero-padded to twice the frame length, a Parseval tight frame), sparsity
step s = 1 every r = 1 iteration, ε = 0.1, and Hann-weighted overlap-add.

### 2.4 SPADE engine: as implemented

`sharpunclip.aspade` and `_aspade_frames`, plus the OpenCL kernel in `sharpunclip_cl.py`:

- **Frames:** rectangular frames of w = 512, 1024 or 2048 samples (*Lookahead*
  0/1/2), hop w/2 to w/16 (*Quality* 0–6). Frames with no clipped sample are
  skipped.
- **Transform:** DFT of length N = 2w (redundancy 2), scaled by 1/√N so the
  adjoint is the scaled inverse.
- **Iteration (A-SPADE):** with z = A·x̂ and dual variable u,
  `z̄ = H_k(z + u)`, `x̂ = proj(Aᴴ(z̄ − u))`, `u ← u + A·x̂ − z̄`, then k ← k + 1.
  It stops when ‖A·x̂ − z̄‖ ≤ ε = 0.1, or after 1000 iterations.
- **Recombination:** Hann-weighted overlap-add, then one last projection onto
  the constraints in float64. The GPU runs in float32, which matched float64
  to within 0.01 dB in every test.

### 2.5 SPADE engine: modifications to SPADE

1. **Clip detection relative to the measured peak.** SPADE assumes the
   clipping threshold θ is known. Here the peak is measured from the signal:
   the (1 − *peak ignore*) quantile of |x| per *peak time* block, or per section
   between silent gaps with *Different-clips detection*. A sample counts as
   clipped when |x| ≥ peak × *Clip above* %. Further options:
   - *Dynamic threshold* off: also flags the band down to *Not-clip below* %.
   - *Not-clip level*: samples below this level are never flagged.
   - *Remove glitches*: drops single-sample runs.
   - *Soft-clip detection*: also flags |x| ≥ 0.92 × peak.
2. **Per-sample lower bound instead of a single threshold.** A flagged sample
   y keeps its sign and must come back with |x̂| ≥ |y|, not just ≥ θ. That is
   the right constraint when the clipping level varies (limited or
   soft-clipped material) and the same thing when it doesn't.
3. **Upper bound (Max restore).** A box constraint |x̂| ≤ m·|y| is added to the
   projection, with m = 2.5 in the Accurate SPADE preset. Plain SPADE has no upper bound and
   overshoots badly on long clipped runs. On real clipped songs the bound
   roughly doubled the improvement (section 6.1).
4. **Linked stereo (joint sparsity across channels).** SPADE is single-channel.
   Here each frame of every channel is recovered together: one shared set of
   *k* DFT bins is chosen from the channels' summed energy Σ_c |z_c + u_c|²,
   and a frame group stops when every channel has converged. A channel with
   no clipping in that frame still contributes its spectrum, so it guides its
   clipped partner. Independent per-channel recovery made the stereo image
   swirl in rebuilt regions, which was heard as a "watery" sound. Linking
   fixed most of it (section 6.2).
5. **Level normalization.** ε is an absolute residual, so the original
   algorithm gives different results for the same audio at different levels.
   Here the signal is scaled to peak 1 before SPADE and back after, making the
   result independent of input level (a test checks this to 1e-6).
6. **Streaming.** Frames are built and solved in time-ordered batches of 2048
   frame positions, so memory stays constant with song length. Each batch's
   finished part is passed to a progress callback; the GUI uses it to draw
   the output as it renders.
7. **GPU kernel (OpenCL, any vendor).** One work-group runs entire A-SPADE
   iterations for one frame group in on-chip memory:
   - Both transforms use a radix-2 Cooley–Tukey FFT.
   - The k-th largest energy is found by a 4-bit radix select on the float
     bit patterns.
   - Each launch runs 25 iterations, which keeps every launch far below
     Windows' 2-second GPU watchdog and lets the user cancel.

   It produces the same scores as the PyTorch/NumPy path and runs about 4×
   faster than PyTorch CUDA on an RTX 4070 laptop GPU.

---

## 3. Unlimit: De-limiter

### 3.1 The published method

Jeon & Lee (2023) model a limiter as a sample-wise gain reduction and train
a network to invert it ("sample-wise gain inversion"). A ConvTasNet-style
encoder/masker/decoder predicts one gain per sample in (0, 1), shared by both
channels, and the output is the input times that gain. It was trained on
musdb18-HQ mixtures limited with a commercial limiter (musdb-XL) at 44.1 kHz,
with an SI-SDR loss. The authors publish code and weights under the MIT
license.

### 3.2 Modifications

1. **Re-implemented without its framework.** The De-limiter network is a
   Conv-TasNet-style model (Luo & Mesgarani 2019) built with the Asteroid
   toolkit (Pariente et al. 2020, MIT license). It is rebuilt in plain PyTorch
   (`unlimit.py`), mirroring Asteroid's free filterbank encoder/decoder and
   TDConvNet masker (global layer norm, dilated depthwise blocks, skip
   connections) with the same layer names, so the published
   weights (`weight/all.pth` in the authors' repository, shipped here as
   `weights/delimiter.pth`) load unchanged. Checked against the original asteroid-based model on random
   stereo input: maximum gain difference 6 × 10⁻⁸.
2. **Gain applied at the file's own sample rate.** The network still runs at
   44.1 kHz (input resampled with `torchaudio`). Only the predicted gain
   curve, which is smooth, is interpolated back to the file's rate and
   applied to the original-rate audio. The audio itself is never resampled.
3. **Body level kept, Amount in dB.** The paper's output is the input times a
   gain in (0, 1), then loudness-normalized. Here the gain curve is divided by
   its median, so the song's body stays at its level and peaks get louder,
   then raised to the power *Amount*/100 (a dB scaling: 200% doubles every dB
   of correction). 50% was the most accurate when Unlimit runs alone; the
   Unmaster preset uses 15%, because its declip stage already undoes part of
   the limiting (section 6.3).
4. **Chunked inference.** 6-second chunks are crossfaded over 1 second, so any
   song length fits in memory.
5. **Pipeline placement.** The gain is predicted from the input (before
   declipping) and applied to the input, and the declip stage's correction is
   added unscaled: out = y·g + (x̂ − y). Multiplying the declipped signal
   instead applied the gain a second time to the rebuilt peaks (section 6.3).
   With declip off this is exactly y·g.

---

## 4. Uncompress: model-based compressor inversion

### 4.1 The published methods

Giannoulis, Massberg & Reiss (2012) describe the standard digital compressor:
a level detector, a static gain computer (threshold T, ratio R, knee), and
gain smoothing with separate attack and release. Gorlow & Reiss (2013) show
that a known feed-forward compressor can be inverted exactly. Given the
compressor's parameters and the state of its smoother, the input level that
produced each output sample can be solved for, so the original is recovered
sample by sample.

### 4.2 As implemented and modified

`sharpunclip.uncompress_gain`:

1. **Model chosen by fitting the author's compressor.** The author's
   compressor renders were compared with the unmastered originals and several
   variants of the design were fitted. Best fit: peak detector, hard knee,
   gain smoothed in the dB domain, attack/release defined as 10–90% rise
   times, each channel separate (stereo link optional), makeup gain. At 2:1
   with 20 ms attack it reproduces the compressor's gain to 0.25 dB RMS.
2. **Inversion.** Per sample, given the previous smoother state s and the
   output level Y, the gain computer is piecewise linear in dB. So the input
   level X has a closed form for the attack branch and for the release
   branch, and the branch that is consistent with the smoother's direction is
   used. Synthetic compress → uncompress round trip: maximum error 3 × 10⁻¹⁵.
3. **Partial inversion (Amount).** Real compressors differ slightly from any
   model, and inverting ratio R multiplies that difference by R. Most
   noticeably, a 1 ms attack moves the gain within a single waveform cycle in
   ways the model doesn't reproduce exactly. *Amount* applies a fraction of
   the restoring gain in dB.
   - Slow attack (≥ 10 ms): 50–75% is best.
   - Very fast attack (≤ 5 ms): 15–25%; 100% made those renders worse
     (section 6.4).
4. **Generic times.** With the right threshold and ratio, attack/release of
   10/150 ms did as well as the true times on all seven test renders. Those
   are the defaults.
5. **Process rate.** The optional *Process at Hz* setting runs the whole
   pipeline as if the samples had another rate, then writes them back at the
   original rate. The entered compressor times are converted so they still
   describe the real compressor; a test checks the round trip stays exact.

---

## 5. Testing methodology

### 5.1 Test material

All of the material below is private and not in the repository (unreleased
music, and too large for GitHub). `abtest.py` reproduces the measurements on
any material you have.

**Ground truth:** two songs, *MD* (5:34) and *dchc* (2:18), 48 kHz stereo,
rendered by the author from their own projects. Each unmastered mix was
exported as 32-bit float, so peaks above full scale were kept (MD peaks at
+10.5 dB, dchc at +17.7 dB). These are the ground truth.

**Degraded versions, rendered from the same projects:**
- **Clipped:** an oversampled hard clipper at 0 dBFS (4.7% and 10.5% of
  samples pinned at the ceiling). MD's clipped render matches a plain
  sample-by-sample hard clip of its original at 47.8 dB SNR, so for scoring
  it behaves like a plain hard clip.
- **Mastered:** a limiting chain with a −0.01 dBFS ceiling.
- **Compressed:** compressor only, with no limiter, at seven settings (dB,
  ratio, attack/release in ms):

  | Song | Threshold | Ratio | Attack / release |
  |---|---|---|---|
  | MD | −6 | 2:1 | 20 / 200 |
  | MD | −6 | 6:1 | 1 / 60 |
  | dchc | −6 | 2:1 | 20 / 200 |
  | dchc | −6 | 6:1 | 1 / 60 |
  | dchc | −10 | 3.5:1 | 0.725 / 100 |
  | dchc | −8 | 8:1 | 10 / 200 |
  | dchc | −12 | 5:1 | 40 / 50 |

  The −6 dB settings on each song were also rendered followed by the
  limiter.

**Commercial reference:** iZotope Ozone's Unlimiter on both masters, with
*Learn Threshold* and Amount at 100% and 200%.

**Third song, one stage at a time:** *dhdpmtws* (3:40, 48 kHz float), rendered
from a project whose truth has no instrument clippers, no master processing,
no sidechain ducking and no pre-clipped samples (its distorted bass, which is
sound design, stays on in every version). It renders deterministically: the
sidechain version is bit-identical to the truth wherever the ducking is idle.
The truth peaks at +17.5 dBFS, so every master stage here works far harder
than on MD.

| Version | Processing |
|---|---|
| Instrument Clipper | the project's per-instrument clippers on; changes 86% of samples (error energy −7.6 dB) |
| Clipper | master clipper only: PeakEater, cubic (soft) curve, no oversampling, +0.97 dB drive; 10.3% of samples at the ceiling |
| Limiter | master Ozone Maximizer only: IRC 5 (multiband), True Peak, +1 dB gain, character 0.25, stereo independence 100% |
| Limiter + Clipper | the same Maximizer with Soft Clip 25% |
| Full Master | instrument clippers, then the Maximizer |
| Truth + Sidechain | the truth with kick sidechain ducking |

**Listening material:** commercially released clipped songs, with no ground
truth, used only for listening and for checking how loud rebuilt peaks get.

### 5.2 Integrity checks

Before scoring, every degraded render was checked against its original. The
same checks are built into `abtest.py`.

- **Alignment:** cross-correlation showed lag 0 everywhere.
- **Muted sections:** none. An early set of dchc compressor renders had about
  28 s muted; they were re-rendered.
- **Polarity:** a compressor, limiter or clipper never flips a sample's sign.
  MD agreed with its original on 99.7–100% of samples. dchc agreed on only
  about 89% in every render. Its project evidently renders differently each
  time (free-running or random effects): even more than 256 samples away from
  any clipped sample, dchc's clipped render matches its original at only
  5.6 dB SNR (MD: 49.2 dB). That caps what any declipper can gain on dchc, so
  dchc scores only confirm the direction of a change; MD is the yardstick.
- **Model checks:** the compressor model was verified on synthetic round
  trips, and the De-limiter port against the original implementation.

### 5.3 Scores

| Score | Used for | Meaning |
|---|---|---|
| SNR improvement (dB) | declipping | Sample-exact accuracy vs the original, minus the degraded file's. Declipping doesn't change the level, so plain SNR is right. |
| SI-SDR improvement (dB) | unlimit, uncompress, chains | Scale-invariant SDR (Le Roux et al. 2019; also the De-limiter paper's measure), because the overall level may legitimately differ. |
| Stereo-side SNR (dB) | stereo swirl | SNR of L − R. Independent per-channel rebuilds lower it. |
| HF LSD (dB) | "watery" artifacts | Log-spectral distance to the original, 2–16 kHz, only in frames containing rebuilt samples. Spectral junk raises it. |
| Rebuilt peak height | loudness of restored hits | Per clipped run, the rebuilt maximum vs the true maximum, in dB over the clip level. |

### 5.4 Protocol

1. **Excerpt sweeps.** Parameters were compared on the loudest stretches of
   the clipped files (8–10 s each, 2–3 per song, ranked by fraction of
   clipped samples), since every setting has to be scored many times.
2. **Full-song confirmation.** Every default and every number in the README
   was then confirmed on full songs.
3. **Controls.** Any blind estimator was also run on the unprocessed
   originals, where it must find nothing. Model ports were checked against
   their reference implementations.
4. **Commercial reference.** iZotope's renders were scored with exactly the
   same SI-SDR as SharpUnClip's output.
5. **Adoption rule.** A change was adopted only if it improved the scores on
   both songs, or was neutral on one and clearly better on the other.

### 5.5 Automated tests

`python test_sharpunclip.py` runs every engine test on every backend present
(the SPADE tests use the Accurate SPADE settings):
CPU, PyTorch's GPU (CUDA, XPU or MPS) if present, and OpenCL once per GPU (on
the development machine: an NVIDIA RTX 4070 laptop GPU and an AMD Radeon
integrated GPU).

| Test | What it checks |
|---|---|
| `test_nmf_engine_auto` | NMF engine with its own analysis: samples below the ceiling are bit-exact, SNR improves by more than 3 dB, progress ends at 100%; on the CPU and PyTorch's GPU. |
| `test_torch_gpu_pick` | With CUDA absent, an Intel (XPU) or Apple (MPS) GPU is picked for both engines (simulated). |
| `test_gpu_probe_order` | ROCm (simulated): PyTorch's GPU is found before OpenCL is loaded, auto then skips OpenCL, and MIOpen is off. |
| `test_nmf_engine_manual_detection_bounds` | NMF engine with SharpUnClip's detection: unflagged samples bit-exact, rebuilt ones keep their sign and stay within \|y\| ≤ \|x̂\| ≤ m\|y\|. |
| `test_lsb_estimate_ignores_silent_intro` | A silent intro no longer makes 16-bit audio look like 8-bit to the NMF engine's analysis (a bug fixed in the included engine). |
| `test_spade_progress_snapshots_are_final` | Every progress snapshot already equals the final output up to its "done" point, and reporting progress changes nothing; on every backend. |
| `test_declip_below_clip_above_is_restored` | Samples below the threshold are bit-exact; rebuilt samples are never quieter; SNR improves by more than 3 dB. Also guards a past bug where rounded clip tops were missed. |
| `test_max_restore_cap_and_level_invariance` | The bound \|y\| ≤ \|x̂\| ≤ m\|y\| holds, and a quieter copy gives the same result scaled. |
| `test_linked_stereo_uses_the_clean_channel` | A clipped L guided by a clean R improves by more than 3 dB over solo recovery; R is untouched. |
| `test_declip_improves_hard_clip` | SNR improves on a synthetic kick. |
| `test_declip_improves_soft_clip` | SNR improves on a tanh-saturated signal. |
| `test_declip_diff_clips` | Per-section detection works, including a song ending in silence (a past crash). |
| `test_unlimit_linked_gain` | One gain per sample for both channels, with the median kept at 1. |
| `test_unlimit_adds_declip_correction` | Declip + unlimit equals y·g + (declip − y). |
| `test_cubic_soft_clip_inverse` | The cubic inverse round-trips exactly; `--soft-clip cubic` equals declipping the inverted input; off changes nothing. |
| `test_nmf_normal_plus_spade` | *normal+spade* keeps the engine's invariants (untouched samples bit-exact, sign kept, \|y\| ≤ \|x̂\| ≤ m\|y\|) and improves SNR by more than 3 dB. |
| `test_uncompress_inverts_compressor` | Exact inversion of the model, including at a different process rate; 50% gives exactly half the correction in dB. |
| `test_process_hints_equivalent`, `test_stage_split_equals_one_call` | Precomputed masks, gains or analyses (the GUI's region previews) change nothing, and processing in two calls (declip, then the rest) equals one. |
| `test_spade_region_preview_exact` | A SPADE region preview equals the render's samples on every backend, also across a GPU batch boundary; a span 7 samples off does not. |
| `test_uncompress_warmup` | An uncompress preview that starts before the region is within 0.001 dB of the render there (ratios 2 and 10); the lead-in is capped at 10 s. |
| `test_nmf_preview_close` | A render reusing the GUI's NMF analysis equals the CLI's; an NMF region preview is within 30 dB SNR of the render. |
| `test_cli_*` | End to end: presets, `.json` presets, logging (also of a failed run, with its traceback), stereo float output, uncompress flags. |

`python gui.py --selfcheck` drives the real window's input fields with every
preset and checks they reach the engine unchanged. It also checks the GUI's
detection display against the engine's masks, that its region previews and
renders equal `process()`, and its playback buffers (offline: no sound is
played). `python abtest.py
--selftest` checks the scoring code.

**Development environment:** Windows 11, Python 3.14, NumPy, SciPy 1.18,
PyTorch 2.11 (CUDA 12.8), pyopencl 2026.1. A tester also ran it on Linux (AMD gfx1201,
Mesa OpenCL).

---

## 6. Results

### 6.1 Declip (clipped renders, SNR improvement in dB)

**Engines, full songs** (side = stereo-side SNR change; HF LSD after
processing, lower is better; the clipped files score 3.57 / 5.09):

| Engine and settings | MD | dchc | MD side | MD HF LSD |
|---|---|---|---|---|
| **NMF, auto detect, normal, no cap (Accurate)** | **+8.15** | **+1.95** | **+8.34** | **1.67** |
| NMF, auto detect, normal, cap 2.5× | +8.10 | +1.77 | +8.38 | 1.67 |
| NMF, manual 99.9%, normal, no cap | +8.02 | +1.95 | +8.16 | 1.84 |
| NMF, manual 99.9%, normal, cap 2.5× | +7.96 | +1.77 | +8.20 | 1.84 |
| NMF, auto detect, high, no cap | +8.18 | +1.93 | +8.44 | 1.67 |
| NMF, auto detect, fast, no cap | +6.97 | +1.63 | +8.01 | 1.68 |
| SPADE, Accurate SPADE preset (99.9%, cap 2.5×, linked stereo) | +5.70 | +1.83 | +4.64 | 2.42 |

The NMF engine is about 2.5 dB more accurate on MD, nearly doubles the
stereo-side accuracy and cuts the high-frequency error (the "watery"
measure) by a third. *High* is no better than *normal* and slower; *fast*
gives up 1.2 dB on MD for about 4× the speed. The cap that SPADE needs
slightly hurts this engine, so Accurate has none.

**normal+spade** (section 2.2, item 4) against *normal*. "Synthetic" means
the unprocessed original hard-clipped to 10 dB input SNR; dh Clipper is the
third song's cubic soft clip (section 6.6); dchc counts for its sign only.

| Render | SNR | Side SNR | HF LSD | Peak error, median per run |
|---|---|---|---|---|
| MD clipped | +8.15 → **+8.56** | +8.34 → **+8.54** | 1.67 → 1.66 | −0.30 → −0.22 dB |
| MD synthetic | +8.11 → **+8.43** | +7.66 → **+7.90** | 2.97 → 2.97 | −0.31 → −0.29 dB |
| dh synthetic | +7.61 → **+7.85** | +5.99 → **+6.22** | 2.77 → 2.75 | −0.66 → −0.62 dB |
| dh Clipper | +0.00 change | | | |
| dchc clipped | +0.10 change | | | |

The third model's weight was chosen on MD (0.1 / 0.2 / 0.3: +0.24 / +0.41 /
+0.36 dB) and held on the other renders (0.3 lost 0.02 on dh Clipper).
Using our SPADE *in place of* the engine's scored up to +8.63 on MD but lost
0.1–0.7 dB on the others, so it was rejected (section 7). The cost is one
extra SPADE pass, about 40–90 s per song on the RTX 4070, so it is an option
rather than the default.

**SPADE engine, detection threshold.** Loudest-excerpt sweeps, before the
upper bound existed:

| Clip above | MD | dchc |
|---|---|---|
| 99.9% (only samples at the ceiling) | **+3.12** | **+1.15** |
| 95% | +1.82 | +0.59 |
| 90% | +0.46 | 0.00 |
| 85% | −0.81 | −0.56 |
| 78% | −2.59 | −1.30 |
| 70% | −4.57 | −2.08 |

Flagging well below the ceiling makes the result *less* like the original:
the extra rebuilt material is invented. 78% is kept as the **SharpUnClip**
preset for its louder, punchier sound.

**Upper bound and frame length** (Clip above 99.9%):

| Setting | MD | dchc |
|---|---|---|
| No bound, frame 1024 | +3.12 | +1.15 |
| Frame 512 (Lookahead 0) | +3.91 | +2.55 |
| Frame 512, bound 2× | **+5.98** | +2.31 |
| Frame 512, bound 2.5× | +5.01 | +2.86 |
| Frame 512, bound 3× | +4.37 | **+2.97** |
| Frame 512, bound 2.5×, Quality 5 | +5.46 | +2.93 |

The best bound depends on how hard a song is clipped: 2× for MD, 3× for
dchc. 2.5× is the default compromise.

**SPADE engine, full songs, final presets (with linked stereo):**

| Preset | MD | dchc |
|---|---|---|
| Accurate SPADE | **+5.70** | **+1.83** |
| SharpUnClip (78%, no bound) | −3.26 | −0.90 |

**Peak height.** On runs whose true peak was 9 dB or more over the clip
level (median true height 9.8 dB on MD, 10.7 dB on dchc), the SharpUnClip
settings rebuilt them at 10.2 and 11.3 dB. So loud hits come back at about
their real height, not limited by the algorithm.

### 6.2 Linked stereo (loudest excerpts)

| Preset, song | SNR | Side SNR | HF LSD (lower is better) |
|---|---|---|---|
| Accurate, MD | +5.32 → **+6.22** | +2.15 → **+4.66** | 3.42 → **3.06** |
| Accurate, dchc | +2.90 → **+2.99** | −0.55 → **−0.31** | 6.24 → **6.08** |
| SharpUnClip, MD | −2.89 → **−2.23** | −4.57 → **−1.41** | 4.65 → **4.24** |
| SharpUnClip, dchc | −0.63 → **−0.39** | −2.88 → **−1.45** | 6.43 → **6.26** |

Full MD song with the Accurate preset: +4.71 → **+5.70 dB**, at the same speed.

### 6.3 Unlimit (mastered renders, full songs, SI-SDR in dB)

| | MD | dchc |
|---|---|---|
| Master as-is | 15.81 | 3.37 |
| Unlimit only, 30 / 50 / 100% | 16.45 / 16.63 / 15.92 | 3.65 / 3.83 / 4.19 |
| SPADE declip at 93% only | 16.20 | 4.68 |
| First Unmaster preset (SPADE declip 93% + unlimit 50%) | 16.74 (+0.93) | 4.96 (+1.59) |
| Previous Unmaster preset (NMF declip, manual 75%, no cap + unlimit 50%, declipped signal × gain) | 17.71 (+1.90) | 5.08 (+1.71) |
| **Unmaster preset** (the same declip + unlimit 15%, declip correction added, section 3.2) | **18.16 (+2.35)** | **5.04 (+1.67)** |
| iZotope Ozone Unlimiter, 100% | 15.74 (−0.07) | 4.65 (+1.28) |
| iZotope Ozone Unlimiter, 200% | 12.59 (−3.22) | 4.38 (+1.01) |

**NMF engine on the masters.** The engine's own analysis finds no ceiling on
the MD master (a limiter's ceiling is touched too rarely), so it changes
nothing there; on dchc it finds a soft shoulder with its knee at 50% of the
ceiling, flags 38% of samples and scores worse (+0.99 with unlimit). Manual
detection with SharpUnClip's threshold, sweeping *Clip above* (SI-SDR change,
with unlimit 50%):

| Clip above | 65% | 70% | **75%** | 80% | 85% | 88% | 90% |
|---|---|---|---|---|---|---|---|
| MD | +1.16 | +1.68 | **+1.90** | +1.81 | +1.52 | +1.32 | +1.19 |
| dchc | +1.49 | +1.66 | **+1.71** | +1.70 | +1.64 | +1.57 | +1.54 |

75% is best on both songs. Without unlimit, lower thresholds score higher on
MD (+2.40 at 70%), because the declip stage then takes over part of the
limiter's undoing; with unlimit, 75% is the balance. BetterDeclipper's own
"soft" mode (knee at 80% of the ceiling) scored the same as manual 80% on MD.

That sweep used the previous combination. With the current one (unlimit 15%,
declip correction added), 65 / 70 / 75 / 80 / 85% give +2.30 / +2.51 /
+2.35 / +1.90 / +1.34 on MD and +1.51 / +1.64 / +1.67 / +1.63 / +1.54 on
dchc. 70% is 0.16 dB better on MD's master, but takes MD 6:1 + limiter to
−0.79 and changes the untouched MD original more (31.7 vs 34.0 dB SI-SDR
against itself), so 75% stays.

**Unmaster, previous → current, on every limited render** (SI-SDR change):
MD 2:1 + limiter +0.78 → +1.29, MD 6:1 + limiter −1.62 → +0.14, dh Limiter
−0.01 → +0.09, dh Limiter + Clipper +0.49 → +0.55, dh Full Master +0.09 →
+0.16. Run on the untouched originals it now changes them less (MD 32.74 →
33.99 dB, dh 30.20 → 30.66 dB SI-SDR against themselves) and leaves lower
peaks. Most of this comes from the lower amount: adding the declip
correction instead of multiplying gains 0.06–0.33 dB on its own (both at
50%). Unmaster SPADE keeps 50% (25% was mixed) and gains 0.02–0.11 dB from
the new combination.

Applying iZotope's *own* gain curve at half strength after our SPADE declip
scored 16.93 and 5.10, also better than its 100% render. Over-correction is a
common failure, which is why *Amount* defaults to 50% for Unlimit alone, and
Unmaster, whose declip already does part of the work, uses 15%.

### 6.4 Uncompress (compressed renders, full songs, SI-SDR in dB)

| Render | As is | Amount 15% | 25% | 50% | 75% | Generic times, 50% |
|---|---|---|---|---|---|---|
| MD 2:1, 20/200 ms | 22.44 | 24.15 | 25.40 | 28.57 | **29.37** | 28.91 |
| MD 6:1, 1/60 ms | 12.42 | 13.72 | **13.96** | 12.48 | 10.01 | 13.33 |
| dchc 2:1, 20/200 ms | 6.92 | 7.20 | 7.36 | 7.65 | **7.76** | 7.86 |
| dchc 6:1, 1/60 ms | 3.95 | **4.56** | 4.40 | 2.40 | −0.31 | 2.92 |
| dchc 3.5:1, 0.725/100 ms | 2.83 | | 3.31 | **3.38** | 2.99 | 2.73 |
| dchc 8:1, 10/200 ms | 3.15 | | **3.35** | 2.51 | 0.69 | 2.55 |
| dchc 5:1, 40/50 ms | 2.83 | | 2.95 | 3.06 | 3.17 | **3.83** |

"Generic times" means the true threshold and ratio with 10/150 ms
attack/release.

**Uncompress preset** (nothing known: −12 dB, 2:1, 10/150 ms, 50%), change in
SI-SDR: MD 2:1 +4.01, MD 6:1 +0.92, dchc 2:1 +0.94, dchc 6:1 +0.31,
dchc 3.5:1 +0.21, dchc 8:1 +0.49, dchc 5:1 +0.41; with the limiter
MD 2:1 +1.48, MD 6:1 +0.81, dchc 2:1 +0.41, dchc 6:1 +0.34. All 11 renders
improve, including the fast 6:1 ones where the true settings at 50% do not,
because a gentle inversion cannot overshoot far. Control: on the uncompressed
originals it gives 18.8 / 13.3 dB SI-SDR against themselves, so it is only for
audio that was compressed.

**Full chain** (compressor + limiter renders, unlimit 50% then uncompress 50%):

| Render | Limited | After |
|---|---|---|
| MD 2:1 | 15.80 | 17.39 |
| dchc 2:1 | 3.49 | 4.24 |
| dchc 6:1 | 2.90 | 4.46 |
| MD 6:1 | 11.81 | 11.16 (worse at 50%; use 15–25%) |

### 6.5 Speed

MD (5:34 stereo, 48 kHz), on AC power, nothing else running:

| Engine | RTX 4070 laptop GPU | Laptop CPU, 16 threads (from a 20 s excerpt) |
|---|---|---|
| NMF, normal (Accurate) | 118 s, CUDA | 2.6× the audio's length (about 14 min) |
| NMF, fast | 26 s, CUDA | 0.66× (about 4 min) |
| SPADE (Accurate SPADE) | 79 s, OpenCL | 6.4× (about 36 min) |

A 2-compute-unit AMD integrated GPU runs the SPADE engine at about 5× the
audio's length. Until this version the command line and GUI ran the SPADE
engine about 2× slower than these figures (164 s on MD), because every
progress update rebuilt the whole song; they now only finalize the new part
(82 s, bit-identical output). Uncompress runs on the CPU at about 30× realtime. The same
laptop in its power-saving state was about 2× slower on the GPU (SPADE:
166–182 s).

### 6.6 Third song, one stage at a time (dhdpmtws)

Changes vs the degraded file, in dB (section 5.1 describes the versions).
The Clipper version is scored against the truth times its drive (×1.118),
since a declipper should restore the level the clipper was fed.

| Degradation | Preset | SNR | SI-SDR | Side SNR |
|---|---|---|---|---|
| Master clipper (cubic, about 18 dB over) | **Accurate (NMF)** | **+2.88** | **+2.66** | +0.72 |
| | NMF, fast | +2.44 | +2.36 | +0.73 |
| | Accurate SPADE | +1.97 | +2.32 | +0.81 |
| Maximizer (IRC 5, true peak) | Unmaster | +0.34 | +0.09 | +0.03 |
| | Unmaster SPADE | +0.10 | −0.12 | −0.04 |
| | Unlimit 50% only | +0.12 | +0.11 | +0.03 |
| | Accurate (finds no ceiling) | 0 | 0 | 0 |
| Maximizer + Soft Clip 25% | Accurate | −0.34 | **+0.87** | −0.88 |
| | Unmaster | +0.36 | +0.55 | −0.10 |
| Full Master, vs truth | Accurate | −0.97 | +0.62 | −0.63 |
| | Unmaster | −0.89 | +0.16 | −0.06 |
| Full Master, vs the instrument-clipped mix | Accurate | −0.95 | +0.40 | −0.53 |
| Sidechain ducking | Unlimit 50% / 100% | 0.00 | 0.00 / −0.01 | — |
| | Uncompress preset | −6.78 | −0.50 | −4.85 |

- **Clipping:** the NMF engine is again the best, but a soft clipper 18 dB
  into the material leaves far less to recover than MD's hard clip (input SNR
  5.4 dB).
- **This limiter is out of reach today.** Ozone's IRC 5 mode limits each
  frequency band separately, and 100% stereo independence limits each
  channel separately, by up to 17 dB here. De-limiter's gain barely follows
  it: it correlates with the real gain at only 0.03–0.22 and moves about
  ±0.7 dB, where the real gain spans −1.1 to +6.2 dB. The shared gain is not
  the main problem: a correct linked gain (fitted on one channel, scored on
  the other) would still recover about +3 dB, and per-band, per-channel
  gains add only about 0.5–1 dB on top of it.
- The Unmaster SPADE row predates the current unlimit combination (section
  3.2), which moves its SI-SDR to −0.10.
- **Sidechain ducking** is not undone by Unlimit, and the Uncompress preset
  makes it worse (its threshold is in absolute dBFS, and this unmastered file
  peaks at +17.5 dBFS).
- **Controls on the untouched truth:** Accurate (NMF, auto detect) changes
  nothing. The manual-threshold presets do: Accurate SPADE raises peaks from
  7.5 to 13.7 (0.15% of samples, 30 dB SNR), Unmaster changes 37% of samples
  (30 dB SNR). They are only for audio that is known to be clipped or limited.
- **Full Master** cannot be split into its parts: its quiet intro, where the
  truth and the instrument-clipped mix are identical, is 2.5 dB louder in RMS
  but lower in peak than the Limiter + Clipper version's.

### 6.7 Soft-clip undo (`--soft-clip cubic`, command line only)

A cubic soft clipper (PeakEater's *Cubic*) maps u to c·(w − w³/6.75), w = u/c,
and is flat at ±c from |w| = 1.5. Its slope at 0 is 1, so the ceiling c =
max|y| fixes the whole curve. `--soft-clip cubic` inverts it exactly below the
ceiling, leaving a flat ceiling at 1.5c, which the unchanged declip stage
then rebuilds like a hard clip. Every later stage sees the inverted signal.

| Case | Off | Cubic |
|---|---|---|
| dh Clipper (real PeakEater render), Accurate, SNR | +2.88 | **+3.49** |
| same, Accurate SPADE | +1.97 | **+3.24** |
| same, Unmaster (previous unlimit), SI-SDR | +2.53 | **+3.02** |
| Synthetic cubic clips, Accurate: dh 5% / MD 5% / MD 1% | +6.65 / +3.28 / +1.25 | +11.61 / +14.45 / +18.71 |
| MD clipped (hard clip), SNR | +8.15 | **−6.10** |
| Ozone Maximizer + Soft Clip, Accurate, SI-SDR | +0.87 | +0.14 |

On dh Clipper it also brings big hits closer to their true height (runs at
least 9 dB over the ceiling: median error −2.18 → −1.21 dB with Accurate,
−4.45 → −1.10 dB with SPADE). Used on anything else it does real harm: hard
clips come back 3 dB too loud, and untouched audio is rewritten (peaks ×1.5).
An automatic detector failed (section 7), so the option is off by default,
command line only, and only for audio known to be cubic soft clipped.

---

## 7. Tried and rejected

Each idea below was measured with the same protocol and dropped.

| Idea | Result |
|---|---|
| Synthesis SPADE (Záviška et al. 2019) instead of analysis | Same scores (+5.04 vs +5.01 / +2.86 vs +2.86) |
| Stopping threshold ε = 0.03 or 0.3 | No change |
| Per-run upper bound from a parabola through each clipped run's edges | +5.20 / +2.70 vs the fixed 2.5× bound's +5.01 / +2.86: mixed, so the simpler fixed bound was kept |
| Leaving 1–4 samples next to each clipped run unconstrained (aimed at oversampled clippers' smear) | Worse: −0.00 to −6.33 dB |
| Narrower overlap-add weighting (Hann², Hann⁴) | Slightly worse; peak heights unchanged |
| DFT redundancy 4, longer frames, more overlap (against "watery" sound) | Mixed or worse; linked stereo fixed it |
| Blind compressor estimation from the kink in the level distribution | Failed its control: it reported ratio 20 and 5.4 on the *uncompressed* originals. Not shipped. |
| De-limiter as a blind uncompressor | No effect on compressed audio (±0.03 dB) |
| Processing at a relabelled sample rate | No effect on declip (frames are counted in samples); ±0.3 dB on unlimit. Kept as an option because users asked for it. |
| NMF engine with the 2.5× cap SPADE uses | −0.05 / −0.18 dB vs no cap (section 6.1) |
| NMF engine at *high* quality | +0.03 / −0.02 dB vs *normal*, 40% slower |
| NMF engine's automatic analysis on masters | No change on MD, worse than manual 75% on dchc (section 6.3) |
| Our SPADE *replacing* the engine's SPADE in the NMF fusion (instead of joining it, section 6.1) | Up to +0.48 dB on MD, but −0.1 to −0.7 dB on the synthetic clips and dh Clipper |
| Multiplying the declipped signal by the unlimit gain (the previous combination) | Never better than adding the declip correction, up to 1.04 dB worse |
| Unlinked or half-linked De-limiter gain (each channel from itself) | Within ±0.05 dB of linked |
| De-limiter on band-split input (2–4 bands) | The network returns "no change" above 1.5 kHz; 2 bands at 1.5 kHz halve MD's gain (+0.36 vs +0.82) |
| Holding the unlimit gain for 10/20/50 ms, or flooring it at the body level | 0.03–0.8 dB worse; holding takes MD 6:1 + limiter down to −2.8 |
| Unlimit above 50% on raw masters | dh Limiter tops out at +0.12 (75%); MD falls to +0.68 |
| Unmaster at Clip above 65–70% or 80–85% | Lower: MD 6:1 + limiter falls below its input; higher: MD's master 0.3–0.9 dB worse |
| Unmaster SPADE at 25% unlimit | Mixed: MD +0.83 vs +0.93, dchc +1.47 vs +1.59 |
| Automatic soft-clip detection (flat ceiling + pile-up under it) | Also fires on limited masters clipped by 0.02–0.3 dB, where the inversion costs Unmaster 0.6–0.7 dB on MD |
| Cubic inversion on hard clips, Ozone's soft clip or the SharpUnClip preset | MD clipped −6.10 dB SNR; Ozone soft clip +0.14 vs +0.87 SI-SDR; big SharpUnClip hits overshoot by 2 dB (section 6.7) |
| Per-band restoring gain on MD-type masters (fitted to the truth, cross-validated) | 0.10–0.39 dB worse than one broadband gain |

---

## 8. Limitations

- **Small test set.** Three songs, all very loud electronic music, one of
  which renders non-deterministically. The defaults may not be best for other
  genres. More original/degraded pairs would make them more trustworthy.
- **Objective scores only.** There was no formal listening test. The author's
  listening drove the investigation of the "watery" sound and the peak
  height.
- **NMF engine GPU support is tested on NVIDIA here, and on AMD by a tester.**
  It runs on whatever GPU PyTorch supports: AMD through a ROCm build (CUDA
  graphs and MIOpen are switched off there), Intel through XPU, Apple through
  MPS. A tester's RX 9070 (Linux, PyTorch 2.13 with ROCm 7.1) rendered on
  ROCm once the GPU was asked for before OpenCL: loading Mesa's OpenCL
  (Rusticl) into the process made PyTorch report no GPU, so with a ROCm build
  the SPADE engine now runs through PyTorch as well. Intel and Apple are
  untested; their device choice is tested by simulation.
  GPUs PyTorch doesn't support (e.g. Radeons older than RX 7000 on Windows)
  run the NMF engine on the CPU. The SPADE engine runs on any GPU vendor
  through OpenCL.
- **Very heavy clipping.** When more than about 70% of samples are flagged,
  no sparse declipper can recover the signal; SPADE then invents loud peaks.
  Short frames also overshoot on long, decaying pure low tones.
- **Clipper smear is untested.** The test clipper's oversampling left no
  measurable damage below the ceiling (section 5.1), and dchc improves less
  than MD because its renders are not repeatable (section 5.2), not because of
  smear. Clippers or lossy codecs that do smear the ceiling are handled by the
  NMF engine's smeared-ceiling detection, but no such pair was tested.
- **Unlimit's training data.** The model was trained on pop/rock stems
  limited with one commercial limiter. Out of that domain it helps less: on
  the third song's multiband, unlinked limiter its gain barely followed the
  real one, although a correct gain would have recovered about 3 dB
  (section 6.6).
- **Manual-threshold presets change unclipped audio.** Accurate SPADE,
  Unmaster and Unmaster SPADE rebuild whatever crosses their threshold, so use
  them only on songs that are clipped or limited. Accurate (NMF auto) left an
  unclipped song untouched (section 6.6).
- **Instrument-level clipping is not undone.** Clipping on single instruments
  inside the mix leaves no ceiling in the master, so both engines treat it as
  part of the song. On the third song it accounts for as much damage as the
  master clipper (section 5.1).
- **Uncompress works best with the threshold and ratio** (the Uncompress
  preset is a safe but smaller generic fallback), and models a generic
  feed-forward digital compressor. Program-dependent or analog-style
  compressors are only approximated.

---

## 9. How this project was made (AI assistance)

SharpUnClip was developed with heavy AI assistance:

- **AI-written:** the code, the experiments and this documentation were
  written by AI coding agents working under the direction of the project
  author.
- **Author's role:** the author set the goals, rendered every test file from
  their own projects (including the iZotope reference renders), listened to
  the results and reported problems, which led to the stereo-swirl and
  peak-height investigations.
- **History:** an early prototype was made with the open-source coding agent
  *opencode* running a locally hosted open-weight model. Most of it was later
  replaced; parts of the clip-detection logic date from that prototype. The
  current code, GPU kernel, experiments and documentation were produced with
  Claude (Anthropic) in Claude Code.
- **Third-party engine:** the NMF declip engine (`betterdeclipper/`) is not
  SharpUnClip's work. It is BetterDeclipper by Qupci, included under its MIT
  license; only its integration here (section 2.2) was written for this
  project.
- **How claims were checked:** everything was verified by the automated tests
  and the ground-truth measurements above, not by the AI's own claims. Review
  the code before relying on it for anything important.

---

## 10. References

**Used:**

- Qupci, *BetterDeclipper*, 2026, v0.3.0, <https://github.com/Qupci/BetterDeclipper> (MIT): the NMF declip engine, included in `betterdeclipper/`.
- K. Siedenburg, M. Kowalski, M. Dörfler, "Audio declipping with social sparsity," *IEEE ICASSP*, 2014 (PEW shrinkage in the NMF engine).
- D. D. Lee, H. S. Seung, "Algorithms for non-negative matrix factorization," *Advances in Neural Information Processing Systems 13*, 2001 (the NMF engine's spectrogram model).
- A. Beck, M. Teboulle, "A fast iterative shrinkage-thresholding algorithm for linear inverse problems," *SIAM Journal on Imaging Sciences*, 2009 (FISTA momentum in the NMF engine).
- S. V. Venkatakrishnan, C. A. Bouman, B. Wohlberg, "Plug-and-play priors for model based reconstruction," *IEEE GlobalSIP*, 2013 (the NMF engine's iteration scheme).
- P. Záviška, P. Rajmic, Z. Průša, V. Veselý, "Revisiting synthesis model in sparse audio declipper," *LVA/ICA*, 2018 (SPADE variant in the NMF engine).
- S. Kitić, N. Bertin, R. Gribonval, "Sparsity and cosparsity for audio declipping: a flexible non-convex approach," *LVA/ICA 2015* (Latent Variable Analysis and Signal Separation), 2015.
- P. Záviška, P. Rajmic, O. Mokrý, Z. Průša, "A proper version of synthesis-based sparse audio declipper," *IEEE ICASSP*, 2019.
- P. Záviška, P. Rajmic, A. Ozerov, L. Rencker, "A survey and an extensive evaluation of popular audio declipping methods," *IEEE Journal of Selected Topics in Signal Processing*, 2021.
- C.-B. Jeon, K. Lee, "Music De-limiter Networks via Sample-wise Gain Inversion," *IEEE WASPAA*, 2023. arXiv:2308.01187. Code and weights: <https://github.com/jeonchangbin49/De-limiter> (MIT).
- S. Gorlow, J. D. Reiss, "Model-Based Inversion of Dynamic Range Compression," *IEEE Transactions on Audio, Speech, and Language Processing*, 2013.
- D. Giannoulis, M. Massberg, J. D. Reiss, "Digital Dynamic Range Compressor Design—A Tutorial and Analysis," *Journal of the Audio Engineering Society*, 2012.
- Y. Luo, N. Mesgarani, "Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2019 (the architecture De-limiter is built on).
- M. Pariente et al., "Asteroid: the PyTorch-based audio source separation toolkit for researchers," *Interspeech*, 2020. <https://github.com/asteroid-team/asteroid> (MIT): `unlimit.py` mirrors its module structure, and it was used to verify the rebuild.
- J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, "SDR – Half-baked or Well Done?," *IEEE ICASSP*, 2019 (SI-SDR).
- Z. Rafii, A. Liutkus, F.-R. Stöter, S. I. Mimilakis, R. Bittner, "MUSDB18-HQ - an uncompressed version of MUSDB18," 2019 (the data De-limiter's weights were trained on, limited into musdb-XL by its authors).
- J. W. Cooley, J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series," *Mathematics of Computation*, 1965 (the GPU kernel's FFT).

**Software:** NumPy, SciPy (NMF engine's analysis and band split),
soundfile/libsndfile (audio I/O), PyTorch and torchaudio (NMF engine,
Unlimit, GPU/CPU backend, resampling), PyOpenCL (SPADE GPU backend), Tkinter
(GUI). Asteroid and asteroid-filterbanks were used during development only, to
check the De-limiter rebuild.

**Comparison product:** iZotope Ozone's Unlimiter was used only as a
reference, through renders made by the author. iZotope and Ozone are
trademarks of their owner; this project is not affiliated with or endorsed by
them.

**Considered, not used** (no released code at the time; the blind-estimation
idea was tested independently in section 7):

- H. Sun, D. Fourer, H. Maaref, "Neural-Enhanced Dynamic Range Compression Inversion: A Hybrid Approach for Restoring Audio Dynamics," arXiv:2411.04337, 2024/2025.
- H. Sun, D. Fourer, H. Maaref, "Black-Box Optimization for Identifying and Inverting Audio Dynamic Range Control Effects," arXiv:2607.19645, 2026.
