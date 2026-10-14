<!--
SPDX-FileCopyrightText: 2026 AG89

SPDX-License-Identifier: MIT
-->

# SharpUnClip: methods, modifications and testing

This document describes what SharpUnClip computes, which published methods it
is built on, exactly how those methods were changed, and how every default
was chosen and checked. All numbers below were measured on the material
described in [Testing methodology](#5-testing-methodology).

Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Declip: SPADE](#2-declip-spade)
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
| Declip | SPADE (Kitić et al. 2015), frame-wise A-SPADE | nothing (detects clipping itself) |
| Unlimit | De-limiter (Jeon & Lee 2023), pretrained network | PyTorch |
| Uncompress | Model-based inversion (Gorlow & Reiss 2013) of the compressor design of Giannoulis, Massberg & Reiss (2012) | the compressor's threshold and ratio |

Output is always 32-bit float, because restored peaks are louder than full scale.

---

## 2. Declip: SPADE

### 2.1 The published method

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

### 2.2 As implemented

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

### 2.3 Modifications to SPADE

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
   projection, with m = 2.5 by default. Plain SPADE has no upper bound and
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
   of correction). 50% was the most accurate on the test masters (section 6.3).
4. **Chunked inference.** 6-second chunks are crossfaded over 1 second, so any
   song length fits in memory.
5. **Pipeline placement.** The gain is predicted from the input (before
   declipping) and applied to the declipped signal.

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
  samples pinned at the ceiling).
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
  about 89% in every render. Its project evidently renders slightly
  differently each time (free-running or random effects), so dchc scores are
  lower bounds and MD is the more reliable yardstick.
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

`python test_sharpunclip.py` runs every engine test on every backend present:
CPU, CUDA, and OpenCL once per GPU (on the development machine: an NVIDIA RTX
4070 laptop GPU and an AMD Radeon integrated GPU).

| Test | What it checks |
|---|---|
| `test_declip_below_clip_above_is_restored` | Samples below the threshold are bit-exact; rebuilt samples are never quieter; SNR improves by more than 3 dB. Also guards a past bug where rounded clip tops were missed. |
| `test_max_restore_cap_and_level_invariance` | The bound \|y\| ≤ \|x̂\| ≤ m\|y\| holds, and a quieter copy gives the same result scaled. |
| `test_linked_stereo_uses_the_clean_channel` | A clipped L guided by a clean R improves by more than 3 dB over solo recovery; R is untouched. |
| `test_declip_improves_hard_clip` | SNR improves on a synthetic kick. |
| `test_declip_improves_soft_clip` | SNR improves on a tanh-saturated signal. |
| `test_declip_diff_clips` | Per-section detection works, including a song ending in silence (a past crash). |
| `test_unlimit_linked_gain` | One gain per sample for both channels, with the median kept at 1. |
| `test_uncompress_inverts_compressor` | Exact inversion of the model, including at a different process rate; 50% gives exactly half the correction in dB. |
| `test_cli_*` | End to end: presets, `.json` presets, logging, stereo float output, uncompress flags. |

`python gui.py --selfcheck` drives the real window's input fields with every
preset and checks they reach the engine unchanged. `python abtest.py
--selftest` checks the scoring code.

**Development environment:** Windows 11, Python 3.14, NumPy, PyTorch 2.11
(CUDA 12.8), pyopencl 2026.1. A tester also ran it on Linux (AMD gfx1201,
Mesa OpenCL).

---

## 6. Results

### 6.1 Declip (clipped renders, SNR improvement in dB)

**Detection threshold.** Loudest-excerpt sweeps, before the upper bound
existed:

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

**Full songs, final presets (with linked stereo):**

| Preset | MD | dchc |
|---|---|---|
| Accurate | **+5.70** | **+1.83** |
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
| Declip at 93% only | 16.20 | 4.68 |
| **Unmaster preset** (declip 93% + unlimit 50%) | **16.74 (+0.93)** | **4.96 (+1.59)** |
| iZotope Ozone Unlimiter, 100% | 15.74 (−0.07) | 4.65 (+1.28) |
| iZotope Ozone Unlimiter, 200% | 12.59 (−3.22) | 4.38 (+1.01) |

Applying iZotope's *own* gain curve at half strength after our declip scored
16.93 and 5.10, also better than its 100% render. Over-correction is a
common failure, which is why *Amount* defaults to 50%.

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

**Full chain** (compressor + limiter renders, unlimit 50% then uncompress 50%):

| Render | Limited | After |
|---|---|---|
| MD 2:1 | 15.80 | 17.39 |
| dchc 2:1 | 3.49 | 4.24 |
| dchc 6:1 | 2.90 | 4.46 |
| MD 6:1 | 11.81 | 11.16 (worse at 50%; use 15–25%) |

### 6.5 Speed

Accurate preset, 5:34 stereo song, RTX 4070 laptop GPU via OpenCL: about
75 s. A 2-compute-unit AMD integrated GPU runs at about 5× slower than
realtime. CPU only is roughly 20× slower than the RTX. Uncompress runs on
the CPU at about 30× realtime.

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

---

## 8. Limitations

- **Small test set.** Two songs, both very loud electronic music, one of which
  renders non-deterministically. The defaults may not be best for other
  genres. More original/degraded pairs would make them more trustworthy.
- **Objective scores only.** There was no formal listening test. The author's
  listening drove the investigation of the "watery" sound and the peak
  height.
- **Very heavy clipping.** When more than about 70% of samples are flagged,
  no sparse declipper can recover the signal; SPADE then invents loud peaks.
  Short frames also overshoot on long, decaying pure low tones.
- **Oversampled clippers.** They smear damage below the ceiling, which the
  per-sample consistency constraint can't fully model. That is why dchc
  improves less than MD.
- **Unlimit's training data.** The model was trained on pop/rock stems
  limited with one commercial limiter. Out of that domain it helps less.
- **Uncompress needs the threshold and ratio,** and models a generic
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
- **How claims were checked:** everything was verified by the automated tests
  and the ground-truth measurements above, not by the AI's own claims. Review
  the code before relying on it for anything important.

---

## 10. References

**Used:**

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

**Software:** NumPy, soundfile/libsndfile (audio I/O), PyTorch and torchaudio
(Unlimit, CUDA/CPU backend, resampling), PyOpenCL (GPU backend), Tkinter
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
