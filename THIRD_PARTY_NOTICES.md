<!--
SPDX-FileCopyrightText: 2026 thatrandomalt
SPDX-License-Identifier: MIT
-->

# Third-party notices

SharpUnClip is released under the MIT License ([LICENSE](LICENSE),
[LICENSES/MIT.txt](LICENSES/MIT.txt)). Every file states its copyright holders
and license in an SPDX header; binary files do so in a `.license` file next to
them. The project follows the [REUSE](https://reuse.software) specification.

## Third-party material in this repository

| Material | Where | License | Copyright |
|---|---|---|---|
| Pretrained De-limiter model weights, redistributed unchanged from <https://github.com/jeonchangbin49/De-limiter> (file `weight/all.pth`, commit `e15f80b`) | `weights/delimiter.pth` (see `weights/delimiter.pth.license`) | MIT | © 2023 jeonchangbin49 |
| Network structure (layer layout and names) of Asteroid's TDConvNet masker and asteroid-filterbanks' free filterbank, mirrored so the weights load | `unlimit.py` | MIT | © 2019 Pariente Manuel |
| BetterDeclipper's restoration engine (the NMF Declip engine), from <https://github.com/Qupci/BetterDeclipper> v0.3.0, commit `f9ff7f4`: the modules the engine needs, without its CLI. Changed: a progress callback inside its SPADE stage (`methods/spade.py`, `engine.py`), the quantization-step estimator now samples the whole file and skips exact zeros so a silent intro no longer looks like 8-bit audio (`detect.py`), CUDA graphs are off on ROCm builds of PyTorch (`methods/pnp.py`), the analysis pass and the lambda reference are split out of `declip()` as `fast_pass()` and `lam_refs_for()` (which also accepts a non-integer sample rate), with `declip(lam_refs=)` to reuse a file-global reference (`engine.py`), and the package docstring (`__init__.py`) | `betterdeclipper/` | MIT | © 2026 Qupci |

The original license texts follow, exactly as published.

### BetterDeclipper (<https://github.com/Qupci/BetterDeclipper>)

```
MIT License

Copyright (c) 2026 Qupci

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### De-limiter (<https://github.com/jeonchangbin49/De-limiter>)

```
MIT License

Copyright (c) 2023 jeonchangbin49

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Asteroid and asteroid-filterbanks (<https://github.com/asteroid-team/asteroid>)

Both packages carry this same text.

```
MIT License

Copyright (c) 2019 Pariente Manuel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Papers

SharpUnClip implements methods described in the papers below, from the
descriptions in the publications. **No text, figures, code or data from the
papers are included**, so no license from them applies to this repository. A
published method is not licensed the way code is. The papers are cited, with how
each was used and changed, in [METHODS.md](METHODS.md#10-references):

- Kitić, Bertin & Gribonval 2015: SPADE (Declip)
- Záviška, Rajmic, Mokrý & Průša 2019, and Záviška, Rajmic, Ozerov & Rencker 2021: SPADE conventions
- Siedenburg, Kowalski & Dörfler 2014: social sparsity declipping; Záviška, Rajmic, Průša & Veselý 2018: S-SPADE; Lee & Seung 2001: NMF; Beck & Teboulle 2009: FISTA; Venkatakrishnan, Bouman & Wohlberg 2013: plug-and-play priors (all used by the BetterDeclipper engine above)
- Jeon & Lee 2023: De-limiter (Unlimit); its *weights* are licensed as above
- Luo & Mesgarani 2019: Conv-TasNet (the architecture De-limiter uses)
- Gorlow & Reiss 2013, and Giannoulis, Massberg & Reiss 2012: compressor inversion (Uncompress)
- Le Roux, Wisdom, Erdogan & Hershey 2019: SI-SDR (scoring)
- Cooley & Tukey 1965: FFT (GPU kernel)

## Training data of the De-limiter weights

The De-limiter weights were trained by their authors on MUSDB18-HQ (Rafii et al.,
2019), limited into musdb-XL. No audio from that dataset is included here. The
dataset's tracks have their own terms, intended for research use, so check them
before using the Unlimit stage commercially.

## Dependencies

NumPy, SciPy, soundfile/libsndfile, PyTorch, torchaudio, PyOpenCL and sounddevice
(with PortAudio) are installed separately by the user (see the README) and are
not redistributed here; each is under its own license. Tkinter ships with Python.
