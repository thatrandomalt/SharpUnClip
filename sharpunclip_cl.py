# SPDX-FileCopyrightText: 2026 thatrandomalt
#
# SPDX-License-Identifier: MIT

"""OpenCL A-SPADE backend (Kitić et al. 2015; radix-2 Cooley-Tukey FFT) for any GPU vendor.
Needs pyopencl.

Each work-group runs whole A-SPADE iterations for one frame (of every linked channel)
in local memory. SHARPUNCLIP_OPENCL_DEVICE picks a GPU by part of its name.
"""
import os
import warnings
import numpy as np
import pyopencl as cl

_SRC = r"""
inline float2 cmul(float2 a, float2 b) { return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }

// In-place radix-2 FFT of buf[0..N) by one work-group; inv conjugates the twiddles (unnormalized).
inline void fft(__local float2 *buf, __global const float2 *TW, __global const int *REV, int inv, int l) {
    for (int i = l; i < N; i += LS) {
        int j = REV[i];
        if (i < j) { float2 t = buf[i]; buf[i] = buf[j]; buf[j] = t; }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int hs = 1; hs < N; hs <<= 1) {
        int step = N / (2 * hs);
        for (int t = l; t < N / 2; t += LS) {
            int pos = t & (hs - 1), i = ((t - pos) << 1) + pos, j = i + hs;
            float2 w = TW[pos * step];
            if (inv) w.y = -w.y;
            float2 v = cmul(buf[j], w), u = buf[i];
            buf[i] = u + v; buf[j] = u - v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// `iters` A-SPADE iterations (sparsity k = k0, k0+1, ...) on group get_group_id(0): its C
// channel rows (C*g .. C*g+C-1) share one support, picked from their summed |V|^2.
__kernel __attribute__((reqd_work_group_size(LS, 1, 1)))
void aspade_iters(__global float2 *Z, __global float2 *U, __global float2 *ZB,
                  __global float *X, __global const float *LO, __global const float *HI,
                  __global int *ACTIVE, __global const float2 *TW, __global const int *REV,
                  const int k0, const int iters, const float eps2) {
    __local float2 buf[N];
    __local float en[NB];
    __local uint hist[16], sh[3];
    __local float red[LS];
    __local int act;
    const int g = get_group_id(0), l = get_local_id(0);
    if (l == 0) act = ACTIVE[g];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int it = 0; it < iters && act; it++) {
        for (int j = l; j < NB; j += LS) {                              // shared energy of V = Z + U
            float e = 0.0f;
            for (int c = 0; c < C; c++) {
                float2 v = Z[(C * g + c) * NB + j] + U[(C * g + c) * NB + j];
                e += v.x * v.x + v.y * v.y;
            }
            en[j] = e;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // k-th largest energy: float bits of non-negatives sort like uints; 4-bit radix select
        uint prefix = 0, mask = 0, kk = min(k0 + it, NB);
        for (int shift = 28; shift >= 0; shift -= 4) {
            if (l < 16) hist[l] = 0;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int j = l; j < NB; j += LS) {
                uint b = as_uint(en[j]);
                if ((b & mask) == prefix) atomic_inc(&hist[(b >> shift) & 15]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (l == 0) {
                uint d = 15;
                while (kk > hist[d]) { kk -= hist[d]; d--; }
                sh[0] = prefix | (d << shift); sh[1] = mask | (15u << shift); sh[2] = kk;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            prefix = sh[0]; mask = sh[1]; kk = sh[2];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float worst = 0.0f;
        for (int c = 0; c < C; c++) {
            const int r = C * g + c;
            __global float2 *z = Z + r * NB, *u = U + r * NB, *zb = ZB + r * NB;
            __global float *x = X + r * W;
            __global const float *lo = LO + r * W, *hi = HI + r * W;
            // zb = V on the shared support; buf = zb - U, Hermitian-extended for the real inverse
            for (int j = l; j < NB; j += LS) {
                float2 v = z[j] + u[j];
                float2 s = as_uint(en[j]) >= prefix ? v : (float2)(0.0f, 0.0f);
                zb[j] = s; buf[j] = s - u[j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int j = NB + l; j < N; j += LS) { float2 q = buf[N - j]; buf[j] = (float2)(q.x, -q.y); }
            barrier(CLK_LOCAL_MEM_FENCE);
            fft(buf, TW, REV, 1, l);
            // x = project(adjoint) onto lo <= x <= hi, zero-padded for the forward transform
            for (int j = l; j < N; j += LS) {
                float v = 0.0f;
                if (j < W) { v = fmin(fmax(buf[j].x * ISQ, lo[j]), hi[j]); x[j] = v; }
                buf[j] = (float2)(v, 0.0f);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            fft(buf, TW, REV, 0, l);
            float s = 0.0f;
            for (int j = l; j < NB; j += LS) {
                float2 zl = buf[j] * ISQ, d = zl - zb[j];
                u[j] += d; z[j] = zl; s += d.x * d.x + d.y * d.y;
            }
            red[l] = s;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int o = LS / 2; o > 0; o >>= 1) {
                if (l < o) red[l] += red[l + o];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            worst = fmax(worst, red[0]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (l == 0 && worst <= eps2) act = 0;                          // every channel converged
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0) ACTIVE[g] = act;
}
"""

_cache = {}


def device(name=None):
    """The OpenCL GPU with the most compute units (optionally matching `name`), or None."""
    name = name if name is not None else os.environ.get("SHARPUNCLIP_OPENCL_DEVICE")
    gpus = []
    for p in cl.get_platforms():
        try:
            gpus += p.get_devices(cl.device_type.GPU)
        except cl.Error:
            pass
    if name:
        gpus = [d for d in gpus if name.lower() in d.name.lower()]
    return max(gpus, key=lambda d: d.max_compute_units, default=None)


def _setup(dev, w, N, C):
    key = (dev.int_ptr, w, N, C)
    if key not in _cache:
        ls = min(256, dev.max_work_group_size)
        if 8 * N + 4 * (N // 2 + 1) + 4 * ls + 128 > dev.local_mem_size:
            raise RuntimeError(f"{dev.name}: not enough local memory for {N}-point frames "
                               f"(try a lower Lookahead)")
        ctx = cl.Context([dev])
        opts = [f"-DN={N}", f"-DNB={N // 2 + 1}", f"-DW={w}", f"-DC={C}", f"-DLS={ls}",
                f"-DISQ={N ** -0.5!r}f"]
        with warnings.catch_warnings():  # NVIDIA prints a harmless inlining note
            warnings.simplefilter("ignore", cl.CompilerWarning)
            kern = cl.Kernel(cl.Program(ctx, _SRC).build(options=opts), "aspade_iters")
        logn = N.bit_length() - 1
        i = np.arange(N)
        rev = sum(((i >> b) & 1) << (logn - 1 - b) for b in range(logn)).astype(np.int32)
        tw = np.exp(-2j * np.pi * np.arange(N // 2) / N).astype(np.complex64)
        mf = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
        _cache[key] = (ctx, cl.CommandQueue(ctx), kern, ls,
                       cl.Buffer(ctx, mf, hostbuf=tw), cl.Buffer(ctx, mf, hostbuf=rev))
    return _cache[key]


def aspade_frames(Y, LO, HI, N, eps, maxit, tick=None, group=1, dev=None, chunk=25):
    """sharpunclip._aspade_frames on the GPU. Runs `chunk` iterations per launch to stay
    under the Windows GPU watchdog and to report progress between launches."""
    rows, w = Y.shape
    assert N & (N - 1) == 0, "N must be a power of two"
    assert rows % group == 0
    ctx, q, kern, ls, tw, rev = _setup(dev or device(), w, N, group)
    nb, groups = N // 2 + 1, rows // group
    mf = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
    put = lambda a: cl.Buffer(ctx, mf, hostbuf=np.ascontiguousarray(a))
    Z = put((np.fft.rfft(Y, N) * N ** -0.5).astype(np.complex64))
    U, ZB = put(np.zeros((rows, nb), np.complex64)), put(np.zeros((rows, nb), np.complex64))
    X, LOb, HIb = put(Y.astype(np.float32)), put(LO.astype(np.float32)), put(HI.astype(np.float32))
    act = np.ones(groups, np.int32)
    A = put(act)
    for start in range(0, maxit, chunk):
        if tick:
            tick(start / maxit)
        kern(q, (groups * ls,), (ls,), Z, U, ZB, X, LOb, HIb, A, tw, rev,
             np.int32(start + 1), np.int32(min(chunk, maxit - start)), np.float32(eps * eps))
        cl.enqueue_copy(q, act, A)
        if not act.any():
            break
    out = np.empty((rows, w), np.float32)
    cl.enqueue_copy(q, out, X)
    return out.astype(np.float64)
