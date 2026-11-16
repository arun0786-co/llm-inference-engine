"""Paged attention for decode steps, as a CUDA kernel. Same design as the
Metal kernel in metal.py: one block per (sequence, kv head), 4 warps; each
warp takes chunks of 32 keys, lane j scores key j against the kv head's G
query heads (q in shared memory), an online softmax runs over the chunk
with warp shuffles, then lanes switch to 8 head dims each and read
32 / (D / 8) value rows at once. The 4 warps' partial results are merged
through shared memory.

Compiled on first use with torch.utils.cpp_extension.load_inline (needs
nvcc). Set TORCH_EXTENSIONS_DIR to choose where the build goes.
"""

import torch

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int NW = 4;     // warps per block
constexpr int MAXG = 8;   // query heads per kv head

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// 8 consecutive elements (16-byte aligned) to float
__device__ __forceinline__ void load8(const float* p, float* o) {
    const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
    o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
__device__ __forceinline__ void load8(const __half* p, float* o) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&raw);
    for (int i = 0; i < 4; i++) { const float2 f = __half22float2(h[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
}
__device__ __forceinline__ float to_f(float x) { return x; }
__device__ __forceinline__ float to_f(__half x) { return __half2float(x); }
__device__ __forceinline__ void store(float* p, float v) { *p = v; }
__device__ __forceinline__ void store(__half* p, float v) { *p = __float2half(v); }

template <typename T, int D>
__global__ void paged_decode_kernel(
    const T* __restrict__ q, const T* __restrict__ cache, const int* __restrict__ tables,
    const int* __restrict__ ctx_lens, T* __restrict__ out, int nh, int nkv, int bs, int maxb, float scale)
{
    constexpr int LPK = D / 8;      // lanes per value row, 8 dims each
    constexpr int KPI = 32 / LPK;   // value rows read at once per warp
    const int kvh = blockIdx.x, s = blockIdx.y;
    const int G = nh / nkv;
    const int ctx = ctx_lens[s];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int sub = lane / LPK, li = lane % LPK;
    const long row = 2L * nkv * D;

    __shared__ float qs[MAXG * D];
    __shared__ float ps[NW][MAXG][32];
    __shared__ long sl[NW][32];
    __shared__ float red_m[MAXG], red_l[MAXG];
    __shared__ float red_acc[MAXG * D];

    for (int i = tid; i < G * D; i += NW * 32) qs[i] = to_f(q[((long)s * nh + kvh * G) * D + i]) * scale;
    __syncthreads();

    float m[MAXG], l[MAXG], acc[MAXG][8];
    #pragma unroll
    for (int g = 0; g < MAXG; g++) {
        m[g] = -INFINITY; l[g] = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[g][i] = 0.f;
    }
    for (int c0 = warp * 32; c0 < ctx; c0 += NW * 32) {
        // 1. lane j scores key c0 + j against all G heads
        const int j = c0 + lane;
        float sc[MAXG];
        if (j < ctx) {
            const long slot = (long)tables[(long)s * maxb + j / bs] * bs + j % bs;
            sl[warp][lane] = slot;
            const T* k = cache + slot * row + (long)kvh * D;
            #pragma unroll
            for (int g = 0; g < MAXG; g++) sc[g] = 0.f;
            #pragma unroll 2
            for (int i = 0; i < D; i += 8) {
                float kk[8];
                load8(k + i, kk);
                _Pragma("unroll") for (int g = 0; g < MAXG; g++) if (g < G) {
                    const float* qg = qs + g * D + i;
                    #pragma unroll
                    for (int t = 0; t < 8; t++) sc[g] += kk[t] * qg[t];
                }
            }
        } else {
            #pragma unroll
            for (int g = 0; g < MAXG; g++) sc[g] = -INFINITY;
        }
        // 2. online softmax over the chunk
        _Pragma("unroll") for (int g = 0; g < MAXG; g++) if (g < G) {
            const float mn = fmaxf(m[g], warp_max(sc[g]));
            const float c = __expf(m[g] - mn);
            const float e = j < ctx ? __expf(sc[g] - mn) : 0.f;
            l[g] = l[g] * c + warp_sum(e);
            m[g] = mn;
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[g][i] *= c;
            ps[warp][g][lane] = e;
        }
        __syncwarp();
        // 3. weighted values: KPI rows at a time, LPK lanes per row
        const int n = min(32, ctx - c0);
        for (int t = sub; t < n; t += KPI) {
            float vv[8];
            load8(cache + sl[warp][t] * row + (long)(nkv + kvh) * D + li * 8, vv);
            _Pragma("unroll") for (int g = 0; g < MAXG; g++) if (g < G) {
                const float pg = ps[warp][g][t];
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[g][i] += pg * vv[i];
            }
        }
        __syncwarp();
    }
    // lanes with the same li hold sums over different keys
    _Pragma("unroll") for (int g = 0; g < MAXG; g++) if (g < G)
        #pragma unroll
        for (int i = 0; i < 8; i++)
            for (int off = LPK; off < 32; off *= 2) acc[g][i] += __shfl_xor_sync(0xffffffffu, acc[g][i], off);

    // merge warps one at a time into warp 0
    for (int w = 1; w < NW; w++) {
        if (warp == w && sub == 0) {
            _Pragma("unroll") for (int g = 0; g < MAXG; g++) if (g < G) {
                if (li == 0) { red_m[g] = m[g]; red_l[g] = l[g]; }
                #pragma unroll
                for (int i = 0; i < 8; i++) red_acc[g * D + li * 8 + i] = acc[g][i];
            }
        }
        __syncthreads();
        if (warp == 0) {
            _Pragma("unroll") for (int g = 0; g < MAXG; g++) if (g < G) {
                const float mn = fmaxf(m[g], red_m[g]);
                const float c0 = m[g] == -INFINITY ? 0.f : __expf(m[g] - mn);
                const float c1 = red_m[g] == -INFINITY ? 0.f : __expf(red_m[g] - mn);
                l[g] = l[g] * c0 + red_l[g] * c1;
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[g][i] = acc[g][i] * c0 + red_acc[g * D + li * 8 + i] * c1;
                m[g] = mn;
            }
        }
        __syncthreads();
    }
    if (warp == 0 && sub == 0) {
        _Pragma("unroll") for (int g = 0; g < MAXG; g++) if (g < G) {
            T* op = out + ((long)s * nh + kvh * G + g) * D + li * 8;
            const float inv = 1.f / l[g];
            #pragma unroll
            for (int i = 0; i < 8; i++) store(op + i, acc[g][i] * inv);
        }
    }
}

torch::Tensor paged_decode(torch::Tensor q, torch::Tensor cache, torch::Tensor tables, torch::Tensor ctx_lens,
                           int64_t nkv, int64_t bs, double scale) {
    TORCH_CHECK(q.is_cuda() && cache.is_cuda(), "cuda tensors expected");
    q = q.contiguous();
    const int b = q.size(0), nh = q.size(1), d = q.size(2);
    TORCH_CHECK(nh % nkv == 0 && nh / nkv <= MAXG, "unsupported head grouping");
    auto out = torch::empty_like(q);
    const dim3 grid(nkv, b), block(NW * 32);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int maxb = tables.size(1);
#define LAUNCH(T, ST, DD) paged_decode_kernel<T, DD><<<grid, block, 0, stream>>>( \
        reinterpret_cast<const T*>(q.data_ptr<ST>()), reinterpret_cast<const T*>(cache.data_ptr<ST>()), \
        tables.data_ptr<int>(), ctx_lens.data_ptr<int>(), reinterpret_cast<T*>(out.data_ptr<ST>()), \
        nh, (int)nkv, (int)bs, maxb, (float)scale)
    if (q.scalar_type() == at::kHalf) {
        if (d == 64) LAUNCH(__half, at::Half, 64); else if (d == 128) LAUNCH(__half, at::Half, 128);
        else TORCH_CHECK(false, "head dim must be 64 or 128");
    } else if (q.scalar_type() == at::kFloat) {
        if (d == 64) LAUNCH(float, float, 64); else if (d == 128) LAUNCH(float, float, 128);
        else TORCH_CHECK(false, "head dim must be 64 or 128");
    } else {
        TORCH_CHECK(false, "float16 or float32 only");
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

CPP_DECL = "torch::Tensor paged_decode(torch::Tensor q, torch::Tensor cache, torch::Tensor tables, " \
           "torch::Tensor ctx_lens, int64_t nkv, int64_t bs, double scale);"

_mod = None


def available(dtype, head_dim, group):
    return (torch.cuda.is_available() and dtype in (torch.float16, torch.float32)
            and head_dim in (64, 128) and group <= 8)


def _module():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        _mod = load_inline("paged_decode_ext", cpp_sources=CPP_DECL, cuda_sources=CUDA_SRC,
                           functions=["paged_decode"], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    return _mod


def paged_decode(q, cache, tables, ctx_lens, num_kv_heads, block_size, scale):
    """Same contract as metal.paged_decode."""
    return _module().paged_decode(q, cache, tables, ctx_lens, num_kv_heads, block_size, float(scale))
