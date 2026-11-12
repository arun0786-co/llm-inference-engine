"""Paged attention for decode steps, as a Metal kernel.

The portable path gathers every sequence's keys and values into a padded
[b, max_ctx] tensor, then calls scaled_dot_product_attention with a mask.
That copies the whole context once per layer and pads short sequences to
the longest one. This kernel reads keys and values straight from the
cache blocks through the block table, and each sequence stops at its own
length.

Layout: one threadgroup per (sequence, kv head), 4 simdgroups of 32
lanes. The kv head's G query heads (grouped-query attention) are handled
together, so each key and value is read once. The simdgroups take turns
on chunks of 32 keys. In a chunk:

1. lane j reads key j and computes its G scores (q sits in threadgroup
   memory, so every lane reads it by broadcast). The 32 key loads are in
   flight at the same time.
2. simd_max and simd_sum over the chunk update each head's running max
   and sum (online softmax). The probabilities go to threadgroup memory.
3. lanes switch to owning D/32 head dims each and walk the 32 values,
   so each value row is one coalesced load.

The four simdgroups' partial results are merged at the end.
"""

import torch

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr int NSG = 4;    // simdgroups per threadgroup
constant constexpr int MAXG = 8;   // query heads per kv head

template <typename T, int D>
void paged_decode(
    device const T* q, device const T* cache, device const int* tables, device const int* ctx_lens,
    device T* out, int nh, int nkv, int bs, int maxb, float scale,
    uint2 tg, uint sg, uint lane, threadgroup float* qs, threadgroup float* ps, threadgroup float* red,
    threadgroup long* slots)
{
    constexpr int LPK = D / 8;    // lanes per value row in the value pass, 8 dims each
    constexpr int KPI = 32 / LPK; // value rows read at once by one simdgroup
    const int s = tg.y, kvh = tg.x;
    const int G = nh / nkv;
    const int ctx = ctx_lens[s];
    const int tid = sg * 32 + lane;
    const long row = 2L * nkv * D;

    for (int i = tid; i < G * D; i += NSG * 32)
        qs[i] = float(q[(long(s) * nh + kvh * G) * D + i]) * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int sub = lane / LPK, li = lane % LPK;
    float m[MAXG], l[MAXG];
    float4 acc[MAXG][2];
    for (int g = 0; g < MAXG; g++) {
        m[g] = -INFINITY; l[g] = 0.0f;
        acc[g][0] = acc[g][1] = float4(0.0f);
    }
    threadgroup float* p = ps + sg * MAXG * 32;
    threadgroup long* sl = slots + sg * 32;  // cache slot of each key in the chunk
    for (int c0 = sg * 32; c0 < ctx; c0 += NSG * 32) {
        // 1. lane j scores key c0 + j against all G heads
        const int j = c0 + lane;
        float sc[MAXG];
        if (j < ctx) {
            const long slot = long(tables[long(s) * maxb + j / bs]) * bs + j % bs;
            sl[lane] = slot;
            device const vec<T, 4>* k = (device const vec<T, 4>*)(cache + slot * row + kvh * D);
            threadgroup const float4* q4 = (threadgroup const float4*)qs;
            for (int g = 0; g < G; g++) sc[g] = 0.0f;
            for (int i = 0; i < D / 4; i++) {
                const float4 kk = float4(k[i]);
                for (int g = 0; g < G; g++) sc[g] += dot(kk, q4[g * (D / 4) + i]);
            }
        } else {
            for (int g = 0; g < G; g++) sc[g] = -INFINITY;
        }
        // 2. online softmax over the chunk
        for (int g = 0; g < G; g++) {
            const float mn = max(m[g], simd_max(sc[g]));
            const float c = exp(m[g] - mn);
            const float e = j < ctx ? exp(sc[g] - mn) : 0.0f;
            l[g] = l[g] * c + simd_sum(e);
            m[g] = mn;
            acc[g][0] *= c;
            acc[g][1] *= c;
            p[g * 32 + lane] = e;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // 3. weighted sum of values: KPI rows at a time, LPK lanes per row
        const int n = min(32, ctx - c0);
        for (int t = sub; t < n; t += KPI) {
            device const vec<T, 4>* v = (device const vec<T, 4>*)(cache + sl[t] * row + (nkv + kvh) * D + li * 8);
            const float4 v0 = float4(v[0]), v1 = float4(v[1]);
            for (int g = 0; g < G; g++) {
                const float pg = p[g * 32 + t];
                acc[g][0] += pg * v0;
                acc[g][1] += pg * v1;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // lanes with the same li hold partial sums over different keys
    for (int g = 0; g < G; g++)
        for (int off = LPK; off < 32; off *= 2) {
            acc[g][0] += simd_shuffle_xor(acc[g][0], off);
            acc[g][1] += simd_shuffle_xor(acc[g][1], off);
        }

    // merge simdgroups one at a time into simdgroup 0 through red[]
    threadgroup float* rm = red;                          // [MAXG] running max
    threadgroup float* rl = red + MAXG;                   // [MAXG] sum
    threadgroup float4* ra = (threadgroup float4*)(red + 2 * MAXG);  // [MAXG * D / 4] acc
    for (int w = 1; w < NSG; w++) {
        if (sg == uint(w) && sub == 0) {
            for (int g = 0; g < G; g++) {
                if (li == 0) { rm[g] = m[g]; rl[g] = l[g]; }
                ra[(g * D + li * 8) / 4] = acc[g][0];
                ra[(g * D + li * 8) / 4 + 1] = acc[g][1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            for (int g = 0; g < G; g++) {
                const float mn = max(m[g], rm[g]);
                const float c0 = m[g] == -INFINITY ? 0.0f : exp(m[g] - mn);
                const float c1 = rm[g] == -INFINITY ? 0.0f : exp(rm[g] - mn);
                l[g] = l[g] * c0 + rl[g] * c1;
                acc[g][0] = acc[g][0] * c0 + ra[(g * D + li * 8) / 4] * c1;
                acc[g][1] = acc[g][1] * c0 + ra[(g * D + li * 8) / 4 + 1] * c1;
                m[g] = mn;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0 && sub == 0) {
        for (int g = 0; g < G; g++) {
            device T* op = out + (long(s) * nh + kvh * G + g) * D + li * 8;
            const float inv = 1.0f / l[g];
            for (int i = 0; i < 4; i++) { op[i] = T(acc[g][0][i] * inv); op[4 + i] = T(acc[g][1][i] * inv); }
        }
    }
}

#define INST(T, D, NAME)                                                                          \
kernel void NAME(device const T* q [[buffer(0)]], device const T* cache [[buffer(1)]],           \
                 device const int* tables [[buffer(2)]], device const int* ctx_lens [[buffer(3)]], \
                 device T* out [[buffer(4)]], constant int& nh [[buffer(5)]],                    \
                 constant int& nkv [[buffer(6)]], constant int& bs [[buffer(7)]],                \
                 constant int& maxb [[buffer(8)]], constant float& scale [[buffer(9)]],          \
                 uint2 tg [[threadgroup_position_in_grid]],                                       \
                 uint sg [[simdgroup_index_in_threadgroup]],                                      \
                 uint lane [[thread_index_in_simdgroup]]) {                                       \
    threadgroup float4 qs4[MAXG * D / 4];                                                             \
    threadgroup float ps[NSG * MAXG * 32];                                                        \
    threadgroup float4 red4[(2 * MAXG + MAXG * D) / 4];                                                \
    threadgroup long slots[NSG * 32];                                                             \
    paged_decode<T, D>(q, cache, tables, ctx_lens, out, nh, nkv, bs, maxb, scale, tg, sg, lane, (threadgroup float*)qs4, ps, (threadgroup float*)red4, slots); \
}

INST(float, 64, paged_decode_f32_64)
INST(float, 128, paged_decode_f32_128)
INST(half, 64, paged_decode_f16_64)
INST(half, 128, paged_decode_f16_128)
INST(bfloat, 64, paged_decode_bf16_64)
INST(bfloat, 128, paged_decode_bf16_128)
"""

_lib = None
_NAMES = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}


def available(dtype, head_dim, group):
    return (
        torch.backends.mps.is_available()
        and hasattr(torch.mps, "compile_shader")
        and dtype in _NAMES
        and head_dim in (64, 128)
        and group <= 8
    )


def paged_decode(q, cache, tables, ctx_lens, num_kv_heads, block_size, scale):
    """q [b, nh, d]; cache [slots, 2 * nkv, d]; tables [b, max_blocks] int32;
    ctx_lens [b] int32. Returns [b, nh, d]."""
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(SOURCE)
    b, nh, d = q.shape
    out = torch.empty_like(q)
    fn = getattr(_lib, f"paged_decode_{_NAMES[q.dtype]}_{d}")
    fn(
        q.contiguous(), cache, tables, ctx_lens, out,
        nh, num_kv_heads, block_size, tables.shape[1], float(scale),
        threads=(num_kv_heads * 128, b, 1), group_size=(128, 1, 1),
    )
    return out
