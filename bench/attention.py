"""Decode attention alone: the metal kernel against index_select gather +
scaled_dot_product_attention, Qwen2.5-0.5B shapes (14 query heads, 2 kv
heads, head dim 64), bf16, random block tables."""

import random
import time

import torch
import torch.nn.functional as F

from common import save, table
from infer import metal


def bench(fn, n=50):
    for _ in range(5):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    dtype, nh, nkv, d, bs = torch.bfloat16, 14, 2, 64, 16
    rows = []
    for b, lo, hi in [(1, 200, 200), (8, 500, 500), (16, 500, 500), (64, 200, 200), (64, 100, 1000), (256, 100, 1000), (256, 1000, 1000)]:
        rng = random.Random(0)
        lens = [rng.randint(lo, hi) for _ in range(b)]
        nblocks = sum(-(-n // bs) for n in lens) + 1
        cache = torch.randn(nblocks * bs, 2 * nkv, d, device="mps", dtype=dtype)
        maxb = max(-(-n // bs) for n in lens)
        perm = torch.randperm(nblocks).tolist()
        tables = torch.zeros(b, maxb, dtype=torch.long)
        for i, n in enumerate(lens):
            for j in range(-(-n // bs)):
                tables[i, j] = perm.pop()
        q = torch.randn(b, nh, d, device="mps", dtype=dtype)
        maxc = max(lens)
        j = torch.arange(maxc)
        kv_slots = (tables[:, j // bs] * bs + j % bs).to("mps")
        mask = (j[None, :] < torch.tensor(lens)[:, None])[:, None, None].to("mps")

        def gather():
            kv = cache.index_select(0, kv_slots.view(-1)).view(b, maxc, 2 * nkv, d)
            k, v = kv[:, :, :nkv].transpose(1, 2), kv[:, :, nkv:].transpose(1, 2)
            return F.scaled_dot_product_attention(q[:, :, None], k, v, attn_mask=mask, enable_gqa=True)

        t32 = tables.to(torch.int32).to("mps")
        l32 = torch.tensor(lens, dtype=torch.int32, device="mps")
        a = bench(gather)
        k = bench(lambda: metal.paged_decode(q, cache, t32, l32, nkv, bs, d**-0.5))
        kv_mb = sum(lens) * 2 * nkv * d * 2 / 2**20
        rows.append({"batch": b, "context": f"{lo}-{hi}" if lo != hi else lo, "kv read MB": round(kv_mb, 1),
                     "gather + sdpa ms": round(a, 3), "kernel ms": round(k, 3), "speedup": round(a / k, 2),
                     "kernel GB/s": round(kv_mb / 1024 / (k / 1000), 1)})
        print(rows[-1], flush=True)
    save("attention", {"rows": rows})
    table(rows, list(rows[0]))


if __name__ == "__main__":
    main()
