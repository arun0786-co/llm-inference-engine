"""Which scaled_dot_product_attention backends accept the decode
baseline's call (float16, a padding mask, 14 query heads on 2 kv heads) on
this GPU, and how long each takes at batch 256 with 1,000 keys. Explains the
slow gather + SDPA rows on the T4.

    python bench/sdpa_backends.py
"""

import torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
print("torch", torch.__version__, torch.cuda.get_device_name(0))
b, nh, nkv, d, L = 256, 14, 2, 64, 1000
dev, dt = "cuda", torch.float16
q = torch.randn(b, nh, 1, d, device=dev, dtype=dt)
k = torch.randn(b, nkv, L, d, device=dev, dtype=dt)
v = torch.randn(b, nkv, L, d, device=dev, dtype=dt)
mask = torch.ones(b, 1, 1, L, device=dev, dtype=torch.bool)
mask[:, :, :, 900:] = False
kr, vr = k.repeat_interleave(nh // nkv, 1), v.repeat_interleave(nh // nkv, 1)

def time_it(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 20

cases = {
    "enable_gqa, mask (the baseline)": lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True),
    "kv repeated, mask": lambda: F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask),
    "enable_gqa, no mask": lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True),
}
for name, fn in cases.items():
    print(f"{name}: default {time_it(fn):.3f} ms")
    for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH):
        try:
            with sdpa_kernel(be):
                t = time_it(fn)
            print(f"  {be.name}: {t:.3f} ms")
        except RuntimeError as ex:
            print(f"  {be.name}: not available ({str(ex).splitlines()[0][:90]})")
