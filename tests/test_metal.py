"""The metal paged-attention kernel. Needs an apple gpu; skipped elsewhere."""

import random

import pytest
import torch
import torch.nn.functional as F

from conftest import make_tiny
from infer import metal
from infer.engine import Engine
from infer.sequence import SamplingParams
from test_engine import prompts

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs mps")


@pytest.mark.parametrize("nh,nkv,d", [(14, 2, 64), (12, 2, 128), (4, 4, 64), (16, 2, 64)])
def test_kernel_matches_reference(nh, nkv, d):
    torch.manual_seed(0)
    bs, nblocks = 16, 400
    cache = torch.randn(nblocks * bs, 2 * nkv, d, device="mps")
    lens = [1, 5, 16, 17, 300, 777, 64, 31, 33]
    perm = torch.randperm(nblocks).tolist()
    maxb = max(-(-n // bs) for n in lens)
    tables = torch.zeros(len(lens), maxb, dtype=torch.int32)
    for i, n in enumerate(lens):
        for j in range(-(-n // bs)):
            tables[i, j] = perm.pop()
    q = torch.randn(len(lens), nh, d, device="mps")
    ctx = torch.tensor(lens, dtype=torch.int32, device="mps")
    out = metal.paged_decode(q, cache, tables.to("mps"), ctx, nkv, bs, d**-0.5)
    for i, n in enumerate(lens):
        pos = torch.arange(n)
        slots = (tables[i, pos // bs].long() * bs + pos % bs).to("mps")
        kv = cache[slots].transpose(0, 1)
        ref = F.scaled_dot_product_attention(q[i][:, None], kv[:nkv], kv[nkv:], enable_gqa=True)[:, 0]
        assert (out[i] - ref).abs().max() < 1e-5


def test_engine_same_output_with_kernel(tmp_path):
    path = str(tmp_path)
    make_tiny(path, "qwen2", hidden=256, layers=2)  # head_dim 64
    ps = prompts(12, 30, 3, 60)
    sp = SamplingParams(max_tokens=16, ignore_eos=True)
    kw = dict(device="mps", dtype=torch.float32, num_blocks=256, block_size=4)
    on = Engine(path, **kw)
    off = Engine(path, paged_kernel=False, **kw)
    assert on.runner.paged_kernel
    on.runner.KERNEL_MIN_BATCH = 1  # use the kernel for every decode batch
    assert on.generate(ps, sp) == off.generate(ps, sp)
