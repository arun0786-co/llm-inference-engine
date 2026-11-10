"""End-to-end generation on random checkpoints, float32 on cpu."""

import random

import torch

from infer.engine import Engine
from infer.sequence import SamplingParams


def greedy_reference(hf, prompt, n):
    """Greedy decoding with full forward passes and no cache at all."""
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            ids.append(int(hf(torch.tensor([ids])).logits[0, -1].argmax()))
    return ids[len(prompt) :]


def prompts(n, seed, lo=3, hi=40, vocab=256):
    rng = random.Random(seed)
    return [[rng.randrange(3, vocab) for _ in range(rng.randint(lo, hi))] for _ in range(n)]


GREEDY = SamplingParams(max_tokens=12, ignore_eos=True)


def test_greedy_batch_matches_reference(tiny):
    path, hf = tiny
    ps = prompts(6, 0)
    # small budget and blocks: prompts get chunked, sequences join mid-flight
    eng = Engine(path, device="cpu", num_blocks=64, block_size=4, max_batched_tokens=16, max_num_seqs=4)
    out = eng.generate(ps, GREEDY)
    for p, o in zip(ps, out):
        assert o == greedy_reference(hf, p, 12)


def test_preemption_does_not_change_output(tiny):
    path, _ = tiny
    ps = prompts(8, 1)
    big = Engine(path, device="cpu", num_blocks=256, block_size=4).generate(ps, GREEDY)
    small = Engine(path, device="cpu", num_blocks=30, block_size=4, max_num_seqs=8)
    out = small.generate(ps, GREEDY)
    assert small.sched.preemptions > 0
    assert out == big
    small.bm.check()
    assert small.bm.num_free() == small.bm.num_blocks


def test_prefix_cache_hits_and_same_output(tiny):
    path, _ = tiny
    rng = random.Random(2)
    system = [rng.randrange(3, 256) for _ in range(37)]
    ps = [system + q for q in prompts(6, 3, 1, 9)]
    off = Engine(path, device="cpu", num_blocks=256, block_size=4, prefix_caching=False)
    on = Engine(path, device="cpu", num_blocks=256, block_size=4, max_num_seqs=2)
    assert on.generate(ps, GREEDY) == off.generate(ps, GREEDY)
    # the first two run together; the other four reuse the 9 full blocks
    assert on.sched.cached_tokens == 4 * 36
    on.bm.check()


def test_sampling_is_per_request(tiny):
    """A seeded request gives the same tokens alone or in a busy batch."""
    path, _ = tiny
    sp = SamplingParams(temperature=0.9, top_p=0.9, top_k=50, max_tokens=10, seed=7, ignore_eos=True)
    p = prompts(1, 4)[0]
    alone = Engine(path, device="cpu", num_blocks=128, block_size=4).generate([p], sp)[0]
    others = SamplingParams(temperature=1.2, max_tokens=10, ignore_eos=True)
    eng = Engine(path, device="cpu", num_blocks=256, block_size=4)
    ps = prompts(5, 5) + [p]
    out = eng.generate(ps, [others] * 5 + [sp])
    assert out[-1] == alone


def test_stop_on_eos(tiny):
    path, hf = tiny
    p = prompts(1, 6)[0]
    ref = greedy_reference(hf, p, 12)
    # stop on whatever token the model emits third
    sp = SamplingParams(max_tokens=12, stop_token_ids=(ref[2],))
    out = Engine(path, device="cpu", num_blocks=64, block_size=4).generate([p], sp)[0]
    assert out == ref[: ref.index(ref[2]) + 1]
