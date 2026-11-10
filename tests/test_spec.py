"""Speculative decoding must not change what the target model generates."""

import itertools
import random
from collections import Counter

import pytest
import torch

from conftest import make_tiny
from infer.engine import Engine
from infer.sequence import SamplingParams
from test_engine import GREEDY, greedy_reference, prompts


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    target = tmp_path_factory.mktemp("target")
    draft = tmp_path_factory.mktemp("draft")
    hf = make_tiny(str(target), "qwen2", seed=0, layers=3)
    make_tiny(str(draft), "qwen2", seed=1, layers=1)
    return str(target), str(draft), hf


@pytest.mark.parametrize("k", [1, 3, 5])
def test_greedy_spec_matches_target(pair, k):
    target, draft, hf = pair
    ps = prompts(5, 10)
    eng = Engine(target, device="cpu", num_blocks=128, block_size=4, draft_model_path=draft, num_spec_tokens=k)
    out = eng.generate(ps, GREEDY)
    for p, o in zip(ps, out):
        assert o == greedy_reference(hf, p, 12)
    eng.bm.check()
    eng.draft_bm.check()
    assert eng.bm.num_free() == eng.bm.num_blocks


def test_self_draft_accepts_everything(pair):
    target, _, _ = pair
    eng = Engine(target, device="cpu", num_blocks=128, block_size=4, draft_model_path=target, num_spec_tokens=4)
    eng.generate(prompts(4, 11), SamplingParams(max_tokens=21, ignore_eos=True))
    st = eng.stats()
    assert st["spec_acceptance"] == 1.0
    assert st["spec_tokens_per_pass"] == 5.0


def test_spec_with_preemption(pair):
    target, draft, _ = pair
    ps = prompts(8, 12)
    ref = Engine(target, device="cpu", num_blocks=256, block_size=4).generate(ps, GREEDY)
    eng = Engine(target, device="cpu", num_blocks=40, block_size=4, draft_model_path=draft, num_spec_tokens=3)
    assert eng.generate(ps, GREEDY) == ref
    assert eng.sched.preemptions > 0


def joint_target_probs(hf, prompt, n, vocab):
    """Exact probability of every n-token continuation at temperature 1."""
    out = {}
    with torch.no_grad():
        for cont in itertools.product(range(vocab), repeat=n):
            p = 1.0
            for i in range(n):
                lg = hf(torch.tensor([prompt + list(cont[:i])])).logits[0, -1]
                p *= float(torch.softmax(lg.double(), -1)[cont[i]])
            out[cont] = p
    return out


def test_sampled_spec_preserves_distribution(tmp_path):
    """With temperature 1, the 3-token continuations produced with a
    (bad) draft model follow the target model's exact joint distribution.
    A vocabulary of 4 keeps the 64 outcomes countable. The draft alone has
    a clearly different distribution, so the test can tell them apart."""
    vocab, n_tok, n = 4, 3, 12000
    hf = make_tiny(str(tmp_path / "t"), "qwen2", vocab=vocab, seed=3, layers=2)
    draft_hf = make_tiny(str(tmp_path / "d"), "qwen2", vocab=vocab, seed=4, layers=1)
    prompt = [0, 3, 1]
    exact = joint_target_probs(hf, prompt, n_tok, vocab)
    draft_exact = joint_target_probs(draft_hf, prompt, n_tok, vocab)

    eng = Engine(
        str(tmp_path / "t"),
        device="cpu",
        num_blocks=4096,
        block_size=4,
        max_num_seqs=2048,
        max_batched_tokens=8192,
        draft_model_path=str(tmp_path / "d"),
        num_spec_tokens=2,
    )
    rng = random.Random(0)
    params = [SamplingParams(temperature=1.0, max_tokens=n_tok, ignore_eos=True, seed=rng.getrandbits(32)) for _ in range(n)]
    outs = eng.generate([prompt] * n, params)
    counts = Counter(tuple(o) for o in outs)

    def tv(p):
        return 0.5 * sum(abs(counts.get(c, 0) / n - p[c]) for c in p)

    def chi2(p):
        return sum((counts.get(c, 0) - n * p[c]) ** 2 / (n * p[c]) for c in p)

    assert eng.stats()["spec_acceptance"] < 0.95  # the draft really is wrong sometimes
    # 63 degrees of freedom: the 99.9th percentile of chi-square is 103.4
    assert chi2(exact) < 103.4, chi2(exact)
    assert tv(exact) < 0.03
    assert chi2(draft_exact) > 300  # the test has power
