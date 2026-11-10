import torch

from infer.sampler import probs_from_logits, sample_from_probs
from infer.sequence import SamplingParams


def test_top_k_and_top_p():
    logits = torch.log(torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05]] * 3))
    params = [
        SamplingParams(temperature=1.0, top_k=2),
        SamplingParams(temperature=1.0, top_p=0.8),  # 0.5 + 0.2 < 0.8, so 3 tokens
        SamplingParams(temperature=1.0, top_p=0.01),  # only the top token
    ]
    p = probs_from_logits(logits, params)
    assert torch.allclose(p[0], torch.tensor([5 / 7, 2 / 7, 0, 0, 0]))
    assert torch.allclose(p[1], torch.tensor([0.5, 0.2, 0.15, 0, 0]) / 0.85)
    assert torch.allclose(p[2], torch.tensor([1.0, 0, 0, 0, 0]))


def test_inverse_cdf_draws_follow_probs():
    torch.manual_seed(0)
    probs = torch.tensor([0.1, 0.0, 0.6, 0.3])
    n = 200000
    u = torch.rand(n)
    idx = sample_from_probs(probs.expand(n, 4), u)
    freq = torch.bincount(idx, minlength=4).float() / n
    assert freq[1] == 0  # a zero-probability token is never drawn
    assert (freq - probs).abs().max() < 0.005
