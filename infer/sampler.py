"""Token sampling.

Each request draws its random numbers from its own generator, and a token
is drawn by inverting the CDF with one uniform number: token i is chosen
when cdf[i-1] <= u < cdf[i]. That costs one float per request per token,
instead of a vocabulary-sized noise tensor, and the result depends only on
the request's seed and logits, not on its neighbours in the batch.
"""

import torch


def probs_from_logits(logits, params):
    """Applies temperature, top-k and top-p. logits [n, V] float32, one
    SamplingParams per row (none greedy). Returns probabilities [n, V]."""
    dev = logits.device
    temp = torch.tensor([p.temperature for p in params], device=dev)
    logits = logits / temp[:, None]
    ks = [p.top_k for p in params]
    if any(k > 0 for k in ks):
        kmax = min(max(ks), logits.shape[-1])
        top = torch.topk(logits, kmax, dim=-1).values
        kk = torch.tensor([(k if k > 0 else kmax) - 1 for k in ks], device=dev).clamp(max=kmax - 1)
        thresh = top.gather(1, kk[:, None])
        thresh = torch.where(torch.tensor([k > 0 for k in ks], device=dev)[:, None], thresh, -float("inf"))
        logits = logits.masked_fill(logits < thresh, -float("inf"))
    probs = torch.softmax(logits, dim=-1)
    tps = [p.top_p for p in params]
    if any(tp < 1.0 for tp in tps):
        sp, si = torch.sort(probs, dim=-1, descending=True)
        before = sp.cumsum(-1) - sp  # mass of strictly more likely tokens
        tp = torch.tensor(tps, device=dev)[:, None]
        sp = sp.masked_fill(before >= tp, 0.0)  # the top token always survives
        probs = torch.zeros_like(probs).scatter_(1, si, sp)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs


def sample_from_probs(probs, u):
    """Inverse-CDF draw: probs [n, V], u [n] uniform in [0, 1)."""
    cdf = probs.cumsum(-1)
    target = u.to(probs.device)[:, None] * cdf[:, -1:]
    idx = (cdf <= target).sum(-1)
    return idx.clamp(max=probs.shape[-1] - 1)


def sample(logits, seqs):
    """One next token per row of logits [n, V]."""
    out = [0] * len(seqs)
    greedy = [i for i, s in enumerate(seqs) if s.params.greedy]
    rand = [i for i, s in enumerate(seqs) if not s.params.greedy]
    if greedy:
        g = logits[greedy].argmax(-1).tolist()
        for i, t in zip(greedy, g):
            out[i] = t
    if rand:
        probs = probs_from_logits(logits[rand], [seqs[i].params for i in rand])
        u = torch.tensor([seqs[i].rng.random() for i in rand], dtype=torch.float32)
        r = sample_from_probs(probs, u).tolist()
        for i, t in zip(rand, r):
            out[i] = t
    return out
