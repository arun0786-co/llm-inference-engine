"""The model against transformers on random checkpoints, float32 on cpu."""

import random

import torch

from infer.model import Model
from infer.runner import Item, Runner

BS = 4


def reference_logits(hf, ids):
    with torch.no_grad():
        return hf(torch.tensor([ids])).logits[0].float()


def alloc(n_tokens, free):
    return [free.pop() for _ in range(-(-n_tokens // BS))]


def test_prefill_matches(tiny):
    path, hf = tiny
    m = Model.from_dir(path)
    r = Runner(m, 64, BS)
    ids = [random.Random(1).randrange(256) for _ in range(23)]
    free = list(range(64))
    (got,) = r.run([Item(alloc(23, free), 0, ids, len(ids))])
    ref = reference_logits(hf, ids)
    assert (got - ref).abs().max() < 1e-4


def test_chunked_and_decode_match(tiny):
    """Prefill in uneven chunks, then decode one token at a time: every
    position's logits equal a single full forward pass."""
    path, hf = tiny
    m = Model.from_dir(path)
    r = Runner(m, 64, BS)
    rng = random.Random(2)
    ids = [rng.randrange(256) for _ in range(40)]
    ref = reference_logits(hf, ids)
    free = list(range(64))
    table = alloc(40, free)
    got = []
    pos = 0
    for n in [7, 1, 9, 3] + [1] * 20:
        (lg,) = r.run([Item(table, pos, ids[pos : pos + n], n)])
        got.append(lg)
        pos += n
    got = torch.cat(got)
    assert got.shape[0] == 40
    assert (got - ref).abs().max() < 1e-4


def test_mixed_batch_matches(tiny):
    """Several sequences in one call, some prefilling, some decoding, with
    shuffled physical blocks: each row matches its own reference."""
    path, hf = tiny
    m = Model.from_dir(path)
    r = Runner(m, 128, BS)
    rng = random.Random(3)
    free = list(range(128))
    rng.shuffle(free)
    seqs = [[rng.randrange(256) for _ in range(n)] for n in (5, 17, 30, 12, 9)]
    tables = [alloc(len(s), free) for s in seqs]
    # first pass: everything but the last 1 or 3 tokens
    tail = [1, 3, 1, 1, 3]
    r.run([Item(t, 0, s[: len(s) - k], 1) for s, t, k in zip(seqs, tables, tail)])
    out = r.run([Item(t, len(s) - k, s[len(s) - k :], k) for s, t, k in zip(seqs, tables, tail)])
    for s, k, lg in zip(seqs, tail, out):
        ref = reference_logits(hf, s)[-k:]
        assert (lg - ref).abs().max() < 1e-4
