"""Turns a list of scheduled sequences into the tensors of one forward pass."""

from collections import defaultdict
from dataclasses import dataclass

import torch

from .model import AttnBatch, AttnGroup, Model


@dataclass
class Item:
    block_table: list
    start: int  # position of the first new token (= tokens already cached)
    tokens: list  # new tokens to run
    n_logits: int  # logits wanted for this many trailing tokens


class Runner:
    def __init__(self, model: Model, num_blocks, block_size):
        self.model = model
        self.block_size = block_size
        self.kv = model.new_kv_cache(num_blocks, block_size)
        self.forward_calls = 0
        self.forward_tokens = 0

    def _slots(self, block_table, start, end):
        bs = self.block_size
        p = torch.arange(start, end)
        bt = torch.tensor(block_table, dtype=torch.long)
        return bt[p // bs] * bs + p % bs

    def run(self, items):
        """Returns one float32 logits tensor [n_logits, V] per item."""
        dev = self.model.device
        bs = self.block_size
        # items with the same number of new tokens go in one attention call;
        # laying them out next to each other makes each group a [b, q] slice
        by_q = defaultdict(list)
        for i, it in enumerate(items):
            by_q[len(it.tokens)].append(i)

        ids, pos, slots, logit_index = [], [], [], []
        rows = [None] * len(items)
        groups = []
        t = 0
        for q in sorted(by_q):
            members = by_q[q]
            group_start = t
            for i in members:
                it = items[i]
                ids.extend(it.tokens)
                pos.extend(range(it.start, it.start + q))
                slots.append(self._slots(it.block_table, it.start, it.start + q))
                rows[i] = (len(logit_index), len(logit_index) + it.n_logits)
                logit_index.extend(range(t + q - it.n_logits, t + q))
                t += q
            b = len(members)
            if all(items[i].start == 0 for i in members):
                groups.append(AttnGroup(group_start, b, q, None, None))
                continue
            ctx = torch.tensor([items[i].start + q for i in members])
            maxc = int(ctx.max())
            nb = -(-maxc // bs)
            bt = torch.zeros(b, nb, dtype=torch.long)
            for r, i in enumerate(members):
                tab = items[i].block_table[:nb]
                bt[r, : len(tab)] = torch.tensor(tab, dtype=torch.long)
            j = torch.arange(maxc)
            kv_slots = bt[:, j // bs] * bs + j % bs
            mask = None
            if b > 1 or q > 1:
                # query r of a sequence sits at position ctx - q + r and sees
                # every key at or before it
                last = (ctx - q)[:, None, None] + torch.arange(q)[None, :, None]
                mask = (j[None, None, :] <= last)[:, None].to(dev)
            groups.append(AttnGroup(group_start, b, q, kv_slots.to(dev), mask))

        batch = AttnBatch(torch.cat(slots).to(dev), groups)
        logits = self.model.forward(
            torch.tensor(ids, device=dev),
            torch.tensor(pos, device=dev),
            batch,
            self.kv,
            torch.tensor(logit_index, dtype=torch.long, device=dev),
        )
        self.forward_calls += 1
        self.forward_tokens += t
        return [logits[a:b] for a, b in rows]
