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
        order = [i for q in sorted(by_q) for i in by_q[q]]

        ids, pos, slots, logit_rows = [], [], [], {}
        groups = []
        t = 0
        n_rows = 0
        for q in sorted(by_q):
            members = by_q[q]
            start_t = t
            for i in members:
                it = items[i]
                ids.extend(it.tokens)
                pos.extend(range(it.start, it.start + q))
                slots.append(self._slots(it.block_table, it.start, it.start + q))
                logit_rows[i] = (n_rows, n_rows + it.n_logits, list(range(t + q - it.n_logits, t + q)))
                n_rows += it.n_logits
                t += q
            b = len(members)
            if all(items[i].start == 0 for i in members):
                groups.append(AttnGroup(start_t, b, q, None, None))
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
            # query r of a sequence sits at position ctx - q + r and sees
            # every key at or before it
            last = (ctx - q)[:, None, None] + torch.arange(q)[None, :, None]
            mask = (j[None, None, :] <= last)[:, None]
            groups.append(AttnGroup(start_t, b, q, kv_slots.to(dev), mask.to(dev)))

        flat_index = []
        for i in order:
            flat_index.extend(logit_rows[i][2])
        # logit rows come out in flat order; map each item to its slice
        offsets, r = {}, 0
        for i in order:
            n = items[i].n_logits
            offsets[i] = (r, r + n)
            r += n

        batch = AttnBatch(torch.cat(slots).to(dev), groups)
        logits = self.model.forward(
            torch.tensor(ids, device=dev),
            torch.tensor(pos, device=dev),
            batch,
            self.kv,
            torch.tensor(flat_index, dtype=torch.long, device=dev),
        )
        self.forward_calls += 1
        self.forward_tokens += t
        return [logits[offsets[i][0] : offsets[i][1]] for i in range(len(items))]
