"""Continuous batching.

Every step the scheduler picks which sequences run and how many new
tokens each brings, under two limits: a token budget per step and a cap
on running sequences. Running sequences go first, oldest first. A
sequence still prefilling gets a chunk of its prompt; a decoding one gets
one token (or 1 + k with speculative decoding). Leftover budget admits
waiting requests, whose prompts may be split over several steps (chunked
prefill), so a long prompt never stalls the decodes already in flight.

If the KV cache runs out, the newest running sequence is preempted: its
blocks are freed and it goes back to the front of the queue. When it is
scheduled again its prompt and the tokens it already generated are
recomputed, mostly from the prefix cache.
"""

from collections import deque

from .block_manager import BlockManager
from .sequence import RUNNING, WAITING, Sequence


class Scheduler:
    def __init__(
        self,
        bm: BlockManager,
        max_num_seqs=64,
        max_batched_tokens=2048,
        chunked_prefill=True,
        lookahead=0,
        watermark=0.01,
    ):
        self.bm = bm
        self.max_num_seqs = max_num_seqs
        self.max_batched_tokens = max_batched_tokens
        self.chunked_prefill = chunked_prefill
        self.lookahead = lookahead  # extra slots a decode needs (speculative tokens)
        self.reserve = int(watermark * bm.num_blocks)
        self.waiting = deque()
        self.running = []
        self.preemptions = 0
        self.prompt_tokens = 0  # prompt tokens admitted (first admission only)
        self.cached_tokens = 0  # of those, served from the prefix cache
        self.on_preempt = None  # called with a preempted sequence

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def has_work(self):
        return bool(self.waiting or self.running)

    def _preempt(self, seq):
        self.bm.free(seq.kv)
        if self.on_preempt:
            self.on_preempt(seq)
        seq.status = WAITING
        seq.preemptions += 1
        self.preemptions += 1
        self.waiting.appendleft(seq)

    def _need(self, seq, budget):
        """Tokens to run now and cache slots to hold, for one sequence."""
        todo = len(seq.ids) - seq.kv.num_computed
        if todo == 1:
            return 1, 1 + self.lookahead
        if self.chunked_prefill:
            n = min(todo, budget)
            return n, n
        return todo, todo

    def schedule(self):
        """Returns [(seq, n_new_tokens)] for this step."""
        budget = self.max_batched_tokens
        out, kept = [], []
        queue = list(self.running)
        preempted = False
        while queue:
            seq = queue.pop(0)
            n, need = self._need(seq, budget)
            if n <= 0 or need > budget:
                kept.append(seq)  # no room this step; stays running
                continue
            while not self.bm.ensure(seq.kv, seq.kv.num_computed + need):
                preempted = True
                victim = queue.pop() if queue else seq
                self._preempt(victim)
                if victim is seq:
                    break
            if seq.status != RUNNING:
                break
            kept.append(seq)
            out.append((seq, n))
            budget -= need
        self.running = kept

        # admit new requests, unless memory was just short
        while not preempted and self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            hit = 0
            if not seq.kv.block_table:
                hit = self.bm.match_prefix(seq.kv, seq.ids)
            n, need = self._need(seq, budget)
            if need > budget or not self.bm.ensure(seq.kv, seq.kv.num_computed + need, reserve=self.reserve):
                self.bm.free(seq.kv)  # give the matched prefix back
                break
            self.waiting.popleft()
            seq.status = RUNNING
            if seq.preemptions == 0:
                seq.cached_tokens = hit
                self.prompt_tokens += len(seq.ids)
                self.cached_tokens += hit
            self.running.append(seq)
            out.append((seq, n))
            budget -= need
        return out

    def finish(self, seq):
        self.bm.free(seq.kv)
        self.running.remove(seq)
