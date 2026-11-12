"""The engine: requests in, tokens out, one scheduler step at a time."""

import time

import torch

from .block_manager import BlockManager
from .model import Model
from .runner import Item, Runner
from .sampler import probs_from_logits, sample, sample_from_probs
from .scheduler import Scheduler
from .sequence import FINISHED, SamplingParams, Sequence


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class Engine:
    def __init__(
        self,
        model_path,
        device=None,
        dtype=None,
        kv_cache_gb=2.0,
        num_blocks=None,
        block_size=16,
        max_num_seqs=64,
        max_batched_tokens=2048,
        prefix_caching=True,
        chunked_prefill=True,
        draft_model_path=None,
        num_spec_tokens=4,
        paged_kernel=True,
    ):
        self.device = device or default_device()
        if dtype is None:
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.model = Model.from_dir(model_path, self.device, dtype)
        self.eos_ids = self.model.cfg.eos_token_ids
        if num_blocks is None:
            num_blocks = int(kv_cache_gb * 2**30 // (self.model.kv_bytes_per_token() * block_size))
        self.block_size = block_size
        self.bm = BlockManager(num_blocks, block_size, prefix_caching)
        self.runner = Runner(self.model, num_blocks, block_size, paged_kernel)

        self.k = num_spec_tokens if draft_model_path else 0
        self.draft = None
        if draft_model_path:
            self.draft = Model.from_dir(draft_model_path, self.device, dtype)
            if self.draft.cfg.vocab_size != self.model.cfg.vocab_size:
                raise ValueError("draft and target vocabularies differ")
            # the draft caches the same tokens plus at most k, and shares
            # prefixes the same way; the margin covers timing differences
            self.draft_bm = BlockManager(num_blocks + num_blocks // 4, block_size, prefix_caching)
            self.draft_runner = Runner(self.draft, self.draft_bm.num_blocks, block_size, paged_kernel)

        self.sched = Scheduler(self.bm, max_num_seqs, max_batched_tokens, chunked_prefill, lookahead=self.k)
        if self.draft:
            self.sched.on_preempt = lambda s: self.draft_bm.free(s.draft)
        self.steps = 0
        self.spec_steps = 0  # per sequence verify passes
        self.spec_proposed = 0
        self.spec_accepted = 0

    # requests

    def add_request(self, prompt_ids, params: SamplingParams | None = None):
        params = params or SamplingParams()
        seq = Sequence(prompt_ids, params, self.eos_ids)
        cap = self.bm.num_blocks * self.block_size
        if len(prompt_ids) + params.max_tokens + self.k > cap:
            raise ValueError(f"prompt + max_tokens exceeds the kv cache ({cap} tokens)")
        self.sched.add(seq)
        return seq

    def has_work(self):
        return self.sched.has_work()

    def abort(self, seq):
        if seq in self.sched.waiting:
            self.sched.waiting.remove(seq)
            self.bm.free(seq.kv)
        elif seq in self.sched.running:
            self.sched.finish(seq)
        if self.draft:
            self.draft_bm.free(seq.draft)
        seq.status = FINISHED
        seq.finish_reason = seq.finish_reason or "abort"

    # stepping

    def step(self):
        """Runs one scheduler step. Returns [(seq, new_token_ids)]."""
        plan = self.sched.schedule()
        if not plan:
            return []
        self.steps += 1
        events = {}
        normal = plan
        spec = []
        if self.draft:
            normal = [(s, n) for s, n in plan if len(s.ids) - s.kv.num_computed != 1]
            spec = [s for s, n in plan if len(s.ids) - s.kv.num_computed == 1]

        if normal:
            items = []
            for s, n in normal:
                c = s.kv.num_computed
                done = c + n == len(s.ids)
                items.append(Item(s.kv.block_table, c, s.ids[c : c + n], 1 if done else 0))
            logits = self.runner.run(items)
            ready, rows = [], []
            for (s, n), lg in zip(normal, logits):
                s.kv.num_computed += n
                if lg.shape[0]:
                    ready.append(s)
                    rows.append(lg)
            if ready:
                toks = sample(torch.cat(rows), ready)
                now = time.perf_counter()
                for s, t in zip(ready, toks):
                    events[s] = [t]
                    if s.append(t, now):
                        s.status = FINISHED
            for s, _ in normal:
                self.bm.commit(s.kv, s.ids)

        if spec:
            for s, toks in self._spec_step(spec).items():
                events[s] = toks

        for s in list(events):
            if s.status == FINISHED:
                self.sched.finish(s)
                if self.draft:
                    self.draft_bm.free(s.draft)
        return list(events.items())

    def _spec_step(self, seqs):
        """Speculative decoding for sequences whose prompt is done.

        The draft model proposes k tokens one by one. The target model then
        runs the last real token plus the k proposals in one pass, which
        gives its distribution at all k+1 positions. Proposals are accepted
        left to right while the target agrees (greedy) or by rejection
        sampling (Leviathan et al. 2023), which leaves the output
        distribution exactly that of the target model. Each pass emits
        between 1 and k+1 tokens.
        """
        k = self.k
        n = len(seqs)
        for s in seqs:
            if not s.draft.block_table:
                self.draft_bm.match_prefix(s.draft, s.ids)
            if not self.draft_bm.ensure(s.draft, len(s.ids) + k - 1):
                raise RuntimeError("draft kv cache full; raise kv_cache_gb")

        greedy = [s.params.greedy for s in seqs]
        sampled = [i for i in range(n) if not greedy[i]]
        drafts = [[] for _ in range(n)]
        q_probs = [[] for _ in range(n)]
        inputs = [s.ids[s.draft.num_computed :] for s in seqs]
        for step in range(k):
            items = [Item(s.draft.block_table, s.draft.num_computed, inputs[i], 1) for i, s in enumerate(seqs)]
            logits = torch.cat(self.draft_runner.run(items))
            toks = [0] * n
            if any(greedy):
                g = [i for i in range(n) if greedy[i]]
                for i, t in zip(g, logits[g].argmax(-1).tolist()):
                    toks[i] = t
            if sampled:
                probs = probs_from_logits(logits[sampled], [seqs[i].params for i in sampled])
                u = torch.tensor([seqs[i].rng.random() for i in sampled])
                for j, (i, t) in enumerate(zip(sampled, sample_from_probs(probs, u).tolist())):
                    toks[i] = t
                    q_probs[i].append(probs[j])
            for i, s in enumerate(seqs):
                s.draft.num_computed += len(inputs[i])
                inputs[i] = [toks[i]]
                drafts[i].append(toks[i])

        items = [Item(s.kv.block_table, s.kv.num_computed, [s.ids[-1]] + drafts[i], k + 1) for i, s in enumerate(seqs)]
        target = self.runner.run(items)  # one [k+1, V] per sequence

        out = {}
        g = [i for i in range(n) if greedy[i]]
        if g:
            am = torch.stack([target[i] for i in g]).argmax(-1).tolist()
            for i, row in zip(g, am):
                acc = 0
                while acc < k and drafts[i][acc] == row[acc]:
                    acc += 1
                out[i] = (acc, drafts[i][:acc] + [row[acc]])
        for i in sampled:
            s = seqs[i]
            p = probs_from_logits(target[i], [s.params] * (k + 1))
            q = torch.stack(q_probs[i])
            d = torch.tensor(drafts[i], device=p.device)
            pd = p[:k].gather(1, d[:, None])[:, 0].tolist()
            qd = q.gather(1, d[:, None])[:, 0].tolist()
            acc = 0
            while acc < k and s.rng.random() * qd[acc] < pd[acc]:
                acc += 1
            if acc < k:
                resid = (p[acc] - q[acc]).clamp(min=0)
                if resid.sum() <= 0:  # p == q here, the draft could not be rejected
                    resid = p[acc]
                dist = resid / resid.sum()
            else:
                dist = p[k]
            last = sample_from_probs(dist[None], torch.tensor([s.rng.random()])).item()
            out[i] = (acc, drafts[i][:acc] + [last])

        events = {}
        now = time.perf_counter()
        for i, s in enumerate(seqs):
            acc, toks = out[i]
            self.spec_steps += 1
            self.spec_proposed += k
            self.spec_accepted += acc
            old = len(s.ids)
            emitted = []
            for t in toks:
                emitted.append(t)
                if s.append(t, now):
                    s.status = FINISHED
                    break
            # the target cached the last real token and all k proposals; only
            # those that were accepted (and kept) are valid
            s.kv.num_computed = old + len(emitted) - 1
            s.draft.num_computed = min(s.draft.num_computed, s.kv.num_computed)
            if s.status != FINISHED:
                self.bm.trim(s.kv, len(s.ids))
                self.draft_bm.trim(s.draft, len(s.ids))
                self.bm.commit(s.kv, s.ids)
                self.draft_bm.commit(s.draft, s.ids)
            events[s] = emitted
        return events

    # convenience

    def generate(self, prompts, params=None):
        """Runs prompts (lists of token ids) to completion. Returns output ids."""
        if isinstance(params, SamplingParams) or params is None:
            params = [params or SamplingParams()] * len(prompts)
        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, params)]
        while self.has_work():
            self.step()
        return [s.output_ids for s in seqs]

    def stats(self):
        st = {
            "steps": self.steps,
            "preemptions": self.sched.preemptions,
            "prompt_tokens": self.sched.prompt_tokens,
            "prefix_cache_hit_tokens": self.sched.cached_tokens,
            "kv_blocks": self.bm.num_blocks,
            "kv_usage": round(self.bm.usage(), 4),
            "running": len(self.sched.running),
            "waiting": len(self.sched.waiting),
        }
        if self.draft:
            st["spec_acceptance"] = self.spec_accepted / max(1, self.spec_proposed)
            st["spec_tokens_per_pass"] = (self.spec_accepted + self.spec_steps) / max(1, self.spec_steps)
        return st
