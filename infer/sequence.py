import itertools
import random
import time
from dataclasses import dataclass

from .block_manager import KVState

WAITING, RUNNING, FINISHED = "waiting", "running", "finished"


@dataclass
class SamplingParams:
    temperature: float = 0.0  # 0 means greedy
    top_p: float = 1.0
    top_k: int = 0  # 0 means no limit
    max_tokens: int = 128
    seed: int | None = None
    ignore_eos: bool = False
    stop_token_ids: tuple = ()

    @property
    def greedy(self):
        return self.temperature <= 0.0


_ids = itertools.count()
_seed_rng = random.Random(0)


class Sequence:
    def __init__(self, prompt_ids, params: SamplingParams, eos_ids=()):
        if not prompt_ids:
            raise ValueError("empty prompt")
        self.id = next(_ids)
        self.ids = list(prompt_ids)
        self.prompt_len = len(self.ids)
        self.params = params
        self.status = WAITING
        self.finish_reason = None
        self.kv = KVState()
        self.draft = KVState()
        self.stop_ids = set(params.stop_token_ids) | (set() if params.ignore_eos else set(eos_ids))
        # every random draw of this request comes from its own stream, so
        # its output does not depend on what else is in the batch
        seed = params.seed if params.seed is not None else _seed_rng.getrandbits(63)
        self.rng = random.Random(seed)
        self.cached_tokens = 0  # prompt tokens served from the prefix cache
        self.preemptions = 0
        self.arrival = time.perf_counter()
        self.token_times = []

    @property
    def output_ids(self):
        return self.ids[self.prompt_len :]

    @property
    def num_output(self):
        return len(self.ids) - self.prompt_len

    @property
    def finished(self):
        return self.status == FINISHED

    def append(self, tok, now=None):
        """Adds a generated token. Returns True if the sequence is done."""
        self.ids.append(tok)
        self.token_times.append(now if now is not None else time.perf_counter())
        if tok in self.stop_ids:
            self.finish_reason = "stop"
        elif self.num_output >= self.params.max_tokens:
            self.finish_reason = "length"
        return self.finish_reason is not None
