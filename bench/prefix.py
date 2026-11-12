"""Prefix caching: 64 requests share a 2,000-token system prompt (think of
a long instruction block or a document) and each adds its own short
question. They arrive at once."""

import random
import statistics

import torch

from common import SMALL, Timer, release, resolve, save, table
from infer.engine import Engine
from infer.sequence import SamplingParams


def main():
    path = resolve(SMALL)
    rng = random.Random(0)
    system = [rng.randrange(1000, 30000) for _ in range(2000)]
    prompts = [system + [rng.randrange(1000, 30000) for _ in range(rng.randint(10, 40))] for _ in range(64)]
    sp = SamplingParams(max_tokens=32, ignore_eos=True)
    rows = []
    outs = {}
    for on in (False, True):
        release()
        eng = Engine(path, kv_cache_gb=4, prefix_caching=on, max_num_seqs=64, max_batched_tokens=4096)
        eng.generate([system[:32]], SamplingParams(max_tokens=4))
        eng.runner.forward_tokens = 0
        with Timer(eng.device) as t:
            seqs = [eng.add_request(p, sp) for p in prompts]
            while eng.has_work():
                eng.step()
        ttft = [s.token_times[0] - s.arrival for s in seqs]
        outs[on] = [s.output_ids for s in seqs]
        rows.append({
            "prefix cache": "on" if on else "off",
            "seconds": round(t.s, 2),
            "tokens run through the model": eng.runner.forward_tokens,
            "prompt tokens from cache": eng.sched.cached_tokens - 32 * on,
            "mean ttft s": round(statistics.mean(ttft), 2),
            "max ttft s": round(max(ttft), 2),
        })
        print(rows[-1], flush=True)
    same = sum(a == b for a, b in zip(outs[False], outs[True]))
    print(f"identical outputs with and without the cache: {same} of {len(prompts)}")
    # bf16: the cached keys were computed in a different batch shape than
    # a fresh prefill, so rounding differs. float32 on 16 requests:
    out32 = {}
    for on in (False, True):
        release()
        eng = Engine(path, dtype=torch.float32, kv_cache_gb=2, prefix_caching=on, max_num_seqs=64, max_batched_tokens=4096)
        out32[on] = eng.generate(prompts[:16], sp)
        del eng
    same32 = sum(a == b for a, b in zip(out32[False], out32[True]))
    print(f"float32: {same32} of 16 identical")
    save("prefix", {"model": SMALL, "requests": 64, "shared_prefix_tokens": 2000, "rows": rows,
                    "identical_outputs_bf16": f"{same}/64", "identical_outputs_float32": f"{same32}/16"})
    table(rows, list(rows[0]))


if __name__ == "__main__":
    main()
