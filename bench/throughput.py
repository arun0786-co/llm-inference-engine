"""Offline throughput: 256 requests with mixed prompt and output lengths,
all submitted at once, against transformers' generate with static
batches. Output lengths are fixed per request (eos ignored), so every
system produces exactly the same number of useful tokens."""

import argparse
import random

import torch

import json
import os

from common import RESULTS, SMALL, Timer, release, resolve, save, table
from infer.engine import Engine
from infer.sequence import SamplingParams


def workload(n, seed=0):
    rng = random.Random(seed)
    prompts = [[rng.randrange(1000, 30000) for _ in range(rng.randint(32, 512))] for _ in range(n)]
    outs = [rng.randint(16, 512) for _ in range(n)]
    return prompts, outs


def run_engine(path, prompts, outs, **kw):
    release()
    eng = Engine(path, kv_cache_gb=4, **kw)
    eng.generate([prompts[0][:16]], SamplingParams(max_tokens=8))  # warm up
    params = [SamplingParams(max_tokens=o, ignore_eos=True) for o in outs]
    peak = 0
    with Timer(eng.device) as t:
        seqs = [eng.add_request(p, sp) for p, sp in zip(prompts, params)]
        while eng.has_work():
            eng.step()
            peak = max(peak, len(eng.sched.running))
    assert all(s.num_output == o for s, o in zip(seqs, outs))
    return t.s, {"peak_running": peak, "preemptions": eng.sched.preemptions, "steps": eng.steps}


def run_hf(path, prompts, outs, batch):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("mps").eval()
    pad = 0

    def gen(ps, n):
        m = max(len(p) for p in ps)
        ids = torch.tensor([[pad] * (m - len(p)) + p for p in ps], device="mps")
        mask = torch.tensor([[0] * (m - len(p)) + [1] * len(p) for p in ps], device="mps")
        return model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=pad
        )

    gen([prompts[0][:16]], 8)
    with Timer("mps") as t:
        for i in range(0, len(prompts), batch):
            ps, os_ = prompts[i : i + batch], outs[i : i + batch]
            # a static batch runs until its longest request is done
            gen(ps, max(os_))
    return t.s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--skip-hf", action="store_true")
    a = ap.parse_args()
    path = resolve(SMALL)
    prompts, outs = workload(a.n)
    useful = sum(outs)
    rows = []

    def add(name, secs, extra=None):
        rows.append({"system": name, "seconds": round(secs, 1), "output tok/s": round(useful / secs), **(extra or {})})
        print(rows[-1], flush=True)

    for name, kw in [
        ("engine: continuous batching, <= 256 seqs", {}),
        ("engine: same, gather + sdpa attention instead of the metal kernel", {"paged_kernel": False}),
        ("engine: same, no chunked prefill", {"chunked_prefill": False}),
        ("engine: <= 64 seqs", {"max_num_seqs": 64}),
    ]:
        s, ex = run_engine(path, prompts, outs, **({"max_num_seqs": 256, "max_batched_tokens": 4096} | kw))
        add(name, s, ex)
    if not a.skip_hf:
        for b in (64, 16):
            release()
            add(f"transformers generate, static batches of {b}", run_hf(path, prompts, outs, b))
    else:  # keep the earlier transformers rows
        with open(os.path.join(RESULTS, "throughput.json")) as f:
            rows += [r for r in json.load(f)["rows"] if r["system"].startswith("transformers")]
    save("throughput", {"model": SMALL, "requests": a.n, "useful_output_tokens": useful,
                        "prompt_tokens": sum(map(len, prompts)), "rows": rows})
    table(rows, ["system", "seconds", "output tok/s"])


if __name__ == "__main__":
    main()
