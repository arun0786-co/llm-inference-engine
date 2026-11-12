"""Speculative decoding: Qwen2.5-Coder-1.5B-Instruct as the target,
Qwen2.5-Coder-0.5B-Instruct as the draft, on 12 short coding prompts."""

import argparse

import torch

from common import LARGE, SMALL, Timer, release, resolve, save, table
from infer.engine import Engine
from infer.sequence import SamplingParams
from infer.tokenizer import Tokenizer

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Write a Python function to check whether a string is a palindrome, ignoring case and spaces.",
    "Implement binary search over a sorted list in Python.",
    "Write a JavaScript function that debounces another function.",
    "Write a SQL query that returns the top 5 customers by total order amount.",
    "Implement a stack class in Python with push, pop and peek.",
    "Write a Python function that merges two sorted lists into one sorted list.",
    "Write a C function that reverses a linked list.",
    "Write a Python function that counts word frequencies in a text file.",
    "Implement quicksort in Python.",
    "Write a bash script that prints the 10 largest files in a directory.",
    "Write a Python class for an LRU cache with get and put.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()
    target, draft = resolve(LARGE), resolve(SMALL)
    tok = Tokenizer(target)
    prompts = [tok.chat([{"role": "user", "content": p}]) for p in PROMPTS]
    rows = []
    reference = None

    def run(k, batch, temperature=0.0, model=None):
        nonlocal reference
        release()
        eng = Engine(model or target, kv_cache_gb=1, draft_model_path=draft if k else None, num_spec_tokens=k or 4)
        eng.generate([prompts[0]], SamplingParams(max_tokens=8))  # warm up
        sp = SamplingParams(max_tokens=a.max_tokens, temperature=temperature, seed=1)
        n_tok, outs = 0, []
        with Timer(eng.device) as t:
            for i in range(0, len(prompts), batch):
                o = eng.generate(prompts[i : i + batch], sp)
                outs += o
                n_tok += sum(map(len, o))
        st = eng.stats()
        row = {
            "draft k": "0.5B model alone" if model else (k or "none"),
            "batch": batch,
            "temperature": temperature,
            "tok/s": round(n_tok / t.s, 1),
            "acceptance": round(st.get("spec_acceptance", 0), 3) if k else "",
            "tokens per target pass": round(st.get("spec_tokens_per_pass", 1), 2) if k else 1,
        }
        if temperature == 0 and not model:
            if reference is None:
                reference = outs
            row["same text as target alone"] = f"{sum(x == y for x, y in zip(outs, reference))}/{len(outs)}"
        rows.append(row)
        print(row, flush=True)

    run(0, 1)
    run(0, 1, model=draft)  # how expensive one draft step is
    for k in (2, 3, 4, 6, 8):
        run(k, 1)
    run(0, 12)
    run(4, 12)
    run(0, 1, 0.7)
    run(4, 1, 0.7)
    # in bf16 the verify pass (k+1 tokens at once) rounds differently from
    # one-token decode steps, so greedy outputs can drift apart; in float32
    # they should not
    release()
    fp32 = {}
    for k in (0, 4):
        release()
        eng = Engine(target, dtype=torch.float32, kv_cache_gb=1, draft_model_path=draft if k else None, num_spec_tokens=4)
        fp32[k] = eng.generate(prompts[:6], SamplingParams(max_tokens=128))
        del eng
    same32 = sum(x == y for x, y in zip(fp32[0], fp32[4]))
    print(f"float32, k=4: {same32} of 6 outputs identical to the target alone")
    save("spec", {"target": LARGE, "draft": SMALL, "prompts": len(PROMPTS), "max_tokens": a.max_tokens, "rows": rows,
                  "float32_identical_k4": f"{same32}/6"})
    table(rows, ["draft k", "batch", "temperature", "tok/s", "acceptance", "tokens per target pass", "same text as target alone"])


if __name__ == "__main__":
    main()
