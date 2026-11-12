"""Greedy outputs of the engine (all prompts batched, paged kernel on)
against transformers' generate (one prompt at a time) on a real
checkpoint, in float32 and bf16."""

import torch
from transformers import AutoModelForCausalLM

from common import SMALL, resolve, save
from infer.engine import Engine
from infer.sequence import SamplingParams
from infer.tokenizer import Tokenizer
from spec import PROMPTS


def main():
    path = resolve(SMALL)
    tok = Tokenizer(path)
    prompts = [tok.chat([{"role": "user", "content": p}]) for p in PROMPTS]
    n = 128
    res = {}
    for dtype in (torch.float32, torch.bfloat16):
        eng = Engine(path, dtype=dtype, kv_cache_gb=1)
        eng.runner.KERNEL_MIN_BATCH = 1  # 12 prompts: make sure the kernel runs
        ours = eng.generate(prompts, SamplingParams(max_tokens=n, ignore_eos=True))
        hf = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to("mps").eval()
        same, first_diff = 0, []
        for p, o in zip(prompts, ours):
            ref = hf.generate(torch.tensor([p], device="mps"), max_new_tokens=n, min_new_tokens=n, do_sample=False,
                                 repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
            ref = ref[0, len(p) :].tolist()
            k = next((i for i, (a, b) in enumerate(zip(o, ref)) if a != b), n)
            same += k == n
            first_diff.append(k)
        name = str(dtype).replace("torch.", "")
        res[name] = {"identical": same, "prompts": len(prompts), "tokens": n, "first_difference_at": first_diff}
        print(name, res[name], flush=True)
        del hf
    save("check_hf", {"model": SMALL, **res})


if __name__ == "__main__":
    main()
