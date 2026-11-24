"""The throughput workload of bench/throughput.py on vLLM, for comparison:
the same 256 token-id prompts, the same per-request output lengths with
eos ignored, greedy. Needs `pip install vllm` (its own environment: vLLM
pins its torch version).

    python bench/vllm_throughput.py --model /path/to/Qwen2.5-Coder-0.5B-Instruct
"""

import argparse
import json
import os
import platform
import time

from throughput import RESULTS, workload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    a = ap.parse_args()

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    prompts, outs = workload(a.n)
    useful = sum(outs)
    llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=a.gpu_memory_utilization,
              max_num_seqs=256, max_num_batched_tokens=4096, seed=0)
    llm.generate([{"prompt_token_ids": prompts[0][:16]}], SamplingParams(max_tokens=8), use_tqdm=False)
    params = [SamplingParams(max_tokens=o, ignore_eos=True, temperature=0.0) for o in outs]
    # twice: the first run can include Triton compiling kernels for shapes
    # the warm-up did not hit
    rows = []
    for run in (1, 2):
        torch.cuda.synchronize()
        t = time.perf_counter()
        res = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t
        produced = sum(len(r.outputs[0].token_ids) for r in res)
        assert produced == useful, (produced, useful)
        rows.append({"system": f"vLLM {vllm.__version__}, run {run}", "seconds": round(secs, 1),
                     "output tok/s": round(useful / secs)})
        print(rows[-1], flush=True)
    gpu = torch.cuda.get_device_name(0).lower().replace("nvidia", "").replace("geforce", "")
    name = "vllm_throughput_" + "".join(c for c in gpu if c.isalnum())
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name + ".json"), "w") as f:
        json.dump({"machine": {"gpu": torch.cuda.get_device_name(0), "host": platform.node(),
                               "torch": torch.__version__, "vllm": vllm.__version__},
                   "model": a.model, "requests": a.n, "useful_output_tokens": useful, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
