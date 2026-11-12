"""What chunked prefill buys: 32 requests are decoding when eight
3,000-token prompts arrive, one every 20 steps. Without chunking each
long prompt runs in a single step and every decoding request waits for
it. With a 512-token budget the prompt is spread over six steps."""

import random
import statistics

from common import SMALL, resolve, save, table
from infer.engine import Engine, sync
from infer.sequence import SamplingParams


def run(path, chunked):
    rng = random.Random(0)
    eng = Engine(path, kv_cache_gb=4, max_num_seqs=64, chunked_prefill=chunked,
                 max_batched_tokens=512 if chunked else 4096)
    eng.generate([[1000] * 16], SamplingParams(max_tokens=4))
    short = [eng.add_request([rng.randrange(1000, 30000) for _ in range(64)], SamplingParams(max_tokens=300, ignore_eos=True))
             for _ in range(32)]
    long = []
    step = 0
    while eng.has_work():
        if step >= 20 and step % 20 == 0 and len(long) < 8:
            long.append(eng.add_request([rng.randrange(1000, 30000) for _ in range(3000)], SamplingParams(max_tokens=16, ignore_eos=True)))
        eng.step()
        sync(eng.device)
        step += 1
    gaps = [b - a for s in short for a, b in zip(s.token_times, s.token_times[1:])]
    gaps.sort()
    ttft = [s.token_times[0] - s.arrival for s in long]
    return {
        "chunked prefill": "on, 512-token budget" if chunked else "off",
        "inter-token ms p50": round(1000 * gaps[len(gaps) // 2], 1),
        "p99": round(1000 * gaps[int(len(gaps) * 0.99)], 1),
        "max": round(1000 * gaps[-1], 1),
        "long prompt ttft s (mean)": round(statistics.mean(ttft), 2),
        "total s": round(max(s.token_times[-1] for s in short + long) - min(s.arrival for s in short), 1),
    }


def main():
    path = resolve(SMALL)
    rows = [run(path, False), run(path, True)]
    for r in rows:
        print(r)
    save("latency", {"model": SMALL, "rows": rows})
    table(rows, list(rows[0]))


if __name__ == "__main__":
    main()
