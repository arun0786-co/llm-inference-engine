"""Draw the README figures from the saved benchmark results in results/.

Nothing is re-run: every number comes from a json file written by a bench/ script.

    uv run --with matplotlib python scripts/make_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "images")

BLUE = "#257de0"     # this engine
ORANGE = "#f47230"   # vllm
GRAY = "#a3a3a3"     # transformers baselines
LIGHT = "#9cc3ef"    # engine variants
TEXT = "#191919"
DIM = "#666666"

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 10,
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": TEXT,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": DIM,
    "ytick.color": DIM,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.dpi": 150,
})


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(path, os.path.getsize(path) // 1024, "KB")


def short(system):
    s = system.replace("engine: ", "engine, ").replace("transformers generate, ", "transformers, ")
    s = s.replace("continuous batching, ", "").replace("same, ", "")
    s = s.replace("gather + sdpa attention instead of the metal kernel", "gather + SDPA, no kernel")
    s = s.replace("gather + sdpa attention instead of the cuda kernel", "gather + SDPA, no kernel")
    s = s.replace("static batches of", "static batch")
    s = s.replace(", run 1", ", first run").replace(", run 2", ", second run")
    return s


def colour(system):
    if system.startswith("vLLM"):
        return ORANGE
    if system.startswith("transformers"):
        return GRAY
    if system == "engine: continuous batching, <= 256 seqs":
        return BLUE
    return LIGHT


def throughput():
    mac = load("throughput.json")["rows"]
    t4 = load("throughput_teslat4.json")["rows"] + load("vllm_throughput_teslat4.json")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, rows, title in [(axes[0], mac, "Apple M4 Pro, bf16"), (axes[1], t4, "NVIDIA T4, float16")]:
        rows = sorted(rows, key=lambda r: r["output tok/s"])
        y = range(len(rows))
        vals = [r["output tok/s"] for r in rows]
        ax.barh(y, vals, color=[colour(r["system"]) for r in rows], height=0.7)
        ax.set_yticks(list(y), [short(r["system"]) for r in rows])
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.01, i, f"{v:,}", va="center", fontsize=9, color=TEXT)
        ax.set_xlim(0, max(vals) * 1.15)
        ax.set_xlabel("output tokens / s")
        ax.set_title(title)
        ax.grid(axis="x", color="#eeeeee")
        ax.set_axisbelow(True)
    fig.suptitle("256 requests, 66,214 output tokens, Qwen2.5-Coder-0.5B", x=0.01, ha="left",
                 fontsize=10, color=DIM, y=1.0)
    fig.tight_layout()
    save(fig, "throughput.png")


def prefix():
    rows = load("prefix.json")["rows"]
    off, on = rows
    metrics = [("total time", "seconds"), ("mean time to\nfirst token", "mean ttft s"),
               ("max time to\nfirst token", "max ttft s")]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = range(len(metrics))
    w = 0.36
    for k, (row, c, lab) in enumerate([(off, GRAY, "prefix cache off"), (on, BLUE, "prefix cache on")]):
        xs = [i + (k - 0.5) * w for i in x]
        vals = [row[m] for _, m in metrics]
        ax.bar(xs, vals, width=w - 0.03, color=c, label=lab)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.4, f"{v:.2f} s", ha="center", fontsize=9, color=TEXT)
    ax.set_xticks(list(x), [m for m, _ in metrics])
    ax.set_ylabel("seconds")
    ax.set_title("64 requests sharing a 2,000-token prefix (M4 Pro)")
    ax.legend(frameon=False, loc="upper center")
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(off["seconds"], off["max ttft s"]) * 1.15)
    fig.tight_layout()
    save(fig, "prefix_cache.png")


def attention():
    mac = load("attention.json")["rows"]
    t4 = load("attention_teslat4.json")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, rows, title, name in [(axes[0], mac, "Metal kernel, M4 Pro, bf16", "Metal"),
                                  (axes[1], t4, "CUDA kernel, T4, float16", "CUDA")]:
        labels = [f"{r['batch']} x {str(r['context']).replace('-', ' to ')}" for r in rows]
        x = range(len(rows))
        w = 0.38
        base = [r["gather + sdpa ms"] for r in rows]
        kern = [r["kernel ms"] for r in rows]
        ax.bar([i - w / 2 for i in x], base, width=w - 0.03, color=GRAY, label="gather + SDPA")
        ax.bar([i + w / 2 for i in x], kern, width=w - 0.03, color=BLUE, label=f"{name} kernel")
        for i, r in enumerate(rows):
            top = max(base[i], kern[i])
            ax.text(i, top * 1.12, f"{r['speedup']:.2f}x", ha="center", fontsize=8.5, color=TEXT)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks(list(x), labels, rotation=25, ha="right")
        ax.set_xlabel("sequences x context tokens")
        ax.set_ylabel("ms per call (log)")
        ax.set_title(title)
        ax.set_ylim(min(kern + base) * 0.5, max(kern + base) * 3)
        ax.legend(frameon=False, loc="upper left")
        ax.grid(axis="y", color="#eeeeee", which="major")
        ax.set_axisbelow(True)
    fig.suptitle("decode attention for one layer, Qwen2.5-0.5B shapes; label = kernel speedup",
                 x=0.01, ha="left", fontsize=10, color=DIM, y=1.0)
    fig.tight_layout()
    save(fig, "attention_kernel.png")


def latency():
    rows = load("latency.json")["rows"]
    metrics = [("p50", "inter-token ms p50"), ("p99", "p99"), ("max", "max")]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.36
    for k, (row, c, lab) in enumerate([(rows[0], GRAY, "chunked prefill off"),
                                       (rows[1], BLUE, "on, 512-token budget")]):
        xs = [i + (k - 0.5) * w for i in range(len(metrics))]
        vals = [row[m] for _, m in metrics]
        ax.bar(xs, vals, width=w - 0.03, color=c, label=lab)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 8, f"{v:.0f}", ha="center", fontsize=9, color=TEXT)
    ax.set_xticks(range(len(metrics)), [m for m, _ in metrics])
    ax.set_ylabel("ms between tokens")
    ax.set_title("decode stalls while 3,000-token prompts arrive (M4 Pro)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    ax.set_ylim(0, 620)
    fig.tight_layout()
    save(fig, "latency.png")


if __name__ == "__main__":
    throughput()
    prefix()
    attention()
    latency()
