import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from infer.engine import sync  # noqa: E402
from infer.hub import resolve  # noqa: E402

SMALL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
LARGE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def machine():
    chip = platform.processor()
    if sys.platform == "darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    # load average at the start: other jobs on the machine slow the python side
    return {"chip": chip, "torch": torch.__version__, "python": platform.python_version(), "load": os.getloadavg()[0]}


class Timer:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        sync(self.device)
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        sync(self.device)
        self.s = time.perf_counter() - self.t


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    obj = {"machine": machine(), **obj}
    with open(os.path.join(RESULTS, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)


def table(rows, cols):
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
