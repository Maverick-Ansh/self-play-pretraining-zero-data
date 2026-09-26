"""Re-bake DCLM and arithmetic with the bake scripts' DEFAULT sample counts (8192), as the
paper did, and keep the first 256 windows: data/c4096_default/*.npy. Used only by the gate.

(Our training-time eval corpora keep the 256-count bakes the sweep started with: same
scripts and sources, different draws, so their windows differ from the paper's.)
"""
import json
import os
import subprocess
import sys

import numpy as np

SCORING = "ext/self_play_pretraining/datasets"
os.makedirs("data/c4096_default", exist_ok=True)
for name, cmd in {"dclm": "prepare_dclm_benchmark", "arithmetic": "prepare_arithmetic_benchmark"}.items():
    out = os.path.abspath(f"data/c4096_default/{name}.jsonl")
    if not os.path.exists(out):
        subprocess.run(f"{sys.executable} -m src.scripts.{cmd} --context-length 4096 --output {out}",
                       shell=True, check=True, cwd=SCORING)
    rows = [json.loads(l)["sequence"] for _, l in zip(range(256), open(out))]
    np.save(f"data/c4096_default/{name}.npy", np.array(rows, dtype=np.uint8))
    print(name, len(rows))
