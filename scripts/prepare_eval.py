"""Build the held-out evaluation corpora with the authors' own bake scripts.

Every corpus is baked at the paper's context (4096 -> 4095-byte windows) by the
pinned, checksummed scripts in nourya-aliz/Solomonoff-Figures/scoring, so the
bytes are identical to the paper's. They are then written twice:

    data/c4096/<name>.npy  uint8[<=256, 4095]  scoring the released checkpoints (smoke gate)
    data/c512/<name>.npy   uint8[256, 511]     our context: each 4095 window cut into 8x511

None of this data is ever used for a gradient step or for choosing a hyperparameter.

    python scripts/prepare_eval.py
"""
import json
import os
import subprocess
import sys

import numpy as np

EXT = "ext/Solomonoff-Figures"
SCORING = f"{EXT}/scoring"
BAKES = {                                   # our name -> bake command (run inside scoring/)
    "dclm": "prepare_dclm_benchmark --num-sequences 256",
    "cifar10_rgb_hwc": "prepare_cifar10_benchmark --datasets cifar10_rgb_hwc --num-sequences 256",
    "mutopia_melody_16th": "prepare_mutopia_benchmark",
    "arithmetic": "prepare_arithmetic_benchmark --num-samples 256",
}
PREBAKED = ["dna", "audio_8bit", "dclm_ranked"]          # shipped already baked in scoring/data/c4096


def sh(cmd, cwd=None):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def read_jsonl(path, n=256):
    rows = []
    for line in open(path):
        rows.append(json.loads(line)["sequence"])
        if len(rows) == n:
            break
    return np.array(rows, dtype=np.uint8)


def rewindow(a, w=511, n=256):
    """Cut 4095-byte windows into consecutive w-byte windows; keep n spread evenly."""
    k = a.shape[1] // w
    win = a[:, :k * w].reshape(-1, w)
    idx = np.linspace(0, len(win) - 1, min(n, len(win))).astype(int)
    return win[idx]


def main():
    if not os.path.isdir(EXT):
        sh(f"git clone -q --depth 1 https://github.com/nourya-aliz/Solomonoff-Figures {EXT}")
    sh(f"{sys.executable} -m pip -q install zstandard mido requests")
    os.makedirs("data/c4096", exist_ok=True)
    os.makedirs("data/c512", exist_ok=True)
    for name, cmd in BAKES.items():
        path = f"{SCORING}/data/c4096/{name}.jsonl"
        if not os.path.exists(path):
            sh(f"{sys.executable} -m src.scripts.{cmd} --context-length 4096", cwd=SCORING)
    for name in list(BAKES) + PREBAKED:
        a = read_jsonl(f"{SCORING}/data/c4096/{name}.jsonl")
        assert a.shape[1] == 4095, (name, a.shape)
        np.save(f"data/c4096/{name}.npy", a)
        if name != "dclm_ranked":
            np.save(f"data/c512/{name}.npy", rewindow(a))
        print(f"{name:22s} c4096 {a.shape}  c512 {rewindow(a).shape}")
    rnd = np.random.default_rng(12345).integers(0, 256, (256, 511), dtype=np.uint8)
    np.save("data/c512/random.npy", rnd)                     # control: nothing can beat 8.0 bits


if __name__ == "__main__":
    main()
