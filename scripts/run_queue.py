"""(Named run_queue, not queue: scripts/ is sys.path[0], and a file called queue.py
shadows the stdlib module that torch imports.)

Run a list of self-play runs sequentially on one GPU. Finished runs (final.json) are skipped.

    python scripts/run_queue.py --gpu 0 --runs "1M:selfplay:0:2048" "1M:uniform:0:2048"

Each spec is rung:arm:seed:rounds. Launch one detached queue per GPU.
"""
import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--extra", default="")
args = ap.parse_args()

env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), NUMBA_NUM_THREADS="2")
for spec in args.runs:
    rung, arm, seed, rounds = spec.split(":")
    out = f"runs/{rung}_{arm}_s{seed}"
    if os.path.exists(f"{out}/final.json"):
        continue
    cmd = [sys.executable, "-m", "spp.selfplay", "--rung", rung, "--arm", arm, "--seed", seed,
           "--rounds", rounds, "--out", out] + args.extra.split()
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env)
