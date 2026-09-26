"""How often does the fixed prior g0 produce each mathematical family by chance? (Table 1, last column)

    "we sample programs by uniformly sampling the primitive augmented alphabet until an "F"
     symbol is drawn. We draw 1.64 x 10^8 programs in total. [...] The arithmetic family occurs
     1,526 times in the uniform baseline, giving an estimated probability of 9.3 x 10^-6"  (App. C)

Same procedure at this reproduction's tape length (T = 511) and length cap (128). Writes
results/prior_discovery.json with per-family hit counts; the expected first-discovery round at
R programs per round is 1 / (R p), with the rule of three (p <= 3/N) when a family never appears.

    python scripts/prior_discovery.py --n 20000000
"""
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from spp.machine import execute, sample_uniform_prior
from spp.structure import FAMILIES, scan

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20_000_000)
ap.add_argument("--chunk", type=int, default=65_536)
ap.add_argument("--T", type=int, default=511)
ap.add_argument("--out", default="results/prior_discovery.json")
args = ap.parse_args()

rng = np.random.default_rng(2026)
counts, examples, done, t0 = {f: 0 for f in FAMILIES}, {}, 0, time.time()
while done < args.n:
    bodies, _ = sample_uniform_prior(args.chunk, 128, rng)
    for i, h in scan(execute(bodies, args.T, seed=done)["out"]):
        counts[h["family"]] += 1
        examples.setdefault(h["family"], dict(program=bodies[i].decode(), terms=h["first_terms"]))
    done += args.chunk
    if done % (args.chunk * 32) == 0:
        print(f"{done:,} programs  {time.time() - t0:.0f}s  {counts}", flush=True)
        json.dump(dict(samples=done, T=args.T, hit_counts=counts, examples=examples), open(args.out, "w"), indent=1)
json.dump(dict(samples=done, T=args.T, hit_counts=counts, examples=examples), open(args.out, "w"), indent=1)
print("final", done, counts)
