"""The measurement gate: our model code + our eval data on the AUTHORS' released weights
must reproduce the AUTHORS' published bits/byte before any of our own number is trusted.

Published values: figure2/data/selfplay_frontier_perk.csv in acowsik/self_play_pretraining
(K = 1 column = mean over single seeds, 256 sequences of 4095 bytes, context 4096).

    python scripts/release_gate.py --rung 1M --round 16383
"""
import argparse
import csv
import json
import math
import sys
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from spp.evals import with_prefix
from spp.model import LADDER, ByteLlama

RUNG_TAG = {"100k": "d64h1L1", "500k": "d128h2L2", "1M": "d128h2L4", "3M": "d256h4L4",
            "6M": "d256h4L8", "24M": "d512h8L8"}
HF = "https://huggingface.co/nourya-cohen/solomonoff-paper/resolve/main"
CSV = "ext/self_play_pretraining/figures/figure2/data/selfplay_frontier_perk.csv"
CORPORA = ["dclm", "dclm_ranked", "cifar10_rgb_hwc", "audio_8bit", "mutopia_melody_16th", "dna", "arithmetic"]

ap = argparse.ArgumentParser()
ap.add_argument("--rung", default="1M")
ap.add_argument("--round", type=int, default=16383)
ap.add_argument("--n_seq", type=int, default=256)
ap.add_argument("--data", default="data/c4096")
ap.add_argument("--corpora", default=",".join(CORPORA))
args = ap.parse_args()

published = {r["corpus"]: float(r["bpb"]) for r in csv.DictReader(open(CSV))
             if r["rung"] == RUNG_TAG[args.rung] and r["K"] == "1" and int(r["round"]) == args.round}
tree = json.load(urllib.request.urlopen(
    f"https://huggingface.co/api/models/nourya-cohen/solomonoff-paper/tree/main/{args.rung}"))
seeds = [t["path"].split("/")[-1] for t in tree]
CORPORA = args.corpora.split(",")
data = {c: np.load(f"{args.data}/{c}.npy")[:args.n_seq] for c in CORPORA}
per_seed = {c: [] for c in CORPORA}
for s in seeds:
    try:
        path, _ = urllib.request.urlretrieve(f"{HF}/{args.rung}/{s}/learner_{args.round}.pth")
    except Exception:
        continue                                    # not every seed reached this round
    sd = torch.load(path, map_location="cpu", weights_only=False)["learner_state_dict"]
    m = ByteLlama.rung(args.rung, 4096).load_release(sd).cuda().eval()
    for c, a in data.items():
        X = torch.as_tensor(with_prefix(a).astype(np.int64))
        tot = 0.0
        with torch.no_grad():
            for lo in range(0, len(X), 8):
                x = X[lo:lo + 8].cuda()
                lp = torch.log_softmax(m(x[:, :-1]).float(), -1)       # fp32, as their scorer
                tot += -lp.gather(-1, x[:, 1:, None]).sum().item()
        per_seed[c].append(tot / (len(X) * (X.shape[1] - 1)) / math.log(2))
    print(f"{s}: " + " ".join(f"{c[:8]}={per_seed[c][-1]:.3f}" for c in CORPORA), flush=True)

print(f"\n{args.rung} round {args.round}, {len(next(iter(per_seed.values())))} seeds: ours (mean ± sd over seeds) vs published K=1")
res = {}
for c in CORPORA:
    v = np.array(per_seed[c])
    res[c] = dict(ours=float(v.mean()), sd=float(v.std()), published=published.get(c), n_seeds=len(v))
    print(f"  {c:22s} {v.mean():.3f} ± {v.std():.3f}   published {published.get(c, float('nan')):.3f}   "
          f"diff {v.mean() - published.get(c, float('nan')):+.3f}")
json.dump(res, open(f"{args.data}/release_gate_{args.rung}_{args.round}.json", "w"), indent=1)
