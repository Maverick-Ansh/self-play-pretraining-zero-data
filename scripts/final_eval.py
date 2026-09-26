"""Score every finished run's last learner on the full held-out windows and the ICL tasks.

    python scripts/final_eval.py            # writes runs/<run>/final_eval.json, data/kt_baselines.json
"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, ".")
from spp.evals import CORPORA, ICL_MS, ICL_TASKS, bpb, icl_accuracy, kt_bpb, load_corpora
from spp.model import ByteLlama

corp = load_corpora("data/c512", 256)
if not os.path.exists("data/kt_baselines.json"):
    kt = {c: {f"order{k}": kt_bpb(a, k) for k in (0, 1, 2)} for c, a in corp.items()}
    json.dump(kt, open("data/kt_baselines.json", "w"), indent=1)
    print("KT baselines:", json.dumps(kt, indent=1), flush=True)

for run in sorted(glob.glob("runs/*/final.json")):
    d = os.path.dirname(run)
    if os.path.exists(f"{d}/final_eval.json"):
        continue
    cfg = json.load(open(f"{d}/config.json"))
    ckpt = sorted(glob.glob(f"{d}/learner_*.pt"))[-1]
    m = ByteLlama.rung(cfg["rung"], cfg["T"] + 1).cuda()
    m.load_state_dict({k: v.float() for k, v in torch.load(ckpt).items()})
    m.eval()
    res = dict(ckpt=ckpt, bpb={c: bpb(m, a) for c, a in corp.items()}, icl={})
    for t in ICL_TASKS:
        res["icl"][t] = {m_: icl_accuracy(m, t, m_, trials=256, seed=7) for m_ in ICL_MS}
    json.dump(res, open(f"{d}/final_eval.json", "w"), indent=1)
    print(d, {c: round(v, 3) for c, v in res["bpb"].items()}, flush=True)
