"""Scaling analysis: compute-optimal frontier and the power-law fit of Sec. 3.1.

    "For each dataset and algorithm, we construct a compute-optimal frontier over
     model sizes, checkpoints, and ensemble sizes. A point is on the frontier if
     and only if it achieves lower validation loss than every observed
     configuration with less than or equal compute. We fit each compute-optimal
     frontier with the asymptotic power law  L(C) = E + A C^{-alpha}."   (Sec. 3.1)

C = params x tokens (the paper's x-axis, K = 1 here: no ensembles).
"""
import glob
import json
import os

import numpy as np
from scipy.optimize import curve_fit


def load_run(path):
    """-> dict(config, evals=[{round, tokens, compute, bpb{}}], train=[...], final)."""
    ev, tr = [], []
    for line in open(f"{path}/log.jsonl"):
        r = json.loads(line)
        (ev if r["kind"] == "eval" else tr).append(r)
    fin = f"{path}/final.json"
    return dict(path=path, config=json.load(open(f"{path}/config.json")), evals=ev, train=tr,
                final=json.load(open(fin)) if os.path.exists(fin) else None)


def load_runs(root="runs"):
    return [load_run(p) for p in sorted(glob.glob(f"{root}/*")) if os.path.exists(f"{p}/log.jsonl")]


def points(runs, corpus):
    """All (compute, bpb) checkpoints of a set of runs on one corpus; round 0 excluded (C = 0)."""
    return np.array([(e["compute"], e["bpb"][corpus]) for r in runs for e in r["evals"] if e["compute"] > 0])


def frontier(pts):
    """Keep a point iff its loss beats every point with <= compute."""
    pts = pts[np.argsort(pts[:, 0])]
    keep, best = [], np.inf
    for c, l in pts:
        if l < best:
            keep.append((c, l))
            best = l
    return np.array(keep)


def power_law(C, E, A, alpha):
    return E + A * C ** (-alpha)


def fit(front):
    """Fit L = E + A C^-alpha on log-compute; returns (E, A, alpha) or None if it fails."""
    C, L = front[:, 0], front[:, 1]
    c0 = C.min()
    try:
        (E, A, a), _ = curve_fit(lambda x, E, A, a: E + A * (x / c0) ** (-a), C, L,
                                 p0=(L.min() * 0.8, L.max() - L.min() * 0.8, 0.2),
                                 bounds=([0, 0, 0], [8.5, 50, 3]), maxfev=20000)
    except RuntimeError:
        return None
    return E, A * c0 ** a, a
