"""Measurements: zero-shot bits/byte, in-context baselines, and the ICL tasks.

Bits per byte (Sec. 3.1, App. B). The authors' scorer: input = byte 'O' + the
corpus window, loss = mean -log2 p(next byte) over every data byte. Reproduced
exactly here: scripts/release_gate.py scores their released 1M weights with this
code and gets their published numbers to 3 decimals on all 7 corpora.

A floor for "transfer". The paper compares self-play to two other *trained*
distributions (uniform prior, PCFG). It does not compare to a predictor that
has seen nothing at all but adapts in context. `kt_bpb` is that predictor: a
Krichevsky-Trofimov order-k counter, restarted for every window, no training.
Any byte model that beats it is using something beyond "count what you have
seen so far"; any gap it does NOT close is transfer that in-context counting
alone already explains.

ICL tasks (App. D), formats copied from the authors' ICL harness (released in
nourya-aliz/Solomonoff-Figures, which has since gone 404), with the same
cells the paper plots: reverse k=8 (teacher-forced), stack L=4, assoc V=16,
sum/max/min k=2. Sentinel 0 before each example; greedy exact match.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .machine import OUT_PREFIX

CORPORA = ["dclm", "cifar10_rgb_hwc", "audio_8bit", "mutopia_melody_16th", "dna",
           "arithmetic", "random"]
LABEL = {"dclm": "text (DCLM)", "cifar10_rgb_hwc": "images (CIFAR-10)",
         "audio_8bit": "speech (PCM8)", "mutopia_melody_16th": "melody (Mutopia)",
         "dna": "DNA", "arithmetic": "arithmetic", "random": "random bytes"}


def load_corpora(root, n=None, names=CORPORA):
    """{name: uint8[n, window]} from <root>/<name>.npy (written by scripts/prepare_eval.py)."""
    out = {}
    for c in names:
        a = np.load(f"{root}/{c}.npy")
        out[c] = a[:n] if n else a
    return out


def with_prefix(arr):
    return np.concatenate([np.full((len(arr), 1), OUT_PREFIX, np.uint8), arr], 1)


@torch.no_grad()
def bpb(model, arr, batch=32):
    """Mean bits per byte of `model` on windows `arr` [n, w] (zero-shot)."""
    dev = model.wte.weight.device
    X = torch.as_tensor(with_prefix(arr).astype(np.int64))
    tot, cnt = 0.0, 0
    for lo in range(0, len(X), batch):
        x = X[lo:lo + batch].to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(x[:, :-1])
        ce = F.cross_entropy(logits.float().transpose(1, 2), x[:, 1:], reduction="sum")
        tot += ce.item()
        cnt += x[:, 1:].numel()
    return tot / cnt / math.log(2)


def kt_bpb(arr, order=0, alpha=0.5):
    """In-context KT estimator of order k, fresh per window: p = (c + a) / (n + 256a)."""
    bits = 0.0
    for row in arr.astype(np.int64):
        counts, totals = {}, {}
        for t, b in enumerate(row):
            ctx = tuple(row[max(0, t - order):t]) if order else ()
            c = counts.setdefault(ctx, np.zeros(256))
            n = totals.get(ctx, 0)
            bits -= math.log2((c[b] + alpha) / (n + 256 * alpha))
            c[b] += 1
            totals[ctx] = n + 1
    return bits / arr.size


# ---------------------------------------------------------------- ICL tasks
def _fits(toks, ctx):
    return len(toks) <= ctx


def icl_task(name, m, rng, ctx=512):
    """One trial: (tokens, answer_positions, answers). Returns None if it overflows ctx."""
    O = [OUT_PREFIX]
    if name in ("sum", "max", "min"):
        f = {"sum": lambda x: int(x.sum() % 256), "max": lambda x: int(x.max()),
             "min": lambda x: int(x.min())}[name]
        toks = list(O)
        for _ in range(m):
            x = rng.integers(1, 256, 2)
            toks += [0] + x.tolist() + [f(x)]
        xq = rng.integers(1, 256, 2)
        toks += [0] + xq.tolist() + [f(xq)]
        return (toks, [len(toks) - 1], [f(xq)]) if _fits(toks, ctx) else None
    if name == "reverse":                                   # teacher-forced over the k=8 output slots
        k, toks = 8, list(O)
        for _ in range(m):
            x = rng.integers(1, 256, k)
            toks += [0] + x.tolist() + x[::-1].tolist()
        xq = rng.integers(1, 256, k)
        toks += [0] + xq.tolist() + xq[::-1].tolist()
        pos = list(range(len(toks) - k, len(toks)))
        return (toks, pos, xq[::-1].tolist()) if _fits(toks, ctx) else None
    if name == "stack":                                     # PUSH=250 v / POP=251 a, L=4 ops
        def trace(L=4):
            while True:
                t, st = [], []
                for _ in range(L - 1):
                    if st and rng.random() < 0.5:
                        t += [251, st.pop()]
                    else:
                        v = int(rng.integers(1, 250))
                        st.append(v)
                        t += [250, v]
                if st:
                    return t + [251], st[-1]
        toks = list(O)
        for _ in range(m):
            t, a = trace()
            toks += [0] + t + [a]
        t, a = trace()
        toks += [0] + t + [a]
        return (toks, [len(toks) - 1], [a]) if _fits(toks, ctx) else None
    if name == "assoc":                                     # V=16 dictionary, first V demos cover it
        V = 16
        if m < V:
            return None
        keys = rng.choice(np.arange(1, 256), V, replace=False)
        vals = rng.integers(1, 256, V)
        toks, shown = list(O), set()
        for j in range(m):
            i = int(rng.integers(0, V)) if j >= V else j
            toks += [0, int(keys[i]), int(vals[i])]
            shown.add(i)
        iq = int(rng.choice(sorted(shown)))
        toks += [0, int(keys[iq]), int(vals[iq])]
        return (toks, [len(toks) - 1], [int(vals[iq])]) if _fits(toks, ctx) else None
    raise ValueError(name)


ICL_TASKS = ["reverse", "stack", "assoc", "sum", "max", "min"]
ICL_MS = [0, 1, 2, 4, 8, 16, 32, 64, 120]


@torch.no_grad()
def icl_accuracy(model, name, m, trials=128, seed=0, ctx=512):
    """Greedy exact-match accuracy at the answer positions (mean over slots and trials)."""
    rng = np.random.default_rng(seed)
    rows = [icl_task(name, m, rng, ctx) for _ in range(trials)]
    if rows[0] is None:
        return None
    dev = model.wte.weight.device
    by_len = {}
    for toks, pos, ans in rows:                             # ragged (stack) -> batch by length
        by_len.setdefault(len(toks), []).append((toks, pos, ans))
    hits = tot = 0
    for group in by_len.values():
        X = torch.tensor([g[0] for g in group], device=dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            pred = model(X[:, :-1]).argmax(-1).cpu().numpy()
        for (toks, pos, ans), p in zip(group, pred):
            for q, a in zip(pos, ans):
                hits += int(p[q - 1] == a)
                tot += 1
    return hits / tot
