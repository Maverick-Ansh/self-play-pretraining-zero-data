"""Assert the paper's rules, not just tensor shapes. CPU is enough; ~1 minute.

    python smoke.py
"""
import math

import numpy as np
import torch

from spp import generator as G
from spp.machine import ALPHABET, execute, log_g0, sample_uniform_prior
from spp.model import LADDER, LADDER_PARAMS, ByteLlama
from spp.reward import Snapshots, preconditioner, rewards, seq_losses
from spp.structure import scan

T = 511
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- the machine (App. E)
r = execute([b"+++[>+.<-]"], T)
assert r["out"][0][:6].tolist() == [1, 2, 3, 0, 0, 0] and r["n_out"][0] == 3, \
    "App. E: '+++[>+.<-]F' emits y = (1, 2, 3, 0, ..., 0)"
assert len(ALPHABET) == 19, "8 BF instructions + 10 macros + F (Sec. 2.3)"
bodies, _ = sample_uniform_prior(2000, 128, rng)
ex = execute(bodies + [b"[[[", b"]]]+.", b""], T)
assert ex["n_out"][-3] == 0 and ex["out"][-2][0] == 1 and ex["n_out"][-1] == 0, \
    "unmatched brackets are no-ops; the empty program emits nothing"
assert (ex["n_out"] <= T).all() and (ex["steps"] <= 2048 * (T + 1)).all(), \
    "execution always terminates within the step budget / T bytes"
table1 = {b"+[.++]": "arithmetic", b"+[[.C>.C>]]": "fibonacci", b"+[.L>]": "geometric",
          b"+++++++++.[<C>>VX<RX++]": "quadratic"}
out = execute(list(table1), T)["out"]
found = {i: h["family"] for i, h in scan(out)}
assert [found.get(i) for i in range(4)] == list(table1.values()), f"Table 1 families: {found}"
assert out[3][:4].tolist() == [9, 25, 59, 111], "Table 1 quadratic output"
assert not [h for _, h in scan(ex["out"]) if h["family"] != "arithmetic"], \
    "no Fibonacci/geometric/quadratic/cubic among 2000 uniform-prior programs (App. C)"
print("machine ok")

# ---------------------------------------------------------------- the transformer (Sec. 2.3)
for k in LADDER:
    assert ByteLlama.rung(k).n_params() == LADDER_PARAMS[k], f"{k} param count != release README"
print("param counts match the six released rungs exactly")

# ---------------------------------------------------------------- the generator starts at g0
gen = ByteLlama.rung("100k", 130)
torch.nn.init.zeros_(gen.lm_head.weight)
bodies, lp, charged = G.sample(gen, 512, 128)
assert np.allclose(lp, log_g0(charged), atol=1e-3), "zero lm_head => g_phi == g0 exactly"
assert all(set(b) <= set(ALPHABET[:-1].tobytes()) for b in bodies), "logits restricted to A"
assert abs(np.mean([len(b) for b in bodies]) - 18) < 2, "g0 length ~ Geometric(1/19): mean 18"
lp2 = G.logprob(gen, bodies[:64], 128).detach().numpy()
assert np.allclose(lp2, lp[:64], atol=1e-3), "sampling log-prob == scoring log-prob"
long = [bytes([ord("+")]) * 128]
assert abs(G.logprob(gen, long, 128).item() - log_g0([128])[0]) < 1e-3, "forced F at the cap costs 0"
print("generator ok")

# ---------------------------------------------------------------- the reward (Eq. 2) via forward mode
torch.manual_seed(0)
m = ByteLlama(64, 4, 2, max_len=64)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
X = torch.randint(0, 256, (6, 33))
X[:, 0] = ord("O")
n_out = torch.tensor([32, 5, 0, 32, 17, 1])
snaps = Snapshots(every=1)
for e in range(4):
    snaps.maybe_save(e, m)
    loss = seq_losses(m, dict(m.named_parameters()), X, n_out, manual=False).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
rj = rewards(m, opt, snaps, 4, X, n_out, chunk=4, signed=True)
P, (_, past) = preconditioner(opt, m), snaps.lookback(4)
ref = []
for i in range(6):
    m.zero_grad()
    seq_losses(m, dict(m.named_parameters()), X[i:i + 1], n_out[i:i + 1], manual=False).sum().backward()
    ref.append(sum((p.grad * P[n] * (past[n] - p.detach())).sum()
                   for n, p in m.named_parameters() if p.grad is not None))
ref = torch.stack(ref).detach()
assert torch.allclose(rj, ref, rtol=1e-4, atol=1e-7), "JVP reward == per-example grad . (P * delta)"
assert rj[2] == 0, "a program that emitted nothing earns no reward"
assert snaps.lookback(4)[0] == 2 and Snapshots(8).lookback(40)[0] == 16, "p(e) = floor(e/2)"
print("reward ok: forward-mode JVP matches per-example autograd")

# ---------------------------------------------------------------- the metric's ceiling
from spp.evals import bpb, kt_bpb
rnd = np.random.default_rng(1).integers(0, 256, (8, 511), dtype=np.uint8)
b = bpb(ByteLlama.rung("100k"), rnd)
assert b > 7.9, "no predictor gets below ~8 bits/byte on uniform random bytes"
assert kt_bpb(rnd, 0) > 8.0, "KT on random bytes pays the adaptation cost"
print(f"eval ok (random-init model on random bytes: {b:.3f} bits)")
print("ALL SMOKE CHECKS PASSED")
