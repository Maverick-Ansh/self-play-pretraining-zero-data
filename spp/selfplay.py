"""One round of self-play, and the loop that runs it: Sec. 2 of arXiv:2609.30063.

    "Each round of self-play proceeds as follows:
     1. Program generation: Sample N programs from the generator.
     2. Execution: Run each program on U to obtain output sequences y_i = U(x_i, w_i).
     3. Learner and Generator update: The learner takes one gradient step on the
        output sequences, optimizing the standard next-token loss. The generator
        takes a policy gradient step with a learning-progress reward [...] In
        addition, the generator is updated via a supervised fine-tuning objective
        on existing programs to mitigate catastrophic forgetting and on mutated
        programs to promote exploration."                                 (Sec. 2)

Generator objective (Eqs. 3-5):
    A_i   = (r_i - mean_e r) / (std_e r + eps)  -  beta (log g(x_i) - log g0(x_i))
    L_PG  = -1/|B \\ B_mut| sum_{i not mut} sg(rho_i) log g(x_i) sg(A_i),
            rho_i = g(x_i) / g_old(x_i) clipped to [e^-20, e^20]
    L_EI  = -sum_i w_i log g(x_i),   w_i = [r_i]_+ / sum_j [r_j]_+
    L_gen = L_PG + lambda_EI L_EI,   lambda_EI = 1.0

Arms (every arm shares the machine, learner, pool size and token budget):
    selfplay   the method
    uniform    programs drawn i.i.d. from g0; no generator (the Sec. 3.1 control)
    shuffle    rewards permuted across the pool before every use (Table 5)
    signed     r_i without the absolute value (Table 5)

Run:  python -m spp.selfplay --rung 1M --arm selfplay --rounds 2048 --out runs/1M_sp_s0
"""
import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import generator as G
from .evals import bpb, load_corpora, with_prefix
from .machine import execute, log_g0, sample_uniform_prior
from .model import ByteLlama
from .pool import Archive, ReplayBank
from .reward import Snapshots, rewards, seq_losses
from .structure import FAMILIES, scan


@dataclass
class Config:
    rung: str = "1M"
    arm: str = "selfplay"
    rounds: int = 2048
    T: int = 511                  # output bytes per program; context = T + 1 with the O prefix
    pool: int = 256               # M_e programs per round
    mut_frac: float = 0.125
    replay_frac: float = 0.125
    max_body: int = 128           # program length cap L (a capped program gets a forced F)
    lr: float = 2e-3              # learner AdamW, constant after warmup
    gen_lr_ratio: float = 0.25    # generator lr = ratio * learner lr
    beta: float = 0.02            # KL(g || g0) coefficient
    lambda_ei: float = 1.0
    warmup: int = 32
    wd: float = 0.1
    snap_every: int = 8
    micro: int = 256              # learner micro-batch (the whole pool: these models are launch-bound)
    reward_chunk: int = 128       # JVP rows per forward (fp32 explicit attention)
    eval_every: int = 128
    eval_seqs: int = 64
    eval_dir: str = "data/c512"
    seed: int = 0
    out: str = "runs/debug"


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


def learner_step(learner, opt, scaler, X, n_out, micro):
    """One AdamW step on Eq. 1: (1/M) sum_i L(y_i). Returns mean loss in bits over emitting rows."""
    opt.zero_grad(set_to_none=True)
    params = dict(learner.named_parameters())
    M, tot = X.shape[0], 0.0
    for lo in range(0, M, micro):
        with torch.autocast("cuda", dtype=torch.float16):
            L = seq_losses(learner, params, X[lo:lo + micro], n_out[lo:lo + micro], manual=False)
        scaler.scale(L.sum() / M).backward()
        tot += L.detach().sum().item()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    return tot / max(int((n_out > 0).sum()), 1) / math.log(2)


def generator_step(gen, opt, scaler, bodies, kind, lp_old, charged, r, cfg):
    """One step on L_PG + lambda_EI L_EI (Eqs. 3-5). kind: 0 fresh, 1 mutation, 2 replay."""
    dev = r.device
    lp = G.logprob(gen, bodies, cfg.max_body)                                  # log g_phi(x), with grad
    kind_t = torch.as_tensor(kind, device=dev)
    with torch.no_grad():
        lg0 = torch.as_tensor(log_g0(charged), device=dev, dtype=torch.float32)
        A = (r - r.mean()) / (r.std() + 1e-8) - cfg.beta * (lp.detach() - lg0)
        log_rho = (lp.detach() - torch.as_tensor(lp_old, device=dev, dtype=torch.float32)).clamp(-20, 20)
        rho = torch.where(kind_t == 0, torch.ones_like(log_rho), log_rho.exp())
        w = r.clamp(min=0)
        w = w / w.sum() if w.sum() > 0 else w
    pg = kind_t != 1
    L_pg = -(rho * lp * A)[pg].mean()
    L_ei = -(w * lp).sum()
    loss = L_pg + cfg.lambda_ei * L_ei
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    kl = (lp.detach() - lg0)[kind_t == 0].mean().item()                      # MC estimate of KL(g||g0)
    return dict(L_pg=L_pg.item(), L_ei=L_ei.item(), kl=kl, gnorm=gn.item(),
                rho_rep_max=rho[kind_t == 2].max().item() if (kind_t == 2).any() else 1.0), lp.detach()


def charged_len(bodies, max_body):
    return np.array([len(b) + 1 if len(b) < max_body else len(b) for b in bodies])


def run(cfg: Config):
    os.makedirs(cfg.out, exist_ok=True)
    if os.path.exists(f"{cfg.out}/final.json"):
        print("done already:", cfg.out)
        return
    json.dump(dataclasses.asdict(cfg), open(f"{cfg.out}/config.json", "w"), indent=1)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = "cuda"
    selfplay = cfg.arm != "uniform"

    learner = ByteLlama.rung(cfg.rung, cfg.T + 1).to(dev)
    opt_l = torch.optim.AdamW(learner.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.wd)
    sc_l = torch.amp.GradScaler()
    if selfplay:
        gen = ByteLlama.rung(cfg.rung, cfg.max_body + 2).to(dev)
        torch.nn.init.zeros_(gen.lm_head.weight)                   # policy starts exactly at g0
        opt_g = torch.optim.AdamW(gen.parameters(), lr=cfg.lr * cfg.gen_lr_ratio,
                                  betas=(0.9, 0.95), weight_decay=0.0)
        sc_g = torch.amp.GradScaler()
        archive, bank, snaps = Archive(), ReplayBank(), Snapshots(cfg.snap_every)
    evalset = load_corpora(cfg.eval_dir, cfg.eval_seqs) if os.path.isdir(cfg.eval_dir) else {}
    first_seen, hit_counts = {}, {f: 0 for f in FAMILIES}
    logf = open(f"{cfg.out}/log.jsonl", "a")
    n_params = learner.n_params()
    tokens = 0
    t_start = time.time()

    for e in range(cfg.rounds + 1):
        # -------------------------------------------------- evaluation (before round e's update)
        if e % cfg.eval_every == 0 or e in (8, 16, 32, 64) or e == cfg.rounds:   # log-spaced early points
            learner.eval()
            ev = {c: bpb(learner, a) for c, a in evalset.items()}
            learner.train()
            rec = dict(kind="eval", round=e, tokens=tokens, compute=n_params * tokens, bpb=ev,
                       wall=time.time() - t_start)
            if selfplay:
                progs, _, _ = G.sample(gen, 32, cfg.max_body)
                rec["programs"] = [p.decode() for p in progs]
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            torch.save({k: v.half() for k, v in learner.state_dict().items()},
                       f"{cfg.out}/learner_{e:05d}.pt")
            print(f"[{cfg.out}] round {e} tokens {tokens/1e6:.1f}M  " +
                  " ".join(f"{c[:6]}={v:.3f}" for c, v in ev.items()), flush=True)
        if e == cfg.rounds:
            break

        t0 = time.time()
        lr = cfg.lr * min(1.0, (e + 1) / cfg.warmup)
        set_lr(opt_l, lr)
        # -------------------------------------------------- 1. program generation (the pool B_e)
        if selfplay:
            set_lr(opt_g, lr * cfg.gen_lr_ratio)
            n_mut = int(cfg.pool * cfg.mut_frac) if len(archive) else 0
            rep, rep_lp = bank.draw(int(cfg.pool * cfg.replay_frac), rng)
            fresh, fresh_lp, _ = G.sample(gen, cfg.pool - n_mut - len(rep), cfg.max_body)
            mut = archive.mutants(n_mut, rng, cfg.max_body)
            bodies = fresh + mut + rep
            kind = np.array([0] * len(fresh) + [1] * len(mut) + [2] * len(rep))
            lp_old = np.concatenate([fresh_lp, np.zeros(len(mut)), rep_lp])
        else:
            bodies, _ = sample_uniform_prior(cfg.pool, cfg.max_body, rng)
            kind = np.zeros(len(bodies), dtype=int)
        t1 = time.time()
        # -------------------------------------------------- 2. execution on U with fresh random tapes
        ex = execute(bodies, cfg.T, seed=cfg.seed * 1_000_003 + e)
        X = torch.as_tensor(with_prefix(ex["out"]).astype(np.int64), device=dev)
        n_out = torch.as_tensor(ex["n_out"], device=dev)
        t2 = time.time()
        # -------------------------------------------------- 3a. reward at theta_e (Eq. 2)
        stats = {}
        if selfplay:
            snaps.maybe_save(e, learner)
            r = rewards(learner, opt_l, snaps, e, X, n_out, cfg.reward_chunk, signed=cfg.arm == "signed")
            if cfg.arm == "shuffle":
                r = r[torch.randperm(len(r), device=dev)]
        t3 = time.time()
        # -------------------------------------------------- 3b. learner step (Eq. 1)
        loss_bits = learner_step(learner, opt_l, sc_l, X, n_out, cfg.micro)
        tokens += int(X.shape[0] * cfg.T)
        t4 = time.time()
        # -------------------------------------------------- 3c. generator step (Eqs. 3-5) + pool upkeep
        if selfplay:
            charged = charged_len(bodies, cfg.max_body)
            stats, lp_now = generator_step(gen, opt_g, sc_g, bodies, kind, lp_old, charged, r, cfg)
            rc = r.cpu().numpy()
            archive.step_decay()
            archive.add(bodies, rc, ex["depth"])
            nonrep = kind != 2
            lp_bank = np.where(kind == 0, lp_old, lp_now.cpu().numpy())
            bank.add([b for b, k in zip(bodies, nonrep) if k], lp_bank[nonrep], rng)
            stats.update(r_mean=float(rc.mean()), r_std=float(rc.std()), r_max=float(rc.max()),
                         archive=len(archive), niches=sum(1 for c in archive.cells.values() if c),
                         bank=len(bank))
        # -------------------------------------------------- discovery of mathematical structure (App. C)
        nonrep = np.nonzero(kind != 2)[0]
        for i, h in scan(ex["out"][nonrep]):
            f = h["family"]
            hit_counts[f] += 1
            if f not in first_seen:
                first_seen[f] = dict(round=e, program=bodies[nonrep[i]].decode(), terms=h["first_terms"])
        t5 = time.time()
        if e % 8 == 0:
            lens = np.array([len(b) for b in bodies])
            rec = dict(kind="train", round=e, tokens=tokens, loss_bits=loss_bits, lr=lr,
                       body_len=float(lens[kind == 0].mean()), emit_frac=float((ex["n_out"] > 0).mean()),
                       full_frac=float((ex["n_out"] == cfg.T).mean()), n_out=float(ex["n_out"].mean()),
                       hits=dict(hit_counts), t_sample=t1 - t0, t_exec=t2 - t1, t_reward=t3 - t2,
                       t_learn=t4 - t3, t_gen=t5 - t4, **stats)
            logf.write(json.dumps(rec) + "\n")
        if e % 64 == 0:
            logf.flush()
            print(f"[{cfg.out}] r{e} loss {loss_bits:.3f}b  len {np.mean([len(b) for b in bodies]):.1f} "
                  f"full {(ex['n_out'] == cfg.T).mean():.2f}  {t5 - t0:.2f}s/round  "
                  + (f"kl {stats['kl']:.1f} r {stats['r_mean']:.2e}" if selfplay else ""), flush=True)

    final = dict(config=dataclasses.asdict(cfg), n_params=n_params, tokens=tokens,
                 first_seen=first_seen, hit_counts=hit_counts, wall=time.time() - t_start)
    if selfplay:
        torch.save(gen.state_dict(), f"{cfg.out}/generator.pt")
    json.dump(final, open(f"{cfg.out}/final.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    run(Config(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
