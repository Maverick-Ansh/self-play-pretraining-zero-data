"""The generator g_phi: a policy over programs (Sec. 2.2-2.3).

    "g0(x) = |A|^{-l(x)}. Here l(x) is the number of tokens up to and including
     the terminating F. This is the natural analog of the Solomonoff prior
     2^{-|p|}. The generator is initialized near g0 and regularized toward it
     throughout training."                                               (Sec. 2.2)

A program row is  S x_1 ... x_l F <pad>.  The policy is the ByteLlama's
next-byte distribution restricted to the 19-token alphabet A. Position L_max
(the cap) allows only F, so a capped program's final F has probability 1 under
both g_phi and g0; this is how "every program terminates" is made exact.

"Initialized near g0": the generator's lm_head starts at zero, so its first
policy is exactly uniform over A at every position, i.e. exactly g0.
"""
import numpy as np
import torch

from .machine import ALPHABET_MASK, END, PROG_PREFIX

NEG = -1e4          # fp16-safe "minus infinity" for masked logits


def position_mask(n_pos, max_body, device):
    """[n_pos, 256] bool: tokens allowed at predicted position t (t = 1 .. n_pos)."""
    m = torch.as_tensor(ALPHABET_MASK, device=device).expand(n_pos, 256).clone()
    if n_pos >= max_body + 1:                      # position max_body+1 is the forced F
        m[max_body:] = False
        m[max_body:, END] = True
    return m


@torch.no_grad()
def sample(gen, n, max_body, temperature=1.0):
    """Sample n programs with a KV cache. Returns (bodies, logp_old, charged_len).

    logp_old  : log g_phi(x) at sampling time (the PG importance-ratio denominator)
    charged   : l(x) minus a forced F, i.e. the tokens g0 actually pays for
    """
    dev = gen.wte.weight.device
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    cache = gen.new_cache(n, max_body + 2, dtype)
    mask = position_mask(max_body + 1, max_body, dev)
    tok = torch.full((n, 1), PROG_PREFIX, device=dev, dtype=torch.long)
    out = torch.zeros(n, max_body + 1, dtype=torch.long, device=dev)
    logp = torch.zeros(n, device=dev)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    with torch.autocast(dev.type, dtype=dtype, enabled=dev.type == "cuda"):
        for t in range(max_body + 1):
            logits = gen(tok, cache=cache, pos=t)[:, -1].float() / temperature
            lp = logits.masked_fill(~mask[t], NEG).log_softmax(-1)
            nxt = torch.multinomial(lp.exp(), 1)
            out[:, t] = torch.where(alive, nxt[:, 0], 0)
            logp += torch.where(alive, lp.gather(1, nxt)[:, 0], 0.0)
            alive &= nxt[:, 0] != END
            tok = nxt
            if not alive.any():
                break
    out = out.cpu().numpy().astype(np.uint8)
    bodies, charged = [], []
    for row in out:
        L = int(np.argmax(row == END))              # always found: position max_body forces F
        bodies.append(row[:L].tobytes())
        charged.append(L + 1 if L < max_body else L)
    return bodies, logp.cpu().numpy(), np.array(charged)


def pack(bodies, max_body):
    """bytes -> LongTensor rows  S body F 0...0  of width max_body + 2."""
    X = np.zeros((len(bodies), max_body + 2), dtype=np.int64)
    X[:, 0] = PROG_PREFIX
    for i, b in enumerate(bodies):
        X[i, 1:1 + len(b)] = np.frombuffer(b, dtype=np.uint8)
        X[i, 1 + len(b)] = END
    return X


def logprob(gen, bodies, max_body):
    """log g_phi(x) for arbitrary programs, differentiable. Returns [N] fp32."""
    dev = gen.wte.weight.device
    X = torch.as_tensor(pack(bodies, max_body), device=dev)
    lens = torch.tensor([len(b) + 1 for b in bodies], device=dev)       # tokens incl. F
    inp, tgt = X[:, :-1], X[:, 1:]
    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        logits = gen(inp)
    mask = position_mask(inp.shape[1], max_body, dev)
    lp = logits.float().masked_fill(~mask, NEG).log_softmax(-1)
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    valid = torch.arange(tgt.shape[1], device=dev)[None] < lens[:, None]
    return (tok_lp * valid).sum(1)
