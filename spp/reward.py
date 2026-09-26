"""The learning-progress reward: Eq. 2 of arXiv:2609.30063.

    "p(e) = floor(e/2),   delta_theta_e = theta_{p(e)} - theta_e,

         r_i = | < grad_theta L(y_i; theta_e),  P_e (.) delta_theta_e > |,     (2)

     where P_e = lr / (sqrt(v_hat_e) + eps) is the diagonal AdamW step operator
     obtained from the learner's optimizer state. [...] We avoid materializing
     full gradients by using forward mode automatic differentiation."   (Sec. 2.2)

Why forward mode: the direction v = P (.) delta_theta is the SAME for every
program in the pool, while the gradient differs per program. Reverse mode would
need one backward pass per program to get N per-example gradients. Forward mode
pushes the single tangent v through one forward pass and returns the whole
vector of directional derivatives  [<grad L_1, v>, ..., <grad L_N, v>]  at once:

    torch.func.jvp( theta -> [L_1(theta), ..., L_N(theta)],  (theta,), (v,) )

Snapshots. delta_theta needs theta at round floor(e/2). Storing every round is
O(e) memory, so snapshots are kept every `every` rounds and the newest one at or
before floor(e/2) is used: the window is e/2 rounds plus at most `every - 1`.
Snapshots older than that can never be needed again (floor(e/2) only grows)
and are dropped.
"""
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp


def seq_losses(model, params, X, n_out, manual=True):
    """Per-sequence mean CE over content tokens (Eq. 1 summand). X = [B, T+1] incl. O prefix.

    Only emitted bytes count (positions c < n_out): the zero padding after a
    program halts is not content. A program that emitted nothing has loss 0.
    """
    logits = functional_call(model, params, (X[:, :-1],), {"manual_attn": manual})
    ce = F.cross_entropy(logits.float().transpose(1, 2), X[:, 1:], reduction="none")
    valid = torch.arange(ce.shape[1], device=X.device)[None] < n_out[:, None]
    return (ce * valid).sum(1) / n_out.clamp(min=1)


def preconditioner(opt, model):
    """P = lr / (sqrt(v_hat) + eps) per parameter, from torch AdamW state. None before step 1."""
    P = {}
    for group in opt.param_groups:
        b2, eps, lr = group["betas"][1], group["eps"], group["lr"]
        for p in group["params"]:
            st = opt.state.get(p)
            if not st:
                return None
            step = float(st["step"])
            v_hat = st["exp_avg_sq"] / (1 - b2 ** step)
            P[id(p)] = lr / (v_hat.sqrt() + eps)
    return {n: P[id(p)] for n, p in model.named_parameters()}


class Snapshots:
    """theta_{p(e)} with p(e) = floor(e/2), stored every `every` rounds on the model's device."""

    def __init__(self, every=8):
        self.every, self.store = every, {}

    def maybe_save(self, e, model):
        if e % self.every == 0:
            self.store[e] = {n: p.detach().clone() for n, p in model.named_parameters()}

    def lookback(self, e):
        target = (e // 2) // self.every * self.every
        for k in [k for k in self.store if k < target]:
            del self.store[k]
        return target, self.store.get(target)


def rewards(model, opt, snaps, e, X, n_out, chunk=64, signed=False):
    """r_i for every sequence in the pool at the current learner theta_e. Returns [B] fp32.

    Zero at the first rounds, where delta_theta = 0 or the optimizer has no state yet.
    """
    P = preconditioner(opt, model)
    _, past = snaps.lookback(e)
    if P is None or past is None:
        return torch.zeros(X.shape[0], device=X.device)
    params = {n: p.detach() for n, p in model.named_parameters()}
    v = {n: P[n] * (past[n] - params[n]) for n in params}            # P (.) delta_theta
    out = []
    for lo in range(0, X.shape[0], chunk):
        f = lambda prm: seq_losses(model, prm, X[lo:lo + chunk], n_out[lo:lo + chunk])
        _, dL = jvp(f, (params,), (v,))
        out.append(dL)
    r = torch.cat(out).float()
    return r if signed else r.abs()
