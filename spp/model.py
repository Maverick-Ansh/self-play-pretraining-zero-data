"""Learner pi_theta and generator g_phi: the same byte-level Llama (Sec. 2.3).

    "The learner and generator are independently parameterized decoder-only
     Llama transformers with identical architecture. We use byte-level
     tokenization with a fixed vocabulary of 256 byte values [...] Programs and
     outputs are prefixed by the bytes S and O, respectively. During program
     generation, logits are restricted to the eight Brainf*ck instructions, ten
     canonical single-byte macro instructions, and the end-of-program token F,
     whereas output sequences may contain any byte value."               (Sec. 2.3)

Block anatomy, identical to the authors' released inference class
(nourya-aliz/Solomonoff-Figures, scoring/src/framework/model.py; that repo went
404 on 2026-09-26, after this was written against it), with the same
parameter names so their checkpoints load directly:

    bytes -> wte -> [ x + Attn(RMSNorm(x)) ; x + SwiGLU(RMSNorm(x)) ] x L -> RMSNorm -> lm_head
    Attn:   RoPE (theta 1e4), grouped-query (4 query heads per kv head), no biases
    SwiGLU: hidden = 8/3 d rounded up to a multiple of 256

Two additions the release does not need but training does:
  * `manual_attn=True` routes attention through explicit softmax(QK^T)V so that
    forward-mode autodiff (torch.func.jvp) works; fused SDPA kernels have no
    forward derivative. The paper used a custom JVP flash kernel for this.
  * a preallocated KV cache for sampling programs from the generator.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# the six rungs of the released ladder: name -> (d_model, n_heads, n_layers)
LADDER = {"100k": (64, 1, 1), "500k": (128, 2, 2), "1M": (128, 2, 4),
          "3M": (256, 4, 4), "6M": (256, 4, 8), "24M": (512, 8, 8)}
# total parameter counts from the release README; asserted in smoke.py
LADDER_PARAMS = {"100k": 98_496, "500k": 557_696, "1M": 1_049_728,
                 "3M": 3_148_032, "6M": 6_164_736, "24M": 24_388_096}


def swiglu_hidden(d, multiple=256):
    return multiple * ((int(8 * d / 3) + multiple - 1) // multiple)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(dt)


def rope(positions, head_dim, dtype):
    inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim))
    f = torch.outer(positions.float(), inv)
    e = torch.cat((f, f), -1)[None, None]
    return e.cos().to(dtype), e.sin().to(dtype)


def rotate(x, cos, sin):
    h = x.shape[-1] // 2
    return (x * cos + torch.cat((-x[..., h:], x[..., :h]), -1) * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.kv = n_heads, max(1, n_heads // 4)
        self.hd = d // n_heads
        self.q_proj = nn.Linear(d, n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.kv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.kv * self.hd, bias=False)
        self.o_proj = nn.Linear(n_heads * self.hd, d, bias=False)

    def forward(self, x, cos, sin, cache=None, pos=0, manual=False):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        if cache is not None:                       # sampling: append to the KV cache
            cache[0][:, :, pos:pos + T] = k
            cache[1][:, :, pos:pos + T] = v
            k, v = cache[0][:, :, :pos + T], cache[1][:, :, :pos + T]
        g = self.h // self.kv
        if g > 1:
            k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
        if manual:                                  # jvp-friendly explicit attention
            s = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
            Tk = k.shape[2]
            mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(Tk - T)
            y = s.masked_fill(~mask, float("-inf")).softmax(-1) @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None or T > 1)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        h = swiglu_hidden(d)
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.input_layernorm = RMSNorm(d)
        self.attn = Attention(d, n_heads)
        self.post_attention_layernorm = RMSNorm(d)
        self.mlp = MLP(d)

    def forward(self, x, cos, sin, cache=None, pos=0, manual=False):
        x = x + self.attn(self.input_layernorm(x), cos, sin, cache, pos, manual)
        return x + self.mlp(self.post_attention_layernorm(x))


class ByteLlama(nn.Module):
    """ProgramLanguageModel with training-side extras. vocab = 256 bytes."""

    def __init__(self, d_model, n_heads, n_layers, max_len=512, vocab_size=256):
        super().__init__()
        self.cfg = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        max_len=max_len, vocab_size=vocab_size)
        self.wte = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.head_dim = d_model // n_heads
        self.apply(self._init)
        for n, p in self.named_parameters():        # GPT-2 style depth scaling of residual writes
            if n.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    @classmethod
    def rung(cls, name, max_len=512):
        d, h, L = LADDER[name]
        return cls(d, h, L, max_len)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, cache=None, pos=0, manual_attn=False):
        T = idx.shape[1]
        x = self.wte(idx)
        cos, sin = rope(torch.arange(pos, pos + T, device=idx.device), self.head_dim, x.dtype)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, None if cache is None else cache[i], pos, manual_attn)
        return self.lm_head(self.ln_f(x))

    def new_cache(self, B, T, dtype):
        kv = self.blocks[0].attn.kv
        mk = lambda: torch.zeros(B, kv, T, self.head_dim, device=self.wte.weight.device, dtype=dtype)
        return [(mk(), mk()) for _ in self.blocks]

    def load_release(self, state_dict):
        """Load a released learner_*.pth state dict (ignores its vocab-mask buffer)."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        assert not missing, missing
        assert set(unexpected) <= {"program_vocab_mask"}, unexpected
        return self
