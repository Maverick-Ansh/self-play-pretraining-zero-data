---
# Part 2 · The byte transformer

> *"The learner and generator are independently parameterized decoder-only Llama transformers with identical architecture. We use byte-level tokenization with a fixed vocabulary of 256 byte values."* (§2.3)

One class, used twice. The **learner** reads rows `O y_1 y_2 … y_T` (the machine's output) and predicts every next byte. The **generator** reads rows `S x_1 … x_l F` (a program) and is only allowed to put probability on the 19 program tokens.

```
bytes ─► wte (256 × d) ─► ┌─────────────────────────────────────────┐ × L ─► RMSNorm ─► lm_head (d × 256)
                          │ x ← x + Attn( RMSNorm(x) )               │
                          │      RoPE θ=10⁴ · causal · GQA (4 q/kv)  │
                          │ x ← x + SwiGLU( RMSNorm(x) )             │
                          │      hidden = ⌈8d/3⌉ rounded up to 256   │
                          └─────────────────────────────────────────┘
```

The exact shapes come from the authors' release (`config.json` of every checkpoint on `nourya-cohen/solomonoff-paper`), and the parameter names match their inference class, so **their trained weights load into this module unchanged**. Part 7 uses that to test the measurement before trusting it.

| rung | $d$ | heads | layers | released parameter count |
|---|---|---|---|---|
| 100k | 64 | 1 | 1 | 98,496 |
| 500k | 128 | 2 | 2 | 557,696 |
| 1M | 128 | 2 | 4 | 1,049,728 |
| 3M | 256 | 4 | 4 | 3,148,032 |
| 6M | 256 | 4 | 8 | 6,164,736 |
| 24M | 512 | 8 | 8 | 24,388,096 |

Two additions over the release, both needed only for training:
* `manual_attn=True`: attention written out as $\mathrm{softmax}(QK^\top/\sqrt{d_h})V$. Fused attention kernels have no *forward-mode* derivative, and the reward (Part 4) needs one. The authors wrote a custom JVP flash-attention kernel; explicit attention is the simple equivalent at context 512.
* a key/value cache, so the generator can sample programs one token at a time.
