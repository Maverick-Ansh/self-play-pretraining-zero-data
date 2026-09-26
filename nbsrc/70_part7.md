---
# Part 7 · The measurement, and a gate on it

**Bits per byte.** Every held-out corpus is cut into windows; the learner reads `O` + window and pays $-\log_2 p(\text{next byte})$ for every byte. Uniform guessing costs exactly **8.0 bits/byte**; anything above 8 is *worse than knowing nothing* (a young model often is, because it is confidently wrong).

| corpus | modality | source (baked by the authors' own scripts) |
|---|---|---|
| `dclm` | web text | DCLM-Baseline, 10 global shards |
| `cifar10_rgb_hwc` | images | CIFAR-10 test batch, pixels interleaved R,G,B |
| `audio_8bit` | speech | LibriSpeech PCM, 8-bit |
| `mutopia_melody_16th` | melody | 40 public-domain scores, one byte per 16th note |
| `dna` | DNA | GRCh38, 8 symbols (soft-masked bases) |
| `arithmetic` | synthetic | $(a + b\,j) \bmod 256$ |
| `random` | control | uniform random bytes: **no model can go below 8.0** |

None of these bytes is ever used for a gradient step or to pick a hyperparameter. (The paper selected hyperparameters on DCLM + DNA validation loss; this reproduction fixes them a priori.)

### 7.2 The gate: our code on the authors' weights must give the authors' numbers

Before trusting a single number we produce, we score **the authors' released 1M-parameter checkpoints** (round 16383, every seed that reached it) with *our* model class and *our* baked corpora at their context of 4096, and compare with the bits/byte they published (`figure2/data/selfplay_frontier_perk.csv`).

### 7.3 A floor the paper does not draw: learn nothing, count in context

A **Krichevsky-Trofimov (KT) counter** predicts the next byte from counts of what has appeared so far *in the same window*, $p(b) = (c_b + \tfrac12)/(n + 128)$, optionally conditioned on the previous 1 or 2 bytes. It has **no training at all**. On DNA it learns within a few dozen bytes that only 8 symbols occur. A learner that merely matches it has learned "count what you see" (real in-context learning, but not transfer of structure from the programs).

### 7.4 In-context learning tasks (App. D)

Formats copied from the authors' harness, the cells the paper plots: `reverse` (k = 8, teacher-forced over the 8 output slots), `stack` (L = 4 push/pop ops, 250 = PUSH, 251 = POP), `assoc` (a 16-entry dictionary), `sum`, `max`, `min` (k = 2). Each example is `0, inputs, answer`; the model sees $m$ demonstrations and must produce the answer to the next query. Score: greedy exact match.
