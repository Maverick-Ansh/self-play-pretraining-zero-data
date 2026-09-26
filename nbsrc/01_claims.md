## What the paper claims, stated so it can fail

| # | Claim | Where | Confirmed here if |
|---|---|---|---|
| **C1** | Zero-shot bits/byte on held-out natural data falls as a power law in self-play compute, across modalities | Fig. 1, Fig. 2, Table 2 | the compute-optimal frontier over sizes × checkpoints falls on every modality and $L(C)=E+AC^{-\alpha}$ fits with $\alpha>0$ |
| **C2** | Self-play beats the *non-adaptive* universal prior over the **same** program space | Fig. 2, §3.1 | at equal size and tokens, self-play < uniform-prior bits/byte beyond seed spread |
| **C3** | What the reward says about each program matters, not just its distribution | Table 5 (`shuffle`) | permuting rewards across the pool loses to the canonical reward beyond seed spread |
| **C4** | The learner learns in context on tasks it never saw; the uniform-prior learner does not | Fig. 1, Fig. 4 | exact-match accuracy rises with the number of demonstrations $m$ for self-play, not for uniform |
| **C5** | The generator finds Fibonacci / geometric / quadratic / cubic sequences far sooner than uniform sampling | Table 1 | non-arithmetic families appear in self-play pools; none in the same number of uniform programs |

**The primary comparison is C2.** Self-play and the uniform prior differ in exactly one thing (whether the program distribution adapts to the learner); machine, alphabet, learner, pool size and token budget are identical.

**One question the paper does not ask (C6).** How much of the "transfer" does a predictor with *no training at all* already get, just by counting bytes inside the window? A Krichevsky-Trofimov counter is that floor. The self-play gain only means something relative to it.

## How it was resized for 2 × T4

| | Paper | Here | Why the claim is still testable |
|---|---|---|---|
| context | 4096 | **512** | same for every arm; eval windows are cut from the paper's own 4096 bakes |
| tokens per run | up to 34.36 B | 268 M | the claims are about *slopes and gaps*, measured within this range |
| model sizes | 99k to 24.4M (6 rungs) | 99k to 3.1M (4 rungs) | identical architectures to the released rungs |
| programs per round | 1024 to 2048 | 256 | one learner step + one generator step per round, as in the paper |
| hyperparameters | tuned per scale on DCLM + DNA validation loss | fixed a priori, never tuned on natural data | removes the paper's (acknowledged) leakage |
| seeds / ensembles | up to 10 seeds, ensembles $K$ | 1 to 2 seeds, $K=1$ | seed spread is reported next to every gap |
| PCFG baseline | yes | no | not needed for C1 to C5 |
