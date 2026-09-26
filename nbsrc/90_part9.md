---
# Part 9 · The experiments, on 2 × T4

One detached queue per GPU, one run at a time per GPU (a second process on the same T4 only slows both). Every run is skippable (`final.json`), so a killed queue resumes instead of restarting. The notebook never blocks on training: it launches, then reads logs.

**Phase A: the ladder, self-play vs the uniform prior (C1, C2, C5).** 2048 rounds each = 268 M learner tokens.

| GPU 0 | GPU 1 |
|---|---|
| 3M self-play (~2 h) | 1M self-play (~80 min) |
| 100k self-play | 1M uniform |
| 3M uniform | 500k self-play, 500k uniform, 100k uniform |

**Phase B: noise and the reward ablation at 1M (C2 seed spread, C3).** Starts on each GPU the moment its Phase A queue finishes.

| GPU 0 | GPU 1 |
|---|---|
| 1M self-play, seed 1 | 1M shuffle, seed 0 |
| 1M shuffle, seed 1 | 1M uniform, seed 1 |

Measured cost per round (fp16 learner, fp32 JVP reward): 100k 0.44 s · 1M 2.2 s · 3M 3.6 s for self-play; the uniform arm skips sampling, reward and generator (~3× cheaper).

**Scale note.** A paper round is 1024 programs × 4095 bytes = 4.2 M tokens; ours is 256 × 511 = 131 k. Our whole 2048-round run equals roughly the paper's first 64 rounds in learner tokens (while the generator takes 32× more RL steps per learner token). Everything below is compared at **matched compute** $C$ = params × tokens, never at matched round number.
