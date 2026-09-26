# Self-Play Pretraining with Zero Data, rebuilt from scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maverick-Ansh/self-play-pretraining-zero-data/blob/main/self_play_pretraining.ipynb)

**Paper:** Cowsik, Dolev, Li, De Luca, Cohen, Goodman, Levine. *Self-Play Pretraining with Zero Data.* arXiv:2609.30063 (Sep 2026).
**Hardware:** 2 × Tesla T4 (Kaggle backend, driven from Colab).  **Repo:** [Maverick-Ansh/self-play-pretraining-zero-data](https://github.com/Maverick-Ansh/self-play-pretraining-zero-data)

Two transformers start from random weights. Neither ever sees a single byte of natural data.

* The **generator** $g_\phi$ writes short programs for a tiny universal computer (Brainf\*ck plus ten macros).
* The machine runs each program and prints bytes.
* The **learner** $\pi_\theta$ learns to predict those bytes, one at a time.
* The generator is paid for programs whose outputs push the learner *further along the direction it is already learning*.

The paper's question: after this closed loop, can the learner predict **real** text, images, speech, music and DNA, and does that improve predictably with compute?

```
         +------------------ RL reward  r_i = |< grad L(y_i), P * (theta_past - theta_now) >| -----------------+
         |                                                                                                       |
 generator g_phi --- programs x_i --->  universal machine U --- bytes y_i = U(x_i, w_i) --->  learner pi_theta ---+
 (GRPO + KL to g0 + expert iteration)    (BF + 10 macros)                                     (next-byte CE)
```

## How this notebook is built

Bottom up. Each part writes one module of the `spp/` package with `%%writefile`, then tests it against a rule from the paper before anything is stacked on top of it.

| Part | Builds | Checked against |
|---|---|---|
| 1 | the universal machine $U$ | App. E example · Table 1 outputs · **6,143 programs the authors' generator discovered** |
| 2 | the byte transformer (learner and generator share it) | parameter counts of all 6 released models |
| 3 | the generator and the prior $g_0$ | at initialisation $g_\phi = g_0$ exactly |
| 4 | the reward (Eq. 2) by forward-mode autodiff | per-example autograd |
| 5 | the program pool (MAP-Elites mutation + replay) | App. G |
| 6 | the structure detector (Table 1) | the authors' detector, verbatim |
| 7 | evaluation: zero-shot bits/byte, a zero-training floor, ICL tasks | **the authors' released weights reproduce their published bits/byte to 3 decimals** |
| 8 | one round of self-play, then the loop | a live mini-run |
| 9 | the experiments on 2 GPUs | |
| 10 | results and verdicts | |
