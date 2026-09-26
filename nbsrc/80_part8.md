---
# Part 8 · One round of self-play, then the loop

Everything above is a part. This is the machine they make together. One round $e$:

```
 1. POOL       192 fresh programs ~ g_phi    32 mutants of archive elites    32 replays from the bank
 2. EXECUTE    y_i = U(x_i, w_i)   (fresh random tape w_i every time)                        -> 256 x 511 bytes
 3a. REWARD    r_i = |< grad L(y_i; theta_e), P_e * (theta_{e/2} - theta_e) >|  (one JVP pass, BEFORE the learner moves)
 3b. LEARNER   one AdamW step on  (1/M) sum_i L(y_i)                                          (Eq. 1)
 3c. GENERATOR one AdamW step on  L_PG + L_EI                                                  (Eqs. 3-5)
     UPKEEP    decay archive rewards x0.97, admit r>0 programs to their niche, bank the non-replays
     SCAN      every non-replay tape through the structure detector                             (Table 1)
```

**The generator's objective.** GRPO-style: rewards are standardised over the whole pool, and the KL pull toward $g_0$ is folded into the advantage.

$$A_i = \frac{r_i - \bar r_e}{\sigma_{r,e} + \epsilon} \;-\; \beta\,\big(\log g_\phi(x_i) - \log g_0(x_i)\big)$$

$$\mathcal{L}_{PG} = -\frac{1}{|\mathcal{B}_e \setminus \mathcal{B}^{mut}_e|}\sum_{i \notin \text{mut}} \mathrm{sg}(\rho_i)\,\log g_\phi(x_i)\,\mathrm{sg}(A_i), \qquad \rho_i = \frac{g_\phi(x_i)}{g_{\text{old}}(x_i)} \in [e^{-20}, e^{20}]$$

$$\mathcal{L}_{EI} = -\sum_i w_i \log g_\phi(x_i),\quad w_i = \frac{[r_i]_+}{\sum_j [r_j]_+} \qquad\qquad \mathcal{L}_{gen} = \mathcal{L}_{PG} + 1.0\,\mathcal{L}_{EI}$$

* Mutants are **excluded from $\mathcal{L}_{PG}$**: nobody sampled them from a policy, so they have no $g_{\text{old}}$. They still enter $\mathcal{L}_{EI}$, which is how a good mutation gets *taught* to the generator.
* $\mathcal{L}_{EI}$ ("expert iteration") is reward-weighted imitation of the best programs of the round, the paper's guard against the generator forgetting what worked.

**The arms.** Identical machine, learner, pool size and token budget. Only the program source differs.

| arm | where programs come from | tests |
|---|---|---|
| `selfplay` | the generator, trained with the canonical reward | the method |
| `uniform` | $g_0$ i.i.d., no generator | **C2**: does adaptation matter? |
| `shuffle` | the generator, but rewards permuted across the pool every round | **C3**: does *which* program earns the reward matter? |

Hyperparameters (the paper tunes these per scale on natural-data validation loss and does not publish them; here they are fixed before any run and never tuned): learner AdamW lr $2\times10^{-3}$, $\beta = (0.9, 0.95)$, weight decay 0.1, 32-round warmup then constant; generator lr $= 0.25\times$ learner; $\beta_{KL} = 0.02$; program cap 128 tokens; 256 programs/round (192 fresh, 32 mutated, 32 replayed).
