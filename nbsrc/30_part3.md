---
# Part 3 · The generator $g_\phi$ and the prior $g_0$

> *"$g_0(x) = |\mathcal{A}|^{-\ell(x)}$. Here $\ell(x)$ is the number of tokens up to and including the terminating F. This is the natural analog of the Solomonoff prior $2^{-|p|}$. The generator is initialized near $g_0$ and regularized toward it throughout training."* (§2.2)

$g_0$ is "type random keys until you hit F": every program of length $\ell$ has probability $19^{-\ell}$, so short programs are exponentially more likely. It is the **non-adaptive control** of claim C2.

The generator is the same byte transformer with its output restricted to the 19 program tokens. Its probability of a whole program is the product of its next-token probabilities:

$$\log g_\phi(x) = \sum_{t=1}^{\ell} \log \frac{\exp z_t[x_t]}{\sum_{a \in \mathcal{A}} \exp z_t[a]}$$

Three details that make this exact rather than approximate:

* **"Initialized near $g_0$"**: the generator's output layer starts at **zero**, so every allowed token gets the same logit and $g_\phi = g_0$ exactly at step 0.
* **Length cap.** Programs are capped at $L = 128$ tokens (the paper does not state $L$). At position 128 the only allowed token is F, so a capped program's F has probability 1 under both $g_\phi$ and $g_0$ and neither is charged for it.
* **Same function for sampling and scoring.** The log-probability recorded while sampling (needed later for importance ratios) must equal the log-probability recomputed by a training forward pass. The test below checks it.
