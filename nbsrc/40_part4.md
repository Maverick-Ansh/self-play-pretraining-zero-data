---
# Part 4 · The reward: learning progress, measured by forward-mode autodiff

> *"we reward the generator for producing programs whose learner gradients align with the learner's current learning trajectory. Let $p(e)=\lfloor e/2\rfloor$, $\delta\theta_e=\theta_{p(e)}-\theta_e$ [...]*
> $$r_i = \big|\langle \nabla_\theta \mathcal{L}(y_i;\theta_e),\; P_e \odot \delta\theta_e \rangle\big| \qquad (2)$$
> *where $P_e = \mathrm{lr}/(\sqrt{\hat v_e}+\epsilon)$ is the diagonal AdamW step operator."* (§2.2)

Read it in three pieces:

1. $\delta\theta_e = \theta_{\lfloor e/2\rfloor} - \theta_e$ is **where the learner has been going** over the last half of training (pointing backwards; the absolute value makes the sign irrelevant).
2. $P_e \odot$ rescales each parameter the way AdamW would: parameters with a small gradient history get a larger step. So the inner product is measured in *the geometry the optimizer actually uses*.
3. $\langle \nabla \mathcal{L}(y_i), \cdot \rangle$ asks: if the learner trained on program $i$'s output, how much would that push it **along the direction it is already moving?**

A program the learner has already mastered has $\nabla\mathcal{L}\approx 0$: no reward. A program full of unlearnable noise has a large gradient in a random direction, nearly orthogonal to $\delta\theta$: little reward. The reward concentrates on **learnable-but-not-yet-learned** structure. (The paper's first attempt rewarded pure difficulty, and the generator learned to inject random bytes.)

### Why forward mode

The direction $v = P_e \odot \delta\theta_e$ is **the same for every program**; only the gradient differs. With reverse mode you would need one backward pass per program to get $N$ separate gradients. Forward mode pushes the single tangent $v$ through the network and gets every program's directional derivative in **one forward pass**:

$$\texttt{jvp}\big(\theta \mapsto [\mathcal{L}(y_1;\theta),\dots,\mathcal{L}(y_N;\theta)],\ \theta,\ v\big) = \big[\langle\nabla\mathcal{L}_1, v\rangle, \dots, \langle\nabla\mathcal{L}_N, v\rangle\big]$$

**Snapshots.** $\delta\theta_e$ needs $\theta$ from round $\lfloor e/2\rfloor$. Keeping every round is $O(e)$ memory, so $\theta$ is saved every 8 rounds and the newest snapshot at or before $\lfloor e/2\rfloor$ is used (the window is $e/2$ plus at most 7 rounds). Older snapshots can never be needed again and are dropped.
