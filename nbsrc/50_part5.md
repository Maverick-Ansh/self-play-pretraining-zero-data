---
# Part 5 · The program pool: fresh, mutated, replayed

> *"Fresh samples provide global exploration, mutations refine promising regions of program space, and replay preserves useful structures discovered earlier in training."* (§2.2)

Every round trains on a pool $\mathcal{B}_e = \mathcal{B}^{\text{fresh}}_e \,\dot\cup\, \mathcal{B}^{\text{mut}}_e \,\dot\cup\, \mathcal{B}^{\text{replay}}_e$ (here 192 + 32 + 32 = 256 programs; the paper does not give its split).

**Mutation (MAP-Elites, App. G).** An archive keeps good programs in **36 niches**: 9 bins of *maximum loop depth reached while running* (0 to 8+) × 4 bins of *length* (≤8, ≤16, ≤32, >32 tokens). Each niche keeps its top 8 programs by reward; stored rewards decay by 0.97 per round so stale elites get replaced. A mutation parent is drawn by picking an **occupied niche uniformly**, then a program in it, and applying one random substitution, insertion or deletion. Picking niches uniformly keeps rare behaviours (deeply nested loops) in play instead of mutating only the single best program.

**Replay.** Every non-replay program enters a bank with the log-probability it had when it entered. Replayed programs are re-run with a **fresh random tape**, so a program that reads `,` gives new data every time.

**Why store the log-probability.** A replayed program was sampled by an *older* generator. Its policy-gradient term is reweighted by the importance ratio $\rho_i = g_\phi(x_i)/g_{\text{old}}(x_i)$ (Part 8), and $g_{\text{old}}$ is exactly that stored number.
