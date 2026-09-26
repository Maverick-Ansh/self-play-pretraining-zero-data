# Report: Self-Play Pretraining with Zero Data, reproduced on 2 × T4

Paper: Cowsik, Dolev, Li, De Luca, Cohen, Goodman, Levine, *Self-Play Pretraining with Zero Data*,
arXiv:2609.30063 (24 Sep 2026). Authors' release: figure data + learner checkpoints
(`acowsik/self_play_pretraining`, `nourya-cohen/solomonoff-paper`); **no training code**.
Everything here (machine, models, reward, generator objective, pool, loop) was written from the paper.

## 1. Claims, stated so they can fail

| # | Claim | Where | Confirmed if |
|---|---|---|---|
| C1 | zero-shot bits/byte on held-out natural data falls as a power law in self-play compute, across modalities | Fig. 1, 2, Table 2 | compute-optimal frontier falls on every modality, $L=E+AC^{-\alpha}$ with $\alpha>0$ |
| C2 | self-play beats the non-adaptive uniform prior over the same program space | Fig. 2 | equal size + tokens: self-play < uniform beyond seed spread |
| C3 | which program earns which reward matters | Table 5 (shuffle) | shuffled rewards lose beyond seed spread |
| C4 | in-context learning on held-out tasks; the uniform-prior learner has none | Fig. 1, 4 | accuracy rises with demonstrations for self-play, not uniform |
| C5 | the generator finds Fibonacci / geometric / quadratic / cubic sequences far sooner than chance | Table 1 | self-play pool rate well above $g_0$'s chance rate at the same tape length |
| C6 | *(ours)* how much of the "transfer" does an untrained in-context counter already get? | — | reported, not a pass/fail |

## 2. How it was resized, and every deviation

| | Paper | Here | Why it is still a test |
|---|---|---|---|
| context | 4096 | 512 | same for every arm; eval windows cut from the paper's own 4096 bakes |
| tokens per run | up to 34.36 B | 268 M (2048 rounds × 256 × 511) | claims are about slopes and gaps inside this range; see §5 for what this range cannot reach |
| model sizes | 6 rungs, 99k to 24.4M | 4 rungs, 99k to 3.1M | identical architectures (param counts match the release exactly) |
| programs per round | 1024 to 2048 | 256 (192 fresh, 32 mutated, 32 replayed) | the paper does not give its fresh/mut/replay split |
| hyperparameters | tuned per scale on DCLM + DNA val loss | fixed a priori (lr 2e-3, gen/learner 0.25, β_KL 0.02, warmup 32, wd 0.1) | removes the paper's acknowledged leakage; **material**: untuned |
| seeds / ensembles | up to 10 seeds, ensembles K | 1 seed per rung, 2 seeds at 1M, K = 1 | seed spread reported next to every gap |
| reward snapshots | θ at ⌊e/2⌋ | θ at ⌊e/2⌋ rounded down to a multiple of 8 | window e/2 + ≤7 rounds, O(e/16) memory |
| program length cap L | not stated | 128 tokens (forced F at the cap, charged 0 under g0 and gφ) | g0 puts 0.08% of its mass above 128 |
| tape size | not stated ("finite") | 2048 cells | recovered from the released programs; see §3.1 |
| step budget | not stated | 2048 × (T+1) | recovered from the released programs; see §3.1 |
| JVP kernel | custom forward-mode flash attention | explicit attention in fp32 + `torch.func.jvp` | equal to per-example autograd to 9e-8 relative |
| PCFG baseline | yes | no | not needed for C1 to C5 |
| structure detector | `detect.py` | same file, trailing window 2048 → n//2 | see §3.3; unchanged at n = 4095 |

## 3. What broke, and what it would have cost

### 3.1 The machine's two missing numbers, recovered from the authors' own outputs
The paper gives neither the tape size nor the step budget. The release includes `hits.jsonl`: 20,045 programs
the authors' generator discovered, with the bytes each printed; 6,143 are deterministic.
* **Step budget.** A 32·T budget (my first guess) silences the paper's own Fibonacci example after 14 bytes:
  macro `C` loops x times, ~1000 steps per byte. The released programs need up to 8,319,126 steps for 4095 bytes,
  just below 2²³ = 2048 × 4096, so the budget is 2048 per context position.
* **Tape size.** 128 and 2048 cells reproduce all 6,143 recorded first bytes; 4096 misses 3 (programs that walk
  left and wrap). But checked to the *end* of the tape, 35 (128 cells) and 33 (2048 cells) of 5,955 arithmetic
  tapes leave their recurrence after the head wraps, while 4096 keeps all 5,955. No single circular size explains
  every record, so the authors' tape is not a plain circular array of one size. 2048 is used; it differs from
  theirs only for programs whose head wraps (0.5% of the released hits).

### 3.2 The measurement gate
Our model class + our bakes on the authors' released 1M checkpoints (round 16383, 3 seeds, context 4096)
reproduce their published per-seed-mean bits/byte **to three decimals on all seven corpora**, after one fix:
DCLM and arithmetic have to be baked with the scripts' default sample count (8192), not 256, or the windows
differ (a first pass was off by 0.057 and 0.023 bits). The first local check (1 seed, 24 windows) was 0.3 bits
lower than published on every corpus; that was seed spread, which is **0.08 to 0.5 bits in the authors' own 1M
models**.

### 3.3 The structure detector breaks on short tapes
The authors' degenerate-tape guard takes the period of the last 2048 bytes. At 511-byte tapes that window is the
whole tape, and `[5, 0, 0, …]` passes as "geometric, ratio 0": **1,256 junk geometric discoveries in 4,096
uniform programs**. The window is now half the tape (2048 exactly at n = 4095).

### 3.4 At 511-byte tapes every family occurs by chance
The paper reports no non-arithmetic family in 1.64 × 10⁸ uniform programs (4095-byte tapes). At 511-byte tapes,
10 M uniform programs contain all five: arithmetic 141, geometric 94, Fibonacci 24, cubic 5, quadratic 1. Shorter
tapes only have to keep a recurrence for 511 bytes, and a right-walking program never wraps a 2048-cell tape.
So C5 is tested as an enrichment over the measured chance rate, not as "a family appeared".

### 3.5 Engineering failures (cost only time)
* The companion repository with the authors' model class and scorer (`nourya-aliz/Solomonoff-Figures`) went 404
  mid-session; everything needed was already mirrored or re-derived.
* `scripts/queue.py` shadowed the stdlib `queue` module that torch imports (scripts/ is `sys.path[0]`).
* A `pgrep -f` wait loop matched its own shell's command line and would have waited forever.
* Files written with Python text mode on Windows were committed with CRLF endings.

## 4. Results

*(filled in from `runs/` when the sweep completes)*

## 5. Verdicts

*(filled in when the sweep completes)*

## 6. What was not tested

* Anything above 3.1M parameters or 268 M tokens per run. The paper's own released learners only drop below 8 bits
  on text after ~4 B tokens (1M model) and on images after ~4 B; our budget is 15 to 30 times smaller.
* Context 4096 for our own models (512 here); ensembles of seeds (K > 1); the PCFG baseline.
* The `signed`, `last_step`, `loss_delta`, `negate` reward arms of Table 5 (only `shuffle`).
* Hyperparameter tuning of any kind.
* Pre-pretraining (App. A.1) and the epiplexity analysis (Fig. 3).
