# Self-Play Pretraining with Zero Data, rebuilt from scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maverick-Ansh/self-play-pretraining-zero-data/blob/main/self_play_pretraining.ipynb)

A from-scratch reproduction of **Cowsik et al., *Self-Play Pretraining with Zero Data*
([arXiv:2609.30063](https://arxiv.org/abs/2609.30063))** on two Tesla T4s.

Two transformers start from random weights and never see natural data. A **generator** writes programs for a
tiny universal machine (Brainf\*ck plus ten macros); the machine runs them; a **learner** predicts the bytes they
print. The generator is rewarded, by a forward-mode directional derivative, for programs whose outputs push the
learner *along the direction it is already learning*. The question: does the learner then predict real text,
images, speech, music and DNA, and does that improve predictably with compute?

The authors released figure data and trained learner checkpoints, **not training code**. Everything here was
written from the paper and checked against the authors' own artefacts.

## What was checked before anything was trained

| check | result |
|---|---|
| machine semantics vs the **6,143 deterministic programs the authors' generator discovered** (`hits.jsonl`) | every recorded first byte reproduced; tape size and step budget recovered from them |
| model parameter counts vs all 6 released sizes | exact (98,496 … 24,388,096) |
| forward-mode JVP reward (Eq. 2) vs per-example autograd | 9 × 10⁻⁸ relative error |
| **our eval code on the authors' released 1M weights vs their published bits/byte** | **identical to 3 decimals on all 7 corpora** |
| generator at initialisation vs the prior $g_0$ | exactly equal |

## Findings

*(filled in from the sweep; see [REPORT.md](REPORT.md))*

## Layout

```
spp/                    the package, one module per part of the paper
  machine.py            the universal machine U (numba), g0, mutation, MAP-Elites niches
  model.py              ByteLlama: the learner/generator architecture (loads the authors' weights)
  generator.py          sampling programs and their log-probabilities under the 19-token mask
  reward.py             Eq. 2 by torch.func.jvp, AdamW preconditioner, theta snapshots
  pool.py               MAP-Elites archive + replay bank (App. G)
  structure.py          the authors' Table 1 detector (+ short-tape repair)
  evals.py              zero-shot bits/byte, KT in-context floor, the six ICL tasks
  selfplay.py           one round and the loop (Eqs. 1-5), CLI
  analysis.py           compute-optimal frontier + power-law fit
scripts/                prepare_eval, release_gate, bake_default, run_queue, final_eval, prior_discovery
nbsrc/ + build_notebook.py -> self_play_pretraining.ipynb   (the notebook is a build artefact)
smoke.py                asserts the paper's rules (CPU, ~1 min)
REPORT.md               claims, deviations, what broke, results, verdicts
results/                small JSON results committed to the repo
```

## Run it

```bash
pip install numba zstandard mido scipy
python smoke.py                       # the paper's rules, on CPU
python scripts/prepare_eval.py        # bake held-out corpora with the authors' scripts (~5 min)
python scripts/run_queue.py --gpu 0 --runs 1M:selfplay:0:2048 1M:uniform:0:2048
python scripts/final_eval.py          # bits/byte on all windows + ICL tasks for every finished run
```

Or open the notebook: it writes the package cell by cell, tests each part, and runs the experiments.
