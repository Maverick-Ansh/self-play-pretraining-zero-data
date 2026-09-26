"""The program pool B_e = B_fresh + B_mut + B_replay: Sec. 2.2 and App. G.

    "Fresh samples provide global exploration, mutations refine promising
     regions of program space, and replay preserves useful structures
     discovered earlier in training."                                   (Sec. 2.2)

    "For mutation, we use a MAP-Elites-style algorithm, so that mutation does
     not concentrate exclusively on the highest-reward programs. [...] each
     program is assigned to a niche according to two descriptors: (i) its
     maximum dynamic loop depth, binned from 0 through 8 with larger depths
     clamped to the final bin, and (ii) its program-body length, bucketed at 8,
     16, and 32 tokens. This produces at most 36 niches in total. Only programs
     receiving positive generator reward are admitted to the archive, and
     within each niche we retain the top 8 programs according to their stored
     reward. [...] stored rewards are decayed by a factor of 0.97 each round.
     Mutation parents are selected uniformly across occupied niches."   (App. G)

    "The replay pool is formed by drawing programs uniformly without
     replacement from the bank of non-replay programs produced in earlier
     rounds. These programs are re-executed with fresh random tapes."   (App. G)

The replay bank stores each program's log-probability under the generator of
the round it entered: the importance-ratio denominator of Eq. 4.
"""
import numpy as np

from .machine import mutate, niche


class Archive:
    """MAP-Elites archive: niche -> {body: stored_reward}, top-k per niche."""

    def __init__(self, per_niche=8, decay=0.97):
        self.cells, self.k, self.decay = {}, per_niche, decay

    def __len__(self):
        return sum(len(c) for c in self.cells.values())

    def step_decay(self):
        for c in self.cells.values():
            for b in c:
                c[b] *= self.decay

    def add(self, bodies, rewards, depths):
        for b, r, d in zip(bodies, rewards, depths):
            if r <= 0:
                continue
            cell = self.cells.setdefault(niche(d, len(b)), {})
            if r > cell.get(b, -1.0):
                cell[b] = float(r)
            if len(cell) > self.k:
                del cell[min(cell, key=cell.get)]

    def mutants(self, n, rng, max_body):
        occupied = [c for c in self.cells.values() if c]
        if not occupied:
            return []
        out = []
        for _ in range(n):
            cell = occupied[rng.integers(len(occupied))]
            parents = list(cell)
            out.append(mutate(parents[rng.integers(len(parents))], rng, max_body))
        return out


class ReplayBank:
    """All non-replay programs of earlier rounds with their sampling log-prob (random eviction at cap)."""

    def __init__(self, cap=200_000):
        self.bodies, self.logp, self.cap = [], [], cap

    def __len__(self):
        return len(self.bodies)

    def add(self, bodies, logp, rng):
        for b, lp in zip(bodies, logp):
            if len(self.bodies) < self.cap:
                self.bodies.append(b)
                self.logp.append(float(lp))
            else:
                j = int(rng.integers(self.cap))
                self.bodies[j], self.logp[j] = b, float(lp)

    def draw(self, n, rng):
        if len(self.bodies) < n:
            return [], np.zeros(0)
        idx = rng.choice(len(self.bodies), n, replace=False)
        return [self.bodies[i] for i in idx], np.array([self.logp[i] for i in idx])
