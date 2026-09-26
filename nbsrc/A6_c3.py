# 10.4  C3: the reward ablation at 1M. `shuffle` keeps the generator, the pool, the reward's
#       distribution, everything, but permutes which program earned which reward every round.
arms = {"selfplay": [final_bpb("1M", "selfplay", s) for s in (0, 1)],
        "shuffle": [final_bpb("1M", "shuffle", s) for s in (0, 1)]}
print(f"{'corpus':20s} {'canonical (2 seeds)':>22s} {'shuffled (2 seeds)':>22s} {'gap':>7s} {'spread':>7s}")
for c in SHOW:
    if any(x is None for v in arms.values() for x in v):
        print("waiting for the Phase B runs"); break
    a = np.array([x[c] for x in arms["selfplay"]]); b = np.array([x[c] for x in arms["shuffle"]])
    gap, spread = b.mean() - a.mean(), max(np.ptp(a), np.ptp(b))
    print(f"{LABEL[c]:20s} {a[0]:10.3f} {a[1]:10.3f}  {b[0]:10.3f} {b[1]:10.3f}  {gap:+7.3f} {spread:7.3f}   "
          + ("canonical better" if gap > spread else "shuffled better" if gap < -spread else "within seed noise"))
print("\npaper, Table 5 (1M, 4-seed ensembles): shuffle is worse than canonical on 9 of 10 datasets, by 0.2 to 1.2 bits")
