# 10.3  C2: self-play against the uniform prior at the SAME size and token budget (final checkpoint,
#       all 256 held-out windows). At 1M both arms have two seeds, which sets the noise level.
def final_bpb(rung, arm, seed):
    e = FINAL.get((rung, arm, seed))
    return e["bpb"] if e else None
print(f"{'':20s}" + "".join(f"{rung:>22s}" for rung in RUNG_ORDER))
print(f"{'corpus':20s}" + "".join(f"{'self-play / uniform':>22s}" for _ in RUNG_ORDER))
for c in SHOW + ["random"]:
    row = f"{LABEL[c]:20s}"
    for rung in RUNG_ORDER:
        sp, un = final_bpb(rung, "selfplay", 0), final_bpb(rung, "uniform", 0)
        row += f"{sp[c]:9.3f} / {un[c]:5.3f} {'<' if sp[c] < un[c] else '>'}  " if sp and un else f"{'-':>22s}"
    print(row)
print("\nAt 1M, two seeds per arm: gap = mean(self-play) - mean(uniform); spread = largest seed-to-seed difference within an arm")
for c in SHOW:
    sp = [final_bpb("1M", "selfplay", s) for s in (0, 1)]
    un = [final_bpb("1M", "uniform", s) for s in (0, 1)]
    if None in sp + un:
        continue
    sp, un = np.array([x[c] for x in sp]), np.array([x[c] for x in un])
    gap, spread = sp.mean() - un.mean(), max(np.ptp(sp), np.ptp(un))
    verdict = "self-play better" if gap < -spread else "uniform better" if gap > spread else "within seed noise"
    print(f"  {LABEL[c]:20s} gap {gap:+.3f}   seed spread {spread:.3f}   -> {verdict}")
