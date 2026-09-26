# 10.2  C1: the compute-optimal frontier (over sizes and checkpoints, seed 0) and its power-law fit.
PAPER_ALPHA = {"dclm": 0.123, "cifar10_rgb_hwc": 0.066, "audio_8bit": 0.260, "mutopia_melody_16th": 0.249, "dna": 0.435}
print(f"{'corpus':20s} {'arm':15s} {'points':>6s} {'E':>6s} {'alpha':>7s}  {'frontier bpb: first -> last':>28s}   paper alpha")
FITS = {}
for c in SHOW:
    for arm in ("selfplay", "uniform"):
        rs = [r for (rung, a, s), r in by.items() if a == arm and s == 0]
        if not rs:
            continue
        f = frontier(points(rs, c))
        p = fit(f) if len(f) >= 4 else None
        FITS[(c, arm)] = (f, p)
        E, A, a = p if p else (np.nan,) * 3
        print(f"{LABEL[c]:20s} {ARM_NAME[arm]:15s} {len(f):6d} {E:6.2f} {a:7.3f}  {f[0, 1]:13.3f} -> {f[-1, 1]:.3f}   "
              + (f"{PAPER_ALPHA[c]:.3f}" if arm == "selfplay" and c in PAPER_ALPHA else ""))
