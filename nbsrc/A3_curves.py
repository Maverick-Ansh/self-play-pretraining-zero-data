# 10.1  C1 + C2: zero-shot bits/byte against compute, every checkpoint of every size, both arms (seed 0).
#       Grey crosses: the AUTHORS' released self-play learners of the same sizes (their CSV, context 4096,
#       x = total params x tokens), for orientation only: a longer context makes their numbers lower.
PAPER_RUNG = {"100k": ("d64h1L1", 1024), "1M": ("d128h2L4", 1024), "3M": ("d256h4L4", 2048)}   # programs/round
from spp.model import LADDER_PARAMS
paper = list(csv.DictReader(open("ext/self_play_pretraining/figures/figure2/data/selfplay_frontier_perk.csv")))
fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex=True)
for ax, c in zip(axes.flat, SHOW):
    for arm in ("uniform", "selfplay"):
        for k, rung in enumerate(RUNG_ORDER):
            r = by.get((rung, arm, 0))
            if not r:
                continue
            ev = [e for e in r["evals"] if e["compute"] > 0]
            ax.plot([e["compute"] for e in ev], [e["bpb"][c] for e in ev], color=shade(ARM_COLOR[arm], k),
                    lw=1.3, label=f"{ARM_NAME[arm]} {rung}" if c == SHOW[0] else None)
    for rung, (tag, ppr) in PAPER_RUNG.items():
        pts = sorted((int(p["round"]), float(p["bpb"])) for p in paper
                     if p["rung"] == tag and p["corpus"] == c and p["K"] == "1" and 0 < int(p["round"]) <= 2048)
        ax.scatter([LADDER_PARAMS[rung] * rd * ppr * 4095 for rd, _ in pts], [b for _, b in pts], marker="x",
                   s=14, lw=0.9, color=MUTED, zorder=3, label="paper's released learners (ctx 4096)"
                   if c == SHOW[0] and rung == "100k" else None)
    ax.axhline(KT[c]["order0"], color=MUTED, ls="--", lw=1)
    ax.annotate("KT floor (no training)", (0.02, KT[c]["order0"]), xycoords=("axes fraction", "data"),
                color=MUTED, fontsize=7, va="bottom")
    ax.set_xscale("log")
    ax.set_title(LABEL[c], color=INK, fontsize=10, loc="left")
for ax in axes[1]: ax.set_xlabel("compute  C = parameters × tokens")
for ax in axes[:, 0]: ax.set_ylabel("bits / byte (zero-shot)")
fig.legend(loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Held-out bits/byte during self-play vs uniform-prior training (darker = larger model)",
             x=0.01, ha="left", color=INK)
fig.tight_layout()
save(fig, "bpb_vs_compute")
