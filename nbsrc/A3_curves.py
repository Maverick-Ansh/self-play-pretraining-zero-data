# 10.1  C1 + C2: zero-shot bits/byte against compute, every checkpoint of every size, both arms (seed 0).
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
    ax.axhline(KT[c]["order0"], color=MUTED, ls="--", lw=1)
    ax.annotate("KT floor (no training)", (0.02, KT[c]["order0"]), xycoords=("axes fraction", "data"),
                color=MUTED, fontsize=7, va="bottom")
    ax.set_xscale("log")
    ax.set_title(LABEL[c], color=INK, fontsize=10, loc="left")
for ax in axes[1]: ax.set_xlabel("compute  C = parameters × tokens")
for ax in axes[:, 0]: ax.set_ylabel("bits / byte (zero-shot)")
fig.legend(loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.07))
fig.suptitle("Held-out bits/byte during self-play vs uniform-prior training (darker = larger model)",
             x=0.01, ha="left", color=INK)
fig.tight_layout()
save(fig, "bpb_vs_compute")
