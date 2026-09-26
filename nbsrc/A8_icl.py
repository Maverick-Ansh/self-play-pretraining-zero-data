# 10.5  C4: in-context learning on held-out tasks, final 1M and 3M learners. Run the final evaluation first
#       (it scores every finished run once; ~1 min per run):   !python scripts/final_eval.py
from spp.evals import ICL_TASKS, ICL_MS
fig, axes = plt.subplots(1, 6, figsize=(14, 2.9), sharey=True)
for ax, t in zip(axes, ICL_TASKS):
    for arm in ("uniform", "shuffle", "selfplay"):
        for k, rung in ((2, "1M"), (3, "3M")):
            e = FINAL.get((rung, arm, 0))
            if not e:
                continue
            ms = [m for m in ICL_MS if e["icl"][t].get(str(m)) is not None]
            ax.plot([max(m, 0.5) for m in ms], [e["icl"][t][str(m)] for m in ms], marker="o", ms=3,
                    color=shade(ARM_COLOR[arm], k), label=f"{ARM_NAME[arm]} {rung}" if t == ICL_TASKS[0] else None)
    ax.set_xscale("log"); ax.set_title(t, loc="left", color=INK)
    ax.set_xlabel("demonstrations m")
axes[0].set_ylabel("exact-match accuracy")
fig.legend(loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.12))
fig.suptitle("In-context learning on tasks never seen in training (m = 0 drawn at 0.5)", x=0.01, ha="left", color=INK)
fig.tight_layout()
save(fig, "icl")
