# 10.7  Inside the loop: how the generator moves away from g0, and what it writes.
panels = [("kl", "KL(g_phi || g0)  [nats]"), ("body_len", "mean program length"),
          ("full_frac", "fraction of full 511-byte tapes"), ("r_mean", "mean reward |r|")]
fig, axes = plt.subplots(1, 4, figsize=(14, 2.9))
for ax, (key, title) in zip(axes, panels):
    for (rung, arm, seed), r in sorted(by.items()):
        if arm == "uniform" or rung != "1M" and not (rung == "3M" and seed == 0):
            continue
        tr = r["train"]
        ax.plot([t["round"] for t in tr], [t[key] for t in tr], lw=1.1,
                color=shade(ARM_COLOR[arm], 3 if rung == "3M" else 2 - seed),
                label=f"{ARM_NAME[arm]} {rung} s{seed}" if key == "kl" else None)
    ax.set_title(title, loc="left", color=INK); ax.set_xlabel("round")
fig.legend(loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
save(fig, "generator_dynamics")
r = by.get(("1M", "selfplay", 0))
if r:
    for e in r["evals"]:
        if e["round"] in (0, 256, 1024, 2048):
            print(f"round {e['round']:4d}:", "  ".join(p for p in e["programs"][:6]))
