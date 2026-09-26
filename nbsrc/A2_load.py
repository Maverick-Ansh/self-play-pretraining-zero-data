# 10.0  Load every run; one figure style for the whole section.
import matplotlib, matplotlib.pyplot as plt
from spp.analysis import load_runs, points, frontier, fit, power_law
ARM_COLOR = {"selfplay": "#2a78d6", "uniform": "#eb6834", "shuffle": "#1baf7a"}
ARM_NAME = {"selfplay": "self-play", "uniform": "uniform prior", "shuffle": "shuffled reward"}
RUNG_ORDER = ["100k", "500k", "1M", "3M"]
SHOW = ["dclm", "cifar10_rgb_hwc", "audio_8bit", "mutopia_melody_16th", "dna", "arithmetic"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"figure.dpi": 100, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "lines.linewidth": 1.6,
                     "legend.frameon": False})
os.makedirs("figures", exist_ok=True)
def shade(hexcol, k, n=4):               # lightness step k of n within one hue (k = n-1 is the full colour)
    c = np.array(matplotlib.colors.to_rgb(hexcol)); w = 0.6 * (1 - (k + 1) / n)
    return tuple(c * (1 - w) + w)
def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight"); plt.show()
runs = load_runs("runs")
by = {(r["config"]["rung"], r["config"]["arm"], r["config"]["seed"]): r for r in runs}
FINAL = {k: json.load(open(f"{r['path']}/final_eval.json")) for k, r in by.items()
         if os.path.exists(f"{r['path']}/final_eval.json")}
print(f"{len(runs)} runs ({sum(r['final'] is not None for r in runs)} finished, {len(FINAL)} with a final evaluation)")
for (rung, arm, seed), r in sorted(by.items(), key=lambda kv: (RUNG_ORDER.index(kv[0][0]), kv[0][1], kv[0][2])):
    last = r["evals"][-1]
    print(f"  {rung:5s} {ARM_NAME[arm]:16s} seed {seed}  round {last['round']:5d}  {last['tokens']/1e6:5.0f}M tokens  "
          + ("done" if r["final"] else "running"))
