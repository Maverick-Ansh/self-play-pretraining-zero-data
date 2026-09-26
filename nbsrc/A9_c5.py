# 10.6  C5: how often each run's pools contained each mathematical family, against the rate at which
#       the fixed prior g0 produces it BY CHANCE at the same tape length (10M programs, scripts/prior_discovery.py).
#       At 511-byte tapes every family occurs by chance, so the test is a RATE (enrichment), not a first sighting.
from scipy.stats import chi2
from spp.structure import FAMILIES
prior = json.load(open("results/prior_discovery.json"))
N = prior["samples"]
p0 = {f: prior["hit_counts"][f] / N for f in FAMILIES}
def poisson_ci(k, n):
    lo = chi2.ppf(0.025, 2 * k) / 2 / n if k else 0.0
    return lo, chi2.ppf(0.975, 2 * k + 2) / 2 / n
print(f"chance rate under g0 at T = 511 ({N:,} programs):  " + "  ".join(f"{f} {p0[f]:.1e}" for f in FAMILIES))
print(f"\n{'run':24s} {'programs':>9s}  " + "".join(f"{f:>20s}" for f in FAMILIES))
print(f"{'':24s} {'scanned':>9s}  " + "".join(f"{'hits  (x chance)':>20s}" for f in FAMILIES))
ENRICH = {}
for (rung, arm, seed), r in sorted(by.items(), key=lambda kv: (kv[0][1], RUNG_ORDER.index(kv[0][0]), kv[0][2])):
    fin = r["final"]
    if not fin:
        continue
    n = fin["config"]["rounds"] * (256 if arm == "uniform" else 224)   # non-replay programs scanned
    cells = []
    for f in FAMILIES:
        k = fin["hit_counts"][f]
        lo, hi = poisson_ci(k, n)
        ENRICH[(rung, arm, seed, f)] = (k, k / n / p0[f], lo / p0[f], hi / p0[f])
        cells.append(f"{k:5d} ({k / n / p0[f]:5.1f}x)")
    print(f"{rung + ' ' + ARM_NAME[arm] + ' s' + str(seed):24s} {n:9,d}  " + "".join(f"{c:>20s}" for c in cells))
print("\n(x chance) = observed rate / g0 rate; 95% Poisson intervals are in ENRICH. The uniform arm should sit near 1x.")
print("\nFirst program of each non-arithmetic family written by a self-play generator:")
shown = set()
for (rung, arm, seed), r in sorted(by.items()):
    for f, fs in (r["final"] or {}).get("first_seen", {}).items():
        if f != "arithmetic" and f not in shown and arm != "uniform":
            shown.add(f)
            print(f"  {f:10s} {rung} {ARM_NAME[arm]} round {fs['round']}: S{fs['program']}F  ->  {fs['terms']}")
