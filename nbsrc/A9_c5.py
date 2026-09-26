# 10.6  C5: which mathematical families appeared in each run's pools, and when (every round was scanned).
from spp.structure import FAMILIES
print(f"{'run':24s}" + "".join(f"{f:>18s}" for f in FAMILIES) + "     (first round, total tapes)")
for (rung, arm, seed), r in sorted(by.items(), key=lambda kv: (kv[0][1], RUNG_ORDER.index(kv[0][0]), kv[0][2])):
    fin = r["final"]
    if not fin:
        continue
    cells = []
    for f in FAMILIES:
        fs = fin["first_seen"].get(f)
        cells.append(f"r{fs['round']} ({fin['hit_counts'][f]})" if fs else "-")
    print(f"{rung + ' ' + ARM_NAME[arm] + ' s' + str(seed):24s}" + "".join(f"{x:>18s}" for x in cells))
print("\nExample discoveries (first program of each non-arithmetic family, any run):")
shown = set()
for (rung, arm, seed), r in by.items():
    for f, fs in (r["final"] or {}).get("first_seen", {}).items():
        if f != "arithmetic" and f not in shown:
            shown.add(f)
            print(f"  {f:10s} {rung} {ARM_NAME[arm]} round {fs['round']}: S{fs['program']}F  ->  {fs['terms']}")
