# 7.3  The zero-training floor: in-context KT counters on every corpus at OUR context (a minute of CPU)
from spp.evals import kt_bpb
if os.path.exists("data/kt_baselines.json"):
    KT = json.load(open("data/kt_baselines.json"))
else:
    KT = {c: {f"order{k}": kt_bpb(a, k) for k in (0, 1, 2)} for c, a in corp.items()}
    json.dump(KT, open("data/kt_baselines.json", "w"), indent=1)
print(f"{'corpus':18s} {'KT order-0':>10s} {'order-1':>8s} {'order-2':>8s}   (bits/byte, 511-byte windows, no training)")
for c in corp:
    print(f"{LABEL[c]:18s} {KT[c]['order0']:10.3f} {KT[c]['order1']:8.3f} {KT[c]['order2']:8.3f}")
