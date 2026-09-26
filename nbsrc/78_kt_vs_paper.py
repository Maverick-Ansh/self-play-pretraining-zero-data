# 7.3b  The same floor at the PAPER's context (4095-byte windows) against the paper's OWN published numbers.
#       No training anywhere in the KT column; every other column is the authors' released models (their CSV).
import csv
def kt0_fast(arr):                       # order-0 KT, vectorised over positions (identical to kt_bpb(.., 0))
    bits = 0.0
    for row in arr.astype(np.int64):
        onehot = np.zeros((len(row), 256)); onehot[np.arange(len(row)), row] = 1
        seen = np.cumsum(onehot, 0) - onehot                      # counts strictly before t
        bits -= np.log2((seen[np.arange(len(row)), row] + 0.5) / (np.arange(len(row)) + 128)).sum()
    return bits / arr.size
C4096 = {"dclm": "data/c4096_default/dclm.npy", "cifar10_rgb_hwc": "data/c4096/cifar10_rgb_hwc.npy",
         "audio_8bit": "data/c4096/audio_8bit.npy", "mutopia_melody_16th": "data/c4096/mutopia_melody_16th.npy",
         "dna": "data/c4096/dna.npy", "arithmetic": "data/c4096_default/arithmetic.npy"}
KT4096 = {c: kt0_fast(np.load(p)[:256]) for c, p in C4096.items()}
D = "ext/self_play_pretraining/figures/figure2/data"
sp_rows, un_rows = (list(csv.DictReader(open(f"{D}/{f}_frontier_perk.csv"))) for f in ("selfplay", "uniform"))
def best(rows, corpus, rung=None, single=True):
    v = [float(r["bpb"]) for r in rows if r["corpus"] == corpus and (rung is None or r["rung"] == rung)
         and (r["K"] == "1" or not single)]
    return min(v) if v else float("nan")
star = lambda v, c: f"{v:7.3f}{'*' if v > KT4096[c] else ' '}"
print(f"{'':20s} {'no training':>12s} | {'self-play, one seed':^41s} | {'self-play':>9s} | {'uniform prior':>13s}")
print(f"{'corpus':20s} {'KT order-0':>12s} | {'99k':>8s} {'1M':>9s} {'6M':>9s} {'24M':>9s} | {'best, any K':>11s} | {'best, any K':>13s}")
for c in C4096:
    one = [best(sp_rows, c, r) for r in ("d64h1L1", "d128h2L4", "d256h4L8", "d512h8L8")]
    print(f"{LABEL[c]:20s} {KT4096[c]:12.3f} | " + " ".join(star(v, c) for v in one)
          + f" | {star(best(sp_rows, c, single=False), c)}  | {star(best(un_rows, c, single=False), c)}")
print("\n* = worse than an untrained in-context byte counter on the same 4095-byte windows.")
print("'best, any K' is the minimum over every released size, checkpoint and ensemble of up to 10 seeds.")
