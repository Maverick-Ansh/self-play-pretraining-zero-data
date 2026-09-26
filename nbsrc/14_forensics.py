# Forensics: replay the programs the authors' generator discovered (released hits.jsonl) and
# demand their exact recorded bytes. This is how the tape size and step budget were pinned down.
import urllib.request
url = "https://raw.githubusercontent.com/acowsik/self_play_pretraining/main/tables/table1/hits.jsonl"
H = [json.loads(l) for l in urllib.request.urlopen(url).read().decode().splitlines()]
det = [h for h in H if "," not in h["program"]]                   # no random input: output is deterministic
P = [h["program"][1:].encode() for h in det]                       # strip the 'S' row tag
ref = [h["prefix"] + h["first_terms"] for h in det]
arith = [i for i, h in enumerate(det) if h["family"] == "arithmetic"]
print(f"{len(H):,} released hits, {len(det):,} deterministic ({len(arith):,} of them arithmetic)\n")
print(f"{'tape':>10s}  {'recorded first bytes reproduced':>32s}  {'arithmetic tapes arithmetic to the END':>40s}")
for cells in (128, 2048, 4096):
    out = execute(P, 4095, max_steps=2048 * 4096, tape_cells=cells)["out"].astype(np.int64)
    ok = sum(o[:len(f)].tolist() == f for o, f in zip(out, ref))
    # the detector only calls a tape arithmetic if the recurrence holds through its last byte
    whole = sum(bool((np.diff(out[i, det[i]["start_offset"]:]) % 256 == det[i]["params"]["diffs"][0]).all()) for i in arith)
    print(f"{cells:6d} cells  {ok:>24,}/{len(det):,}  {whole:>32,}/{len(arith):,}")
steps = execute(P, 4095, max_steps=10**9)["steps"]
print(f"\nmost steps any released program needed: {steps.max():,} for 4095 bytes = {steps.max() / 4095:.0f} per byte "
      f"(< 2^23 = {2**23:,}: the budget is 2048 per context position)")
