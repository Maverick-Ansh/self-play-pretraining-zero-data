from collections import Counter
from spp.structure import scan
tapes = execute([b"+[.++]", b"+[[.C>.C>]]", b"+[.L>]", b"+++++++++.[<C>>VX<RX++]", b"+++++.", b",[.,]"], T)["out"]
print("Table 1 programs  ->", [(i, h["family"], h["period"]) for i, h in scan(tapes)])
bodies, _ = sample_uniform_prior(4096, 128, np.random.default_rng(7))
t0 = time.time(); hits = scan(execute(bodies, T, seed=3)["out"])
print(f"4,096 uniform-prior programs -> {dict(Counter(h['family'] for _, h in hits))}  ({(time.time()-t0)*1e3:.0f} ms)")
# the repair: a transient-then-constant tape must NOT count as structure
junk = np.zeros((1, T), np.uint8); junk[0, 0] = 5
print("tape [5, 0, 0, ...]      ->", scan(junk), "(the unrepaired 2048-byte window accepts it as geometric)")
