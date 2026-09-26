from spp.pool import Archive, ReplayBank
from spp.machine import niche
rng = np.random.default_rng(0)
arch = Archive()
progs = [b"+[.++]", b"+[[.C>.C>]]", b"+[.L>]", b"++++++++", b"+" * 40, b"[[[[.]]]]"]
ex = execute(progs, T)
arch.add(progs, rewards=np.array([3.0, 5.0, 4.0, 0.0, 1.0, 2.0]), depths=ex["depth"])
for p, d in zip(progs, ex["depth"]):
    print(f"{p.decode():12s} loop depth {d}  length {len(p):2d}  -> niche {niche(d, len(p)):2d}")
print(f"\narchive holds {len(arch)} programs in {len(arch.cells)} niches (the zero-reward one was refused)")
print("five mutants:", [m.decode() for m in arch.mutants(5, rng, 128)])
