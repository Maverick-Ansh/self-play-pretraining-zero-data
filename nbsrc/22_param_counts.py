from spp.model import ByteLlama, LADDER, LADDER_PARAMS
print(f"{'rung':5s} {'d':>4s} {'heads':>5s} {'layers':>6s} {'ours':>12s} {'released':>12s}  non-embedding")
for k, (d, h, L) in LADDER.items():
    m = ByteLlama.rung(k)
    n = m.n_params()
    emb = m.wte.weight.numel() + m.lm_head.weight.numel()
    print(f"{k:5s} {d:4d} {h:5d} {L:6d} {n:12,d} {LADDER_PARAMS[k]:12,d}  {n - emb:10,d}  {'OK' if n == LADDER_PARAMS[k] else 'MISMATCH'}")
    assert n == LADDER_PARAMS[k]
print("\n", ByteLlama.rung("1M").blocks[0])
