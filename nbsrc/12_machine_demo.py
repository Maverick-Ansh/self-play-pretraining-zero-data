from spp.machine import execute, sample_uniform_prior, ALPHABET

T = 511                                          # output bytes per program in this reproduction
show = {
    "App. E    +++[>+.<-]F":                        b"+++[>+.<-]",
    "Table 1   arithmetic   S+[.++]":               b"+[.++]",
    "Table 1   Fibonacci    S,[[.C>.C>]]  (,=1)":   b"+[[.C>.C>]]",
    "Table 1   geometric    S+[.L>]":               b"+[.L>]",
    "Table 1   quadratic    S,.[<C>>VX<RX++] (,=9)": b"+++++++++.[<C>>VX<RX++]",
}
r = execute(list(show.values()), T)
for (name, _), o, n, s in zip(show.items(), r["out"], r["n_out"], r["steps"]):
    print(f"{name:46s} -> {o[:8].tolist()}   {n:3d} bytes, {s:>9,} steps")
assert r["out"][0][:4].tolist() == [1, 2, 3, 0], "App. E: emits (1, 2, 3, 0, ...)"
assert r["out"][4][:4].tolist() == [9, 25, 59, 111], "Table 1 quadratic, exactly as printed in the paper"
print("\nThe Fibonacci program spends ~1000 steps per byte (macro C loops x times): a small step budget would silence it.")
