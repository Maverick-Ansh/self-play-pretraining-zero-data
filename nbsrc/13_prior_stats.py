# What does the fixed prior g0 (i.i.d. uniform tokens until F) actually produce?
bodies, charged = sample_uniform_prior(20_000, 128, np.random.default_rng(0))
ex = execute(bodies, T, seed=1)
lens = np.array([len(b) for b in bodies])
print(f"program length      mean {lens.mean():.1f} tokens (Geometric(1/19) -> 18), 99th pct {np.percentile(lens, 99):.0f}")
print(f"emit anything       {(ex['n_out'] > 0).mean():6.1%}")
print(f"emit a full tape    {(ex['n_out'] == T).mean():6.1%}   <- the only rows with long-range structure to learn")
print(f"hit step budget     {(ex['steps'] >= 2048 * (T + 1)).mean():6.1%}   (infinite loops that print nothing)")
full = ex["out"][ex["n_out"] == T]
print(f"\nfive random full tapes from g0 (first 24 bytes):")
for row in full[np.random.default_rng(2).choice(len(full), 5, replace=False)]:
    print("  ", row[:24].tolist())
