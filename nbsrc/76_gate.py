# 7.2  The gate. Downloads the released 1M learners (4 MB each) and scores them at context 4096.
MAIN = [f for f in ("data/c4096/release_gate_1M_16383.json", "data/release_gate_1M_16383.json") if os.path.exists(f)]
DEFAULT = "data/c4096_default/release_gate_1M_16383.json"      # DCLM + arithmetic re-baked at the paper's counts
if not MAIN:
    subprocess.run("python scripts/release_gate.py --rung 1M --round 16383 "
                   "--corpora dclm_ranked,cifar10_rgb_hwc,audio_8bit,mutopia_melody_16th,dna", shell=True, check=True)
    MAIN = ["data/c4096/release_gate_1M_16383.json"]
if not os.path.exists(DEFAULT):
    subprocess.run("python scripts/bake_default.py && python scripts/release_gate.py --data data/c4096_default "
                   "--corpora dclm,arithmetic", shell=True, check=True)
gate = {c: v for c, v in json.load(open(MAIN[0])).items() if c not in ("dclm", "arithmetic")}
gate.update(json.load(open(DEFAULT)))
print(f"{'corpus':22s} {'our code on their weights':>27s} {'published':>10s} {'diff':>7s}")
for c, v in gate.items():
    print(f"{c:22s} {v['ours']:13.3f} ± {v['sd']:.3f} ({v['n_seeds']} seeds) {v['published']:9.3f} {v['ours'] - v['published']:+7.3f}")
assert all(abs(v["ours"] - v["published"]) < 5e-3 for v in gate.values()), "measurement does not match the paper's"
print("\nGATE PASSED. Note the authors' own seed spread: 0.08 to 0.5 bits across 3 seeds of the same model.")
