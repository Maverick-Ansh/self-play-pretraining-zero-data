# 7.1  Bake the corpora (downloads DCLM shards, CIFAR-10, 40 Mutopia MIDI files; ~5 min, skipped if present)
if not os.path.exists("data/c512/random.npy"):
    subprocess.run("python -u scripts/prepare_eval.py", shell=True, check=True)
from spp.evals import load_corpora, LABEL
corp = load_corpora("data/c512")
for c, a in corp.items():
    print(f"{LABEL[c]:18s} {str(a.shape):12s} first bytes {a[0, :14].tolist()}")
