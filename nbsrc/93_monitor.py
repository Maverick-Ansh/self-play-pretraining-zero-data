# Progress of every run (safe to re-run any time; it only reads logs).
import glob
print(f"{'run':22s} {'status':>6s} {'tokens':>8s} {'s/round':>8s}  latest bits/byte: text  images speech melody  DNA")
for d in sorted(glob.glob("runs/*")):
    try:
        lines = [json.loads(l) for l in open(f"{d}/log.jsonl")]
    except (FileNotFoundError, json.JSONDecodeError):
        continue
    ev = [l for l in lines if l["kind"] == "eval"]
    tr = [l for l in lines if l["kind"] == "train"]
    if not ev:
        continue
    spr = np.median([sum(v for k, v in t.items() if k.startswith("t_")) for t in tr[-8:]]) if tr else float("nan")
    b = ev[-1]["bpb"]
    status = "done" if os.path.exists(f"{d}/final.json") else f"r{tr[-1]['round'] if tr else 0}"
    print(f"{os.path.basename(d):22s} {status:>6s} {ev[-1]['tokens']/1e6:7.0f}M {spr:8.2f}  " + " ".join(
        f"{b[c]:6.2f}" for c in ["dclm", "cifar10_rgb_hwc", "audio_8bit", "mutopia_melody_16th", "dna"]))
