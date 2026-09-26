# The chance baseline for C5: 10M programs from g0, executed and scanned (~5 min on 16 CPU cores, ~20 on 4).
if not os.path.exists("results/prior_discovery.json"):
    subprocess.run("python -u scripts/prior_discovery.py --n 10000000", shell=True, check=True)
print(json.load(open("results/prior_discovery.json"))["hit_counts"])
