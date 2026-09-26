# 7.3c  Is the ensemble gain calibration or knowledge? One temperature per model, fitted on windows 0-127,
#       scored on 128-255, against the 3-seed ensemble and the KT counter (released 1M learners, context 4096).
if not os.path.exists("results/calibration_probe.json"):
    subprocess.run("python -u scripts/calibration_probe.py", shell=True, check=True)
CAL = json.load(open("results/calibration_probe.json"))
print(f"{'corpus':18s} {'one model':>10s} {'+ fitted T':>11s} {'ensemble(3)':>12s} {'KT-0':>7s}   share of ensemble gain a temperature recovers")
for c, v in CAL.items():
    one, tmp, ens = np.mean(v["single"]), np.mean(v["tempered"]), v["ensemble"]
    print(f"{LABEL[c]:18s} {one:10.3f} {tmp:11.3f} {ens:12.3f} {v['kt0']:7.3f}   {(one - tmp) / (one - ens):6.0%}   (T = {v['tau']})")
