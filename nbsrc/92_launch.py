# Launch (detached): Phase A then Phase B on each GPU, one run at a time.
# A GPU that already has a queue running is left alone (re-running this cell never doubles up).
R = 2048
PHASE_A = {0: [f"3M:selfplay:0:{R}", f"100k:selfplay:0:{R}", f"3M:uniform:0:{R}"],
           1: [f"1M:selfplay:0:{R}", f"1M:uniform:0:{R}", f"500k:selfplay:0:{R}", f"500k:uniform:0:{R}", f"100k:uniform:0:{R}"]}
PHASE_B = {0: [f"1M:selfplay:1:{R}", f"1M:shuffle:1:{R}"],
           1: [f"1M:shuffle:0:{R}", f"1M:uniform:1:{R}"]}

def queue_pids(g):
    """PIDs of python queue processes already serving GPU g (python processes, not their shells)."""
    out = subprocess.run("ps -eo pid,args", shell=True, capture_output=True, text=True).stdout.splitlines()
    return [l.split()[0] for l in out if f"queue.py --gpu {g} " in l and l.split()[1].startswith(("python", "/usr/bin/python"))]

for g in range(torch.cuda.device_count()):
    if queue_pids(g):
        print(f"GPU {g}: a queue is already running (pid {queue_pids(g)}); not launching another")
        continue
    a = f"python -u scripts/run_queue.py --gpu {g} --runs {' '.join(PHASE_A[g])} >> queue{g}.log 2>&1"
    b = f"python -u scripts/run_queue.py --gpu {g} --runs {' '.join(PHASE_B[g])} >> queueB{g}.log 2>&1"
    subprocess.Popen(f"{a}; {b}", shell=True, cwd=REPO, start_new_session=True)
    print(f"GPU {g}: launched Phase A, then Phase B")
