# Part 0 · Setup: one working directory that holds the spp/ package this notebook writes, cell by cell.
import os, sys, subprocess, json, time
REPO = ("/kaggle/working" if os.path.isdir("/kaggle/working") else "/content") + "/self-play-pretraining-zero-data"
os.makedirs(f"{REPO}/spp", exist_ok=True)
os.makedirs(f"{REPO}/scripts", exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, REPO)
open("spp/__init__.py", "a").close()
subprocess.run(f"{sys.executable} -m pip -q install numba zstandard mido", shell=True)
import numpy as np, torch
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                     capture_output=True, text=True).stdout)
print("working in", REPO, "| torch", torch.__version__, "| cpus", os.cpu_count())
