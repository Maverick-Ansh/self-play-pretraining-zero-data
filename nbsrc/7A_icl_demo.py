# 7.4  What the ICL tasks look like (m = 2 demonstrations each; the scored answer is the last byte)
from spp.evals import icl_task, ICL_TASKS
rng = np.random.default_rng(0)
for t in ICL_TASKS:
    toks, pos, ans = icl_task(t, 16 if t == "assoc" else 2, rng)
    shown = toks if len(toks) < 40 else toks[:12] + ["..."] + toks[-14:]
    print(f"{t:8s} {shown}  -> answer {ans}")
