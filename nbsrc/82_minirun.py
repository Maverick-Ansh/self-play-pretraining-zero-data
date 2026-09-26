# A live mini-run: 48 rounds of the smallest model, every piece of Parts 1-7 working together.
import shutil, importlib
import spp.selfplay; importlib.reload(spp.selfplay)
from spp.selfplay import Config, run
torch.cuda.set_device(torch.cuda.device_count() - 1)
cfg = Config(rung="100k", rounds=48, eval_every=16, out="runs_demo/100k_selfplay")
shutil.rmtree(cfg.out, ignore_errors=True)
run(cfg)
log = [json.loads(l) for l in open(f"{cfg.out}/log.jsonl")]
print(f"\n{'round':>5s} {'loss':>6s} {'KL(g||g0)':>9s} {'mean r':>8s} {'prog len':>8s} {'full tapes':>10s} {'archive':>7s} {'niches':>6s}")
for r in (x for x in log if x["kind"] == "train"):
    print(f"{r['round']:5d} {r['loss_bits']:6.2f} {r['kl']:9.2f} {r['r_mean']:8.2e} {r['body_len']:8.1f} {r['full_frac']:10.1%} {r['archive']:7d} {r['niches']:6d}")
ev = [x for x in log if x["kind"] == "eval"][-1]
print("\nprograms the generator writes after 48 rounds:", ev["programs"][:8])
