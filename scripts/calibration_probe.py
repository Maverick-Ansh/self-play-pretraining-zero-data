"""Is the paper's ensemble gain knowledge or calibration? (a mechanism test for C6)

On the authors' released 1M learners (round 16383, context 4096): compare
  (a) one model as released,
  (b) one model with ONE softmax temperature fitted on windows 0-127 and scored on windows 128-255,
  (c) the probability-average of the seeds (the paper's ensemble), scored on windows 128-255,
  (d) the untrained in-context KT order-0 counter on windows 128-255.
If (b) recovers most of (a) -> (c), the ensemble gain is mostly calibration, not extra knowledge.
(b) fits a single scalar per corpus on natural data; nothing else is fitted.

    python scripts/calibration_probe.py
"""
import json
import math
import sys
import urllib.request

import numpy as np
import torch

sys.path.insert(0, ".")
from spp.evals import with_prefix
from spp.model import ByteLlama

HF = "https://huggingface.co/nourya-cohen/solomonoff-paper/resolve/main"
CORP = {"dclm": "data/c4096_default/dclm.npy", "cifar10_rgb_hwc": "data/c4096/cifar10_rgb_hwc.npy",
        "audio_8bit": "data/c4096/audio_8bit.npy", "dna": "data/c4096/dna.npy"}
TAUS = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
tree = json.load(urllib.request.urlopen("https://huggingface.co/api/models/nourya-cohen/solomonoff-paper/tree/main/1M"))
models = []
for s in (t["path"].split("/")[-1] for t in tree):
    try:
        path, _ = urllib.request.urlretrieve(f"{HF}/1M/{s}/learner_16383.pth")
    except Exception:
        continue
    sd = torch.load(path, map_location="cpu", weights_only=False)["learner_state_dict"]
    models.append(ByteLlama.rung("1M", 4096).load_release(sd).cuda().eval())
print(f"{len(models)} released 1M seeds loaded")


@torch.no_grad()
def logprobs(m, X, tau):
    """[n, 4095] log p(actual next byte) at temperature tau."""
    out = []
    for lo in range(0, len(X), 8):
        x = X[lo:lo + 8].cuda()
        lp = torch.log_softmax(m(x[:, :-1]).float() / tau, -1)
        out.append(lp.gather(-1, x[:, 1:, None])[..., 0].cpu())
    return torch.cat(out)


res = {}
for c, p in CORP.items():
    a = np.load(p)[:256]
    X = torch.as_tensor(with_prefix(a).astype(np.int64))
    fit, test = X[:128], X[128:]
    bits = lambda lp: float(-lp.mean() / math.log(2))
    single = [bits(logprobs(m, test, 1.0)) for m in models]
    tau_star = [min(TAUS, key=lambda t: bits(logprobs(m, fit, t))) for m in models]
    tempered = [bits(logprobs(m, test, t)) for m, t in zip(models, tau_star)]
    ens = torch.stack([logprobs(m, test, 1.0).exp() for m in models]).mean(0)
    ensemble = float(-(ens.clamp_min(1e-30).log2()).mean())
    ens_t = torch.stack([logprobs(m, test, t).exp() for m, t in zip(models, tau_star)]).mean(0)
    ensemble_t = float(-(ens_t.clamp_min(1e-30).log2()).mean())
    row = a[128:].astype(np.int64)
    kt = 0.0
    for r in row:
        onehot = np.zeros((len(r), 256)); onehot[np.arange(len(r)), r] = 1
        seen = np.cumsum(onehot, 0) - onehot
        kt -= np.log2((seen[np.arange(len(r)), r] + 0.5) / (np.arange(len(r)) + 128)).sum()
    kt /= row.size
    res[c] = dict(single=single, tau=tau_star, tempered=tempered, ensemble=ensemble, ensemble_tempered=ensemble_t, kt0=kt)
    print(f"{c:18s} one model {np.mean(single):.3f} | one model, fitted tau {np.mean(tempered):.3f} (tau {tau_star}) | "
          f"ensemble of {len(models)} {ensemble:.3f} | tempered ensemble {ensemble_t:.3f} | KT-0 {kt:.3f}", flush=True)
json.dump(res, open("results/calibration_probe.json", "w"), indent=1)
