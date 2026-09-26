from spp import generator as G
from spp.machine import log_g0

torch.manual_seed(0)
gen = ByteLlama.rung("1M", 130)                      # the generator's context: S + 128 tokens + F
torch.nn.init.zeros_(gen.lm_head.weight)             # "initialized near g0": here, exactly g0
bodies, lp, charged = G.sample(gen, 1024, 128)
print(f"max |log g_phi - log g0| at init : {np.abs(lp - log_g0(charged)).max():.2e} nats")
print(f"sampling vs scoring log-prob     : {np.abs(G.logprob(gen, bodies[:256], 128).detach().numpy() - lp[:256]).max():.2e}")
lens = np.array([len(b) for b in bodies])
print(f"length: mean {lens.mean():.1f} (Geometric(1/19) mean 18), capped at 128: {(lens == 128).mean():.2%}")
print("a few programs from g0:", [b.decode() for b in bodies[:6]])
assert np.allclose(lp, log_g0(charged), atol=1e-3)
