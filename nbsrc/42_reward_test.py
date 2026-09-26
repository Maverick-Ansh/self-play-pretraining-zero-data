from spp.reward import Snapshots, preconditioner, rewards, seq_losses
# Check Eq. 2 against brute force: one backward pass PER sequence, dotted with P * delta_theta.
torch.manual_seed(0)
m = ByteLlama(64, 4, 2, max_len=64)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
X = torch.randint(0, 256, (6, 33)); X[:, 0] = ord("O")
n_out = torch.tensor([32, 5, 0, 32, 17, 1])           # row 2 emitted nothing
snaps = Snapshots(every=1)
for e in range(4):                                     # a few real AdamW steps so P and delta_theta exist
    snaps.maybe_save(e, m)
    loss = seq_losses(m, dict(m.named_parameters()), X, n_out, manual=False).mean()
    opt.zero_grad(); loss.backward(); opt.step()
r_jvp = rewards(m, opt, snaps, 4, X, n_out, chunk=4, signed=True)
P, (_, past) = preconditioner(opt, m), snaps.lookback(4)
r_bf = []
for i in range(6):
    m.zero_grad()
    seq_losses(m, dict(m.named_parameters()), X[i:i+1], n_out[i:i+1], manual=False).sum().backward()
    r_bf.append(sum((p.grad * P[n] * (past[n] - p.detach())).sum() for n, p in m.named_parameters() if p.grad is not None))
r_bf = torch.stack(r_bf).detach()
print("forward-mode JVP :", r_jvp.numpy().round(6))
print("brute force      :", r_bf.numpy().round(6))
print(f"max relative error {((r_jvp - r_bf).abs().max() / r_bf.abs().max()).item():.1e}; empty program's reward = {r_jvp[2].item()}")
assert torch.allclose(r_jvp, r_bf, rtol=1e-4, atol=1e-7)
