"""Great-circle / arc Rao-Blackwellisation probe (critique, see CRITIQUE.md).

x(θ) = cosθ g1 + sinθ g2 ~ N(0, I) for every θ, so the correct-fraction of any arc of
fixed length with uniform start is unbiased. Measures variance gain vs the cost of exact
breakpoint tracing (linear pieces crossed). Run from estimate/.
"""
import sys, numpy as np, torch as th
sys.path.insert(0, '.')
from ground_truth import load_weights
th.manual_seed(1)
W_ih, W_hh, W_out = [th.tensor(a, dtype=th.float64) for a in load_weights('weights.npz')]
nc, K = 1024, 8192
g1, g2 = th.randn(nc, 1, 10, dtype=th.float64), th.randn(nc, 1, 10, dtype=th.float64)
t_ = th.linspace(0, 2*np.pi, K+1, dtype=th.float64)[:-1].view(1, K, 1)
x = (th.cos(t_)*g1 + th.sin(t_)*g2).reshape(-1, 10)
h = th.relu(x[:, :1]*W_ih); pats = [h > 0]
for t in range(1, 10):
    h = th.relu(x[:, t:t+1]*W_ih + h @ W_hh.T); pats.append(h > 0)
corr = ((h @ W_out.T).argmax(-1) == x.topk(2, -1).indices[:, 1]).double().view(nc, K)
pat = th.cat(pats, 1).view(nc, K, -1)
kink = (pat != pat.roll(1, 1)).any(-1).double()
p = corr.mean().item(); vb = p*(1-p)
for m in [1, 4, 16, 64, 256, 1024]:
    L = K//m
    f = corr.view(nc, m, L).mean(-1).flatten()
    pieces = kink.view(nc, m, L)[..., 1:].sum(-1).flatten() + 1
    cost = 2*pieces.mean().item()
    print(f"m={m:5d} arc={360/m:6.1f}deg  var gain={vb/f.var().item():7.1f}  cost={cost:6.1f} passes  efficiency={vb/f.var().item()/cost:.2f}")
