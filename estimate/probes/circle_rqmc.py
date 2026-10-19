"""Structured great circles: RQMC over the circle parameters (g1, g2) vs iid circles.

x(θ) = cosθ g1 + sinθ g2. g1 in order-statistic coordinates (perm-first, as rqmc.py),
g2 direct Phi^-1. Arc of length 2π/m from θ0 = 0 (the start is uniform already by
rotation invariance of (g1, g2)). Reports per-circle variance gain vs Bernoulli and vs
iid circles; efficiency uses the exact-tracing cost from great_circle.py (pieces + 1)*2.
Run from estimate/.
"""
import sys, numpy as np, torch as th
from torch.quasirandom import SobolEngine
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
from ground_truth import forward, load_weights
from rqmc import orderstat_permfirst, direct

COST = {1: 465., 16: 31., 256: 3.8}          # passes per arc, measured in great_circle.py

def arcs(g1, g2, W, m, K):
    t = th.linspace(0, 2*np.pi/m, K+1)[:-1] if m > 1 else th.linspace(0, 2*np.pi, K+1)[:-1]
    x = (th.cos(t).view(1, K, 1)*g1[:, None] + th.sin(t).view(1, K, 1)*g2[:, None]).reshape(-1, 10)
    ok = (forward(x, *W).argmax(-1) == x.topk(2, -1).indices[:, 1]).float().view(-1, K)
    return ok.mean(1).double()

def run(W, label, nc=256, reps=32, K=1024):
    W = [th.tensor(a, dtype=th.float32) for a in W]
    for m in [1, 16, 256]:
        Km = K if m == 1 else max(32, K//4)
        iid, qmc = [], []
        for r in range(reps):
            g = th.Generator().manual_seed(r)
            iid.append(arcs(th.randn(nc, 10, generator=g), th.randn(nc, 10, generator=g), W, m, Km).mean().item())
            u = SobolEngine(30, scramble=True, seed=r).draw(nc)
            qmc.append(arcs(orderstat_permfirst(u[:, :20]).float(), direct(u[:, 20:]).float(), W, m, Km).mean().item())
        p = np.mean(iid); vb = p*(1-p)
        vi, vq = np.var(iid, ddof=1)*nc, np.var(qmc, ddof=1)*nc
        print(f"{label} arc=2π/{m:<3d} iid-circle gain {vb/vi:6.1f}  rqmc-circle gain {vb/vq:6.1f} "
              f"(x{vi/vq:.1f} from structure)  efficiency vs MC: iid {vb/vi/COST[m]:.2f}  rqmc {vb/vq/COST[m]:.2f}", flush=True)

run(load_weights('weights.npz'), 'trained')
d = np.load('noisy_sigma0.005.npz')
for i in range(2):
    run((d['W_ih'][i], d['W_hh'][i], d['W_out'][i]), f'noisy{i}')
