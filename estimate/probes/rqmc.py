"""RQMC probe (critique follow-up): scrambled Sobol on the inputs vs plain MC.

Two parametrisations: (a) x = Phi^-1(u) directly; (b) order-statistic coordinates:
u_0..u_8 -> Lehmer code of the rank permutation (position of max first), u_9.. -> values
via the Renyi exponential-spacing representation of sorted uniforms (11 dims total).
Variance gain = (p(1-p)/N) / Var_rep(p_hat). Run from estimate/.
"""
import sys, numpy as np, torch as th
from torch.quasirandom import SobolEngine
sys.path.insert(0, '.')
from ground_truth import forward, load_weights

def acc(x, W):
    return (forward(x, *W).argmax(-1) == x.topk(2, -1).indices[:, 1]).double().mean().item()

def direct(u):
    return th.distributions.Normal(0., 1.).icdf(u.clamp(1e-12, 1 - 1e-12))

def orderstat(u):
    # values: sorted uniforms descending via exponential spacings (dims 0..9 + 1 extra)
    e = -th.log(u[:, :11].clamp(1e-12, 1))
    s = th.cumsum(e, 1); vals_u = 1 - s[:, :10] / s[:, 10:11]   # descending uniforms
    vals = direct(vals_u)
    # permutation: dims 11..19 Lehmer code, slot k chooses among remaining positions
    n = u.shape[0]; rem = th.arange(10).repeat(n, 1); pos = th.empty(n, 10, dtype=th.long)
    for k in range(10):
        m = 10 - k
        j = (u[:, 11 + k] * m).long().clamp(max=m - 1) if k < 9 else th.zeros(n, dtype=th.long)
        pos[:, k] = rem.gather(1, j[:, None])[:, 0]
        mask = th.ones_like(rem, dtype=th.bool); mask.scatter_(1, j[:, None], False)
        rem = rem[mask].view(n, m - 1) if m > 1 else rem
    x = th.empty(n, 10); x.scatter_(1, pos, vals.float())
    return x

def orderstat_permfirst(u):
    # same map, Sobol dims reordered: positions of the top ranks get the best dims
    return orderstat(th.cat([u[:, 9:20], u[:, :9]], 1))

def run(W, label, reps=256):
    W = [th.tensor(a, dtype=th.float32) for a in W]
    for lg in [12, 15]:
        N = 1 << lg
        res = {}
        for name, dim, f in [('mc', 10, None), ('rqmc-x', 10, direct), ('rqmc-os', 20, orderstat), ('rqmc-os-perm1st', 20, orderstat_permfirst)]:
            ps = []
            for r in range(reps):
                if f is None:
                    x = th.randn(N, 10, generator=th.Generator().manual_seed(r + 1000 * lg))
                else:
                    u = SobolEngine(dim, scramble=True, seed=r + 1000 * lg).draw(N)
                    x = f(u).float()
                ps.append(acc(x, W))
            res[name] = (np.mean(ps), np.var(ps, ddof=1))
        p = res['mc'][0]; vb = p * (1 - p) / N
        print(f"{label} N=2^{lg} p={p:.4f} " + "  ".join(f"{k}: gain {vb/v:5.2f}" for k, (m, v) in res.items()), flush=True)

if __name__ == '__main__':
    run(load_weights('weights.npz'), 'trained')
    d = np.load('noisy_sigma0.005.npz')
    for i in range(4):
        run((d['W_ih'][i], d['W_hh'][i], d['W_out'][i]), f'noisy{i}')
