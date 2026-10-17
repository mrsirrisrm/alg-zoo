"""RQMC in order-statistic coordinates (probes/rqmc.py, perm-first ordering): bias check
against high-precision truth and growth of the variance gain with N.
Models: trained, noisy0-3, seeds 0-4. Truth: 2^35 (trained), 2^30 (noisy), 2^26 (seeds).
Gain = (p(1-p)/N) / Var_rep(p_hat); z = (mean_rep - p) / se."""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from torch.quasirandom import SobolEngine
from ground_truth import forward, load_weights

ND = th.distributions.Normal(0., 1.)


def orderstat_permfirst(u):
    u = th.cat([u[:, 9:20], u[:, :9]], 1)
    e = -th.log(u[:, :11].clamp(1e-12, 1)); s = th.cumsum(e, 1)
    vals = ND.icdf((1 - s[:, :10] / s[:, 10:11]).clamp(1e-12, 1 - 1e-12))
    n = u.shape[0]; rem = th.arange(10).repeat(n, 1); pos = th.empty(n, 10, dtype=th.long)
    for k in range(10):
        m = 10 - k
        j = (u[:, 11 + k] * m).long().clamp(max=m - 1) if k < 9 else th.zeros(n, dtype=th.long)
        pos[:, k] = rem.gather(1, j[:, None])[:, 0]
        if m > 1:
            mask = th.ones_like(rem, dtype=th.bool); mask.scatter_(1, j[:, None], False); rem = rem[mask].view(n, m - 1)
    return th.empty(n, 10).scatter_(1, pos, vals.float())


d, s = np.load('noisy_sigma0.005.npz'), np.load('seeds_h16.npz')
models = [('trained', load_weights('weights.npz'), 0.9531186)]
models += [(f'noisy{i}', [d[k][i] for k in ('W_ih', 'W_hh', 'W_out')], d['acc'][i]) for i in range(4)]
models += [(f'seed{i}', [s[k][i] for k in ('W_ih', 'W_hh', 'W_out')], s['acc'][i]) for i in range(5)]
reps = 64
with th.no_grad():
    for nm, W, p in models:
        W = [th.tensor(a, dtype=th.float32) for a in W]
        line = []
        for lg in (12, 15, 18, 20):
            N = 1 << lg
            ps = [(forward(x := orderstat_permfirst(SobolEngine(20, scramble=True, seed=r + 97 * lg).draw(N)), *W).argmax(-1)
                   == x.topk(2, -1).indices[:, 1]).double().mean().item() for r in range(reps)]
            v = np.var(ps, ddof=1); z = (np.mean(ps) - p) / np.sqrt(v / reps)
            line.append(f'2^{lg}: gain {p * (1 - p) / N / v:5.2f} (z {z:+.1f})')
        print(nm, '  '.join(line), flush=True)
