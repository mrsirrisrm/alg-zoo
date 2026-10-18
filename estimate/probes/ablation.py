"""Ablation: is the 2-3x (and 3-6x Neyman) gain from understanding THIS network, or from
knowing the task? Run the same unbiased designs on models we never explained:
  - trained M(16,10) (the explained one) and zoo h16 seeds 0-4 (same task, not studied),
  - zoo h8 / h32 at T=10 (best seeds), hand10 (hand-built),
  - 'blur': no network at all, output = 2nd argmax of x + delta*|vM|*eps, delta set so the
    error matches M(16,10) (task-level error structure, no positions, no mechanism),
  - 'random': a random-weight h16 RNN (control; errors unrelated to the task).
Designs: Neyman strata on gaps, and on gaps + positions of the top three (allocation from
split A, variance from split B, as probes/strat_pos.py); RQMC in order-statistic
coordinates (probes/rqmc.py), replicate variance at N = 2^15. Gains vs plain MC.
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from torch.quasirandom import SobolEngine
from alg_zoo import zoo_2nd_argmax, handcrafted
from ground_truth import forward, load_weights
from rqmc import orderstat

N = 1 << 22
edges = [0, .005, .01, .02, .03, .05, .07, .1, .15, .2, .3, .5, 1, np.inf]
coarse = [0, .01, .03, .07, .15, .4, np.inf]


def sd_weights(m):
    sd = m.state_dict()
    return [sd['rnn.weight_ih_l0'][:, 0].float(), sd['rnn.weight_hh_l0'].float(), sd['linear.weight'].float()]


def net(W):
    return lambda x: forward(x, *W).argmax(-1)


def blur(delta, seed=0):
    def f(x):
        g = th.Generator().manual_seed(seed + x.shape[0])
        y = x + delta * x.abs().max(-1, keepdim=True).values * th.randn(x.shape, generator=g)
        return y.topk(2, -1).indices[:, 1]
    return f


models = [('trained', net([th.tensor(a, dtype=th.float32) for a in load_weights('weights.npz')]))]
s = np.load('seeds_h16.npz')
models += [(f'seed{i}', net([th.tensor(s[k][i], dtype=th.float32) for k in ('W_ih', 'W_hh', 'W_out')])) for i in range(5)]
models += [('h8', net(sd_weights(zoo_2nd_argmax(8, 10)))), ('h32', net(sd_weights(zoo_2nd_argmax(32, 10)))),
           ('hand10', net(sd_weights(handcrafted.handcrafted_2nd_argmax_10())))]
x0 = th.randn(1 << 20, 10, generator=th.Generator().manual_seed(9)); l0 = x0.topk(2, -1).indices[:, 1]
lo, hi = 0., .2
for _ in range(25):   # bisect delta so the blur error matches M(16,10)'s 0.0469
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if (blur(mid)(x0) != l0).float().mean() < .0469 else (lo, mid)
models.append((f'blur(d={lo:.4f})', blur(lo)))
g = th.Generator().manual_seed(3)
models.append(('random', net([th.randn(16, generator=g), th.randn(16, 16, generator=g) / 4, th.randn(10, 16, generator=g) / 4])))


def strata(x):
    t = x.topk(3, -1); (vM, vS, vR), (pM, pS, pR) = t.values.T, t.indices.T.numpy()
    g1 = ((vM - vS) / vM.abs()).numpy(); g2 = ((vS - vR) / vM.abs()).numpy(); neg = (vS < 0).numpy()
    fine = (np.digitize(g1, edges) * 14 + np.digitize(g2, edges)) * 2 + neg
    crs = (np.digitize(g1, coarse) * 7 + np.digitize(g2, coarse)) * 2 + neg
    return t.indices[:, 1], {'gaps': fine, 'gaps+pos': crs * 1000 + pM * 100 + pS * 10 + pR}


def neyman(ea, eb, sa_, sb_):
    K = int(max(sa_.max(), sb_.max())) + 1
    ca, ra = np.bincount(sa_, minlength=K), np.bincount(sa_, ea, minlength=K)
    cb, rb = np.bincount(sb_, minlength=K), np.bincount(sb_, eb, minlength=K)
    p = cb / N; qa = ra / np.maximum(ca, 1); qb = rb / np.maximum(cb, 1)
    sa = np.sqrt(np.where(ca > 0, qa * (1 - qa), .25)); sb = np.sqrt(qb * (1 - qb))
    alloc = .9 * p * sa / max((p * sa).sum(), 1e-30) + .1 * p
    m = p > 0; e = eb.mean()
    return e * (1 - e) / (p[m] ** 2 * sb[m] ** 2 / alloc[m]).sum()


def rqmc_gain(f, n=1 << 15, reps=64):
    ps = []
    for r in range(reps):
        x = orderstat(SobolEngine(20, scramble=True, seed=r).draw(n)).float()
        ps.append((f(x) == x.topk(2, -1).indices[:, 1]).double().mean().item())
    p = np.mean(ps)
    return p * (1 - p) / n / np.var(ps, ddof=1)


xa = th.randn(N, 10, generator=th.Generator().manual_seed(1)); la, SA = strata(xa)
xb = th.randn(N, 10, generator=th.Generator().manual_seed(2)); lb, SB = strata(xb)
print(f"gain over plain MC at equal samples; strata N = 2^22 per split; RQMC N = 2^15 x 64 reps")
print(f"{'model':16s} {'err':>7s} {'gaps':>6s} {'gaps+pos':>9s} {'rqmc-os':>8s}")
with th.no_grad():
    for name, f in models:
        ea = (f(xa) != la).numpy().astype(float); eb = (f(xb) != lb).numpy().astype(float)
        print(f"{name:16s} {eb.mean():7.4f} {neyman(ea, eb, SA['gaps'], SB['gaps']):5.2f}x "
              f"{neyman(ea, eb, SA['gaps+pos'], SB['gaps+pos']):8.2f}x {rqmc_gain(f):7.2f}x", flush=True)
