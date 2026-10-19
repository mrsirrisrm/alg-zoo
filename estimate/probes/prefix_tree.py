"""B4 (PLAN.md): prefix sharing. Inputs as a tree: m leaves share the first k steps, so the
recurrence for those steps is computed once per group. Leaves are correlated through the
shared prefix with intra-class correlation rho_k = Var(E[y | x_<k]) / Var(y) (the prefix R2),
so a mean over L leaves has variance (sigma^2 / L)(1 + (m-1) rho_k) while costing
((k/m + 10 - k) * 576 + 320) FLOPs per leaf instead of 5760.
Measures rho_k by pick-freeze (two leaves per prefix, 2^22 prefixes), tabulates the gain over
MC for k = 1..7, m in {2,4,8,16,32}, and checks the best (k, m) by direct simulation (64 reps).
Run from estimate/: python probes/prefix_tree.py
"""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from ground_truth import forward, load_weights

STEP, READ = 2 * 16 * 16 + 2 * 16, 2 * 10 * 16
MS = [2, 4, 8, 16, 32]


def y_of(x, W):
    return (forward(x, *W).argmax(-1) == x.topk(2, -1).indices[:, 1]).double()


def rho(W, k, n=1 << 22, seed=3):
    g = th.Generator().manual_seed(seed)
    pre = th.randn(n, k, generator=g); a = th.randn(n, 10 - k, generator=g); b = th.randn(n, 10 - k, generator=g)
    with th.no_grad():
        y1 = y_of(th.cat([pre, a], 1), W); y2 = y_of(th.cat([pre, b], 1), W)
    p = 0.5 * (y1.mean() + y2.mean()).item(); v = p * (1 - p)
    return ((y1 * y2).mean().item() - p * p) / v, p


def cost(k, m):
    return (k / m + 10 - k) * STEP + READ


def simulate(W, k, m, leaves=1 << 16, reps=64, seed=5):
    g = th.Generator().manual_seed(seed); ests = []
    with th.no_grad():
        for _ in range(reps):
            G = leaves // m
            pre = th.randn(G, k, generator=g).repeat_interleave(m, 0); suf = th.randn(leaves, 10 - k, generator=g)
            ests.append(y_of(th.cat([pre, suf], 1), W).mean().item())
    return np.var(ests, ddof=1), leaves


if __name__ == '__main__':
    d, s = np.load('noisy_sigma0.005.npz'), np.load('seeds_h16.npz')
    models = [('trained', load_weights('weights.npz')), ('noisy0', [d[q][0] for q in ('W_ih', 'W_hh', 'W_out')]),
              ('seed0', [s[q][0] for q in ('W_ih', 'W_hh', 'W_out')])]
    for name, W in models:
        W = [th.tensor(w, dtype=th.float32) for w in W]
        print(f"\n{name}: rho_k (prefix R2) and gain over MC = 5760 / (cost(k,m) (1 + (m-1) rho_k))")
        best = (0, None)
        for k in range(1, 8):
            r, p = rho(W, k)
            gains = [5760 / (cost(k, m) * (1 + (m - 1) * r)) for m in MS]
            print(f"  k={k} rho {r:.4f} | " + "  ".join(f"m={m}: {gn:.2f}x" for m, gn in zip(MS, gains)))
            for m, gn in zip(MS, gains):
                if gn > best[0]:
                    best = (gn, (k, m, r, p))
        gn, (k, m, r, p) = best
        v, L = simulate(W, k, m); vmc = p * (1 - p) / L
        print(f"  best k={k} m={m}: predicted {gn:.2f}x; simulated variance ratio MC/tree at equal leaves {vmc / v:.2f} "
              f"(predicted {1 / (1 + (m - 1) * r):.2f}), FLOP ratio {5760 / cost(k, m):.2f} -> measured gain {vmc / v * 5760 / cost(k, m):.2f}x")
