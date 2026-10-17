"""Does knowing WHERE the top values sit (comparator decay) let us trim strata harder?
Strata: tie gaps g1=(vM-vS)/|vM|, g2=(vS-vR)/|vM|, sign(vS), optionally positions of the
top three. Allocation n_h ∝ p_h * sigma_h is learned on split A (with 10% proportional
floor so no stratum gets zero samples); its variance is evaluated with split B's sigma_h,
so strata that merely look error-free in A are not rewarded. Gain = e(1-e) / variance.
Also: mass of strata with zero errors in A, and the error mass B finds there.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from ground_truth import forward, load_weights

d, s = np.load('noisy_sigma0.005.npz'), np.load('seeds_h16.npz')
models = [('trained', [th.tensor(a) for a in load_weights('weights.npz')])]
models += [(f'noisy{i}', [th.tensor(d[k][i]) for k in ('W_ih', 'W_hh', 'W_out')]) for i in range(4)]
models += [(f'seed{i}', [th.tensor(s[k][i]) for k in ('W_ih', 'W_hh', 'W_out')]) for i in range(5)]
N = 1 << 23
edges = [0, .005, .01, .02, .03, .05, .07, .1, .15, .2, .3, .5, 1, np.inf]
coarse = [0, .01, .03, .07, .15, .4, np.inf]


def draw(seed):
    x = th.randn(N, 10, generator=th.Generator().manual_seed(seed))
    t = x.topk(3, -1); (vM, vS, vR), (pM, pS, pR) = t.values.T, t.indices.T.numpy()
    g1 = ((vM - vS) / vM.abs()).numpy(); g2 = ((vS - vR) / vM.abs()).numpy(); neg = (vS < 0).numpy()
    fine = (np.digitize(g1, edges) * 14 + np.digitize(g2, edges)) * 2 + neg
    crs = (np.digitize(g1, coarse) * 7 + np.digitize(g2, coarse)) * 2 + neg
    S = {'gaps (fine)': fine,
         'gaps + pos(M,S)': crs * 100 + pM * 10 + pS,
         'gaps + pos(S,R)': crs * 100 + pS * 10 + pR,
         'gaps + pos(M,S,R)': crs * 1000 + pM * 100 + pS * 10 + pR}
    return x, t.indices[:, 1], S


xa, la, SA = draw(1); xb, lb, SB = draw(2)
print(f"N = 2^23 per split; gain over MC at equal samples (held-out); trim = mass of strata error-free in A, err = B's error mass there")
for name, W in models:
    ea = (forward(xa, *W).argmax(-1) != la).numpy().astype(float)
    eb = (forward(xb, *W).argmax(-1) != lb).numpy().astype(float)
    e = eb.mean(); out = [f"{name:8s} err {e:.4f}"]
    for k in SA:
        K = int(max(SA[k].max(), SB[k].max())) + 1
        ca, ra = np.bincount(SA[k], minlength=K), np.bincount(SA[k], ea, minlength=K)
        cb, rb = np.bincount(SB[k], minlength=K), np.bincount(SB[k], eb, minlength=K)
        p = cb / N; sa = np.sqrt(np.where(ca > 0, ra / np.maximum(ca, 1) * (1 - ra / np.maximum(ca, 1)), .25))
        sb = np.sqrt(np.where(cb > 0, rb / np.maximum(cb, 1) * (1 - rb / np.maximum(cb, 1)), 0))
        alloc = .9 * p * sa / max((p * sa).sum(), 1e-30) + .1 * p
        m = (p > 0)
        var = (p[m] ** 2 * sb[m] ** 2 / alloc[m]).sum()
        oracle = (p * sb).sum() ** 2
        z = (ra == 0) & (ca > 0)
        out.append(f"| {k}: {e * (1 - e) / var:4.2f}x (oracle {e * (1 - e) / oracle:4.2f}x) trim {p[z].sum():.2f} err {rb[z].sum() / N:.1e}")
    print(' '.join(out), flush=True)
