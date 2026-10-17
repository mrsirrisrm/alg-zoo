"""Upper bound on mechanism-guided stratification: strata on the two tie gaps
g1 = (vM-vS)/|vM|, g2 = (vS-vR)/|vM| and sign(vS). Neyman-allocation variance
(sum_h p_h sqrt(e_h(1-e_h)))^2 vs MC e(1-e). Stratum probabilities are properties of
the input law only (computable offline); e_h from 2^22 samples here."""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from ground_truth import forward, load_weights
d = np.load('noisy_sigma0.005.npz')
models = [('trained', [th.tensor(a) for a in load_weights('weights.npz')])]
models += [(f'noisy{i}', [th.tensor(d[k][i]) for k in ('W_ih', 'W_hh', 'W_out')]) for i in range(8)]
th.manual_seed(5); N = 1 << 22; x = th.randn(N, 10)
top = x.topk(3, -1); vM, vS, vR = top.values.T; S = top.indices[:, 1]
g1 = ((vM - vS) / vM.abs()).numpy(); g2 = ((vS - vR) / vM.abs()).numpy()
for edges in ([0, .1, np.inf], [0, .02, .05, .1, .2, .4, np.inf], [0, .005, .01, .02, .03, .05, .07, .1, .15, .2, .3, .5, 1, np.inf]):
    s1 = np.digitize(g1, edges) - 1; s2 = np.digitize(g2, edges) - 1
    h = (s1 * len(edges) + s2) * 2 + (vS.numpy() < 0)
    gains = []
    for name, W in models:
        e = (forward(x, *W).argmax(-1) != S).numpy().astype(float)
        cnt = np.bincount(h); err = np.bincount(h, e)
        p = cnt[cnt > 0] / N; eh = err[cnt > 0] / cnt[cnt > 0]
        gains.append(e.mean() * (1 - e.mean()) / (p * np.sqrt(eh * (1 - eh))).sum() ** 2)
    print(f'{len(edges) - 1:2d} bins/gap: gain trained {gains[0]:.2f}, noisy mean {np.mean(gains[1:]):.2f} (min {np.min(gains[1:]):.2f})')
