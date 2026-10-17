"""Which background inputs drive the tie margin m0? Total Sobol index per group,
by pick-freeze: resample a group of background inputs from their conditional law
(N(0,1) truncated below the tie value v), keep the rest, ST = E[(m - m')^2] / (2 Var m).
Groups by time relative to the tied pair (first, second) and the max M (k=2)."""
import sys; sys.path.insert(0, '.')
import torch as th
from scipy.stats import norm
from boundary_layer import sample_ties, widths
from ground_truth import load_weights

W = [th.tensor(a) for a in load_weights('weights.npz')]
gen = th.Generator().manual_seed(3)
n = 1 << 15
for k in (1, 2):
    x, a, b = sample_ties(k, n, gen)
    ar = th.arange(n); v = x[ar, a]
    bg = x < v[:, None] - 1e-6
    lo, hi = th.minimum(a, b), th.maximum(a, b)
    t = th.arange(10)[None]
    m = widths(x, a, b, W)[0]; var = m.var()
    Fv = th.tensor(norm.cdf(v.double().numpy()))
    fresh = th.tensor(norm.ppf((th.rand(n, 10, generator=gen, dtype=th.float64) * Fv[:, None]).clamp(1e-300, 1).numpy())).float()
    groups = {'before pair': t < lo[:, None], 'between': (t > lo[:, None]) & (t < hi[:, None]), 'after pair': t > hi[:, None],
              'all': th.ones_like(bg)}
    groups.update({f't={s}': t == s for s in range(10)})
    print(f'k={k}: Var(m0)={var:.3f}')
    for g, mask in groups.items():
        sel = mask & bg
        y = th.where(sel, fresh, x)
        st = ((m - widths(y, a, b, W)[0]) ** 2).mean() / (2 * var)
        print(f'   {g:12s} ST={st:.3f}  (mean #inputs {sel.float().sum(1).mean():.2f})')
