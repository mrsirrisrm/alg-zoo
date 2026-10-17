"""Does the tie margin depend on the background only through its records?

Keep the tied pair, everything above v, and background inputs that are running-max
records at their time; replace the other background inputs by an anchor. If R^2 of
m0 is high, the state space for a deductive average is the record chain.
"""
import sys
sys.path.insert(0, "..")
import torch as th
from scipy.stats import norm

from boundary_layer import sample_ties, widths
from ground_truth import load_weights

W = [th.tensor(a) for a in load_weights("../weights.npz")]
gen = th.Generator().manual_seed(1)
n = 1 << 14
for k in [1, 2]:
    x, a, b = sample_ties(k, n, gen)
    ar = th.arange(n)
    v = x[ar, a]
    bg = x < v[:, None] - 1e-6
    rec = x >= th.cummax(x, 1).values - 1e-7
    m_full, al_full, inp = widths(x, a, b, W)
    r2 = lambda p, q: (1 - ((p - q) ** 2).mean() / q.var()).item()
    for name, fill in [("median", th.tensor(norm.ppf(norm.cdf(v.double().numpy()) / 2)).float()[:, None]),
                       ("-1", -th.ones(n, 1)), ("0", th.zeros(n, 1)), ("-3", -3 * th.ones(n, 1))]:
        for keep_rec in [False, True]:
            drop = bg & ~rec if keep_rec else bg
            y = th.where(drop, fill.expand_as(x), x)
            m, al, _ = widths(y, a, b, W)
            print(f"k={k} fill={name:6s} keep_records={keep_rec}: R2(m0)={r2(m, m_full):.3f} R2(alpha)={r2(al, al_full):.3f}")
