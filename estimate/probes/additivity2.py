"""Second-order cut-HDMR of the tie margin over background inputs (anchor: median)."""
import sys
sys.path.insert(0, "..")
import itertools
import torch as th
from scipy.stats import norm
from boundary_layer import sample_ties, widths
from ground_truth import load_weights

W = [th.tensor(a) for a in load_weights("../weights.npz")]
gen = th.Generator().manual_seed(1)
n = 1 << 13
for k in [1, 2]:
    x, a, b = sample_ties(k, n, gen)
    v = x[th.arange(n), a]
    bg = x < v[:, None] - 1e-6
    anc = th.where(bg, th.tensor(norm.ppf(norm.cdf(v.double().numpy()) / 2)).float()[:, None], x)
    m_full = widths(x, a, b, W)[0]
    m0 = widths(anc, a, b, W)[0]
    one = {}
    for t in range(10):
        y = anc.clone(); y[:, t] = x[:, t]
        one[t] = th.where(bg[:, t], widths(y, a, b, W)[0] - m0, th.zeros_like(m0))
    m1 = m0 + sum(one.values())
    m2 = m1.clone()
    for s, t in itertools.combinations(range(10), 2):
        y = anc.clone(); y[:, s] = x[:, s]; y[:, t] = x[:, t]
        both = bg[:, s] & bg[:, t]
        m2 += th.where(both, widths(y, a, b, W)[0] - m0 - one[s] - one[t], th.zeros_like(m0))
    r2 = lambda p: (1 - ((p - m_full) ** 2).mean() / m_full.var()).item()
    print(f"k={k}: R2 first-order {r2(m1):.3f}, second-order {r2(m2):.3f}")
