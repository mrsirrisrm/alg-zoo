"""Is the tie margin m0 additive over background inputs? (cut-HDMR around an anchor)

Background = the inputs below the tied value v. Anchor: each background input at its
conditional median Phi^-1(Phi(v)/2). First-order model:
  m_add = m(anchor) + sum_t [m(anchor, x_t) - m(anchor)].
If R^2 is high, E|m0| over the background reduces to 1-D integrals per position.
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
    bg = x < v[:, None] - 1e-6               # background mask (below the tie)
    med = th.tensor(norm.ppf(norm.cdf(v.double().numpy()) / 2)).float()
    anc = th.where(bg, med[:, None], x)
    m_full, al_full, inp = widths(x, a, b, W)
    m_anc, al_anc, _ = widths(anc, a, b, W)
    m_add, al_add = m_anc.clone(), al_anc.clone()
    for t in range(10):
        y = anc.clone(); y[:, t] = x[:, t]
        mt, alt, _ = widths(y, a, b, W)
        m_add += th.where(bg[:, t], mt - m_anc, th.zeros_like(mt))
        al_add += th.where(bg[:, t], alt - al_anc, th.zeros_like(alt))
    r2 = lambda p, q: 1 - ((p - q) ** 2).mean() / q.var()
    sgn = 1.0 if k == 2 else -1.0
    def est(m, al):
        good = inp & (sgn * al > 0)
        w = m.abs() / (sgn * al).clamp_min(1e-9)
        return th.where(good, w.clamp_max(0.25), th.zeros_like(w)).mean().item()
    print(f"k={k}: R2(m0)={r2(m_add, m_full):.3f}  R2(alpha)={r2(al_add, al_full):.3f}  "
          f"corr(m)={th.corrcoef(th.stack([m_add, m_full]))[0,1]:.3f}  "
          f"E[w] full={est(m_full, al_full):.5f}  additive={est(m_add, al_full):.5f}  "
          f"additive both={est(m_add, al_add):.5f}  anchor-only={est(m_anc, al_anc):.5f}")
