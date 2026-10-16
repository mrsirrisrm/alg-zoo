"""Boundary-layer estimate of tie-confusion error rates.

For a tie between order statistics k and k+1 (k=1: M/S, k=2: S/R), the first-order
error mass between the true boundary {x_a = x_b} and the network boundary {m = 0}
(m = logit_a - logit_b) is

    err_k  ≈  ½ f_Dk(0) * E_tie[ |m0| / |alpha| ],

(½: f_D counts both orderings of the tied pair; the error region lies on one side),

f_Dk(0): density of X_(k) - X_(k+1) at 0 (analytic, 1-D quadrature),
m0: network margin at the exact tie, alpha = dm/dd across the tie (d = x_a - x_b).
Correct behaviour has sign(alpha) = +1 for k=2 (the higher of a, b should win) and
-1 for k=1 (both tied at the top: the lower one is the 2nd max).
Only ties where the network's argmax is a or b count (others are not boundary-layer).

This version averages the width over tie configurations drawn from the exact
tie-conditional law (a validation of the formula; the average is the part a
mechanistic model must later supply).
"""
import argparse
import json

import numpy as np
import torch as th
from scipy.integrate import quad
from scipy.stats import norm

from ground_truth import forward, load_weights

T = 10


def tie_density(k):
    """f_D(0) for D = X_(k) - X_(k+1), n = 10 iid N(0,1). k-1 above, 9-k below."""
    above, below = k - 1, 9 - k
    from math import factorial
    c = factorial(10) / (factorial(above) * factorial(below))
    g = lambda v: norm.pdf(v) ** 2 * norm.sf(v) ** above * norm.cdf(v) ** below
    return c * quad(g, -10, 10, limit=200)[0]


def sample_ties(k, n, gen):
    """Exact draws from the tie-conditional law: tied value v, positions a, b."""
    above, below = k - 1, 9 - k
    grid = np.linspace(-6, 6, 200001)
    dens = norm.pdf(grid) ** 2 * norm.sf(grid) ** above * norm.cdf(grid) ** below
    cdf = np.cumsum(dens); cdf /= cdf[-1]
    u = th.rand(n, generator=gen, dtype=th.float64).numpy()
    v = th.tensor(np.interp(u, cdf, grid))
    perm = th.argsort(th.rand(n, T, generator=gen), 1)
    a, b = perm[:, 0], perm[:, 1]
    up = perm[:, 2:2 + above]
    Fv = th.tensor(norm.cdf(v.numpy()))
    uu = th.rand(n, T, generator=gen, dtype=th.float64)
    lo = th.tensor(norm.ppf((uu * Fv[:, None]).clamp(1e-300, 1).numpy()))           # below v
    hi = th.tensor(norm.ppf((Fv[:, None] + uu * (1 - Fv[:, None])).clamp(0, 1 - 1e-16).numpy()))  # above v
    x = lo.clone()
    if above:
        x.scatter_(1, up, hi.gather(1, up))
    ar = th.arange(n)
    x[ar, a] = v; x[ar, b] = v
    return x.float(), a, b


def widths(x, a, b, W, eps=1e-3):
    ar = th.arange(x.shape[0])
    def margin(dd):
        y = x.clone(); y[ar, a] += dd / 2; y[ar, b] -= dd / 2
        l = forward(y, *W)
        return l[ar, a] - l[ar, b], l.argmax(-1)
    m0, am = margin(0.0)
    mp, _ = margin(eps); mn, _ = margin(-eps)
    alpha = (mp - mn) / (2 * eps)
    inpair = (am == a) | (am == b)
    return m0, alpha, inpair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights.npz")
    ap.add_argument("--n", type=int, default=1 << 18)
    ap.add_argument("--wmax", type=float, default=0.25, help="cap on |m0/alpha| (beyond: not boundary-layer)")
    args = ap.parse_args()
    W = [th.tensor(a) for a in load_weights(args.weights)]
    gen = th.Generator().manual_seed(0)
    out = {}
    for k in [1, 2]:
        x, a, b = sample_ties(k, args.n, gen)
        m0, alpha, inpair = widths(x, a, b, W)
        sgn = 1.0 if k == 2 else -1.0
        w = m0.abs() / (sgn * alpha).clamp_min(1e-9)
        good = inpair & (sgn * alpha > 0)
        wc = th.where(good, w.clamp_max(args.wmax), th.zeros_like(w))
        f = tie_density(k)
        out[k] = dict(f_D0=f, est=0.5 * f * wc.mean().item(), frac_inpair=inpair.float().mean().item(),
                      frac_alpha_correct_sign=(sgn * alpha > 0).float().mean().item(),
                      frac_capped=(good & (w > args.wmax)).float().mean().item(),
                      median_width=w[good].median().item(), mean_width=wc[good].mean().item())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
