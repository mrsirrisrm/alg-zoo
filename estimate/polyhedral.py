"""Step 3: polyhedral integration of the S/R tie width (k = 2), per skeleton.

Model (the explanation under test): after fixing the skeleton (tied pair a < b at value v,
the max M at its position/value), the background u (7 inputs, iid N(0,1) truncated below v)
is split into regions R by the on/off pattern of the kink neurons K over all 10 steps. In R,
the other neurons' gates are frozen to a per-region pattern, so the network is LINEAR in the
input vector x: logits = A_R x, pre-activations pre_t = B_{R,t} x. Then
  - region membership = linear constraints (kink-neuron signs) + box u < v,
  - margin m = (A_a - A_b) x is affine in u, alpha = (A_a - A_b)(e_a - e_b)/2 is constant,
  - 'in pair' is approximated linearly: (L_a + L_b)/2 >= L_j for all j != a, b,
and E[width; R] = E[min(|m|, 0.25 alpha) / alpha ; u in R] is a Gaussian polytope integral.

Integrator (deterministic): constraints are ordered in time; given u_<i, every constraint
whose last variable is u_i bounds u_i to an interval, so u_1..u_6 are integrated by a fixed
Sobol lattice through the interval CDFs (Genz separation of variables) and u_7 by 16-point
Gauss-Legendre on its interval.

Region discovery and periphery gates are fitted from samples here (flagged: not yet
deductive). Validation: per skeleton, compare with Monte Carlo of the same linear model
(tests the integrator) and of the exact network (tests the model).
"""
import argparse
import json
import time

import numpy as np
import torch as th
from scipy.stats import norm

from boundary_layer import sample_ties, tie_density
from gauss import Phi, Phi_inv, _gl
from ground_truth import load_weights

KSET = [0, 4, 5, 6, 7, 8, 9, 11]
WMAX = 0.25


class Net:
    def __init__(self, W):
        self.W_ih, self.W_hh, self.W_out = [th.as_tensor(a).double() for a in W]
        self.K = th.zeros(16, dtype=th.bool); self.K[KSET] = True

    def bits(self, x):
        h = th.zeros(x.shape[0], 16, dtype=x.dtype); out = []
        for t in range(10):
            pre = x[:, t:t + 1] * self.W_ih + h @ self.W_hh.T
            out.append(pre > 0); h = th.relu(pre)
        return th.stack(out, 1), h @ self.W_out.T

    def linear_maps(self, G):
        """G [R, 10, 16] gates -> A [R, 10(logit), 10(input)], B [R, 10(t), 16, 10(input)]."""
        R = G.shape[0]
        H = th.zeros(R, 16, 10, dtype=th.float64)               # dh/dx
        Bs = []
        for t in range(10):
            P = self.W_hh @ H                                     # [R, 16, 10]
            P[:, :, t] += self.W_ih
            Bs.append(P)
            H = G[:, t, :, None].double() * P
        return self.W_out @ H, th.stack(Bs, 1)


def skeleton_draws(xi, v, n, gen):
    bg = xi < v - 1e-9
    Fv = norm.cdf(float(v))
    u = th.rand(n, 10, generator=gen, dtype=th.float64)
    return th.where(bg, th.tensor(norm.ppf((u * Fv).clamp(1e-300, 1).numpy())), xi.expand(n, 10)), bg


def integrate(net, xi, v, a, b, G, n_qmc):
    """Sum over regions of E[min(|m|, WMAX alpha)/alpha ; region, in-pair] and P(region)."""
    bgpos = th.nonzero(xi < v - 1e-9).flatten()               # time-ordered background positions
    nv = len(bgpos)
    A, B = net.linear_maps(G)                                   # [R,10,10], [R,10,16,10]
    R = G.shape[0]
    x0 = th.where(xi < v - 1e-9, th.zeros_like(xi), xi)        # skeleton part of x
    # constraint rows c . x > 0 in terms of (const, coeffs on u)
    sgn = (2 * G[:, :, KSET].double() - 1)                      # [R,10,|K|]
    rows = (sgn[..., None] * B[:, :, KSET, :]).reshape(R, -1, 10)   # kink signs
    others = [j for j in range(10) if j not in (a, b)]
    pair = 0.5 * (A[:, a] + A[:, b])
    rows = th.cat([rows, pair[:, None, :] - A[:, others]], 1)    # in-pair (linearised)
    c0 = rows @ x0                                              # [R, nc]
    cu = rows[:, :, bgpos]                                      # [R, nc, nv]
    m_row = A[:, a] - A[:, b]
    m0, mu = m_row @ x0, m_row[:, bgpos]                        # [R], [R, nv]
    alpha = 0.5 * (m_row[:, a] - m_row[:, b])                   # [R]
    last = th.where(cu.abs() > 1e-12, th.arange(nv).expand_as(cu), th.full_like(cu, -1, dtype=th.long)).amax(-1)
    const_ok = th.where(last < 0, c0 > 0, th.ones_like(c0, dtype=th.bool)).all(-1)   # [R]
    eng = th.quasirandom.SobolEngine(max(nv - 1, 1), scramble=True, seed=3)
    w = eng.draw(n_qmc, dtype=th.float64).clamp(1e-12, 1 - 1e-12)  # [Q, nv-1]
    gx, gw = _gl(16, 'cpu'); gx, gw = (gx + 1) / 2, gw / 2      # GL on [0,1]
    Q = n_qmc
    u = th.zeros(R, Q, nv, dtype=th.float64)
    weight = th.ones(R, Q, dtype=th.float64)
    vv = th.tensor(float(v), dtype=th.float64)
    for i in range(nv):
        act = (last == i)                                       # [R, nc]
        val = c0[:, None, :] + (u[..., :i] @ cu[:, :, :i].transpose(1, 2) if i else 0)  # [R,Q,nc]
        coef = cu[:, None, :, i]                                # [R,1,nc]
        bound = -val / th.where(coef.abs() > 1e-300, coef, th.ones_like(coef))
        lo = th.where(act[:, None] & (coef > 0), bound, th.full_like(bound, -th.inf)).amax(-1)
        hi = th.where(act[:, None] & (coef < 0), bound, th.full_like(bound, th.inf)).amin(-1)
        hi = th.minimum(hi, vv)
        Plo, Phi_ = Phi(lo), Phi(hi)
        mass = (Phi_ - Plo).clamp_min(0)
        weight = weight * mass / Phi(vv)
        if i < nv - 1:
            u[..., i] = Phi_inv((Plo + w[:, i] * mass).clamp(1e-300, 1 - 1e-16)).clamp(-12, 12)
        else:   # last variable: GL over its interval in probability space
            p = Plo[..., None] + gx * mass[..., None]               # [R,Q,16]
            ui = Phi_inv(p.clamp(1e-300, 1 - 1e-16)).clamp(-12, 12)
            mm = m0[:, None, None] + (u[..., :i] @ mu[:, :i, None]) + mu[:, None, None, i] * ui
            al = alpha[:, None, None]
            f = th.where(al > 0, th.minimum(mm.abs(), WMAX * al) / al.clamp_min(1e-300), th.zeros_like(mm))
            ew = (f * gw).sum(-1)
    pw = weight * const_ok[:, None]
    return (pw * ew).mean(1).sum().item(), pw.mean(1).sum().item()


def mc_linear(net, x, a, b, G_of_x):
    """Monte Carlo of the same linear model (gates from the region lookup per sample)."""
    A, _ = net.linear_maps(G_of_x)
    L = (A @ x[:, :, None])[..., 0]
    m = L[:, a] - L[:, b]
    m_row = A[:, a] - A[:, b]
    alpha = 0.5 * (m_row[:, a] - m_row[:, b])
    others = [j for j in range(10) if j not in (a, b)]
    inpair = (0.5 * (L[:, a] + L[:, b]))[:, None] >= L[:, others]
    f = th.where((alpha > 0) & inpair.all(1), th.minimum(m.abs(), WMAX * alpha) / alpha.clamp_min(1e-300), th.zeros_like(m))
    return f


def mc_exact(net, x, a, b, eps=1e-3):
    ar = th.arange(x.shape[0])
    def marg(dd):
        y = x.clone(); y[:, a] += dd / 2; y[:, b] -= dd / 2
        l = net.bits(y)[1]; return l[:, a] - l[:, b], l
    m0, L = marg(0.0)
    al = (marg(eps)[0] - marg(-eps)[0]) / (2 * eps)
    others = [j for j in range(10) if j not in (a, b)]
    inpair = ((0.5 * (L[:, a] + L[:, b]))[:, None] >= L[:, others]).all(1)
    return th.where((al > 0) & inpair, th.minimum(m0.abs(), WMAX * al) / al.clamp_min(1e-300), th.zeros_like(m0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights.npz')
    ap.add_argument('--skeletons', type=int, default=48)
    ap.add_argument('--fit-log2', type=int, default=12)
    ap.add_argument('--qmc', type=int, default=256)
    ap.add_argument('--mc-log2', type=int, default=14)
    ap.add_argument('--cover', type=float, default=0.999)
    args = ap.parse_args()
    net = Net(load_weights(args.weights))
    gen = th.Generator().manual_seed(31)
    k = 2
    sk = sample_ties(k, args.skeletons, gen)[0].double()
    rows = []
    t_poly = 0.0
    for i in range(args.skeletons):
        xi = sk[i]; v = xi.topk(2).values[1]
        pos = th.nonzero((xi - v).abs() < 1e-9).flatten(); a, b = int(pos[0]), int(pos[1])
        # region discovery + periphery gates (fitted from samples: flagged)
        xf, _ = skeleton_draws(xi, v, 1 << args.fit_log2, gen)
        bf = net.bits(xf)[0]
        key = np.packbits(bf[:, :, KSET].reshape(bf.shape[0], -1).numpy(), axis=1)
        uk, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel(); order = np.argsort(-cnt)
        keep = order[: int(np.searchsorted(np.cumsum(cnt[order]) / cnt.sum(), args.cover) + 1)]
        G = []
        for r in keep:
            sel = th.tensor(inv == r)
            g = bf[sel].double().mean(0) > 0.5
            g[:, KSET] = bf[sel][0][:, KSET]                     # kink gates = the region's pattern
            G.append(g)
        G = th.stack(G)
        t0 = time.time()
        e_poly, p_cov = integrate(net, xi, v, a, b, G, args.qmc)
        t_poly += time.time() - t0
        # references on independent draws
        x, _ = skeleton_draws(xi, v, 1 << args.mc_log2, gen)
        bx = net.bits(x)[0]
        kx = np.packbits(bx[:, :, KSET].reshape(bx.shape[0], -1).numpy(), axis=1)
        lut = {bytes(uk[r]): j for j, r in enumerate(keep)}
        idx = th.tensor([lut.get(bytes(r), -1) for r in kx])
        found = idx >= 0
        f_lin = th.zeros(x.shape[0], dtype=th.float64)
        f_lin[found] = mc_linear(net, x[found], a, b, G[idx[found]])
        e_ex = mc_exact(net, x, a, b)
        rows.append(dict(poly=e_poly, cover=p_cov, mc_lin=f_lin.mean().item(), mc_lin_se=f_lin.std().item() / x.shape[0] ** .5,
                         mc_exact=e_ex.mean().item(), mc_exact_se=e_ex.std().item() / x.shape[0] ** .5,
                         found=found.double().mean().item(), regions=len(keep)))
        r = rows[-1]
        print(f"skel {i:2d} (a={a}, b={b}, v={float(v):+.2f}, R={len(keep):4d}): poly {r['poly']:.4f} cover {r['cover']:.3f} | "
              f"MC linear {r['mc_lin']:.4f}±{r['mc_lin_se']:.4f} (found {r['found']:.3f}) | MC exact {r['mc_exact']:.4f}±{r['mc_exact_se']:.4f}", flush=True)
    f = tie_density(k)
    agg = {key: 0.5 * f * np.mean([r[key] for r in rows]) for key in ('poly', 'mc_lin', 'mc_exact')}
    agg.update(cover=np.mean([r['cover'] for r in rows]), regions_median=float(np.median([r['regions'] for r in rows])),
               poly_seconds=t_poly, qmc=args.qmc)
    print(json.dumps(agg, indent=1))


if __name__ == '__main__':
    main()
