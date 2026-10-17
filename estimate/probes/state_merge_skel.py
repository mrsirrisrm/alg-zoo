"""Atom-merge with the tie skeleton held exact.

Deductive design: quadrature over the skeleton (positions, tie value v, values above v)
and atoms only for the background. Test: per skeleton, a pool of background draws; merge
each target's normalised state at time t with its nearest pool state (same skeleton, so
the tied values are identical), continue with the target's own inputs.
t = 'first' (just after the first tied element) or 'second' (just after both)."""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from scipy.stats import norm
from boundary_layer import sample_ties, tie_density
from ground_truth import load_weights

W_ih, W_hh, W_out = [th.tensor(a).double() for a in load_weights('weights.npz')]


def run_from(x, t_cut, swap):
    h = th.relu(x[:, 0:1] * W_ih)
    for t in range(x.shape[1]):
        if t > 0:
            h = th.relu(x[:, t:t + 1] * W_ih + h @ W_hh.T)
        if swap is not None and t == t_cut:
            s = x[:, :t + 1].abs().max(1, keepdim=True).values
            h = swap(h / s) * s
    return h @ W_out.T


def bl_terms(x, a, b, k, t_cut=None, swap=None):
    ar = th.arange(x.shape[0]); eps = 1e-3
    def marg(dd):
        y = x.clone(); y[ar, a] += dd / 2; y[ar, b] -= dd / 2
        l = run_from(y, t_cut, swap); return l[ar, a] - l[ar, b], l.argmax(-1)
    m0, am = marg(0.0); al = (marg(eps)[0] - marg(-eps)[0]) / (2 * eps)
    sg = 1.0 if k == 2 else -1.0
    good = ((am == a) | (am == b)) & (sg * al > 0)
    w = m0.abs() / (sg * al).clamp_min(1e-9)
    return th.where(good, w.clamp_max(0.25), th.zeros_like(w))


gen = th.Generator().manual_seed(13)
n_skel, n_tgt = int(sys.argv[1]) if len(sys.argv) > 1 else 96, 128
n_pool = 1 << (int(sys.argv[2]) if len(sys.argv) > 2 else 12)
for k in (1, 2):
    f = tie_density(k)
    sk, _, _ = sample_ties(k, n_skel, gen)
    sk = sk.double()
    # recover skeleton: tied positions and value, positions above v
    out = {'full': [], 'first': {2: [], 4: [], 8: [], 16: []}, 'second': {2: [], 4: [], 8: [], 16: []}}
    for i in range(n_skel):
        xi = sk[i]
        v = xi.max() if k == 1 else xi.topk(2).values[1]
        tied = (xi - v).abs() < 1e-9
        pos = th.nonzero(tied).flatten(); a0, b0 = int(pos[0]), int(pos[1])
        bgmask = xi < v - 1e-9
        Fv = norm.cdf(float(v))
        def draw(m):
            u = th.rand(m, 10, generator=gen, dtype=th.float64)
            bgv = th.tensor(norm.ppf((u * Fv).clamp(1e-300, 1).numpy()))
            return th.where(bgmask, bgv, xi.expand(m, 10))
        xt, xp = draw(n_tgt), draw(n_pool)
        a = th.full((n_tgt,), a0); b = th.full((n_tgt,), b0)
        out['full'].append(bl_terms(xt, a, b, k))
        for which, t in (('first', a0), ('second', b0)):
            if t >= 9:
                for kk in out[which]: out[which][kk].append(None)
                continue
            hp = th.relu(xp[:, 0:1] * W_ih)
            for u in range(1, t + 1):
                hp = th.relu(xp[:, u:u + 1] * W_ih + hp @ W_hh.T)
            zp = hp / xp[:, :t + 1].abs().max(1, keepdim=True).values
            mu = zp.mean(0); Vh = th.linalg.svd(zp - mu, full_matrices=False)[2]
            for kk in out[which]:
                P = Vh[:kk].T; fp = (zp - mu) @ P
                swap = lambda z, P=P, fp=fp: zp[th.cdist((z - mu) @ P, fp).argmin(1)]
                out[which][kk].append(bl_terms(xt, a, b, k, t, swap))
    full = th.stack(out['full'])
    print(f'k={k}: full BL {0.5 * f * full.mean():.5f}')
    for which in ('first', 'second'):
        ok = [j for j, r in enumerate(out[which][2]) if r is not None]
        ref = 0.5 * f * full[ok].mean()
        line = '  '.join(f'{kk}:{0.5 * f * th.stack([out[which][kk][j] for j in ok]).mean():.4f}' for kk in out[which])
        print(f'  merge after {which:6s} tied element ({len(ok)} skeletons): full {ref:.4f} | comps:BL {line}', flush=True)
