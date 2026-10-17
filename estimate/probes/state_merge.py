"""Atom-merge feasibility: replace each configuration's normalised state z_t by its
nearest neighbour's (a real reachable state, so the cone is respected) in a k-dim PCA
feature space, from a pool of N atoms, then continue with the configuration's own
remaining inputs. This is what merging atoms at resolution ~ pool spacing does.
Compare the boundary-layer estimate with the full-state one, on configurations whose
tied pair comes after t (otherwise the tie itself, which needs ~1e-3 precision, is
inside the merged state)."""
import sys; sys.path.insert(0, '.')
import torch as th
from boundary_layer import sample_ties, tie_density
from ground_truth import load_weights

W_ih, W_hh, W_out = [th.tensor(a).double() for a in load_weights('weights.npz')]


def run(x, t_cut=None, swap=None):
    h = th.relu(x[:, 0:1] * W_ih)
    for t in range(x.shape[1]):
        if t > 0:
            h = th.relu(x[:, t:t + 1] * W_ih + h @ W_hh.T)
        if t == t_cut:
            s = x[:, :t + 1].abs().max(1, keepdim=True).values
            h = swap(h / s) * s
    return h @ W_out.T


def bl(x, a, b, k, f, **kw):
    ar = th.arange(x.shape[0]); eps = 1e-3
    def marg(dd):
        y = x.clone(); y[ar, a] += dd / 2; y[ar, b] -= dd / 2
        l = run(y, **kw); return l[ar, a] - l[ar, b], l.argmax(-1)
    m0, am = marg(0.0); al = (marg(eps)[0] - marg(-eps)[0]) / (2 * eps)
    sg = 1.0 if k == 2 else -1.0
    good = ((am == a) | (am == b)) & (sg * al > 0)
    w = m0.abs() / (sg * al).clamp_min(1e-9)
    return 0.5 * f * th.where(good, w.clamp_max(0.25), th.zeros_like(w)).mean().item()


gen = th.Generator().manual_seed(12)
n, pool_n = 1 << 13, 1 << 16
for k in (1, 2):
    x, a, b = sample_ties(k, n, gen); x = x.double()
    xp, _, _ = sample_ties(k, pool_n, gen); xp = xp.double()
    f = tie_density(k)
    full = bl(x, a, b, k, f)
    print(f'k={k}: full-state BL {full:.5f}')
    for t in (1, 2, 4, 6):
        hp = th.relu(xp[:, 0:1] * W_ih)
        for u in range(1, t + 1):
            hp = th.relu(xp[:, u:u + 1] * W_ih + hp @ W_hh.T)
        zp = hp / xp[:, :t + 1].abs().max(1, keepdim=True).values
        mu = zp.mean(0); Vh = th.linalg.svd(zp - mu, full_matrices=False)[2]
        sub = th.minimum(a, b) > t
        xs, as_, bs = x[sub], a[sub], b[sub]
        full_t = bl(xs, as_, bs, k, f)
        line = [f'full:{full_t:.4f} (n={int(sub.sum())})']
        for kk in (2, 4, 8, 16):
            P = Vh[:kk].T; fp = (zp - mu) @ P
            def swap(z, P=P, fp=fp):
                q = (z - mu) @ P
                idx = th.cat([th.cdist(c, fp).argmin(1) for c in q.split(2048)])
                return zp[idx]
            line.append(f'{kk}:{bl(xs, as_, bs, k, f, t_cut=t, swap=swap):.4f}')
        print(f'  t={t} pool {pool_n}: comps:BL  ' + '  '.join(line), flush=True)
