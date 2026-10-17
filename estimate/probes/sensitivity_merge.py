"""Sensitivity propagation after the first tied element.

Merge each target's normalised state z at t = (first tied position) with its nearest pool
atom z* (same skeleton), but correct the margin to first order along the target's own
future: m(z) ~ m(z*) + J(z*; future) . (z - z*), J by forward-mode AD (jvp).
If this restores the tie estimate, an atom needs only (state, offset moments) and the
post-tie dynamics can be carried as a linear response instead of a precise state.
Also reports the second-order version (jvp of jvp along the same offset)."""
import sys; sys.path.insert(0, '.')
import torch as th
from torch.func import jvp
from scipy.stats import norm
from boundary_layer import sample_ties, tie_density
from ground_truth import load_weights

W_ih, W_hh, W_out = [th.tensor(a).double() for a in load_weights('weights.npz')]


def head(x, t):
    h = th.relu(x[:, 0:1] * W_ih)
    for u in range(1, t + 1):
        h = th.relu(x[:, u:u + 1] * W_ih + h @ W_hh.T)
    s = x[:, :t + 1].abs().max(1, keepdim=True).values
    return h / s, s


def tail(z, s, x, t):
    h = z * s
    for u in range(t + 1, x.shape[1]):
        h = th.relu(x[:, u:u + 1] * W_ih + h @ W_hh.T)
    return h @ W_out.T


def bl_terms(m0, am, al, a, b, k):
    sg = 1.0 if k == 2 else -1.0
    good = ((am == a) | (am == b)) & (sg * al > 0)
    w = m0.abs() / (sg * al).clamp_min(1e-9)
    return th.where(good, w.clamp_max(0.25), th.zeros_like(w))


gen = th.Generator().manual_seed(int(sys.argv[3]) if len(sys.argv) > 3 else 13)
n_skel, n_tgt = int(sys.argv[1]) if len(sys.argv) > 1 else 96, 128
n_pool = 1 << (int(sys.argv[2]) if len(sys.argv) > 2 else 12)
eps = 1e-3
for k in (1, 2):
    f = tie_density(k)
    sk = sample_ties(k, n_skel, gen)[0].double()
    acc = {'full': [], 'merge': [], 'lin': [], 'quad': []}
    for i in range(n_skel):
        xi = sk[i]
        v = xi.max() if k == 1 else xi.topk(2).values[1]
        pos = th.nonzero((xi - v).abs() < 1e-9).flatten(); a0, b0 = int(pos[0]), int(pos[1])
        t = a0                                    # first tied element (a0 < b0)
        if t >= 9:
            continue
        bgmask = xi < v - 1e-9; Fv = norm.cdf(float(v))
        def draw(m):
            u = th.rand(m, 10, generator=gen, dtype=th.float64)
            return th.where(bgmask, th.tensor(norm.ppf((u * Fv).clamp(1e-300, 1).numpy())), xi.expand(m, 10))
        xt, xp = draw(n_tgt), draw(n_pool)
        zp, _ = head(xp, t)
        a = th.full((n_tgt,), a0); b = th.full((n_tgt,), b0); ar = th.arange(n_tgt)
        z0, _ = head(xt, t)
        nb = zp[th.cdist(z0, zp).argmin(1)]
        res = {key: [] for key in acc}
        for dd in (0.0, eps, -eps):
            y = xt.clone(); y[ar, a] += dd / 2; y[ar, b] -= dd / 2
            z, s = head(y, t)
            marg = lambda zz: (lambda l: l[ar, a] - l[ar, b])(tail(zz, s, y, t))
            m_full = marg(z)
            am_full = tail(z, s, y, t).argmax(-1)
            m_nb, dm = jvp(marg, (nb,), (z - nb,))
            d2 = jvp(lambda zz: jvp(marg, (zz,), (z - nb,))[1], (nb,), (z - nb,))[1]
            res['full'].append((m_full, am_full))
            res['merge'].append((m_nb, am_full))
            res['lin'].append((m_nb + dm, am_full))
            res['quad'].append((m_nb + dm + 0.5 * d2, am_full))
        for key in acc:
            (m0, am), (mp, _), (mn, _) = res[key]
            acc[key].append(bl_terms(m0, am, (mp - mn) / (2 * eps), a, b, k))
    out = {key: 0.5 * f * th.cat(v).mean().item() for key, v in acc.items()}
    print(f'k={k} ({len(acc["full"])} skeletons, pool {n_pool}): ' + '  '.join(f'{key} {v:.4f}' for key, v in out.items()), flush=True)
