"""Effective dimension of the memory state for the tie margin.

At time t, normalise the state z_t = h_t / s_t (s_t = max_{<=t} |x|: homogeneity makes
the future depend on z_t and on the remaining inputs in units of s_t). Project z_t onto
its top-k principal components (fitted per t over tie configurations), continue the RNN
from the compressed state with the same remaining inputs, and recompute the
boundary-layer quantities (m0, alpha). Report, per k and t:
  - BL estimate from compressed states vs from the full state,
  - rms of the margin error |dm0/alpha| relative to the median tie width.
If a few components reproduce the BL estimate, an atom-and-merge propagation of z on a
low-dimensional feature space is feasible.
"""
import sys; sys.path.insert(0, '.')
import json
import torch as th
from boundary_layer import sample_ties, tie_density
from ground_truth import load_weights

W_ih, W_hh, W_out = [th.tensor(a).double() for a in load_weights('weights.npz')]


def run(x, t_cut=None, P=None, mu=None):
    """Forward; if t_cut given, compress h at t_cut: z = h/s -> mu + P P^T (z - mu)."""
    h = th.relu(x[:, 0:1] * W_ih)
    for t in range(x.shape[1]):
        if t > 0:
            h = th.relu(x[:, t:t + 1] * W_ih + h @ W_hh.T)
        if t == t_cut:
            s = x[:, :t + 1].abs().max(1, keepdim=True).values
            z = h / s
            z = mu + (z - mu) @ P @ P.T
            h = z.clamp_min(0) * s
    return h @ W_out.T


def states(x, t):
    h = th.relu(x[:, 0:1] * W_ih)
    for u in range(1, t + 1):
        h = th.relu(x[:, u:u + 1] * W_ih + h @ W_hh.T)
    return h / x[:, :t + 1].abs().max(1, keepdim=True).values


def bl(x, a, b, k, f, **kw):
    ar = th.arange(x.shape[0]); eps = 1e-3
    def marg(dd):
        y = x.clone(); y[ar, a] += dd / 2; y[ar, b] -= dd / 2
        l = run(y, **kw); return l[ar, a] - l[ar, b], l.argmax(-1)
    m0, am = marg(0.0); al = (marg(eps)[0] - marg(-eps)[0]) / (2 * eps)
    sg = 1.0 if k == 2 else -1.0
    good = ((am == a) | (am == b)) & (sg * al > 0)
    w = m0.abs() / (sg * al).clamp_min(1e-9)
    est = 0.5 * f * th.where(good, w.clamp_max(0.25), th.zeros_like(w)).mean().item()
    return est, m0, al, w[good].median().item()


gen = th.Generator().manual_seed(11)
n = 1 << 15
res = []
for k in (1, 2):
    x, a, b = sample_ties(k, n, gen); x = x.double()
    f = tie_density(k)
    e_full, m_full, al_full, wmed = bl(x, a, b, k, f)
    print(f'k={k}: full-state BL {e_full:.5f}, median width {wmed:.4f}')
    for t in (2, 4, 6, 8):
        z = states(x, t); mu = z.mean(0)
        U, S, Vh = th.linalg.svd(z - mu, full_matrices=False)
        ev = (S ** 2).cumsum(0) / (S ** 2).sum()
        line = []
        for kk in (1, 2, 3, 4, 6, 8, 12, 16):
            P = Vh[:kk].T
            e, m, al, _ = bl(x, a, b, k, f, t_cut=t, P=P, mu=mu)
            rel = ((m - m_full) / al_full.abs().clamp_min(1e-6)).pow(2).mean().sqrt().item() / wmed
            line.append(f'{kk}:{e:.4f}/{rel:.1f}')
            res.append(dict(k=k, t=t, comps=kk, var_expl=ev[kk - 1].item(), bl=e, bl_full=e_full, rms_err_over_width=rel))
        print(f'  t={t} var95 at {int((ev < .95).sum()) + 1} comps, var99 at {int((ev < .99).sum()) + 1} | comps:BL/rms(dm/alpha)/width  ' + '  '.join(line), flush=True)
json.dump(res, open('probes/state_dim.json', 'w'), indent=1)
