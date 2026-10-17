"""Which ReLU switches break the linear response after the first tied element?
Same setup as sensitivity_merge (unperturbed runs). For target z and neighbour z*,
compare future activation bits (neuron, time) along the target's own future inputs;
attribute the linearised margin error |m_lin - m_true| to the switched neurons."""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from torch.func import jvp
from scipy.stats import norm
from boundary_layer import sample_ties
from ground_truth import load_weights

W_ih, W_hh, W_out = [th.tensor(a).double() for a in load_weights('weights.npz')]


def head(x, t):
    h = th.relu(x[:, 0:1] * W_ih)
    for u in range(1, t + 1):
        h = th.relu(x[:, u:u + 1] * W_ih + h @ W_hh.T)
    s = x[:, :t + 1].abs().max(1, keepdim=True).values
    return h / s, s


def tail_bits(z, s, x, t):
    h = z * s; bits = th.zeros(x.shape[0], 10, 16, dtype=th.bool)
    for u in range(t + 1, 10):
        pre = x[:, u:u + 1] * W_ih + h @ W_hh.T
        bits[:, u] = pre > 0; h = th.relu(pre)
    return h @ W_out.T, bits


gen = th.Generator().manual_seed(13)
for k in (1, 2):
    sk = sample_ties(k, 96, gen)[0].double()
    rows = []   # (abs err, flip bits [10,16], rel time of second tied elem)
    for i in range(96):
        xi = sk[i]
        v = xi.max() if k == 1 else xi.topk(2).values[1]
        pos = th.nonzero((xi - v).abs() < 1e-9).flatten(); a0, b0 = int(pos[0]), int(pos[1])
        if a0 >= 9: continue
        bgmask = xi < v - 1e-9; Fv = norm.cdf(float(v))
        def draw(m):
            u = th.rand(m, 10, generator=gen, dtype=th.float64)
            return th.where(bgmask, th.tensor(norm.ppf((u * Fv).clamp(1e-300, 1).numpy())), xi.expand(m, 10))
        xt, xp = draw(128), draw(4096)
        zp, _ = head(xp, a0); z, s = head(xt, a0)
        nb = zp[th.cdist(z, zp).argmin(1)]
        ar = th.arange(128)
        marg = lambda zz: (lambda l: l[ar, a0] - l[ar, b0])(tail_bits(zz, s, xt, a0)[0])
        m_true = marg(z); m_nb, dm = jvp(marg, (nb,), (z - nb,))
        err = (m_nb + dm - m_true).abs()
        bt = tail_bits(z, s, xt, a0)[1]; bn = tail_bits(nb, s, xt, a0)[1]
        flips = bt ^ bn
        # re-index time relative to the second tied element
        rel = th.zeros(128, 19, 16, dtype=th.bool)
        rel[:, 9 - b0:19 - b0] = flips
        rows.append((err, flips, rel, (bt != bn).any(2).any(1)))
    err = th.cat([r[0] for r in rows]); fl = th.cat([r[1] for r in rows]); rel = th.cat([r[2] for r in rows])
    anyf = fl.any(2).any(1)
    print(f'k={k}: targets {len(err)}, any switch {anyf.float().mean():.2f}; share of total |err| from targets with a switch {err[anyf].sum() / err.sum():.3f}')
    # attribute: error-weighted switch frequency per neuron (any time), and per time rel. to 2nd tied elem
    wn = (err[:, None] * fl.any(1).double()).sum(0) / err.sum()
    print('  neuron   ' + ' '.join(f'n{j:<4d}' for j in range(16)))
    print('  err-wt   ' + ' '.join(f'{w:.2f} ' for w in wn.tolist()))
    print('  freq     ' + ' '.join(f'{w:.2f} ' for w in fl.any(1).double().mean(0).tolist()))
    wt = (err[:, None] * rel.any(2).double()).sum(0) / err.sum()
    print('  err-wt by time rel. to 2nd tied (-9..+9): ' + ' '.join(f'{w:.2f}' for w in wt.tolist()))
    # only-one-neuron switched cases
    single = fl.any(1).sum(1) == 1
    print(f'  single-neuron switches: {single.float().mean():.2f} of targets, {err[single].sum() / err.sum():.2f} of error')
