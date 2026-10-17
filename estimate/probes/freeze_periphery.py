"""Can the non-kink neurons' gating be frozen after the first tied element?
Kink set K = {0,4,6,7,9,11} keeps exact ReLU. For the others, after t0 the gate is fixed:
  'skel'  : per (skeleton, time, neuron) majority gate over background draws,
  'kpat'  : majority gate within the same kink-pattern class (per skeleton),
and pre-activations of frozen neurons pass linearly (gate 1) or are zeroed (gate 0).
Within a kink region the network is then affine in the inputs, so step 3 reduces to a few
polyhedral Gaussian integrals per skeleton. Compare BL tie estimates with the exact net."""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from scipy.stats import norm
from boundary_layer import sample_ties, tie_density
from ground_truth import load_weights

W_ih, W_hh, W_out = [th.tensor(a).double() for a in load_weights('weights.npz')]
import os
K = th.zeros(16, dtype=th.bool); K[[int(j) for j in os.environ.get("KSET", "0,4,6,7,9,11").split(",")]] = True


def run(x, t0, gate=None):
    """gate: [B, 10, 16] bool for frozen neurons (used where ~K, t > t0); returns logits, bits."""
    h = th.relu(x[:, 0:1] * W_ih); bits = th.zeros(x.shape[0], 10, 16, dtype=th.bool)
    bits[:, 0] = (x[:, 0:1] * W_ih) > 0
    for t in range(1, 10):
        pre = x[:, t:t + 1] * W_ih + h @ W_hh.T
        bits[:, t] = pre > 0
        if gate is not None and t > t0:
            g = th.where(K, pre > 0, gate[:, t])
            h = pre * g
        else:
            h = th.relu(pre)
    return h @ W_out.T, bits


def bl(x, a, b, k, t0, gate=None):
    ar = th.arange(x.shape[0]); eps = 1e-3
    def marg(dd):
        y = x.clone(); y[ar, a] += dd / 2; y[ar, b] -= dd / 2
        l = run(y, t0, gate)[0]; return l[ar, a] - l[ar, b], l.argmax(-1)
    m0, am = marg(0.0); al = (marg(eps)[0] - marg(-eps)[0]) / (2 * eps)
    sg = 1.0 if k == 2 else -1.0
    good = ((am == a) | (am == b)) & (sg * al > 0)
    w = m0.abs() / (sg * al).clamp_min(1e-9)
    return th.where(good, w.clamp_max(0.25), th.zeros_like(w))


gen = th.Generator().manual_seed(22)
n_draw = 1 << 12
for k in (1, 2):
    f = tie_density(k)
    sk = sample_ties(k, 96, gen)[0].double()
    acc = {'exact': [], 'kpat': []}
    for i in range(96):
        xi = sk[i]
        v = xi.max() if k == 1 else xi.topk(2).values[1]
        pos = th.nonzero((xi - v).abs() < 1e-9).flatten(); a0, b0 = int(pos[0]), int(pos[1])
        bgmask = xi < v - 1e-9; Fv = norm.cdf(float(v))
        u = th.rand(n_draw, 10, generator=gen, dtype=th.float64)
        x = th.where(bgmask, th.tensor(norm.ppf((u * Fv).clamp(1e-300, 1).numpy())), xi.expand(n_draw, 10))
        a = th.full((n_draw,), a0); b = th.full((n_draw,), b0)
        # fit gates on an independent draw set (out of sample); unseen classes -> skeleton majority
        u2 = th.rand(n_draw, 10, generator=gen, dtype=th.float64)
        xf = th.where(bgmask, th.tensor(norm.ppf((u2 * Fv).clamp(1e-300, 1).numpy())), xi.expand(n_draw, 10))
        bf = run(xf, a0)[1]; bits = run(x, a0)[1]
        g_skel = bf.double().mean(0) > 0.5
        keys = lambda B: [bytes(r) for r in np.packbits(B[:, :, K].reshape(B.shape[0], -1).numpy(), axis=1)]
        table = {}
        for kk, row in zip(keys(bf), bf.double()):
            sm, c = table.get(kk, (0, 0)); table[kk] = (sm + row, c + 1)
        g_kpat = th.stack([(table[kk][0] / table[kk][1] > 0.5) if kk in table else g_skel for kk in keys(bits)])
        seen = np.mean([kk in table for kk in keys(bits)])
        print(f'    skel {i}: seen {seen:.2f}', end='\r')
        acc['exact'].append(bl(x, a, b, k, a0))

        acc['kpat'].append(bl(x, a, b, k, a0, g_kpat))
    print(f'k={k}: ' + '  '.join(f'{key} {0.5 * f * th.cat(v).mean():.4f}' for key, v in acc.items()), flush=True)
