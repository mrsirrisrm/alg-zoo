"""Conditioning ceiling vs feature cost (CRITIQUE.md item 3).

For features Z, R2(Z) = Var(E[y|Z]) / Var(y), y = 1{wrong}. Any control variate or
stratification that uses only Z gains at most 1/(1 - R2) over MC; the gain is reached
only if E[g(Z)] is free. Var(E[y|Z]) = Cov(y, y') with y' recomputed after resampling
everything except Z from its exact conditional law (pick-freeze), so no regression.
Families:
  top-k: positions and values of the k largest inputs (input law only, cost ~0); the
         other 10-k are iid N(0,1) truncated below the k-th value.
  prefix-t: x_0..x_{t-1}, i.e. the hidden state h_t (cost t/10 of a forward pass).
Usage: python probes/cond_ceiling.py [n_log2]
"""
import sys; sys.path.insert(0, '.')
import json
import numpy as np, torch as th
from ground_truth import forward, load_weights
from gauss import Phi, Phi_inv

n = 1 << (int(sys.argv[1]) if len(sys.argv) > 1 else 20)
d, s = np.load('noisy_sigma0.005.npz'), np.load('seeds_h16.npz')
models = [('trained', [th.tensor(a) for a in load_weights('weights.npz')])]
models += [(f'noisy{i}', [th.tensor(d[k][i]) for k in ('W_ih', 'W_hh', 'W_out')]) for i in range(8)]
models += [(f'seed{i}', [th.tensor(s[k][i]) for k in ('W_ih', 'W_hh', 'W_out')]) for i in range(5)]
models = [(nm, [w.float() for w in W]) for nm, W in models]
gen = th.Generator().manual_seed(11)
x = th.randn(n, 10, generator=gen)
lab = x.topk(2, -1).indices[:, 1]


def topk_resample(k):
    srt, idx = x.sort(-1, descending=True)
    vk = srt[:, k - 1:k].double()
    u = th.rand(n, 10 - k, generator=gen, dtype=th.float64)
    new = srt.clone()
    new[:, k:] = Phi_inv((u * Phi(vk)).clamp(1e-300, 1)).clamp_min(-12).float()
    return th.empty_like(x).scatter_(1, idx, new)  # rest iid: any order among free slots


def prefix_resample(t):
    y = x.clone(); y[:, t:] = th.randn(n, 10 - t, generator=gen); return y


xs = {f'top{k}': (topk_resample(k), 0.0) for k in range(1, 10)}
xs.update({f'prefix{t}': (prefix_resample(t), t / 10) for t in range(1, 10)})
out = {}
with th.no_grad():
    for nm, W in models:
        y = (forward(x, *W).argmax(-1) != lab).double()
        row = {}
        for f, (x2, cost) in xs.items():
            y2 = (forward(x2, *W).argmax(-1) != x2.topk(2, -1).indices[:, 1]).double()
            m = (y.mean() + y2.mean()) / 2
            row[f] = ((y * y2).mean() - m * m).item() / (m * (1 - m)).item()
        out[nm] = row
        print(nm, f'err {y.mean():.4f}', ' '.join(f'{f}:{r:.3f}' for f, r in row.items()), flush=True)
json.dump(out, open('probes/cond_ceiling.json', 'w'), indent=1)
