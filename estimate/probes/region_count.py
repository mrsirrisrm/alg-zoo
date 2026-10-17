"""How many linear regions does the post-tie computation visit, per skeleton?
For tie skeletons (k=1,2), draw background inputs, record activation bits for steps
t0+1..9 (t0 = first tied position), and report how many distinct patterns cover
90/99/99.9% of the probability mass, plus the share of that mass with an error-relevant
margin sign ambiguity. If small, step 3 (polyhedral integration per pattern) is feasible."""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from scipy.stats import norm
from boundary_layer import sample_ties
from ground_truth import load_weights

W_ih, W_hh, W_out = [th.tensor(a).double() for a in load_weights('weights.npz')]
gen = th.Generator().manual_seed(21)
n_draw = 1 << 14
import os
SUB = [int(j) for j in os.environ.get('SUB', ','.join(map(str, range(16)))).split(',')]
for k in (1, 2):
    sk = sample_ties(k, 48, gen)[0].double()
    covers = []
    for i in range(48):
        xi = sk[i]
        v = xi.max() if k == 1 else xi.topk(2).values[1]
        pos = th.nonzero((xi - v).abs() < 1e-9).flatten(); a0, b0 = int(pos[0]), int(pos[1])
        bgmask = xi < v - 1e-9; Fv = norm.cdf(float(v))
        u = th.rand(n_draw, 10, generator=gen, dtype=th.float64)
        x = th.where(bgmask, th.tensor(norm.ppf((u * Fv).clamp(1e-300, 1).numpy())), xi.expand(n_draw, 10))
        h = th.relu(x[:, 0:1] * W_ih); bits = []
        for t in range(1, 10):
            pre = x[:, t:t + 1] * W_ih + h @ W_hh.T
            if t > a0: bits.append(pre[:, SUB] > 0)
            h = th.relu(pre)
        if not bits:
            continue
        B = th.cat(bits, 1).numpy()
        key = np.packbits(B, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        c = np.sort(cnt)[::-1].cumsum() / n_draw
        covers.append([int(np.searchsorted(c, q) + 1) for q in (.9, .99, .999)] + [len(cnt), 9 - a0])
    c = np.array(covers)
    print(f'k={k} neurons {SUB}: patterns covering 90/99/99.9% of mass (median over skeletons): '
          f'{np.median(c[:, 0]):.0f} / {np.median(c[:, 1]):.0f} / {np.median(c[:, 2]):.0f}; '
          f'distinct in {n_draw} draws median {np.median(c[:, 3]):.0f}, max {c[:, 3].max()}; mean post-tie steps {c[:, 4].mean():.1f}')
    for steps in range(1, 10):
        sel = c[:, 4] == steps
        if sel.any():
            print(f'   {steps} post-tie steps: 99% cover median {np.median(c[sel, 1]):.0f}, distinct median {np.median(c[sel, 3]):.0f} (n={sel.sum()})')
