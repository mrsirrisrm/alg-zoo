"""Where does the noisy models' extra error come from? Error budget by mechanism
(fractions of all inputs) for the trained model and the noisy copies, plus the
near-tie share (gap/vM < 0.02, 0.1)."""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from ground_truth import forward, load_weights
d = np.load('noisy_sigma0.005.npz')
models = [('trained', [th.tensor(a) for a in load_weights('weights.npz')])]
models += [(f'noisy{i}', [th.tensor(d[k][i]) for k in ('W_ih', 'W_hh', 'W_out')]) for i in range(len(d['acc']))]
th.manual_seed(4); N = 1 << 21; x = th.randn(N, 10)
top = x.topk(3, -1); vM, vS, vR = top.values.T; M, S, R = top.indices.T
tt = th.arange(10).expand(N, 10)
xb = th.where(tt < M[:, None], x, -th.inf); xa = th.where(tt > M[:, None], x, -th.inf)
A = xb.argmax(1); B = xa.argmax(1); both = (M > 0) & (M < 9)
other = th.where(S == A, B, A)
g1 = (vM - vS) / vM.abs(); g2 = (vS - vR) / vM.abs()
print(f"{'model':8s} err    predM  predR  ABswap within  vS<0   tie.02 tie.1  far")
rows = []
for name, W in models:
    p = forward(x, *W).argmax(-1); e = p != S
    c = [e.float().mean(), (e & (p == M)).float().mean(), (e & (p == R)).float().mean(),
         (e & both & (p == other) & (p != R)).float().mean(),
         (e & (p != M) & (p != R) & ~(both & (p == other))).float().mean(),
         (e & (vS < 0)).float().mean(),
         (e & ((g1 < .02) | (g2 < .02))).float().mean(), (e & ((g1 < .1) | (g2 < .1))).float().mean(),
         (e & (g1 >= .1) & (g2 >= .1)).float().mean()]
    c = [v.item() for v in c]; rows.append(c)
    print(f"{name:8s} " + " ".join(f"{v:.4f}" for v in c))
r = np.array(rows); np.save("probes/noisy_residual.npy", r)
print('noisy mean ' + " ".join(f"{v:.4f}" for v in r[1:].mean(0)))
print('corr with err over noisy: ' + " ".join(f"{np.corrcoef(r[1:, 0], r[1:, j])[0, 1]:+.2f}" for j in range(1, r.shape[1])))
