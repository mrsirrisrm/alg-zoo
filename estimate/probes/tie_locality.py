"""Are errors boundary-layer (near-tie) events? Distance of the chosen competitor's
value from vS, relative to vM."""
import sys; sys.path.insert(0, '.')
import torch as th, numpy as np
from ground_truth import forward, load_weights
W = [th.tensor(a) for a in load_weights('weights.npz')]
th.manual_seed(5); N = 1 << 22; x = th.randn(N, 10)
pred = forward(x, *W).argmax(-1)
top = x.topk(2, -1); S = top.indices[:, 1]; vS = top.values[:, 1]; vM = top.values[:, 0]
err = pred != S
vp = x.gather(1, pred[:, None])[:, 0]
d = ((vS - vp).abs() / vM.abs())[err]
print(f'error rate {err.float().mean():.4f}')
for c in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    print(f'  errors with |v_pred - vS|/|vM| < {c:.2f}: {(d < c).float().mean():.3f}')
rank = th.argsort(th.argsort(-x, 1), 1).gather(1, pred[:, None])[:, 0][err]
print('rank of chosen element among errors (0 = max):', (np.bincount(rank.numpy(), minlength=10) / err.sum().item()).round(3))
neg = (vM <= 0.5)
print(f'error rate when vM <= 0.5: {err[neg].float().mean():.3f} (mass {neg.float().mean():.4f});  when vS < 0: {err[vS < 0].float().mean():.3f} (mass {(vS < 0).float().mean():.4f})')
