"""Accuracy of noisy copies vs sigma (justifies the harness's sigma = 0.005)."""
import numpy as np, torch as th
from ground_truth import forward, load_weights
base = load_weights('weights.npz'); rng = np.random.default_rng(1)
th.manual_seed(3); x = th.randn(1 << 21, 10); S = x.topk(2, -1).indices[:, 1]
for sig in [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]:
    accs = []
    for _ in range(8):
        W = [th.tensor(a + sig * np.sqrt((a ** 2).mean()) * rng.standard_normal(a.shape)).float() for a in base]
        accs.append((forward(x, *W).argmax(-1) == S).float().mean().item())
    print(f'sigma {sig:<7} acc mean {np.mean(accs):.4f}  min {np.min(accs):.4f}  max {np.max(accs):.4f}', flush=True)
