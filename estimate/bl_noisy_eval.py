"""Run the boundary-layer estimator unchanged on the noisy-model harness."""
import json

import numpy as np
import torch as th

from boundary_layer import sample_ties, tie_density, widths

d = np.load("noisy_sigma0.005.npz")
n = 1 << 16
f = {k: tie_density(k) for k in (1, 2)}
rows = []
for i in range(len(d["acc"])):
    W = [th.tensor(d[k][i]) for k in ("W_ih", "W_hh", "W_out")]
    gen = th.Generator().manual_seed(100 + i)
    bl = 0.0
    for k in (1, 2):
        x, a, b = sample_ties(k, n, gen)
        m0, al, ip = widths(x, a, b, W)
        sg = 1.0 if k == 2 else -1.0
        w = m0.abs() / (sg * al).clamp_min(1e-9)
        good = ip & (sg * al > 0)
        bl += 0.5 * f[k] * th.where(good, w.clamp_max(0.25), th.zeros_like(w)).mean().item()
    rows.append((1 - d["acc"][i], bl))
    print(f"model {i}: true err {1 - d['acc'][i]:.4f}  BL tie err {bl:.4f}  residual {1 - d['acc'][i] - bl:+.4f}", flush=True)
r = np.array(rows)
print(json.dumps(dict(corr=float(np.corrcoef(r[:, 0], r[:, 1])[0, 1]),
                      mean_true=float(r[:, 0].mean()), mean_bl=float(r[:, 1].mean()),
                      resid_mean=float((r[:, 0] - r[:, 1]).mean()), resid_std=float((r[:, 0] - r[:, 1]).std()),
                      true_err_std=float(r[:, 0].std())), indent=1))
np.save("bl_noisy_eval.npy", r)
