"""Control for probes/strat_tie.py: the same label-free Neyman allocation, but with stratum
error rates predicted by a network-free blurred comparator (2nd argmax of x + d*|vM|*eps,
d = 0.012 fixed, from probes/ablation.py) instead of the tie-layer explanation. If this
matches the tie-predicted gain, the gain is task knowledge, not mechanism."""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import forward
from strat_tie import N, models, xa, xb, FB, KB, var

g = th.Generator().manual_seed(7)
yb = xb + 0.012 * xb.abs().max(-1, keepdim=True).values * th.randn(xb.shape, generator=g)
pb = (yb.topk(2, -1).indices[:, 1] != FB['label']).numpy().astype(float)   # blur errors, no network
with th.no_grad():
    for name, W in models:
        W = [th.tensor(a, dtype=th.float32) for a in W]
        eb = (forward(xb, *W).argmax(-1) != FB['label']).numpy().astype(float); e = eb.mean()
        out = [f"{name:8s}"]
        for k in KB:
            K = int(KB[k].max()) + 1
            cb, rb = np.bincount(KB[k], minlength=K), np.bincount(KB[k], eb, minlength=K)
            p = cb / N; qb = rb / np.maximum(cb, 1); sb = np.sqrt(qb * (1 - qb))
            ph = np.bincount(KB[k], pb, minlength=K) / np.maximum(cb, 1)
            out.append(f"| {k}: blur-predicted {e * (1 - e) / var(p, sb, np.sqrt(ph * (1 - ph))):4.2f}")
        print(' '.join(out), flush=True)
