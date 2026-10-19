"""D1 (PLAN.md): the missing family-table cells. Same lenses as the harness models on the
other T = 10 models: zoo h8 (best seed), zoo h32 (best seed), hand10.
  (ii)  FORM mean-background skeleton level (form_abs_all.deduced at 5x3 and 15x7) vs the
        actual S/R + M/S error and the total error;
  (iii) surprise accounting of the correctness bit (surprise.run);
  (iv)  background-gradient / alpha ratio and active-subspace effective dim (active_subspace.run).
Run from estimate/: python probes/family.py [h8 h32 hand10]
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from alg_zoo import zoo_2nd_argmax, handcrafted
from ground_truth import forward
from ablation import sd_weights
from form_abs_all import deduced
import surprise, active_subspace

N = 1 << 22
names = sys.argv[1:] or ['h8', 'h32', 'hand10']
loaders = {'h8': lambda: sd_weights(zoo_2nd_argmax(8, 10)), 'h32': lambda: sd_weights(zoo_2nd_argmax(32, 10)),
           'hand10': lambda: sd_weights(handcrafted.handcrafted_2nd_argmax_10())}
for name in names:
    W = [w.numpy() for w in loaders[name]()]
    Wt = [th.tensor(a, dtype=th.float32) for a in W]
    x = th.randn(N, 10, generator=th.Generator().manual_seed(3)); t = x.topk(3, -1).indices
    with th.no_grad():
        out = forward(x, *Wt).argmax(-1)
    tot = (out != t[:, 1]).float().mean().item(); srms = ((out == t[:, 2]) | (out == t[:, 0])).float().mean().item()
    print(f"\n==== {name}: h {W[1].shape[0]}, total error {tot:.4f}, S/R+M/S {srms:.4f}, predicts R {(out == t[:, 2]).float().mean():.4f}, predicts M {(out == t[:, 0]).float().mean():.4f}", flush=True)
    try:
        (d1, c1), (d2, _) = deduced(Wt, 5, 3), deduced(Wt, 15, 7)
        print(f"FORM mean-background skeleton: deduced 5x3 {d1:.4f}, 15x7 {d2:.4f}; rel err vs S/R+M/S {d2 / max(srms, 1e-9) - 1:+.1%}, vs total {d2 / tot - 1:+.1%}; MC sd at 5x3 cost {np.sqrt(tot * (1 - tot) / c1):.4f}", flush=True)
    except Exception as e:
        print(f"FORM failed: {e!r}")
    try:
        surprise.run(name, W)
    except Exception as e:
        print(f"surprise failed: {e!r}")
    try:
        active_subspace.run(name, W)
    except Exception as e:
        print(f"active subspace failed: {e!r}")
