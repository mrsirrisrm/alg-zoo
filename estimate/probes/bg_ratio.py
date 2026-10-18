"""Does the mean-background skeleton's accuracy track background sensitivity? Hypothesis:
training suppresses background inputs' effect on the margin (they widen the tie strip),
which is why mean-background estimates work; then estimate quality should follow the
background-gradient / alpha ratio, not training status. Per harness model: the ratio
(rms wall gradient / rms alpha at sampled ties, probes/active_subspace.grads) per tie
kind, the deduced vs actual S/R and M/S error (probes/form_abs machinery, 5x3 nodes), and
the correlation of |relative estimate error| with the ratio. Ten models: suggestive only.
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from scipy.stats import spearmanr, pearsonr
import active_subspace
from active_subspace import grads
from form_bg import form
from boundary_layer import tie_density
from ground_truth import forward
from strat_tie import models

N = 1 << 22
active_subspace.NT = 1 << 14

if __name__ == '__main__':
    rows = []
    print(f"{'model':8s} | {'ratio SR':>8s} {'ratio MS':>8s} | {'SR act':>7s} {'SR ded':>7s} {'rel':>6s} | {'MS act':>7s} {'MS ded':>7s} {'rel':>6s}")
    for name, W in models:
        W = [th.tensor(a, dtype=th.float32) for a in W]
        x = th.randn(N, 10, generator=th.Generator().manual_seed(3)); t = x.topk(3, -1).indices
        with th.no_grad():
            out = forward(x, *W).argmax(-1)
        act = {'S/R': (out == t[:, 2]).float().mean().item(), 'M/S': (out == t[:, 0]).float().mean().item()}
        ratio, ded = {}, {}
        for k, kind in ((2, 'S/R'), (1, 'M/S')):
            _, _, gp, _, al = grads(W, k, th.Generator().manual_seed(k))
            ratio[kind] = np.sqrt((gp ** 2).sum(-1).mean()) / np.sqrt((al ** 2).mean())
            ded[kind] = 0.5 * tie_density(k) * form(W, kind, False, False)[1]
        r = [ratio['S/R'], ratio['M/S'], act['S/R'], ded['S/R'], ded['S/R'] / act['S/R'] - 1, act['M/S'], ded['M/S'], ded['M/S'] / act['M/S'] - 1]
        rows.append(r)
        print(f"{name:8s} | {r[0]:8.3f} {r[1]:8.3f} | {r[2]:7.4f} {r[3]:7.4f} {r[4]:+6.1%} | {r[5]:7.4f} {r[6]:7.4f} {r[7]:+6.1%}", flush=True)
    R = np.array(rows)
    for lab, i, j in (('S/R', 0, 4), ('M/S', 1, 7)):
        print(f"{lab}: |rel err| vs ratio: spearman {spearmanr(R[:, i], np.abs(R[:, j]))[0]:+.2f}, pearson {pearsonr(R[:, i], np.abs(R[:, j]))[0]:+.2f}; "
              f"signed rel err vs ratio: spearman {spearmanr(R[:, i], R[:, j])[0]:+.2f}")
    both = np.r_[R[:, 0], R[:, 1]]; errs = np.r_[np.abs(R[:, 4]), np.abs(R[:, 7])]
    print(f"pooled (20 points): spearman {spearmanr(both, errs)[0]:+.2f}, pearson {pearsonr(both, errs)[0]:+.2f}")
