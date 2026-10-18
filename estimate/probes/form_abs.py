"""Absolute (no fitted scale) boundary-layer error estimates per tie kind from FORM
skeletons, against the actual S/R and M/S error rates and the sampled-tie estimate of
boundary_layer.py (0.5 f_D(0) E[|m0|/alpha], capped at .25). Columns: skeleton at the
mean background without noise, FORM with constant mean background, FORM with the
expected order statistics permuted over background positions. Fully deductive: no
labels, no random inputs.
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import forward, load_weights
from boundary_layer import sample_ties, widths, tie_density
from form_bg import form

N = 1 << 22


def run(name, W):
    W = [th.tensor(a, dtype=th.float32) for a in W]
    x = th.randn(N, 10, generator=th.Generator().manual_seed(3)); t = x.topk(3, -1).indices
    with th.no_grad():
        out = forward(x, *W).argmax(-1)
        act = {'S/R': (out == t[:, 2]).float().mean().item(), 'M/S': (out == t[:, 0]).float().mean().item()}
        meas = {}
        for k, kind in ((2, 'S/R'), (1, 'M/S')):
            xt, a, b = sample_ties(k, 1 << 17, th.Generator().manual_seed(k)); m0, al, inp = widths(xt, a, b, W)
            sgn = 1. if k == 2 else -1.; w = (m0.abs() / (sgn * al).clamp_min(1e-9)).clamp_max(.25)
            meas[kind] = 0.5 * tie_density(k) * th.where(inp & (sgn * al > 0), w, th.zeros_like(w)).mean().item()
    print(f"\n{name}: error rate by kind, actual vs boundary-layer estimates")
    for k, kind in ((2, 'S/R'), (1, 'M/S')):
        f = 0.5 * tie_density(k)
        d = [f * form(W, kind, perm, noise)[1] for perm, noise in ((False, False), (False, True), (True, True))]
        print(f"  {kind}: actual {act[kind]:.4f} | sampled ties {meas[kind]:.4f} | skeleton mean-bg {d[0]:.4f} | FORM const {d[1]:.4f} | FORM perm {d[2]:.4f}", flush=True)


if __name__ == '__main__':
    run('trained M(16,10)', load_weights('weights.npz'))
    d = np.load('noisy_sigma0.005.npz'); s = np.load('seeds_h16.npz')
    run('noisy0', [d[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
    run('seed0', [s[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
