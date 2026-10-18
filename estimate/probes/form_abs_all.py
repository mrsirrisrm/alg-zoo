"""Absolute mean-background skeleton estimate (probes/form_abs.py, no noise term) across
the 10 harness models (trained, noisy0-3, seed0-4), at 5x3 and 15x7 quadrature nodes.
Compared with the actual S/R + M/S error and with the TOTAL error (the target; 'other'
errors are outside any tie skeleton), and with MC at the same compute. Cost: one forward
+ one backward per skeleton, counted as 3 forward passes.
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
import form_bg
from form_bg import form
from boundary_layer import tie_density
from ground_truth import forward
from strat_tie import models

N = 1 << 22
F = {k: 0.5 * tie_density(k) for k in (1, 2)}


def deduced(W, vq, oq):
    form_bg.VQ = list((np.arange(vq) + .5) / vq); form_bg.OQ = list((np.arange(oq) + .5) / oq)
    return F[2] * form(W, 'S/R', False, False)[1] + F[1] * form(W, 'M/S', False, False)[1], 3 * (vq * 90 * 8 * oq + vq * 90)


if __name__ == '__main__':
    print(f"{'model':8s} {'total':>7s} {'SR+MS':>7s} | {'ded 5x3':>8s} {'ded 15x7':>8s} | rel err vs SR+MS, vs total | MC sd at 5x3 cost")
    rows = []
    for name, W in models:
        W = [th.tensor(a, dtype=th.float32) for a in W]
        x = th.randn(N, 10, generator=th.Generator().manual_seed(3)); t = x.topk(3, -1).indices
        with th.no_grad():
            out = forward(x, *W).argmax(-1)
        tot = (out != t[:, 1]).float().mean().item(); srms = ((out == t[:, 2]) | (out == t[:, 0])).float().mean().item()
        (d1, c1), (d2, _) = deduced(W, 5, 3), deduced(W, 15, 7)
        sd = np.sqrt(tot * (1 - tot) / c1)
        rows.append((tot, srms, d1, d2, sd))
        print(f"{name:8s} {tot:7.4f} {srms:7.4f} | {d1:8.4f} {d2:8.4f} | {d2 / srms - 1:+6.1%} {d2 / tot - 1:+6.1%} | {sd:.4f} (n={c1})", flush=True)
    r = np.array(rows)
    print(f"\nrms relative error vs SR+MS: 5x3 {np.sqrt(np.mean((r[:, 2] / r[:, 1] - 1) ** 2)):.3f}, 15x7 {np.sqrt(np.mean((r[:, 3] / r[:, 1] - 1) ** 2)):.3f}")
    print(f"MSE vs total error: deduced(5x3) {np.mean((r[:, 2] - r[:, 0]) ** 2):.2e}, deduced(15x7) {np.mean((r[:, 3] - r[:, 0]) ** 2):.2e}, MC at 5x3 cost {np.mean(r[:, 4] ** 2):.2e}")
    print(f"MSE vs SR+MS:      deduced(5x3) {np.mean((r[:, 2] - r[:, 1]) ** 2):.2e}, deduced(15x7) {np.mean((r[:, 3] - r[:, 1]) ** 2):.2e}")
