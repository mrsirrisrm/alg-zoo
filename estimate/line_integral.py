"""Exact reparametrisation of the total error along a tie direction.

For k in {1, 2} write x_a = c + d/2, x_b = c - d/2 for the k-th / (k+1)-th largest
pair (a higher), d >= 0. The Jacobian is 1 and phi(c+d/2) phi(c-d/2) = phi(c)^2 e^{-d^2/4},
so for ANY input the total error rate is

    err = f_Dk(0) * E_tie[ int_0^inf e^{-d^2/4} 1{order kept at d} 1{wrong at d} dd ]

with the tie-conditional law of boundary_layer.sample_ties. The d-integral is done by
stratified sampling on a fixed grid of cells (one random point per cell: unbiased);
errors live in the tie layer near d = 0, so the cells are dense there.
Reports variance per forward pass relative to plain Monte Carlo.
"""
import argparse
import json

import numpy as np
import torch as th

from boundary_layer import sample_ties, tie_density
from ground_truth import forward, load_weights

EDGES = np.concatenate([np.linspace(0, 0.08, 9), np.linspace(0.08, 0.5, 8)[1:], np.linspace(0.5, 4, 8)[1:]])


def line_estimates(k, n, W, gen, edges=EDGES):
    """Per-config estimates f * J (unbiased for the total error), shape [n]."""
    x, a, b = sample_ties(k, n, gen)
    ar = th.arange(n)
    c = x[ar, a].clone()
    below = th.where(x < c[:, None] - 1e-6, x, -th.inf).max(1).values
    above = th.where(x > c[:, None] + 1e-6, x, th.inf).min(1).values   # k=2: the max; k=1: none
    lo, wd = th.tensor(edges[:-1]).float(), th.tensor(np.diff(edges)).float()
    d = lo + th.rand(n, len(lo), generator=gen) * wd                     # [n, C]
    y = x[:, None, :].repeat(1, len(lo), 1)
    y[ar, :, a] = c[:, None] + d / 2
    y[ar, :, b] = c[:, None] - d / 2
    ok_order = (c[:, None] - d / 2 > below[:, None]) & (c[:, None] + d / 2 < above[:, None])
    pred = forward(y.reshape(-1, 10), *W).argmax(-1).reshape(n, -1)
    target = a if k == 2 else b                                           # the 2nd largest
    wrong = pred != target[:, None]
    J = (wd * th.exp(-d * d / 4) * ok_order * wrong).sum(1)
    return tie_density(k) * J, len(lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 15)
    ap.add_argument("--models", type=int, default=4)
    args = ap.parse_args()
    d = np.load("noisy_sigma0.005.npz")
    models = [("trained", [th.tensor(a) for a in load_weights("weights.npz")], 1 - 0.9531186)]
    models += [(f"noisy{i}", [th.tensor(d[k][i]) for k in ("W_ih", "W_hh", "W_out")], 1 - d["acc"][i])
               for i in range(args.models)]
    out = []
    for name, W, true in models:
        gen = th.Generator().manual_seed(7)
        for k in (1, 2):
            e, cost = line_estimates(k, args.n, W, gen)
            m, v = e.mean().item(), e.var().item()
            gain = true * (1 - true) / (v * cost)                         # MC variance per forward / ours
            out.append(dict(model=name, k=k, est=m, true=true, z=(m - true) / (v / args.n) ** 0.5, gain_per_forward=gain))
            print(json.dumps(out[-1]), flush=True)


if __name__ == "__main__":
    main()
