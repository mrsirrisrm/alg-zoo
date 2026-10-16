"""E0: Gaussian-closure moment propagation estimate of accuracy (no model sampling).

Decomposition (exact):
  acc = sum_{m != s} int_{vM > vS} phi(vM) phi(vS) Phi(vS)^8 * P(correct | m, s, vM, vS)
where, given (m, s, vM, vS), the other 8 inputs are iid N(0,1) truncated to x < vS.

Approximation (the presumption being tested): h_t is Gaussian at every step.
  - fixed input (x_m = vM, x_s = vS): pre ~ N(W_ih x + W_hh mu, W_hh S W_hh^T)
  - background input: x_t integrated by K-node quadrature of the truncated normal;
    ReLU moments computed exactly per node, then moment-matched to a Gaussian.
  - output: P(logit_s > logit_j for all j) from the Gaussian logit law.

Usage: python moment_prop.py --qs 16 --qm 12 --k 12
"""
import argparse
import json
import time

import numpy as np
import torch as th
from scipy.special import roots_jacobi

from gauss import Phi_inv, orthant_prob, relu_moments
from ground_truth import load_weights

T, H = 10, 16


def quadrature(qs, qm, dtype, dev):
    """Nodes (vS, vM, weight) for one ordered position pair; weights sum to 1/90."""
    x, w = roots_jacobi(qs, 1.0, 8.0)  # weight (1-x)^1 (1+x)^8
    u = th.tensor((1 + x) / 2, dtype=dtype, device=dev)
    wu = th.tensor(w / w.sum(), dtype=dtype, device=dev) / 90.0
    g, gw = np.polynomial.legendre.leggauss(qm)
    q = th.tensor((g + 1) / 2, dtype=dtype, device=dev)
    wq = th.tensor(gw / 2, dtype=dtype, device=dev)
    vS = Phi_inv(u)[:, None].expand(qs, qm)
    vM = -Phi_inv((1 - u)[:, None] * (1 - q)[None, :])
    W = wu[:, None] * wq[None, :]
    return vS.reshape(-1), vM.reshape(-1), W.reshape(-1), u.repeat_interleave(qm)


def bg_nodes(u, k, dtype, dev):
    """Quadrature for x ~ N(0,1) | x < Phi^{-1}(u): nodes [P, k], weights [k]."""
    g, gw = np.polynomial.legendre.leggauss(k)
    v = th.tensor((g + 1) / 2, dtype=dtype, device=dev)
    wv = th.tensor(gw / 2, dtype=dtype, device=dev)
    return Phi_inv(u[:, None] * v[None, :]), wv


def propagate(m, s, vM, vS, u, Wi, Wh, Wo, k, n_sobol):
    P = vM.shape[0]
    dt, dev = vM.dtype, vM.device
    mu = th.zeros(P, H, dtype=dt, device=dev)
    S = th.zeros(P, H, H, dtype=dt, device=dev)
    xb, wb = bg_nodes(u, k, dt, dev)  # [P, k]
    for t in range(T):
        x = xb.clone()
        x = th.where((m == t)[:, None], vM[:, None], x)
        x = th.where((s == t)[:, None], vS[:, None], x)
        fixed = ((m == t) | (s == t))[:, None]
        w = th.where(fixed, th.zeros_like(x), wb[None, :].expand_as(x))
        w[:, 0] = th.where(fixed[:, 0], th.ones_like(w[:, 0]), w[:, 0])
        pre_mu = x[..., None] * Wi + (mu @ Wh.T)[:, None, :]  # [P, k, H]
        pre_S = (Wh @ S @ Wh.T)[:, None]  # [P, 1, H, H]
        m1, m2 = relu_moments(pre_mu, pre_S)
        mu = (w[..., None] * m1).sum(1)
        E2 = (w[..., None, None] * m2).sum(1)
        S = E2 - mu[..., :, None] * mu[..., None, :]
        S = 0.5 * (S + S.transpose(-1, -2))
    lm = mu @ Wo.T
    lS = Wo @ S @ Wo.T
    # Y_j = l_s - l_j, j != s
    D = -th.eye(T, dtype=dt, device=dev)[None].repeat(P, 1, 1)
    D[th.arange(P), :, s] += 1.0
    keep = th.ones(P, T, dtype=th.bool, device=dev)
    keep[th.arange(P), s] = False
    D = D[keep].reshape(P, T - 1, T)
    nu = (D @ lm[..., None])[..., 0]
    C = D @ lS @ D.transpose(-1, -2)
    return orthant_prob(nu, C, n=n_sobol)


def estimate(weights, qs, qm, k, n_sobol, chunk, dev, dtype=th.float64):
    Wi, Wh, Wo = (th.tensor(a, dtype=dtype, device=dev) for a in load_weights(weights))
    vS, vM, W, u = quadrature(qs, qm, dtype, dev)
    table = np.zeros((T, T))
    for mi in range(T):
        for si in range(T):
            if mi == si:
                continue
            n = vS.shape[0]
            pc = []
            for c0 in range(0, n, chunk):
                sl = slice(c0, c0 + chunk)
                P = vS[sl].shape[0]
                mm = th.full((P,), mi, device=dev)
                ss = th.full((P,), si, device=dev)
                pc.append(propagate(mm, ss, vM[sl], vS[sl], u[sl], Wi, Wh, Wo, k, n_sobol))
            pc = th.cat(pc)
            table[mi, si] = (90.0 * (W * pc).sum()).item()  # conditional accuracy
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights.npz")
    ap.add_argument("--qs", type=int, default=16)
    ap.add_argument("--qm", type=int, default=12)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--sobol", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--gt", default="gt_2e35.npz")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if th.cuda.is_available() else "cpu"
    t0 = time.time()
    table = estimate(args.weights, args.qs, args.qm, args.k, args.sobol, args.chunk, dev)
    secs = time.time() - t0
    acc = table[~np.eye(T, dtype=bool)].mean()
    g = np.load(args.gt)
    gt_tab = g["pos_c"] / np.maximum(g["pos_n"], 1)
    gt = json.loads(str(g["summary"]))["accuracy"]
    off = ~np.eye(T, dtype=bool)
    err = table[off] - gt_tab[off]
    res = dict(estimate=acc, ground_truth=gt, error=acc - gt, seconds=secs,
               table_rmse=float(np.sqrt((err ** 2).mean())), table_maxabs=float(np.abs(err).max()),
               table_corr=float(np.corrcoef(table[off], gt_tab[off])[0, 1]),
               qs=args.qs, qm=args.qm, k=args.k, sobol=args.sobol)
    print(json.dumps(res, indent=1))
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("estimate - ground truth by (M pos, S pos):")
    d = table - gt_tab
    np.fill_diagonal(d, np.nan)
    print(d)
    if args.out:
        np.savez(args.out, table=table, summary=json.dumps(res))


if __name__ == "__main__":
    main()
