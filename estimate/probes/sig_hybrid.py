"""Hybrid exact-cone + MC estimator, oracle bound. With no biases each full 160-bit
activation signature s is a polyhedral cone C_s on which logits = A_s x, so
P(err) = sum_s P(err, C_s) and each term is a Gaussian polytope integral. Estimator:
exact terms for a set K of signatures plus MC of 1[err, sig not in K]. Even with the
exact terms free, its gain over MC is e(1-e) / (q(1-q)), q = P(err, sig not in K).
K = the top error-carrying signatures on split A; q measured on held-out split B.
(The all-input census is dead end #1 in FINDINGS: 1.65M patterns in 4M samples; this
asks the narrower question of how concentrated the ERROR mass is.)
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from pattern_census import patterns
from strat_tie import models

N, B = 1 << 22, 1 << 20


def err_sigs(W, seed):
    W_ih, W_hh, W_out = W; gen = th.Generator().manual_seed(seed); out = []
    for _ in range(N // B):
        x = th.randn(B, 10, generator=gen); p, h = patterns(x, W_ih, W_hh)
        bad = (h @ W_out.T).argmax(-1) != x.topk(2, -1).indices[:, 1]
        out.append(p[bad].numpy())
    return np.concatenate(out)


def run(name, W):
    W = [th.tensor(a, dtype=th.float32) for a in W]
    with th.no_grad():
        A, Bs = err_sigs(W, 1), err_sigs(W, 2)
    ua, ca = np.unique(A, axis=0, return_counts=True); order = np.argsort(-ca)
    e = len(Bs) / N; ub, cb = np.unique(Bs, axis=0, return_counts=True)
    rank = {tuple(k): r for r, k in enumerate(ua[order])}
    rb = np.array([rank.get(tuple(k), 1 << 40) for k in ub])
    cov = np.cumsum(np.sort(ca)[::-1]) / ca.sum()
    print(f"\n{name}: err {e:.4f}; error signatures distinct {len(ua)} of {len(A)} errors (split A), "
          f"singletons {np.mean(ca == 1) * len(ua) / len(A):.2f} of error mass; "
          f"B errors in a signature unseen on A {cb[rb >= len(ua)].sum() / len(Bs):.2f}")
    print(f"  in-sample coverage of A errors by top K: " + ", ".join(
        f"K={k}: {cov[min(k, len(cov)) - 1]:.2f}" for k in (10, 100, 1000, 10000, 100000)))
    for k in (10, 100, 1000, 10000, 100000):
        q = cb[rb >= k].sum() / N
        print(f"  K={k:6d}: held-out error mass outside K {q / e:.3f}, oracle gain {e * (1 - e) / (q * (1 - q)):.2f}", flush=True)


if __name__ == '__main__':
    for name, W in models[:1] + models[1:2] + models[5:6]:
        run(name, W)
