"""Step 2a: census of ReLU activation-pattern sequences D[0..9] (160 bits).

Within one pattern's cone the network is linear, so accuracy is a sum over
patterns of Gaussian measures of polyhedral cones. This measures how the
probability mass is spread over patterns (sampled — for scoping only).

Usage: python pattern_census.py --n-log2 24
"""
import argparse
import time

import numpy as np
import torch as th

from ground_truth import load_weights

T, H = 10, 16


def patterns(x, W_ih, W_hh):
    """Return packed patterns: int64 [B, 3] (bits t*16+i), plus final h."""
    B = x.shape[0]
    bits = th.zeros(B, T * H, dtype=th.bool, device=x.device)
    h = th.zeros(B, H, device=x.device)
    for t in range(T):
        pre = x[:, t : t + 1] * W_ih + h @ W_hh.T
        on = pre > 0
        bits[:, t * H : (t + 1) * H] = on
        h = pre * on
    w = (1 << th.arange(63, device=x.device, dtype=th.int64))
    packed = th.stack(
        [(bits[:, k * 63 : (k + 1) * 63].long() * w[: bits[:, k * 63 : (k + 1) * 63].shape[1]]).sum(-1)
         for k in range(3)], -1)
    return packed, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights.npz")
    ap.add_argument("--n-log2", type=int, default=22)
    ap.add_argument("--batch-log2", type=int, default=20)
    ap.add_argument("--out", default="census.npz")
    args = ap.parse_args()
    dev = "cuda" if th.cuda.is_available() else "cpu"
    W_ih, W_hh, W_out = (th.tensor(a, device=dev) for a in load_weights(args.weights))
    gen = th.Generator(device=dev).manual_seed(123)
    counts, correct = {}, {}
    B = 1 << args.batch_log2
    t0 = time.time()
    with th.no_grad():
        for b in range((1 << args.n_log2) // B):
            x = th.randn(B, T, device=dev, generator=gen)
            p, h = patterns(x, W_ih, W_hh)
            ok = (h @ W_out.T).argmax(-1) == x.topk(2, -1).indices[:, 1]
            u, inv, c = th.unique(p, dim=0, return_inverse=True, return_counts=True)
            cc = th.zeros(len(u), dtype=th.int64, device=dev).index_add_(0, inv, ok.long())
            for key, n, k in zip(map(tuple, u.cpu().tolist()), c.cpu().tolist(), cc.cpu().tolist()):
                counts[key] = counts.get(key, 0) + n
                correct[key] = correct.get(key, 0) + k
    keys = sorted(counts, key=counts.get, reverse=True)
    n = np.array([counts[k] for k in keys])
    c = np.array([correct[k] for k in keys])
    N = n.sum()
    cum = np.cumsum(n) / N
    print(f"N={N} distinct patterns={len(keys)}  ({time.time()-t0:.0f}s)")
    print(f"singletons={np.sum(n == 1)} (Good-Turing unseen mass ~ {np.sum(n == 1)/N:.4f})")
    for q in [0.5, 0.9, 0.99, 0.999]:
        print(f"  patterns for {q:.3f} of mass: {np.searchsorted(cum, q) + 1}")
    wrong = n - c
    print(f"errors: {wrong.sum()}; patterns containing errors: {np.sum(wrong > 0)}")
    cw = np.cumsum(np.sort(wrong)[::-1]) / wrong.sum()
    for q in [0.5, 0.9, 0.99]:
        print(f"  patterns for {q:.2f} of errors: {np.searchsorted(cw, q) + 1}")
    np.savez(args.out, keys=np.array(keys, dtype=np.int64), n=n, c=c)


if __name__ == "__main__":
    main()
