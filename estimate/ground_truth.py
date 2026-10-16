"""Step 1: Monte Carlo ground truth for the 432-param 2nd-argmax RNN.

Validation target only — nothing here feeds the estimator.

Records, over N iid N(0,1)^10 inputs:
  - overall accuracy
  - counts/correct by (argmax position M, 2nd-argmax position S)  [10x10]
  - counts/correct by value gap (x_M - x_S) and by (x_S - x_3rd), binned
  - failure breakdown: predicted position == M, == 3rd, other

Usage: python ground_truth.py --n-log2 33 --out gt.npz
"""
import argparse
import json
import time

import numpy as np
import torch as th

T = 10
GAP_EDGES = np.concatenate([np.linspace(0, 0.5, 51), np.linspace(0.6, 4.0, 35)])


def load_weights(path):
    w = np.load(path)
    return w["W_ih"], w["W_hh"], w["W_out"]


def forward(x, W_ih, W_hh, W_out):
    h = th.relu(x[:, 0:1] * W_ih)
    for t in range(1, x.shape[1]):
        h = th.relu(x[:, t : t + 1] * W_ih + h @ W_hh.T)
    return h @ W_out.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights.npz")
    ap.add_argument("--n-log2", type=int, default=24)
    ap.add_argument("--batch-log2", type=int, default=22)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gt.npz")
    args = ap.parse_args()

    dev = "cuda" if th.cuda.is_available() else "cpu"
    W_ih, W_hh, W_out = (th.tensor(a, device=dev) for a in load_weights(args.weights))
    gen = th.Generator(device=dev).manual_seed(args.seed)
    edges = th.tensor(GAP_EDGES, device=dev, dtype=th.float32)
    nb = len(GAP_EDGES) + 1

    pos_n = th.zeros(T * T, dtype=th.int64, device=dev)
    pos_c = th.zeros(T * T, dtype=th.int64, device=dev)
    g1_n = th.zeros(nb, dtype=th.int64, device=dev)
    g1_c = th.zeros(nb, dtype=th.int64, device=dev)
    g2_n = th.zeros(nb, dtype=th.int64, device=dev)
    g2_c = th.zeros(nb, dtype=th.int64, device=dev)
    fail = th.zeros(3, dtype=th.int64, device=dev)

    B = 1 << args.batch_log2
    n_batches = max(1, (1 << args.n_log2) // B)
    t0 = time.time()
    with th.no_grad():
        for b in range(n_batches):
            x = th.randn(B, T, device=dev, generator=gen)
            pred = forward(x, W_ih, W_hh, W_out).argmax(-1)
            top = x.topk(3, dim=-1)
            M, S, R = top.indices.unbind(-1)
            vM, vS, vR = top.values.unbind(-1)
            ok = pred == S
            k = M * T + S
            pos_n += th.bincount(k, minlength=T * T)
            pos_c += th.bincount(k[ok], minlength=T * T)
            b1 = th.bucketize(vM - vS, edges)
            b2 = th.bucketize(vS - vR, edges)
            g1_n += th.bincount(b1, minlength=nb)
            g1_c += th.bincount(b1[ok], minlength=nb)
            g2_n += th.bincount(b2, minlength=nb)
            g2_c += th.bincount(b2[ok], minlength=nb)
            bad = ~ok
            fail[0] += (pred[bad] == M[bad]).sum()
            fail[1] += (pred[bad] == R[bad]).sum()
            fail[2] += bad.sum()
            if b % 256 == 0:
                acc = pos_c.sum().item() / pos_n.sum().item()
                print(f"batch {b}/{n_batches} acc={acc:.6f} {time.time()-t0:.0f}s", flush=True)

    to = lambda a: a.cpu().numpy()
    N = pos_n.sum().item()
    C = pos_c.sum().item()
    acc = C / N
    se = (acc * (1 - acc) / N) ** 0.5
    fail = to(fail)
    summary = dict(
        N=N, correct=C, accuracy=acc, stderr=se,
        fail_pred_M=int(fail[0]), fail_pred_3rd=int(fail[1]),
        fail_other=int(fail[2] - fail[0] - fail[1]), fail_total=int(fail[2]),
        seconds=time.time() - t0, device=dev, seed=args.seed,
    )
    print(json.dumps(summary, indent=1))
    np.savez(
        args.out, pos_n=to(pos_n).reshape(T, T), pos_c=to(pos_c).reshape(T, T),
        gap_edges=GAP_EDGES, g1_n=to(g1_n), g1_c=to(g1_c), g2_n=to(g2_n), g2_c=to(g2_c),
        summary=json.dumps(summary),
    )


if __name__ == "__main__":
    main()
