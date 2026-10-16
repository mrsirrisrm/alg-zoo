"""Evaluation harness (ARC protocol): noisy copies of the weights + MC ground truth.

W' = W + sigma * rms(W) * N(0,1), per matrix. Saves all weights and accuracies.
Also reports the sampling baseline: MSE(N) = p(1-p)/N at ~FLOPS_PER_SAMPLE each.

Usage (GPU box): python noisy_models.py --sigma 0.1 --count 32 --n-log2 30
"""
import argparse
import json

import numpy as np
import torch as th

from ground_truth import forward, load_weights

FLOPS_PER_SAMPLE = 10 * (2 * 16 * 16 + 2 * 16) + 2 * 10 * 16  # recurrence + readout


def accuracy(W_ih, W_hh, W_out, n_log2, batch_log2, seed, dev):
    W = [th.tensor(a, device=dev, dtype=th.float32) for a in (W_ih, W_hh, W_out)]
    gen = th.Generator(device=dev).manual_seed(seed)
    B = 1 << batch_log2
    c = 0
    with th.no_grad():
        for _ in range(max(1, (1 << n_log2) // B)):
            x = th.randn(B, 10, device=dev, generator=gen)
            c += (forward(x, *W).argmax(-1) == x.topk(2, -1).indices[:, 1]).sum().item()
    n = max(1, (1 << n_log2) // B) * B
    return c / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights.npz")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--n-log2", type=int, default=30)
    ap.add_argument("--batch-log2", type=int, default=22)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if th.cuda.is_available() else "cpu"
    base = load_weights(args.weights)
    rng = np.random.default_rng(12345)
    Ws, accs = [], []
    for i in range(args.count):
        noisy = [a + args.sigma * np.sqrt((a ** 2).mean()) * rng.standard_normal(a.shape).astype(a.dtype)
                 for a in base]
        acc, n = accuracy(*noisy, args.n_log2, args.batch_log2, seed=1000 + i, dev=dev)
        Ws.append(noisy)
        accs.append(acc)
        print(f"model {i}: acc {acc:.6f}", flush=True)
    accs = np.array(accs)
    summary = dict(sigma=args.sigma, count=args.count, n_per_model=n,
                   acc_mean=float(accs.mean()), acc_std=float(accs.std()),
                   acc_min=float(accs.min()), acc_max=float(accs.max()),
                   flops_per_sample=FLOPS_PER_SAMPLE)
    print(json.dumps(summary, indent=1))
    out = args.out or f"noisy_sigma{args.sigma}.npz"
    np.savez(out, W_ih=np.stack([w[0] for w in Ws]), W_hh=np.stack([w[1] for w in Ws]),
             W_out=np.stack([w[2] for w in Ws]), acc=accs, summary=json.dumps(summary))


if __name__ == "__main__":
    main()
