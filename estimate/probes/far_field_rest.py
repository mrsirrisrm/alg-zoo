"""Far-field errors with npos >= 2 (follow-up to probes/far_field.py): rank of the chosen input,
relative-gap strip tails (gap / M < 0.1), pM = 9, far rate against the scale M; then where
the rank >= 3 picks sit relative to M and S.

Usage: uv run python probes/far_field_rest.py
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from opt_bits import loaders
from ground_truth import forward
N = 1 << 22
x = th.randn(N, 10, generator=th.Generator().manual_seed(17)); xn = x.numpy()
order = np.argsort(-xn, 1); v = -np.sort(-xn, 1); pM, pS = order[:, 0], order[:, 1]
npos = (xn > 0).sum(1); gSR, gMS = v[:, 1] - v[:, 2], v[:, 0] - v[:, 1]
L = loaders()
for n in ['trained', 'seed0', 'seed2', 'h32']:
    Wt = [th.tensor(a, dtype=th.float32) for a in L[n]()]
    with th.no_grad(): out = th.cat([forward(x[i:i + (1 << 20)], *Wt).argmax(-1) for i in range(0, N, 1 << 20)]).numpy()
    rank = np.argmax(order == out[:, None], 1); wrong = rank != 1
    far = wrong & ~((rank == 2) & (gSR < 0.1)) & ~((rank == 0) & (gMS < 0.1)) & (npos >= 2)
    f = far.sum(); r = lambda m: f'{(far & m).sum() / f:.2f}'
    # relative gap: the net is homogeneous, so a strip of relative width w covers gap < w * M
    relSR, relMS = gSR / np.abs(v[:, 0]), gMS / np.abs(v[:, 0])
    print(f'{n:8s} far(npos>=2) {f / N:.4f} = {f / wrong.sum():.2f} of errors | rank 0 {r(rank == 0)} 2 {r(rank == 2)} >=3 {r(rank >= 3)}'
          f' | rel gap < 0.1: S/R {r((rank == 2) & (relSR < 0.1))} M/S {r((rank == 0) & (relMS < 0.1))}'
          f' | pM=9 {r(pM == 9)} | S<0.25M {r(v[:, 1] < 0.25 * v[:, 0])} | M>2.5 {r(v[:, 0] > 2.5)}')
    # error rate vs absolute scale at fixed relative gap: homogeneity predicts flat
    m = (rank != 1)
    for lo, hi in [(0, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 9)]:
        c = (v[:, 0] >= lo) & (v[:, 0] < hi) & (npos >= 2)
        print(f'          M in [{lo},{hi}): far rate {(far & c).sum() / c.sum():.4f}')

for n in ['trained', 'seed0', 'seed2']:  # rank >= 3 picks
    Wt = [th.tensor(a, dtype=th.float32) for a in L[n]()]
    with th.no_grad(): out = th.cat([forward(x[i:i + (1 << 20)], *Wt).argmax(-1) for i in range(0, N, 1 << 20)]).numpy()
    rank = np.argmax(order == out[:, None], 1)
    e = (rank >= 3) & (npos >= 2); f = e.sum()
    r = lambda m: f'{(e & m).sum() / f:.2f}'
    # running 2nd max at the moment the chosen input arrived: is the chosen input the 2nd max "so far" at some step?
    cum = np.maximum.accumulate(xn, 1)
    # chosen input was a record (running max) when it arrived?
    rec = xn[np.arange(N), out] >= cum[np.arange(N), out]
    # chosen is the 2nd-largest among inputs up to and including pM? (i.e. was the running 2nd when M arrived)
    pre = np.where(np.arange(10)[None] <= pM[:, None], xn, -np.inf); pre2 = np.argsort(-pre, 1)[:, 1]
    post = np.where(np.arange(10)[None] > pM[:, None], xn, -np.inf); post1 = np.argmax(post, 1)
    print(f'{n:8s} rank>=3 errors {f / N:.4f} | chosen after M {r(out > pM)} | S after M {r(pS > pM)} | chosen = largest after M {r((out == post1) & (out > pM))}'
          f' | chosen = 2nd among x_<=pM {r(out == pre2)} | chosen was a record {r(rec)} | chosen = pM+1 {r(out == pM + 1)} | chosen = 9 {r(out == 9)}')
    # baseline: how often is S itself the largest after M / the 2nd among x_<=pM
    print(f'          baseline: S after M {(pS > pM).mean():.2f}; S/x_rank3 ratio at errors: median {np.median(v[e, 2] / v[e, 1]):.2f} vs all {np.median(v[npos >= 2, 2] / v[npos >= 2, 1]):.2f}')
