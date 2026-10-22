"""Far-field errors: errors away from the tie layers (follow-up to PLAN J5).

Far field = S/R errors with gap S-R >= 0.1, M/S errors with gap M-S >= 0.1, and errors at
any other position. Per model, on 2^22 inputs, error rate (far-field errors / inputs in
the cell) and share of far-field errors, broken down by:
  type        S/R, M/S, other position (rank of the chosen input)
  npos        number of positive inputs (hand10's only error event is npos <= 1)
  S/M         ratio of 2nd to max (scale-free: the net is positively homogeneous)
  pS, pM      positions of 2nd and max; order (M before S or after)
Enrichment = rate in cell / overall far-field rate.

Usage: uv run python probes/far_field.py [names...]
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from opt_bits import loaders
from ground_truth import forward

N = 1 << 22


def table(title, key, cells, far, base):
    rows = []
    for lab, m in cells:
        n = m.sum(); f = (far & m).sum()
        rows.append(f'{lab}: mass {n / len(m):.3f} rate {f / max(n, 1):.4f} x{f / max(n, 1) / base:5.1f} share {f / far.sum():.2f}')
    print(f'  {title}\n    ' + '\n    '.join(rows))


def run(name, W):
    x = th.randn(N, 10, generator=th.Generator().manual_seed(17))
    Wt = [th.tensor(a, dtype=th.float32) for a in W]
    with th.no_grad():
        out = th.cat([forward(x[i:i + (1 << 20)], *Wt).argmax(-1) for i in range(0, N, 1 << 20)]).numpy()
    xn = x.numpy(); order = np.argsort(-xn, 1); v = -np.sort(-xn, 1)
    pM, pS, pR = order[:, 0], order[:, 1], order[:, 2]
    rank = np.argmax(order == out[:, None], 1)              # rank of the chosen input (0 = max)
    wrong = rank != 1
    gSR, gMS = v[:, 1] - v[:, 2], v[:, 0] - v[:, 1]
    far = wrong & ~((rank == 2) & (gSR < 0.1)) & ~((rank == 0) & (gMS < 0.1))
    base = far.mean()
    print(f'\n{name}: err {wrong.mean():.4f}  far-field {base:.4f} ({far.sum() / wrong.sum():.2f} of errors)')
    print('  chosen rank among far-field errors: ' + ' '.join(f'{r}:{((rank == r) & far).sum() / far.sum():.2f}' for r in range(10)))
    npos = (xn > 0).sum(1)
    table('by number of positive inputs', npos, [(f'npos={k}', npos == k) for k in range(5)] + [('npos>=5', npos >= 5)], far, base)
    ratio = v[:, 1] / v[:, 0]
    edges = [-np.inf, 0, 0.25, 0.5, 0.75, 0.9, 1.01]
    table('by S/M', ratio, [(f'S/M in [{a:.2f},{b:.2f})', (ratio >= a) & (ratio < b)) for a, b in zip(edges, edges[1:])], far, base)
    table('by order and lag of S from the end', pS,
          [('M before S', pM < pS), ('M after S', pM > pS)] + [(f'pS={k}', pS == k) for k in (0, 1, 2, 5, 8, 9)], far, base)
    table('by pM', pM, [(f'pM={k}', pM == k) for k in (0, 1, 2, 5, 8, 9)], far, base)


if __name__ == '__main__':
    from hand_gains import hand
    L = loaders(); L['hand10_k2'] = lambda: hand(-0.0199, 0.0091)   # J5(b) fit
    names = sys.argv[1:] or ['trained', 'seed0', 'seed2', 'h32', 'hand10', 'hand10_k2']
    for n in names:
        run(n, L[n]())
