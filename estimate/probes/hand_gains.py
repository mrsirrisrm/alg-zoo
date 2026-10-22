"""PLAN J5(b): hand10 with k finite gains, fitted to the trained model's error-vs-gap curve.

hand10 is an exact comparator (no tie strip). A single-gain scan shows its comparator
input gain, running-max input gain and the comparators' max term act as ONE ratio and only
open an M/S strip; S/R ties are decided by the delay-line copies. Gains (k = 1, 2, 3):
  e   comparator input gain (rows 10-18, 21 of W_ih scaled by 1 + e): M/S strip width
  g   delay-line gain (W_hh[i, i+1] for the copy chain 0-9 scaled by 1 + g): a leaky
      memory, S/R strip width growing with |pS - pR|
  c   comparator-chain gain (W_hh[i, i+1] for 10-17 scaled by 1 + c)
Target: trained M(16,10) error rate by gap bin, S/R (gap S-R) and M/S (gap M-S), 6 bins
each; loss = sum of squared log ratios (floor 3e-3); Nelder-Mead on common random inputs.
Check: every bin inside the seeds' spread (trained + seeds 0-4, min..max), per-pair
error rates log-correlated with the trained model's, accuracy near 0.953.
Surprise: k gains at the step that moves accuracy by 1e-3, against a +-0.1 prior.

Usage: uv run python probes/hand_gains.py
"""
import sys, math
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from scipy.optimize import minimize
from opt_bits import loaders
from ground_truth import forward
from alg_vars import err_stats

CMP = list(range(10, 19)) + [21]
X = th.randn(1 << 18, 10, generator=th.Generator().manual_seed(4))
XV = th.randn(1 << 20, 10, generator=th.Generator().manual_seed(9))
L = loaders(); W0 = L['hand10']()


def hand(e=0.0, g=0.0, c=0.0):
    Wi, Wh, Wo = (a.copy() for a in W0)
    Wi[CMP] *= 1 + e
    for i in range(9): Wh[i, i + 1] *= 1 + g
    for i in range(10, 18): Wh[i, i + 1] *= 1 + c
    return [Wi, Wh, Wo]


def outputs(W, x):
    Wt = [th.tensor(a, dtype=th.float32) for a in W]
    with th.no_grad():
        return forward(x, *Wt).argmax(-1).numpy()


def curves(W, x=X):
    acc, strip, sr, ms = err_stats(x.double(), outputs(W, x)); return acc, strip, np.array(sr + ms)


def pairs(W, x=XV):
    i = x.topk(3, -1).indices.numpy(); out = outputs(W, x)
    rate = lambda hi, lo, hit: np.bincount(hi * 10 + lo, hit, 100) / np.maximum(np.bincount(hi * 10 + lo, minlength=100), 1)
    off = ~np.eye(10, dtype=bool).ravel()
    return rate(i[:, 1], i[:, 2], out == i[:, 2])[off], rate(i[:, 0], i[:, 1], out == i[:, 0])[off]


def logcorr(a, b, fl=1e-3):
    return np.corrcoef(np.log(a + fl), np.log(b + fl))[0, 1]


def fmt(v): return ' '.join(f'{u:.3f}' for u in v)


def main():
    refs = {n: curves(L[n]()) for n in ['trained'] + [f'seed{i}' for i in range(5)]}
    tgt = refs['trained'][2]; R = np.array([r[2] for r in refs.values()]); lo, hi = R.min(0), R.max(0)
    ptr = pairs(L['trained']())
    print('error rate by gap bin [0,.01) [.01,.03) [.03,.1) [.1,.3) [.3,1) [1,10): S/R x6 | M/S x6')
    for n, (a, s, cv) in refs.items():
        print(f'  {n:8s} acc {a:.4f}  {fmt(cv[:6])} | {fmt(cv[6:])}')
    a, s, cv = curves(W0); print(f'  {"hand10":8s} acc {a:.4f}  {fmt(cv[:6])} | {fmt(cv[6:])}')
    loss = lambda cv: float(((np.log(cv + 3e-3) - np.log(tgt + 3e-3)) ** 2).sum())
    for k in (1, 2, 3):
        f = lambda p: loss(curves(hand(*p))[2])
        best = min((minimize(f, p0[:k], method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-4, 'maxfev': 400})
                    for p0 in (np.array([0.01, 0.01, 0.01]), np.array([-0.01, -0.01, -0.01]))), key=lambda r: r.fun)
        p = best.x; W = hand(*p); a, s, cv = curves(W, XV)
        inside = (cv >= lo - 1e-3) & (cv <= hi + 1e-3)
        psr, pms = pairs(W)
        bits = 0.0
        for j in range(k):
            for d in np.geomspace(1e-6, 0.1, 60):
                q = p.copy(); q[j] += d
                if abs(curves(hand(*q))[0] - curves(hand(*p))[0]) > 1e-3: break
            bits += math.log2(0.2 / d)
        print(f'\nk={k}: gains ' + ' '.join(f'{nm}={v:+.4f}' for nm, v in zip('egc', p)) + f'  loss {best.fun:.2f}  acc {a:.4f}')
        print(f'  fit       {fmt(cv[:6])} | {fmt(cv[6:])}')
        print(f'  in spread {" ".join("  yes" if b else "   no" for b in inside[:6])} | {" ".join("  yes" if b else "   no" for b in inside[6:])}'
              f'   ({inside.sum()}/12)')
        print(f'  per-pair log corr with trained: S/R {logcorr(psr, ptr[0]):.2f}  M/S {logcorr(pms, ptr[1]):.2f};'
              f'  surprise ~{bits:.0f} bits for {k} gains')
    print(f'\nseed spread of the per-pair S/R log corr with trained: ' +
          ' '.join(f'{logcorr(pairs(L[f"seed{i}"]())[0], ptr[0]):.2f}' for i in range(5)))


if __name__ == '__main__':
    main()
