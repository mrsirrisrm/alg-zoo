"""Position dependence of the tie errors, and whether it can be derived without sampling.
Per ordered position pair (S/R errors by (pS, pR), M/S errors by (pM, pS)):
  actual    error rate of that type in the cell, from 2^23 labelled inputs
  measured  mean boundary-layer width at sampled ties (probes/surprise.widths; network
            margins at ties, no labels)
  additive  log actual ≈ c + f(first pos) + g(second pos): 19 numbers instead of 90
  skeleton  deterministic: one input per (pair, other top position, quadrature node) with
            the tied pair at v, the max (S/R) or 3rd (M/S) at a fixed offset, every other
            input at a constant background b; width = -m0/alpha by finite difference.
            No random inputs and no labels (the 'others at a constant' baseline).
By symmetry of iid inputs the gap law is the same for every pair, so the per-pair error
rate should be proportional to the mean width; scored by correlation of logs across pairs
and the rms log ratio after one global scale.
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import forward, load_weights
import surprise

N = 1 << 23


def actual(W):
    x = th.randn(N, 10, generator=th.Generator().manual_seed(11))
    t = x.topk(3, -1); pM, pS, pR = (i.numpy() for i in t.indices.T)
    with th.no_grad():
        out = forward(x, *W).argmax(-1).numpy()
    sr = np.bincount(pS * 10 + pR, out == pR, minlength=100) / np.maximum(np.bincount(pS * 10 + pR, minlength=100), 1)
    ms = np.bincount(pM * 10 + pS, out == pM, minlength=100) / np.maximum(np.bincount(pM * 10 + pS, minlength=100), 1)
    return sr.reshape(10, 10), ms.reshape(10, 10)


def measured(W):
    surprise.NT = 1 << 18
    gen = th.Generator().manual_seed(0)
    with th.no_grad():
        P1, P2 = surprise.widths(W, 1, gen), surprise.widths(W, 2, gen)
    mean = lambda P: P[0] * np.exp(P[1] + P[2] ** 2 / 2)       # E[w] under the lognormal fit
    return mean(P2), mean(P1)                                   # S/R by (hi=S, lo=R); M/S by (hi=M, lo=S)


def skeleton(W, b=-0.5, nodes=((1.0, .1), (1.0, .3), (1.5, .1), (1.5, .3), (0.5, .1), (0.5, .3))):
    """Width per ordered pair from constant-background skeletons, averaged over the third
    top position and a few (tie value v, offset of the other top value) nodes."""
    out = {}
    for kind in ('S/R', 'M/S'):
        xs, his, los = [], [], []
        for hi in range(10):
            for lo in range(10):
                if hi == lo: continue
                for o in range(10):
                    if o in (hi, lo): continue
                    for v, off in nodes:
                        x = np.full(10, b * v); x[hi] = x[lo] = v
                        x[o] = v * (1 + off) if kind == 'S/R' else v * (1 - off)
                        xs.append(x); his.append(hi); los.append(lo)
        x = th.tensor(np.array(xs), dtype=th.float32); hi = th.tensor(his); lo = th.tensor(los); ar = th.arange(len(x))
        tgt, cmp = (hi, lo) if kind == 'S/R' else (lo, hi)
        def diff(dd):
            y = x.clone(); y[ar, hi] += dd / 2; y[ar, lo] -= dd / 2
            l = forward(y, *W); return l[ar, tgt] - l[ar, cmp]
        with th.no_grad():
            m0 = diff(0.); al = (diff(1e-3) - diff(-1e-3)) / 2e-3
        w = (-m0 / al.clamp_min(1e-9)).clamp(0, .25) / x.abs().max(-1).values
        w = th.where(al > 0, w, th.zeros_like(w)).numpy()
        M = np.zeros(100); np.add.at(M, his_lo := (hi * 10 + lo).numpy(), w)
        out[kind] = (M / np.maximum(np.bincount(his_lo, minlength=100), 1)).reshape(10, 10)
    return out['S/R'], out['M/S']


def additive(A):
    m = ~np.eye(10, dtype=bool) & (A > 0); i, j = np.nonzero(m)
    X = np.zeros((len(i), 20)); X[np.arange(len(i)), i] = 1; X[np.arange(len(i)), 10 + j] = 1
    c, *_ = np.linalg.lstsq(X, np.log(A[m]), rcond=None)
    F = np.full((10, 10), np.nan); F[i, j] = np.exp(X @ c); return F


def score(A, P):
    m = ~np.eye(10, dtype=bool) & (A > 0) & (P > 0)
    la, lp = np.log(A[m]), np.log(P[m])
    r = np.corrcoef(la, lp)[0, 1]; rms = np.sqrt(np.mean((la - lp - (la - lp).mean()) ** 2))
    return f"corr {r:5.2f}  rms log-ratio {rms:4.2f} ({m.sum()} pairs)"


def run(name, W):
    W = [th.tensor(a, dtype=th.float32) for a in W]
    (aSR, aMS), (wSR, wMS), (kSR, kMS) = actual(W), measured(W), skeleton(W)
    print(f"\n{name}")
    for kind, A, Wm, K in (('S/R by (pS,pR)', aSR, wSR, kSR), ('M/S by (pM,pS)', aMS, wMS, kMS)):
        m = ~np.eye(10, dtype=bool)
        print(f"  {kind}: actual rate spread across pairs: rms log {np.std(np.log(np.maximum(A[m], 1e-6))):.2f}, max/min {A[m].max() / max(A[m].min(), 1e-6):.0f}")
        print(f"    measured widths  {score(A, Wm)}")
        print(f"    additive (fit)   {score(A, additive(A))}")
        print(f"    skeleton (deduc) {score(A, K)}", flush=True)


if __name__ == '__main__':
    run('trained M(16,10)', load_weights('weights.npz'))
    d = np.load('noisy_sigma0.005.npz'); s = np.load('seeds_h16.npz')
    run('noisy0', [d[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
    run('seed0', [s[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
