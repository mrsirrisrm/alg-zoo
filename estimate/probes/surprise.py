"""Surprise accounting for the network's correctness. The observation is the bit c(x) =
[network output = 2nd argmax] on held-out inputs; a predictor q(x) = P(c=1) costs
residual -log2 q(x)^c (1-q(x))^(1-c) bits per input, and its parameters cost
1/2 log2 N_A bits each (two-part MDL, fitted on split A of N_A inputs). Ladder:
  chance      q = 0.1
  constant    q = accuracy
  gaps-logit  logistic in log tie gaps g1=(vM-vS)/|vM|, g2=(vS-vR)/|vM|, sign(vS)   (task-level)
  gaps-table  error rate per fine gap cell (14x14x2)                                (task-level)
  pos-table   error rate per coarse gap cell x pos(M,S)                             (task-level)
  tie-layer   the explanation: errors are the boundary layers at the two tie walls, of
              width -m0/alpha (network margin and its slope at the exact tie), per ordered
              position pair; widths from the network at sampled ties, NOT fitted to c(x).
              Only 2 background rates (vS<0, other) are fitted.
  tie+gaps    logistic on gaps-logit features plus logit(q_tie-layer)
Reported: residual bits/input, parameter bits, and N at which each beats 'constant'.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from scipy.stats import norm
from ground_truth import forward, load_weights
from boundary_layer import sample_ties

N = 1 << 22; NT = 1 << 18; LOG2 = np.log(2); BITS = 0.5 * np.log2(N)
edges = [0, .005, .01, .02, .03, .05, .07, .1, .15, .2, .3, .5, 1, np.inf]
coarse = [0, .01, .03, .07, .15, .4, np.inf]


def feats(x):
    t = x.topk(3, -1); (vM, vS, vR), (pM, pS, pR) = t.values.T, t.indices.T
    g1 = ((vM - vS) / vM.abs()); g2 = ((vS - vR) / vM.abs()); neg = (vS < 0)
    return dict(label=pS, g1=g1.numpy(), g2=g2.numpy(), neg=neg.numpy(), pM=pM.numpy(), pS=pS.numpy(), pR=pR.numpy())


def design(F):
    a, b = np.log(np.maximum(F['g1'], 1e-6)), np.log(np.maximum(F['g2'], 1e-6))
    return np.stack([np.ones_like(a), a, b, a * a, b * b, a * b, F['neg'], F['neg'] * b], 1)


def logistic(X, c, iters=200):
    X = th.tensor(X, dtype=th.float64); c = th.tensor(c, dtype=th.float64)
    w = th.zeros(X.shape[1], dtype=th.float64, requires_grad=True)
    opt = th.optim.LBFGS([w], max_iter=iters, line_search_fn='strong_wolfe')
    def clo():
        opt.zero_grad(); l = th.nn.functional.binary_cross_entropy_with_logits(X @ w, c); l.backward(); return l
    opt.step(clo)
    return w.detach().numpy()


def table(keyA, cA, keyB, alpha=.5):
    K = int(max(keyA.max(), keyB.max())) + 1
    n, r = np.bincount(keyA, minlength=K), np.bincount(keyA, cA, minlength=K)
    q = (r + alpha * cA.mean()) / (n + alpha)        # shrink empty/small cells to the mean
    return q[keyB], int((n > 0).sum())


def widths(W, k, gen):
    """Relative boundary-layer widths per ordered (hi, lo) position pair at a k-tie
    (k=1: M/S, target lo; k=2: S/R, target hi). Zero where the network's argmax is not in
    the pair or the slope has the wrong sign (no boundary layer there)."""
    x, a, b = sample_ties(k, NT, gen); ar = th.arange(NT); out = []
    for hi, lo in ((a, b), (b, a)):
        tgt, cmp = (lo, hi) if k == 1 else (hi, lo)
        def diff(dd):
            y = x.clone(); y[ar, hi] += dd / 2; y[ar, lo] -= dd / 2
            l = forward(y, *W); return l[ar, tgt] - l[ar, cmp], l.argmax(-1)
        m0, am = diff(0.); al = (diff(1e-3)[0] - diff(-1e-3)[0]) / 2e-3
        w = (-m0 / al.clamp_min(1e-9)).clamp(0, .25) / x.abs().max(-1).values
        w = th.where(((am == hi) | (am == lo)) & (al > 0), w, th.zeros_like(w))
        out.append((hi.numpy(), lo.numpy(), w.numpy()))
    hi, lo, w = (np.concatenate(z) for z in zip(*out))
    P = np.zeros((3, 10, 10))       # frac positive, mu, sigma of log w | w > 0
    for i in range(10):
        for j in range(10):
            s = w[(hi == i) & (lo == j)]; s = s[s > 0]
            if len(s) > 10:
                P[:, i, j] = len(s) / ((hi == i) & (lo == j)).sum(), np.log(s).mean(), np.log(s).std()
    return P


def tie_q(P1, P2, F):
    def p(P, i, j, g):
        f, mu, sd = P[0, i, j], P[1, i, j], np.maximum(P[2, i, j], 1e-3)
        return f * norm.sf((np.log(np.maximum(g, 1e-12)) - mu) / sd)
    return 1 - (1 - p(P1, F['pM'], F['pS'], F['g1'])) * (1 - p(P2, F['pS'], F['pR'], F['g2']))


def bits(q, c):
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return -(c * np.log2(q) + (1 - c) * np.log2(1 - q)).mean()


def run(name, W):
    W = [th.tensor(a, dtype=th.float32) for a in W]
    xa = th.randn(N, 10, generator=th.Generator().manual_seed(1)); xb = th.randn(N, 10, generator=th.Generator().manual_seed(2))
    FA, FB = feats(xa), feats(xb)
    with th.no_grad():
        cA = (forward(xa, *W).argmax(-1) == FA['label']).numpy().astype(float)
        cB = (forward(xb, *W).argmax(-1) == FB['label']).numpy().astype(float)
    rows = [('chance', np.full(N, .1), 0), ('constant', np.full(N, cA.mean()), 1)]
    XA, XB = design(FA), design(FB); w = logistic(XA, cA)
    rows.append(('gaps-logit', 1 / (1 + np.exp(-XB @ w)), len(w)))
    fk = lambda F: (np.digitize(F['g1'], edges) * 14 + np.digitize(F['g2'], edges)) * 2 + F['neg']
    q, k = table(fk(FA), cA, fk(FB)); rows.append(('gaps-table', q, k))
    pk = lambda F: ((np.digitize(F['g1'], coarse) * 7 + np.digitize(F['g2'], coarse)) * 2 + F['neg']) * 100 + F['pM'] * 10 + F['pS']
    q, k = table(pk(FA), cA, pk(FB)); rows.append(('pos-table', q, k))
    gen = th.Generator().manual_seed(0)
    with th.no_grad():
        P1, P2 = widths(W, 1, gen), widths(W, 2, gen)
    npar = int((P1[0] > 0).sum() + (P2[0] > 0).sum()) * 3 + 2
    eA, eB = tie_q(P1, P2, FA), tie_q(P1, P2, FB)
    bg = [1 - cA[(FA['neg'] == v)].mean() - eA[(FA['neg'] == v)].mean() for v in (0, 1)]   # background error rates
    bgB = np.clip(np.where(FB['neg'], bg[1], bg[0]), 1e-6, None)
    qt = (1 - eB) * (1 - bgB); rows.append(('tie-layer', qt, npar))
    lgt = lambda e, bgv: np.log(np.clip((1 - e) * (1 - bgv), 1e-9, 1 - 1e-9) / np.clip(1 - (1 - e) * (1 - bgv), 1e-9, 1))
    bgA = np.clip(np.where(FA['neg'], bg[1], bg[0]), 1e-6, None)
    w2 = logistic(np.c_[XA, lgt(eA, bgA)], cA)
    rows.append(('tie+gaps', 1 / (1 + np.exp(-np.c_[XB, lgt(eB, bgB)] @ w2)), npar + len(w2)))
    H1 = bits(rows[1][1], cB)
    print(f"\n{name}: accuracy {cB.mean():.4f}, N_A = N_B = 2^22, {BITS:.0f} bits/param")
    print(f"{'predictor':12s} {'bits/input':>10s} {'% of const':>10s} {'params':>7s} {'param bits':>10s} {'beats const at N >':>18s}")
    for nm, q, k in rows:
        H = bits(q, cB); be = (k - 1) * BITS / (H1 - H) if H < H1 and k > 1 else float('nan')
        print(f"{nm:12s} {H:10.4f} {100 * H / H1:9.1f}% {k:7d} {k * BITS:10.0f} {be:18.3g}", flush=True)


if __name__ == '__main__':
    run('trained M(16,10)', load_weights('weights.npz'))
    s = np.load('seeds_h16.npz'); d = np.load('noisy_sigma0.005.npz')
    run('seed0', [s[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
    run('noisy0', [d[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
