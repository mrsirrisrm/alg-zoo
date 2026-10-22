"""Surprise accounting with the J5 family as the explanation (follow-up to PLAN J1, J5).

Observation: the correctness bit c(x) of the model, held out (probes/surprise.py, 2^22
inputs per split, fitted parameters charged 1/2 log2 N_A = 11 bits each).
Explanation: hand10 + 2 gains (e, g) fitted per model to its error-vs-gap curve as in
probes/hand_gains.py (description ~100 bits for hand10, ~16 for the gains), run on x:
  fam-agree   q = P(c | family correct at x): 2 fitted rates
  fam-logit   logistic in [family correct, log |family margin| / |vM| split by the
              family's sign, sign(vS)]: 5 fitted weights
  fam+gaps    fam-logit features plus surprise.py's gap-logit design: 13 weights
Ladder from surprise.py for reference: constant, gaps-logit, pos-table (tie-layer: see
FINDINGS "Against ARC's ask", trained 57.3%).
Then where the residual bits sit, by region of x: sign event (npos <= 1), near-tie
(npos >= 2, a relative gap g1 or g2 < 0.1), far (the rest).
J1 bookkeeping: total = description + N * residual bits/input. The weights describe
c(x) exactly with their precision bits (J1, tol 1e-3), so N* = (precision - family bits)
/ (family residual bits/input) is the input count above which the weights are the
shorter account.

Usage: uv run python probes/surprise_family.py [names...]
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from scipy.optimize import minimize
from ground_truth import forward
from opt_bits import loaders
from surprise import feats, design, logistic, table, bits, coarse, N, BITS
from hand_gains import hand, curves

PREC = {'trained': 2429, 'seed0': 2516, 'seed2': 3247}   # J1 precision bits, tol 1e-3 (opt_bits.out)
DESC = 116                                               # hand10 construction + 2 gains


def fit_gains(W):
    tgt = curves(W)[2]
    f = lambda p: float(((np.log(curves(hand(*p))[2] + 3e-3) - np.log(tgt + 3e-3)) ** 2).sum())
    return min((minimize(f, p0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-4, 'maxfev': 300})
                for p0 in (np.array([0.01, 0.01]), np.array([-0.02, 0.01]))), key=lambda r: r.fun).x


def logits(W, x):
    Wt = [th.tensor(a, dtype=th.float32) for a in W]
    with th.no_grad():
        return th.cat([forward(x[i:i + (1 << 20)], *Wt) for i in range(0, len(x), 1 << 20)]).numpy()


def fam_design(Lf, F, x):
    lab = F['label'].numpy(); ar = np.arange(len(lab))
    other = Lf.copy(); other[ar, lab] = -np.inf
    m = (Lf[ar, lab] - other.max(1)) / x.abs().max(-1).values.numpy()
    ok = (m > 0).astype(float); lm = np.log(np.abs(m) + 1e-9)
    return np.stack([np.ones_like(ok), ok, ok * lm, (1 - ok) * lm, F['neg']], 1), ok


def run(name, W):
    p = fit_gains(W); Wf = hand(*p)
    xa = th.randn(N, 10, generator=th.Generator().manual_seed(1)); xb = th.randn(N, 10, generator=th.Generator().manual_seed(2))
    FA, FB = feats(xa), feats(xb)
    cA = (logits(W, xa).argmax(1) == FA['label'].numpy()).astype(float)
    cB = (logits(W, xb).argmax(1) == FB['label'].numpy()).astype(float)
    rows = [('constant', np.full(N, cA.mean()), 1)]
    XA, XB = design(FA), design(FB); w = logistic(XA, cA)
    rows.append(('gaps-logit', 1 / (1 + np.exp(-XB @ w)), len(w)))
    pk = lambda F: ((np.digitize(F['g1'], coarse) * 7 + np.digitize(F['g2'], coarse)) * 2 + F['neg']) * 100 + F['pM'] * 10 + F['pS']
    q, k = table(pk(FA), cA, pk(FB)); rows.append(('pos-table', q, k))
    DA, okA = fam_design(logits(Wf, xa), FA, xa); DB, okB = fam_design(logits(Wf, xb), FB, xb)
    rA = [cA[okA == v].mean() for v in (0, 1)]
    rows.append(('fam-agree', np.where(okB == 1, rA[1], rA[0]), 2))
    w = logistic(DA, cA); rows.append(('fam-logit', 1 / (1 + np.exp(-DB @ w)), len(w)))
    w = logistic(np.c_[XA, DA[:, 1:]], cA); rows.append(('fam+gaps', 1 / (1 + np.exp(-np.c_[XB, DB[:, 1:]] @ w)), len(w)))
    H1 = bits(rows[0][1], cB)
    npos = (xb.numpy() > 0).sum(1)
    reg = {'sign (npos<=1)': npos <= 1, 'near-tie': (npos >= 2) & (np.minimum(FB['g1'], FB['g2']) < 0.1)}
    reg['far'] = ~reg['sign (npos<=1)'] & ~reg['near-tie']
    print(f'\n{name}: acc {cB.mean():.4f}  gains e={p[0]:+.4f} g={p[1]:+.4f}  family agrees with the net on c(x) '
          f'{(okB == cB).mean():.4f}')
    print('  regions (mass / share of net errors): ' + '  '.join(f'{k} {m.mean():.3f}/{(m & (cB == 0)).sum() / (cB == 0).sum():.2f}' for k, m in reg.items()))
    print(f"  {'predictor':11s} {'bits/in':>8s} {'% const':>7s} {'params':>6s} | residual share by region: " + ' '.join(f'{k:>14s}' for k in reg))
    for nm, q, k in rows:
        qq = np.clip(q, 1e-9, 1 - 1e-9); b = -(cB * np.log2(qq) + (1 - cB) * np.log2(1 - qq))
        print(f'  {nm:11s} {b.mean():8.4f} {100 * b.mean() / H1:6.1f}% {k:6d} | ' + ' '.join(f'{b[m].sum() / b.sum():14.2f}' for m in reg.values()))
    Hf = bits(rows[-2][1], cB); desc = DESC + rows[-2][2] * BITS
    if name in PREC:
        print(f'  J1: family account {desc:.0f} bits + N x {Hf:.4f}; weights {PREC[name]} bits + N x 0 '
              f'-> the weights are shorter above N* = {(PREC[name] - desc) / Hf:.0f} inputs')


if __name__ == '__main__':
    L = loaders(); names = sys.argv[1:] or ['trained', 'seed0', 'seed2']
    for n in names:
        run(n, L[n]())
