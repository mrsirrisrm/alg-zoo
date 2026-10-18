"""Can the settling-time mechanism SHORTEN the tie-layer explanation? probes/surprise.py's
tie-layer predictor stores a lognormal width per ordered position pair (542 params). The
settling result says the width is mainly a function of the arrival step of the later of
the two tied values (how many steps remain to settle). Same scoring, widths pooled by
  pair      (hi, lo)                 as before
  later     max(hi, lo)              9 cells per kind
  later+ord max(hi, lo), hi < lo     18 cells per kind (which of the two came later)
  global    one cell per kind
At N = 2^22 the parameter bits hardly matter; the point is the crossover N at which each
explanation beats the constant predictor, i.e. how short the explanation can be.
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from scipy.stats import norm
from ground_truth import forward, load_weights
from boundary_layer import sample_ties
import surprise
from surprise import feats, bits, N, BITS

NT = 1 << 18
POOL = {'pair': lambda hi, lo: hi * 10 + lo, 'later': lambda hi, lo: np.maximum(hi, lo),
        'later+ord': lambda hi, lo: np.maximum(hi, lo) * 2 + (hi < lo), 'global': lambda hi, lo: np.zeros_like(hi)}


def raw_widths(W, k, gen):
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
    return [np.concatenate(z) for z in zip(*out)]


def fit(hi, lo, w, keyfn):
    key = keyfn(hi, lo); P = {}
    for c in np.unique(key):
        s = w[key == c]; sp = s[s > 0]
        if len(sp) > 10: P[c] = (len(sp) / len(s), np.log(sp).mean(), max(np.log(sp).std(), 1e-3))
    return P


def q_err(P, keyfn, pi, pj, g):
    key = keyfn(pi, pj); pe = np.zeros(len(g))
    for c, (f, mu, sd) in P.items():
        m = key == c; pe[m] = f * norm.sf((np.log(np.maximum(g[m], 1e-12)) - mu) / sd)
    return pe


def run(name, W):
    W = [th.tensor(a, dtype=th.float32) for a in W]
    xa = th.randn(N, 10, generator=th.Generator().manual_seed(1)); xb = th.randn(N, 10, generator=th.Generator().manual_seed(2))
    FA, FB = feats(xa), feats(xb)
    with th.no_grad():
        cA = (forward(xa, *W).argmax(-1) == FA['label']).numpy().astype(float)
        cB = (forward(xb, *W).argmax(-1) == FB['label']).numpy().astype(float)
        R1, R2 = raw_widths(W, 1, th.Generator().manual_seed(1)), raw_widths(W, 2, th.Generator().manual_seed(2))
    H1 = bits(np.full(N, cA.mean()), cB)
    print(f"\n{name}: accuracy {cB.mean():.4f}; constant {H1:.4f} bits/input")
    print(f"{'pooling':10s} {'bits/input':>10s} {'% of const':>10s} {'params':>7s} {'beats const at N >':>18s}")
    for nm, kf in POOL.items():
        P1, P2 = fit(*R1, kf), fit(*R2, kf); k = 3 * (len(P1) + len(P2)) + 2
        e = {}
        for F in (FA, FB):
            e[id(F)] = 1 - (1 - q_err(P1, kf, F['pM'], F['pS'], F['g1'])) * (1 - q_err(P2, kf, F['pS'], F['pR'], F['g2']))
        bg = [1 - cA[FA['neg'] == v].mean() - e[id(FA)][FA['neg'] == v].mean() for v in (0, 1)]
        bgB = np.clip(np.where(FB['neg'], bg[1], bg[0]), 1e-6, None)
        H = bits((1 - e[id(FB)]) * (1 - bgB), cB); be = (k - 1) * BITS / (H1 - H) if H < H1 else float('nan')
        print(f"{nm:10s} {H:10.4f} {100 * H / H1:9.1f}% {k:7d} {be:18.3g}", flush=True)


if __name__ == '__main__':
    run('trained M(16,10)', load_weights('weights.npz'))
    s = np.load('seeds_h16.npz'); d = np.load('noisy_sigma0.005.npz')
    run('seed0', [s[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
    run('noisy0', [d[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
