"""Surprise accounting with the FORM skeleton widths as the explanation. Same scoring as
probes/surprise.py (cross-entropy of the correctness bit on split B, 2^22 inputs), with the
tie-layer predictor's per-pair width distribution replaced by the per-skeleton widths of
probes/form_bg.form (mean background, no noise; 15x7 quadrature nodes): p(err | pair, gap)
= fraction of that pair's skeleton widths exceeding the gap. No tie sampling, no labels in
the widths (the two background error rates are from split A, as in surprise.py). Params:
one width per skeleton node is the honest count (the table IS the explanation), reported
both ways. Against: constant, measured tie-layer (2^18 sampled ties).
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import forward, load_weights
import surprise, form_bg
from surprise import feats, widths, tie_q, bits, N, BITS
from form_bg import form


def form_q(W, F, nodes=(15, 7)):
    form_bg.VQ = list((np.arange(nodes[0]) + .5) / nodes[0]); form_bg.OQ = list((np.arange(nodes[1]) + .5) / nodes[1])
    p, npar = {}, 0
    for kind, g, pi, pj in (('S/R', F['g2'], F['pS'], F['pR']), ('M/S', F['g1'], F['pM'], F['pS'])):
        key, w, _ = form(W, kind, False, False, raw=True); npar += len(w)
        ws = [np.sort(w[key == k]) for k in range(100)]
        pe = np.zeros(len(g)); pair = pi * 10 + pj
        for k in range(100):
            m = pair == k
            if m.any() and len(ws[k]): pe[m] = 1 - np.searchsorted(ws[k], g[m], side='right') / len(ws[k])
        p[kind] = pe
    return 1 - (1 - p['S/R']) * (1 - p['M/S']), npar


def run(name, W):
    W = [th.tensor(a, dtype=th.float32) for a in W]
    xa = th.randn(N, 10, generator=th.Generator().manual_seed(1)); xb = th.randn(N, 10, generator=th.Generator().manual_seed(2))
    FA, FB = feats(xa), feats(xb)
    with th.no_grad():
        cA = (forward(xa, *W).argmax(-1) == FA['label']).numpy().astype(float)
        cB = (forward(xb, *W).argmax(-1) == FB['label']).numpy().astype(float)
        surprise.NT = 1 << 18; gen = th.Generator().manual_seed(0); P1, P2 = widths(W, 1, gen), widths(W, 2, gen)
    rows = [('constant', np.full(N, cA.mean()), 1)]
    for nm, (eA, eB), k in (('tie-layer', (tie_q(P1, P2, FA), tie_q(P1, P2, FB)), int((P1[0] > 0).sum() + (P2[0] > 0).sum()) * 3 + 2),
                            ('FORM-tie', *(lambda a, b: ((a[0], b[0]), b[1] + 2))(form_q(W, FA), form_q(W, FB)))):
        bg = [1 - cA[FA['neg'] == v].mean() - eA[FA['neg'] == v].mean() for v in (0, 1)]
        bgB = np.clip(np.where(FB['neg'], bg[1], bg[0]), 1e-6, None)
        rows.append((nm, (1 - eB) * (1 - bgB), k))
        if nm == 'FORM-tie': rows.append(('FORM-tie (200 pair means as params)', rows[-1][1], 202))
    H1 = bits(rows[0][1], cB)
    print(f"\n{name}: accuracy {cB.mean():.4f}")
    print(f"{'predictor':36s} {'bits/input':>10s} {'% of const':>10s} {'params':>7s} {'beats const at N >':>18s}")
    for nm, q, k in rows:
        H = bits(q, cB); be = (k - 1) * BITS / (H1 - H) if H < H1 and k > 1 else float('nan')
        print(f"{nm:36s} {H:10.4f} {100 * H / H1:9.1f}% {k:7d} {be:18.3g}", flush=True)


if __name__ == '__main__':
    run('trained M(16,10)', load_weights('weights.npz'))
    s = np.load('seeds_h16.npz'); d = np.load('noisy_sigma0.005.npz')
    run('seed0', [s[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
    run('noisy0', [d[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
