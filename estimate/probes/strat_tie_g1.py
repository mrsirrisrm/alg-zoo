"""Tie-layer allocation with the S/R width conditioned on the M/S gap. The comparator works
relative to the running max (FINDINGS: accuracy falls as M - S grows), so the S/R boundary
layer width should depend on g1 = (vM - v)/|vM| at the tie, which the tie-conditional law
leaves random. Widths are still from network margins at sampled ties (no labels), now per
(S pos, R pos, g1 bin). Same Neyman scoring as probes/strat_tie.py (2^14 ties per kind).
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from scipy.stats import norm
from ground_truth import forward
from boundary_layer import sample_ties
import surprise
from surprise import widths
from strat_tie import N, models, FB, KB, var

G1 = [0, .05, .15, .3, .6, 1.2, np.inf]
NT = 1 << 16


def widths_sr_g1(W, gen):
    x, a, b = sample_ties(2, NT, gen); ar = th.arange(NT)
    vM = x.max(-1).values; g1 = ((vM - x[ar, a]) / vM.abs()).numpy(); gb = np.digitize(g1, G1) - 1
    rows = []
    for hi, lo in ((a, b), (b, a)):
        def diff(dd):
            y = x.clone(); y[ar, hi] += dd / 2; y[ar, lo] -= dd / 2
            l = forward(y, *W); return l[ar, hi] - l[ar, lo], l.argmax(-1)
        m0, am = diff(0.); al = (diff(1e-3)[0] - diff(-1e-3)[0]) / 2e-3
        w = (-m0 / al.clamp_min(1e-9)).clamp(0, .25) / x.abs().max(-1).values
        w = th.where(((am == hi) | (am == lo)) & (al > 0), w, th.zeros_like(w))
        rows.append((hi.numpy(), lo.numpy(), w.numpy()))
    hi, lo, w = (np.concatenate(z) for z in zip(*rows)); g = np.concatenate([gb, gb])
    P = np.zeros((3, 10, 10, len(G1) - 1)); pooled = np.zeros((3, len(G1) - 1))
    for c in range(len(G1) - 1):          # pooled-over-pairs fallback for sparse cells
        s = w[g == c]; sp = s[s > 0]
        if len(sp) > 10: pooled[:, c] = len(sp) / len(s), np.log(sp).mean(), np.log(sp).std()
    for i in range(10):
        for j in range(10):
            for c in range(len(G1) - 1):
                m = (hi == i) & (lo == j) & (g == c); s = w[m]; sp = s[s > 0]
                P[:, i, j, c] = (len(sp) / m.sum(), np.log(sp).mean(), np.log(sp).std()) if len(sp) > 10 else pooled[:, c]
    return P


def q_err(P_ms, P_sr, F):
    gb = np.clip(np.digitize(F['g1'], G1) - 1, 0, len(G1) - 2)
    f, mu, sd = (P_sr[k, F['pS'], F['pR'], gb] for k in range(3))
    p_sr = f * norm.sf((np.log(np.maximum(F['g2'], 1e-12)) - mu) / np.maximum(sd, 1e-3))
    f, mu, sd = (P_ms[k, F['pM'], F['pS']] for k in range(3))
    p_ms = f * norm.sf((np.log(np.maximum(F['g1'], 1e-12)) - mu) / np.maximum(sd, 1e-3))
    return 1 - (1 - p_sr) * (1 - p_ms)


xb = th.randn(N, 10, generator=th.Generator().manual_seed(2))
print("gains over MC (held-out): tie-layer with S/R width per (pair, g1 bin), label-free")
with th.no_grad():
    for name, W in models:
        W = [th.tensor(a, dtype=th.float32) for a in W]
        eb = (forward(xb, *W).argmax(-1) != FB['label']).numpy().astype(float); e = eb.mean()
        gen = th.Generator().manual_seed(5); surprise.NT = NT
        et = q_err(widths(W, 1, gen), widths_sr_g1(W, gen), FB)
        row = [f"{name:8s}"]
        for k in KB:
            K = int(KB[k].max()) + 1; cb = np.bincount(KB[k], minlength=K); p = cb / N
            q = np.bincount(KB[k], eb, minlength=K) / np.maximum(cb, 1); sb = np.sqrt(q * (1 - q))
            ph = np.bincount(KB[k], et, minlength=K) / np.maximum(cb, 1)
            row.append(f"| {k}: {e * (1 - e) / var(p, sb, np.sqrt(ph * (1 - ph))):4.2f}")
        print(' '.join(row), flush=True)
