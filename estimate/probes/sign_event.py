"""Deduce the sign-event error (at most one positive input) and add it to the tie skeleton.

The tie skeleton (probes/form_abs_all.py) deduces the S/R + M/S boundary layers and leaves
out the 'other' errors; on the trained model 57% of those far-field errors are the sign
event npos <= 1 (FINDINGS "Far-field errors"). Its mass is analytic: P(npos = 1) = 10/1024,
P(npos = 0) = 1/1024. The failure rate there (~0.7) is bulk, not a strip, so instead of a
width the skeleton gives P(correct) directly:
  npos = 1: positive max at pM with value u (half-normal), label = the largest negative at
            pS with value v (law of the max of 9 negatives), the other 8 negatives at
            their conditional mean given < v.
  npos = 0: max w and label v < w (top two of 10 negatives), the other 8 at their mean
            given < v.
  Nodes at CDF midpoints (u 5, v 15 for npos = 1; w 15, v 7 for npos = 0), all position pairs.
  margin m = logit[label] - logit[best other at the skeleton], linearised in the 8
  background inputs: P(correct) = Phi(m0 / s), s^2 = sum (dm/dx_i)^2 var_i ('FORM');
  'step' = 1[m0 > 0] at the mean background.
No labels and no random inputs. MC columns (2^22 inputs) are for validation only.
Deduced total error = tie skeleton (15x7) + sign-event mass x (1 - P(correct)).

Usage: uv run python probes/sign_event.py
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from scipy.stats import norm, truncnorm
from ground_truth import forward
from strat_tie import models
from form_abs_all import deduced

MASS = {1: 10 / 1024, 0: 1 / 1024}


def mid(n): return (np.arange(n) + .5) / n


def skeletons(npos):
    xs, lab, sd, bgm, wt = [], [], [], [], []
    if npos == 1:
        outer = [(norm.ppf(.5 + .5 * a), norm.ppf(.5 * b ** (1 / 9))) for a in mid(5) for b in mid(15)]   # (u, v)
    else:
        outer = []
        for a in mid(15):
            w = norm.ppf(.5 * a ** (1 / 10))
            outer += [(w, norm.ppf(norm.cdf(w) * b ** (1 / 9))) for b in mid(7)]                         # (w, v)
    for top, v in outer:
        t = truncnorm(-np.inf, v); mu, s = t.mean(), t.std()
        for pM in range(10):
            for pS in range(10):
                if pS == pM: continue
                x = np.full(10, mu); x[pM] = top; x[pS] = v
                xs.append(x); lab.append(pS); sd.append(s); bgm.append(~np.isin(np.arange(10), (pM, pS)))
    return np.array(xs), np.array(lab), np.array(sd), np.array(bgm)


def p_correct(W, npos):
    xs, lab, sd, bgm = skeletons(npos)
    x = th.tensor(xs, dtype=th.float32, requires_grad=True); ar = th.arange(len(x)); y = th.tensor(lab)
    l = forward(x, *W); other = l.detach().clone(); other[ar, y] = -np.inf; c = other.argmax(-1)
    m = l[ar, y] - l[ar, c]
    g, = th.autograd.grad(m.sum(), x)
    m0 = m.detach().numpy(); s = np.sqrt(((g.numpy() * bgm) ** 2).sum(1)) * sd
    return norm.cdf(m0 / np.maximum(s, 1e-9)).mean(), (m0 > 0).mean()


def mc(W):
    x = th.randn(1 << 22, 10, generator=th.Generator().manual_seed(23))
    with th.no_grad():
        out = forward(x, *W).argmax(-1)
    lab = x.topk(2, -1).indices[:, 1]; npos = (x > 0).sum(1)
    r = {k: (out[npos == k] == lab[npos == k]).float().mean().item() for k in (0, 1)}
    return r, (out != lab).float().mean().item()


def main():
    print('P(correct | sign event): deduced FORM / step vs MC; error rates: tie skeleton, + sign event (FORM), MC total')
    print(f"{'model':8s} {'npos=1 FORM step MC':>22s} {'npos=0 FORM step MC':>22s} | {'tie':>6s} {'tie+sign':>8s} {'total':>7s} {'rel tie':>8s} {'rel +sign':>9s}")
    rows = []
    for name, W in models:
        W = [th.tensor(np.asarray(a), dtype=th.float32) for a in W]
        (f1, s1), (f0, s0) = p_correct(W, 1), p_correct(W, 0)
        r, tot = mc(W)
        tie = deduced(W, 15, 7)[0]; ded = tie + MASS[1] * (1 - f1) + MASS[0] * (1 - f0)
        rows.append((tie / tot - 1, ded / tot - 1))
        print(f'{name:8s} {f1:6.3f} {s1:6.3f} {r[1]:6.3f}    {f0:6.3f} {s0:6.3f} {r[0]:6.3f} | {tie:.4f} {ded:8.4f} {tot:7.4f} '
              f'{100 * (tie / tot - 1):+7.1f}% {100 * (ded / tot - 1):+8.1f}%', flush=True)
    a = np.array(rows)
    print(f'\nrms relative error vs total: tie skeleton {np.sqrt((a[:, 0] ** 2).mean()):.3f}, + sign event {np.sqrt((a[:, 1] ** 2).mean()):.3f}')


if __name__ == '__main__':
    main()
