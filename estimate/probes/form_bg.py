"""FORM (first-order reliability) per skeleton, with the background as NOISE instead of a
constant. probes/pair_widths.py's skeletons fixed the 7-8 background inputs at a constant
and got the per-pair widths wrong by 2-3x for any constant. Here, per ordered pair and
quadrature node, the background sits at its conditional mean (values below the tie value v)
and the margin m is linearised there: m ~ N(m0, s^2) with s^2 = sum_i (dm/dx_i)^2 var_i over
background inputs i (truncated-normal variance). Expected boundary-layer width is then
  E[(-m0)_+] / alpha = (s phi(m0/s) - m0 Phi(-m0/s)) / alpha,
which replaces the constant-background width (-m0)_+ / alpha. Deductive: no labels, no
random inputs (the 'perm' variant assigns the expected order statistics of the background
to positions by a few fixed permutations instead of a constant). Scored exactly as
pair_widths.py: correlation of logs and rms log ratio against the actual per-pair rates.
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from scipy.stats import norm, truncnorm
from ground_truth import forward, load_weights
from boundary_layer import sample_ties
from pair_widths import actual, measured, score, skeleton

VQ = [.1, .3, .5, .7, .9]          # quantiles of the tie value v
OQ = [.17, .5, .83]                # quantiles of the other top value (M for S/R)
NPERM = 8


def tie_quantiles(k):
    x, _, _ = sample_ties(k, 1 << 16, th.Generator().manual_seed(k)); v = x.max(-1).values if k == 1 else x.topk(2, -1).values[:, 1]
    return np.quantile(v.numpy(), VQ)


def form(W, kind, perm, noise=True, raw=False):
    """Returns the per-pair relative-width matrix and the mean ABSOLUTE expected width
    E[|m0|]/alpha (capped at .25) over skeletons, which times 0.5 f_D(0) is the
    boundary-layer error estimate of boundary_layer.py. noise=False gives the plain
    skeleton at the mean background."""
    k = 2 if kind == 'S/R' else 1
    xs, his, los, sig, bgm = [], [], [], [], []
    rng = np.random.default_rng(0)
    for v in tie_quantiles(k):
        lo_t = truncnorm(-np.inf, v); mu_b, sd_b = lo_t.mean(), lo_t.std()
        nb = 7 if kind == 'S/R' else 8
        eos = lo_t.ppf((np.arange(nb) + .5) / nb)             # expected order statistics (approx)
        for hi in range(10):
            for lo in range(10):
                if hi == lo: continue
                others = [p for p in range(10) if p not in (hi, lo)]
                for o in (others if kind == 'S/R' else [None]):
                    bg = [p for p in others if p != o]
                    for oq in (OQ if kind == 'S/R' else [None]):
                        for r in range(NPERM if perm else 1):
                            x = np.zeros(10); x[hi] = x[lo] = v
                            if kind == 'S/R': x[o] = truncnorm(v, np.inf).ppf(oq)
                            x[bg] = rng.permutation(eos) if perm else mu_b
                            xs.append(x); his.append(hi); los.append(lo); sig.append(sd_b); bgm.append(np.isin(np.arange(10), bg))
    x = th.tensor(np.array(xs), dtype=th.float32, requires_grad=True); hi = th.tensor(his); lo = th.tensor(los); ar = th.arange(len(x))
    tgt, cmp = (hi, lo) if kind == 'S/R' else (lo, hi)
    l = forward(x, *W); m = l[ar, tgt] - l[ar, cmp]
    g, = th.autograd.grad(m.sum(), x); g = g.numpy(); m0 = m.detach().numpy(); am = l.argmax(-1).numpy()
    al = (g[ar, hi] - g[ar, lo]) / 2
    s = np.sqrt(((g * np.array(bgm)) ** 2).sum(-1)) * np.array(sig)
    if not noise: s = np.full_like(s, 1e-9)
    ew = s * norm.pdf(m0 / s) - m0 * norm.sf(m0 / s)          # E[(-m0)_+] under N(m0, s^2)
    ea = 2 * ew + m0                                           # E[|m0|]
    good = (al > 0) & ((am == his) | (am == los))
    wa = np.where(good, np.minimum(ea / np.maximum(al, 1e-9), .25), 0)
    w = np.where(good, np.minimum(ew / np.maximum(al, 1e-9), .25), 0) / np.abs(np.array(xs)).max(-1)
    key = np.array(his) * 10 + np.array(los); M = np.zeros(100); np.add.at(M, key, w)
    if raw: return key, w, wa                                  # per-skeleton (pair key, relative width, absolute width)
    return (M / np.maximum(np.bincount(key, minlength=100), 1)).reshape(10, 10), wa.mean()


def run(name, W):
    W = [th.tensor(a, dtype=th.float32) for a in W]
    with th.no_grad():
        (aSR, aMS), (wSR, wMS) = actual(W), measured(W)
    kSR, kMS = skeleton(W)
    print(f"\n{name}")
    for kind, A, Wm, K in (('S/R', aSR, wSR, kSR), ('M/S', aMS, wMS, kMS)):
        print(f"  {kind}: measured widths     {score(A, Wm)}")
        print(f"       skeleton const b   {score(A, K)}")
        print(f"       FORM const bg      {score(A, form(W, kind, False)[0])}")
        print(f"       FORM perm bg       {score(A, form(W, kind, True)[0])}", flush=True)


if __name__ == '__main__':
    run('trained M(16,10)', load_weights('weights.npz'))
    d = np.load('noisy_sigma0.005.npz'); s = np.load('seeds_h16.npz')
    run('noisy0', [d[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
    run('seed0', [s[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
