"""Is n7 a comparator of x_t against n2's stored value, and does a relative error eps in
the n2->n7 weight move the tie walls by ~eps? Controls: other single weights of the stiff
circuit and a few soft ones, same relative eps."""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from ground_truth import forward, load_weights
from boundary_layer import sample_ties
th.manual_seed(0)
W0 = [th.tensor(a, dtype=th.float32) for a in load_weights('weights.npz')]
Wih, Whh, Wout = W0

def states(x, Wih, Whh):
    h = th.relu(x[:, 0:1] * Wih); H = [h]; P = [x[:, 0:1] * Wih]
    for t in range(1, x.shape[1]):
        p = x[:, t:t + 1] * Wih + h @ Whh.T; h = th.relu(p); H.append(h); P.append(p)
    return th.stack(H, 1), th.stack(P, 1)  # n, T, 16

# ---------- 1. what n2 stores, and whether n7's sign is the comparison 0.674 n2 vs 13.17 x_t
x = th.randn(1 << 18, 10, generator=th.Generator().manual_seed(1))
with th.no_grad(): H, P = states(x, Wih, Whh)
rmax = th.cummax(x, 1).values
print("n2 vs running max (t>=1): corr", np.corrcoef(H[:, :, 2].flatten(), rmax.flatten())[0, 1].round(3),
      "| n2/runmax where runmax>0.5: median", (H[:, :, 2] / rmax)[rmax > .5].median().item().__round__(2),
      "iqr", np.percentile((H[:, :, 2] / rmax)[rmax > .5].numpy(), [25, 75]).round(2))
for t in (3, 6, 9):
    A = np.c_[rmax[:, t - 1], th.topk(x[:, :t], 2, 1).values[:, 1] if t > 1 else 0 * rmax[:, 0], x[:, :t].clamp(min=0).sum(1), np.ones(len(x))]
    c, res, *_ = np.linalg.lstsq(A, H[:, t - 1, 2].numpy(), rcond=None)
    r2 = 1 - res[0] / ((H[:, t - 1, 2] - H[:, t - 1, 2].mean()) ** 2).sum().item()
    print(f"  t={t}: n2 ~ {c[0]:+.2f} max + {c[1]:+.2f} 2nd + {c[2]:+.2f} sum(x+) {c[3]:+.2f}  R2 {r2:.3f}")
p7 = P[:, 1:, 7]; cmp = Whh[7, 2] * H[:, :-1, 2] + Wih[7] * x[:, 1:]
rest = p7 - cmp
print(f"n7 preact: var share of (0.674 n2 + w x_t) {1 - rest.var().item() / p7.var().item():.3f}; "
      f"sign agreement {((p7 > 0) == (cmp > 0)).float().mean().item():.3f}; n7 active {(p7 > 0).float().mean().item():.3f}")
thr = (Whh[7, 2] * H[:, :-1, 2] / -Wih[7]).clamp(max=1e9) if Wih[7] < 0 else None
print(f"n7 fires when x_t < {(Whh[7, 2] / -Wih[7]).item():.4f} * n2_(t-1) (+ rest); "
      f"threshold / running max: median {(thr / rmax[:, :-1])[rmax[:, :-1] > .5].median().item():.3f}")
for j in (1, 6, 8):
    pj = P[:, 1:, j]; terms = {i: (Whh[j, i] * H[:, :-1, i]).var().item() for i in range(16)}; terms['x'] = (Wih[j] * x[:, 1:]).var().item()
    top = sorted(terms.items(), key=lambda kv: -kv[1])[:4]
    print(f"  n{j} preact largest input terms (var): " + ", ".join(f"{k if k == 'x' else 'n%d' % k}:{v / pj.var().item():.2f}" for k, v in top))

# ---------- 2. tie-wall shift from a relative error in single weights
NT = 1 << 16
ties = {k: sample_ties(k, NT, th.Generator().manual_seed(10 + k)) for k in (1, 2)}
def walls(W, k):
    x, a, b = ties[k]; ar = th.arange(NT); out = []
    for hi, lo in ((a, b), (b, a)):
        tgt, cm = (lo, hi) if k == 1 else (hi, lo)
        def m(dd):
            y = x.clone(); y[ar, hi] += dd / 2; y[ar, lo] -= dd / 2
            l = forward(y, *W); return l[ar, tgt] - l[ar, cm]
        m0 = m(0.); al = (m(1e-3) - m(-1e-3)) / 2e-3
        out.append((-m0 / al) / x.abs().max(-1).values)
    w = th.cat(out); return w
with th.no_grad():
    base = {k: walls(W0, k) for k in (1, 2)}
xa = th.randn(1 << 21, 10, generator=th.Generator().manual_seed(3)); lab = xa.topk(2, -1).indices[:, 1]
with th.no_grad(): c0 = (forward(xa, *W0).argmax(-1) == lab)
EPS = 1e-2
cands = [('hh', 7, 2, 'n2->n7'), ('hh', 2, 2, 'n2->n2 self'), ('hh', 8, 7, 'n7->n8'), ('hh', 1, 7, 'n7->n1'),
         ('hh', 6, 2, 'n2->n6'), ('ih', 7, None, 'in->n7'), ('hh', 1, 6, 'n6->n1'),
         ('hh', 3, 5, 'n5->n3 (soft)'), ('hh', 12, 14, 'n14->n12 (soft)'), ('out', 1, 2, 'n2->out1 (soft)')]
wpos = lambda w: w.clamp(0, .25)
print(f"\nrelative error eps = {EPS:+g} in one weight; wall shift in relative-gap units (divide by eps for gain)")
print(f"{'weight':18s} {'value':>8s} {'M/S rms shift/eps':>17s} {'strip chg':>9s} {'S/R rms shift/eps':>17s} {'strip chg':>9s} {'acc change':>10s}")
for kind, i, j, nm in cands:
    for s in (EPS, -EPS):
        W = [w.clone() for w in W0]
        if kind == 'hh': W[1][i, j] *= 1 + s; v = Whh[i, j]
        elif kind == 'ih': W[0][i] *= 1 + s; v = Wih[i]
        else: W[2][i, j] *= 1 + s; v = Wout[i, j]
        with th.no_grad():
            d = {k: (walls(W, k) - base[k]) for k in (1, 2)}
            dacc = ((forward(xa, *W).argmax(-1) == lab).float().mean() - c0.float().mean()).item()
        ok = {k: d[k].isfinite() & (d[k].abs() < .1) & base[k].isfinite() for k in (1, 2)}
        dm = {k: d[k][ok[k]].pow(2).mean().sqrt().item() for k in (1, 2)}                       # rms wall shift
        dw = {k: (wpos(base[k] + d[k])[ok[k]].mean() - wpos(base[k])[ok[k]].mean()).item() / wpos(base[k])[ok[k]].mean().item() for k in (1, 2)}
        print(f"{nm if s > 0 else '':18s} {v.item() if s > 0 else 0:8.3f} {dm[1] / abs(s):17.3f} {dw[1]:+9.1%} {dm[2] / abs(s):17.3f} {dw[2]:+9.1%} {dacc:+10.4f}  (eps {s:+g})")
print("typical |wall| (rms of base widths, relative gap): M/S", base[1][base[1].isfinite() & (base[1].abs() < .1)].pow(2).mean().sqrt().item().__round__(4),
      "S/R", base[2][base[2].isfinite() & (base[2].abs() < .1)].pow(2).mean().sqrt().item().__round__(4))
