"""C1 (PLAN.md): what are the noisy copies' NEW errors? Same 2^22 inputs through the trained
model and 8 copies (sigma = 0.005). A copy's errors split into persistent (trained also
wrong) and new; new errors are classified by mechanism (A/B swap, within-side pick, predicts
M, other), by the TRAINED model's logit margin on that input (is it a near-boundary input
pushed over, as the first-order result says?), and by tie gap. Also the reverse flips
(trained wrong, copy right). Question: is the extra error one event, or the trained
error set scaled up?
Run from estimate/: python probes/copy_failures.py
"""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from ground_truth import forward, load_weights

N = 1 << 22
x = th.randn(N, 10, generator=th.Generator().manual_seed(4))
W0 = [th.tensor(a) for a in load_weights('weights.npz')]
d = np.load('noisy_sigma0.005.npz')
lab = x.topk(2, -1).indices[:, 1]
M = x.argmax(-1); vM = x.max(-1).values; S = lab
tt = th.arange(10).expand(N, 10)
xb = th.where(tt < M[:, None], x, -th.inf); xa = th.where(tt > M[:, None], x, -th.inf)
A = xb.argmax(1); B = xa.argmax(1); both = (M > 0) & (M < 9)
other = th.where(S == A, B, A)
sideS = th.where(S < M, 0, 1)
top3 = x.topk(3, -1).values; g1 = (top3[:, 0] - top3[:, 1]) / vM.abs(); g2 = (top3[:, 1] - top3[:, 2]) / vM.abs()
gap = th.minimum(g1, g2)


def cats(pred):
    err = pred != lab
    sidep = th.where(pred < M, 0, th.where(pred > M, 1, -1))
    return err, {'predM': err & (pred == M), 'ABswap': err & both & (pred == other),
                 'within': err & (sidep == sideS) & (pred != M),
                 'other': err & ~((pred == M) | (both & (pred == other)) | ((sidep == sideS) & (pred != M)))}


with th.no_grad():
    l0 = forward(x, *W0); p0 = l0.argmax(-1)
    e0, c0 = cats(p0)
    o = l0.clone(); o[th.arange(N), lab] = -th.inf
    m0 = (l0[th.arange(N), lab] - o.max(-1).values)                 # trained signed logit margin
margin_edges = [0, .1, .3, 1, 3, np.inf]; gap_edges = [0, .02, .1, 1, np.inf]
print(f"trained: err {e0.float().mean():.4f}; by mechanism " + ", ".join(f"{k} {v.float().mean():.4f}" for k, v in c0.items()))
print(f"trained |margin| distribution: " + ", ".join(f"[{a},{b}) {((m0.abs() >= a) & (m0.abs() < b)).float().mean():.3f}" for a, b in zip(margin_edges[:-1], margin_edges[1:])))
rows = []
print("\ncopy    err    persist  new   | new by mechanism: predM ABswap within other | new by trained |margin|: " + " ".join(f"[{a},{b})" for a, b in zip(margin_edges[:-1], margin_edges[1:])) + " | new by min gap: " + " ".join(f"[{a},{b})" for a, b in zip(gap_edges[:-1], gap_edges[1:])) + " | fixed")
for i in range(8):
    W = [th.tensor(d[k][i]) for k in ('W_ih', 'W_hh', 'W_out')]
    with th.no_grad():
        p1 = forward(x, *W).argmax(-1)
    e1, c1 = cats(p1)
    new = e1 & ~e0; fixed = e0 & ~e1; persist = e1 & e0
    mech = [(c1[k] & new).float().mean().item() for k in c1]
    bym = [(new & (m0.abs() >= a) & (m0.abs() < b)).float().mean().item() for a, b in zip(margin_edges[:-1], margin_edges[1:])]
    byg = [(new & (gap >= a) & (gap < b)).float().mean().item() for a, b in zip(gap_edges[:-1], gap_edges[1:])]
    r = [e1.float().mean().item(), persist.float().mean().item(), new.float().mean().item()] + mech + bym + byg + [fixed.float().mean().item()]
    rows.append(r)
    print(f"noisy{i}  " + " ".join(f"{v:.4f}" for v in r[:3]) + " | " + " ".join(f"{v:.4f}" for v in mech) + " | " + " ".join(f"{v:.4f}" for v in bym) + " | " + " ".join(f"{v:.4f}" for v in byg) + f" | {r[-1]:.4f}")
r = np.array(rows).mean(0)
print("mean    " + " ".join(f"{v:.4f}" for v in r[:3]) + " | " + " ".join(f"{v:.4f}" for v in r[3:7]) + " | " + " ".join(f"{v:.4f}" for v in r[7:12]) + " | " + " ".join(f"{v:.4f}" for v in r[12:16]) + f" | {r[-1]:.4f}")
tm = [c0[k].float().mean().item() for k in c0]
print("\nnew errors by mechanism as a multiple of the trained model's rate in that mechanism: " + ", ".join(f"{k} {r[3 + j] / tm[j]:.2f}x" for j, k in enumerate(c0)))
print("share of new errors with trained |margin| < 0.3: %.2f; < 1: %.2f (trained inputs with |margin| < 0.3: %.3f, < 1: %.3f)" % (
    r[7:9].sum() / r[2], r[7:10].sum() / r[2], (m0.abs() < .3).float().mean(), (m0.abs() < 1).float().mean()))
print("share of new errors within 0.02 / 0.1 of a tie: %.2f / %.2f (trained errors: %.2f / %.2f)" % (
    r[12] / r[2], r[12:14].sum() / r[2], (e0 & (gap < .02)).float().mean() / e0.float().mean(), (e0 & (gap < .1)).float().mean() / e0.float().mean()))
