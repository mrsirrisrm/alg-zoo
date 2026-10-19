"""Can the per-copy accuracy of the sigma=0.005 harness be predicted from the perturbation
and the TRAINED model's sensitivity alone (zero network passes per copy)?

Two predictors, both using only trained-model 'advice':
  Q: quadratic form delta^T H delta, H = Gauss-Newton (softmax Fisher) Hessian in absolute
     weight coordinates; one scalar c fitted across copies (loss ~ c * Q).
  L: linearised flip count: fraction of a fixed input pool whose trained margin m_x is
     overturned by the first-order logit change J_x delta (margin = correct logit - best
     other); per copy cost n*432 FLOPs.
Bar: sd of the loss across copies (= the constant predictor's rms), BL's rms bias 0.025.
"""
import sys
sys.path.insert(0, '.')
import numpy as np, torch as th
from torch.func import vmap, jacrev
from ground_truth import forward, load_weights

E = ''
W0 = [th.tensor(a, dtype=th.float32) for a in load_weights(E + 'weights.npz')]
print('shapes', [tuple(w.shape) for w in W0])
d = np.load(E + 'noisy_sigma0.005.npz')
acc = d['acc']; p0 = 0.9531186; loss = p0 - acc
deltas = np.stack([np.concatenate([(d[k][i] - w.numpy()).ravel() for k, w in zip(('W_ih', 'W_hh', 'W_out'), W0)])
                   for i in range(len(acc))])
sizes = [w.numel() for w in W0]; o = np.cumsum([0] + sizes)
print('delta rms per matrix', [float(np.sqrt((deltas[:, o[j]:o[j + 1]] ** 2).mean())) for j in range(3)])


def unpack(delta):
    return [W0[j] + delta[o[j]:o[j + 1]].view(W0[j].shape) for j in range(3)]


def f(delta, xi):
    return forward(xi[None], *unpack(delta))[0]


def jac(n, seed):
    x = th.randn(n, 10, generator=th.Generator().manual_seed(seed))
    J = th.cat([vmap(jacrev(f), in_dims=(None, 0))(th.zeros(o[-1]), x[i:i + 2048]) for i in range(0, n, 2048)])
    return x, J  # J: n, 10, 432


# --- Q: Gauss-Newton Hessian
x, J = jac(8192, 5)
with th.no_grad():
    p = th.softmax(forward(x, *W0), -1)
    F = th.diag_embed(p) - p[:, :, None] * p[:, None, :]
    H = th.einsum('nia,nij,njb->ab', J, F, J) / len(x)
H = 0.5 * (H + H.T)
q = np.einsum('ia,ab,ib->i', deltas, H.numpy().astype(np.float64), deltas)
lam = np.linalg.eigvalsh(H.numpy()); lam = np.clip(lam, 0, None)
print(f"H: trace {lam.sum():.3g}, participation ratio {lam.sum() ** 2 / (lam ** 2).sum():.1f}; "
      f"predicted sd/mean of Q over copies {np.sqrt(2 * (lam ** 2).sum()) / lam.sum():.2f}, observed {q.std() / q.mean():.2f}")


def report(name, pred, cost):
    A = np.stack([pred, np.ones_like(pred)], 1)
    c, b = np.linalg.lstsq(A, loss, rcond=None)[0]
    fit = c * pred + b; res = loss - fit
    c0 = (pred * loss).sum() / (pred * pred).sum(); res0 = loss - c0 * pred
    r = np.corrcoef(pred, loss)[0, 1]
    print(f"{name}: corr {r:.3f}; 1-param (loss=c*pred) rms {np.sqrt((res0 ** 2).mean()):.4f}; "
          f"2-param rms {np.sqrt((res ** 2).mean()):.4f}; constant predictor rms {loss.std():.4f}; "
          f"per-copy cost {cost:.2g} FLOPs = {cost / 5760:.0f} MC samples (MC sd there {np.sqrt(0.09 * 5760 / cost):.4f})")
    return res0


report('Q (GN Hessian quadratic form)', q, 432 ** 2 * 2)

# --- L: linearised flip count on a larger pool
x, J = jac(65536, 7)
with th.no_grad():
    logit = forward(x, *W0)
lab = x.topk(2, -1).indices[:, 1]
n = len(x); idx = th.arange(n)
Jl = J.numpy().astype(np.float32)                                    # n, 10, 432
logit = logit.numpy(); lab = lab.numpy()
pred_flip = []
for dl in deltas.astype(np.float32):
    new = logit + Jl @ dl                                             # first-order logits
    pred_flip.append((new.argmax(-1) == lab).mean())
pred_flip = p0 - np.array(pred_flip)
report('L (linearised flip count, n=65536)', pred_flip, n * 10 * 432 * 2)
print('per-copy: actual loss, Q-pred (1-param), L-pred')
c0 = (q * loss).sum() / (q * q).sum()
for i in range(len(acc)):
    print(f"  {i:2d} {loss[i]:.4f} {c0 * q[i]:.4f} {pred_flip[i]:.4f}")

# ---- corrections and the flip-concentration measurement
print('\n[L cost corrected] per-copy Jacobian-vector product is n*10*432*2 =', f"{n * 10 * 432 * 2:.2g}",
      'FLOPs =', f"{n * 10 * 432 * 2 / 5760:.0f}", 'MC samples; MC sd there', f"{np.sqrt(0.09 * 5760 / (n * 10 * 432 * 2)):.4f}")
# leave-one-out c for Q
loo = []
for i in range(len(acc)):
    m = np.arange(len(acc)) != i
    c = (q[m] * loss[m]).sum() / (q[m] ** 2).sum(); loo.append(loss[i] - c * q[i])
print(f"Q leave-one-out rms {np.sqrt(np.mean(np.square(loo))):.4f}")

# actual flips on the pool with the exact perturbed network
others = logit.copy(); others[idx, lab] = -np.inf
m = logit[idx, lab] - others.max(-1)                                  # trained margin (signed)
best_other = others.argmax(-1)
g = Jl[idx, lab] - Jl[idx, best_other]                                # grad of margin, n x 432
gnorm = np.linalg.norm(g, axis=1)
y0 = m > 0
rows = []
with th.no_grad():
    for ci, dl in enumerate(deltas.astype(np.float32)):
        Wc = unpack(th.tensor(dl))
        y1 = (forward(x, *Wc).argmax(-1).numpy() == lab)
        flip = y1 != y0
        s = g @ dl; dn = np.linalg.norm(dl)
        row = dict(copy=ci, pool_acc=y1.mean(), truth=acc[ci], nflip=int(flip.sum()))
        for k in (1, 1.5, 2, 3):
            S = np.abs(m) < k * np.abs(s); row[f'f{k}'] = S.mean(); row[f'miss{k}'] = (flip & ~S).sum() / max(1, flip.sum())
            Scs = np.abs(m) < k * gnorm * dn; row[f'fcs{k}'] = Scs.mean(); row[f'misscs{k}'] = (flip & ~Scs).sum() / max(1, flip.sum())
        rows.append(row)
import collections
agg = collections.defaultdict(list)
for r in rows:
    for k, v in r.items(): agg[k].append(v)
print(f"pool-exact estimate vs truth: rms {np.sqrt(np.mean((np.array(agg['pool_acc']) - np.array(agg['truth'])) ** 2)):.4f} (pool sampling sd {np.sqrt(0.09 / n):.4f})")
print(f"flips per copy: mean {np.mean(agg['nflip']) / n:.4f} of pool")
for k in (1, 1.5, 2, 3):
    print(f"k={k}: dot-product set  frac {np.mean(agg[f'f{k}']):.4f} (max {np.max(agg[f'f{k}']):.4f}), flips missed {np.mean(agg[f'miss{k}']):.4f} (max {np.max(agg[f'miss{k}']):.4f}) | "
          f"Cauchy-Schwarz set frac {np.mean(agg[f'fcs{k}']):.4f} (max {np.max(agg[f'fcs{k}']):.4f}), missed {np.mean(agg[f'misscs{k}']):.4f} (max {np.max(agg[f'misscs{k}']):.4f})")
