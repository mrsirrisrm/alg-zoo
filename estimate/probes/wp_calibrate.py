"""A2/A3 (PLAN.md): the GN-Fisher quadratic form as a predictor of per-copy accuracy loss,
calibrated OUT OF SAMPLE, and a picture of H.

A2. Fresh copies W + sigma rms(W) N(0,1) per matrix at sigma in {0.0025, 0.005, 0.01},
    16 each, independent rng seed from the harness, truth by MC at 2^26 (sd ~4e-5).
    Fit c in loss ~ c * Q on each sigma set, score on the 32 harness copies with each c.
    One c for all sigma = quadratic regime; drift = cubic/quartic terms.
A3. H: trace by matrix and by neuron (row = incoming weights of neuron i, column = outgoing),
    top-6 eigenvectors' support.
Run from estimate/: python probes/wp_calibrate.py
"""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from torch.func import vmap, jacrev
from ground_truth import forward, load_weights

P0 = 0.9531186
W0 = [th.tensor(a, dtype=th.float32) for a in load_weights('weights.npz')]
sizes = [w.numel() for w in W0]; o = np.cumsum([0] + sizes)


def unpack(delta):
    return [W0[j] + delta[o[j]:o[j + 1]].view(W0[j].shape) for j in range(3)]


def f(delta, xi):
    return forward(xi[None], *unpack(delta))[0]


def hessian(n=8192, seed=5):
    x = th.randn(n, 10, generator=th.Generator().manual_seed(seed))
    J = th.cat([vmap(jacrev(f), in_dims=(None, 0))(th.zeros(o[-1]), x[i:i + 2048]) for i in range(0, n, 2048)])
    with th.no_grad():
        p = th.softmax(forward(x, *W0), -1)
        F = th.diag_embed(p) - p[:, :, None] * p[:, None, :]
        H = th.einsum('nia,nij,njb->ab', J, F, J) / n
    return (0.5 * (H + H.T)).numpy().astype(np.float64)


def acc(W, n_log2=26, seed=0):
    g = th.Generator().manual_seed(seed); c = 0; B = 1 << 22
    with th.no_grad():
        for _ in range(1 << (n_log2 - 22)):
            x = th.randn(B, 10, generator=g)
            c += (forward(x, *W).argmax(-1) == x.topk(2, -1).indices[:, 1]).sum().item()
    return c / (1 << n_log2)


def qform(H, deltas):
    return np.einsum('ia,ab,ib->i', deltas, H, deltas)


H = hessian()
lam, V = np.linalg.eigh(H); lam = lam[::-1]; V = V[:, ::-1]
print(f"H trace {lam.sum():.3g}; top 1/5/20 share {lam[0] / lam.sum():.2f}/{lam[:5].sum() / lam.sum():.2f}/{lam[:20].sum() / lam.sum():.2f}; participation ratio {lam.sum() ** 2 / (lam ** 2).sum():.1f}")
# A3: where the stiffness sits (diagonal of H by block / neuron)
dg = np.diag(H)
print("trace share by matrix: " + ", ".join(f"{nm} {dg[o[j]:o[j + 1]].sum() / dg.sum():.3f}" for j, nm in enumerate(('W_ih', 'W_hh', 'W_out'))))
hh = dg[o[1]:o[2]].reshape(16, 16)
row = hh.sum(1) + dg[:16]; col = hh.sum(0) + dg[o[2]:].reshape(10, 16).sum(0)
order = np.argsort(-row)
print("per-neuron share of trace (incoming weights: W_ih + W_hh row): " + ", ".join(f"n{i} {row[i] / dg.sum():.3f}" for i in order[:8]))
print("per-neuron share of trace (outgoing: W_hh column + W_out column): " + ", ".join(f"n{i} {col[i] / dg.sum():.3f}" for i in np.argsort(-col)[:8]))
for k in range(6):
    v = V[:, k] ** 2; top = np.argsort(-v)[:6]
    def nm(i):
        if i < 16: return f"ih[{i}]"
        if i < 272: j = i - 16; return f"hh[{j // 16},{j % 16}]"
        j = i - 272; return f"out[{j // 16},{j % 16}]"
    print(f"eigvec {k} (lambda share {lam[k] / lam.sum():.3f}): weights of top-6 entries " + ", ".join(f"{nm(i)} {v[i]:.2f}" for i in top) + f"; W_hh share {v[16:272].sum():.2f}")

# A2: fresh copies, out-of-sample c
rng = np.random.default_rng(2024)
sets = {}
for sigma in (0.0025, 0.005, 0.01):
    deltas, losses = [], []
    for i in range(16):
        dl = np.concatenate([sigma * np.sqrt((w.numpy() ** 2).mean()) * rng.standard_normal(w.numel()) for w in W0]).astype(np.float32)
        a = acc(unpack(th.tensor(dl)), seed=100 + i)
        deltas.append(dl); losses.append(P0 - a)
        print(f"  sigma {sigma} copy {i}: loss {P0 - a:.4f}", flush=True)
    deltas = np.stack(deltas).astype(np.float64); losses = np.array(losses)
    q = qform(H, deltas); c = (q * losses).sum() / (q * q).sum()
    res = losses - c * q
    sets[sigma] = dict(c=c, corr=np.corrcoef(q, losses)[0, 1], rms=np.sqrt((res ** 2).mean()), sd=losses.std(), mean=losses.mean())
    print(f"sigma {sigma}: mean loss {losses.mean():.4f}, sd {losses.std():.4f}, corr(Q, loss) {sets[sigma]['corr']:.3f}, c {c:.4g}, rms residual {sets[sigma]['rms']:.4f}")

d = np.load('noisy_sigma0.005.npz'); hl = P0 - d['acc']
hd = np.stack([np.concatenate([(d[k][i] - w.numpy()).ravel() for k, w in zip(('W_ih', 'W_hh', 'W_out'), W0)]) for i in range(32)]).astype(np.float64)
hq = qform(H, hd)
print("\nharness (32 copies at sigma 0.005), constant-predictor rms %.4f" % hl.std())
for sigma, s in sets.items():
    res = hl - s['c'] * hq
    print(f"  c from sigma {sigma}: rms {np.sqrt((res ** 2).mean()):.4f}, mean bias {res.mean():+.4f}")
