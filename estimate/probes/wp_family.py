"""D2 (PLAN.md): does the GN-Fisher quadratic form predict per-copy accuracy loss away from
M(16,10)? For each base model (seed0 h16, zoo h32, hand10): 16 copies W + 0.005 rms(W) N(0,1)
per matrix, truth at 2^26 (sd ~4e-5), Q = delta^T H delta with H the base model's
Gauss-Newton Hessian (8192 inputs), one scalar c fitted leave-one-out.
Reports corr(Q, loss), LOO rms vs the constant predictor's rms, and H's participation ratio.
Run from estimate/: python probes/wp_family.py [seed0 h32 hand10]
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from torch.func import vmap, jacrev
from alg_zoo import zoo_2nd_argmax, handcrafted
from ground_truth import forward
from ablation import sd_weights

names = sys.argv[1:] or ['seed0', 'h32', 'hand10']
s = np.load('seeds_h16.npz')
loaders = {'seed0': lambda: [th.tensor(s[k][0], dtype=th.float32) for k in ('W_ih', 'W_hh', 'W_out')],
           'h32': lambda: sd_weights(zoo_2nd_argmax(32, 10)), 'hand10': lambda: sd_weights(handcrafted.handcrafted_2nd_argmax_10())}


def acc(W, n_log2=26, seed=0):
    g = th.Generator().manual_seed(seed); c = 0; B = 1 << 22
    with th.no_grad():
        for _ in range(1 << (n_log2 - 22)):
            x = th.randn(B, 10, generator=g)
            c += (forward(x, *W).argmax(-1) == x.topk(2, -1).indices[:, 1]).sum().item()
    return c / (1 << n_log2)


for name in names:
    W0 = loaders[name](); sizes = [w.numel() for w in W0]; o = np.cumsum([0] + sizes); D = int(o[-1])
    unpack = lambda delta: [W0[j] + delta[o[j]:o[j + 1]].view(W0[j].shape) for j in range(3)]
    f = lambda delta, xi: forward(xi[None], *unpack(delta))[0]
    x = th.randn(8192, 10, generator=th.Generator().manual_seed(5))
    J = th.cat([vmap(jacrev(f), in_dims=(None, 0))(th.zeros(D), x[i:i + 1024]) for i in range(0, 8192, 1024)])
    with th.no_grad():
        p = th.softmax(forward(x, *W0), -1); F = th.diag_embed(p) - p[:, :, None] * p[:, None, :]
        H = th.einsum('nia,nij,njb->ab', J, F, J) / 8192
    H = (0.5 * (H + H.T)).numpy().astype(np.float64); lam = np.clip(np.linalg.eigvalsh(H), 0, None)
    p0 = acc(W0, seed=999)
    rng = np.random.default_rng(77); deltas, losses = [], []
    for i in range(16):
        dl = np.concatenate([0.005 * np.sqrt((w.numpy() ** 2).mean()) * rng.standard_normal(w.numel()) for w in W0]).astype(np.float32)
        a = acc(unpack(th.tensor(dl)), seed=200 + i); deltas.append(dl); losses.append(p0 - a)
        print(f"  {name} copy {i}: loss {p0 - a:.4f}", flush=True)
    deltas = np.stack(deltas).astype(np.float64); losses = np.array(losses)
    q = np.einsum('ia,ab,ib->i', deltas, H, deltas)
    loo = [losses[i] - (q[m] * losses[m]).sum() / (q[m] ** 2).sum() * q[i] for i in range(16) for m in [np.arange(16) != i]]
    print(f"{name}: params {D}, base acc {p0:.4f}, mean loss {losses.mean():.4f}, sd {losses.std():.4f}; H participation ratio {lam.sum() ** 2 / (lam ** 2).sum():.1f}; "
          f"corr(Q, loss) {np.corrcoef(q, losses)[0, 1]:.3f}; LOO rms {np.sqrt(np.mean(np.square(loo))):.4f} vs constant {losses.std():.4f}", flush=True)
