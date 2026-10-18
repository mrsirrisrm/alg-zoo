"""Weight precision as description length: how many bits does the mechanism need, and is
that precision intrinsic or an artefact of the raw weight basis?
  A. Isotropic relative noise W -> W (1 + sigma eps): accuracy loss vs sigma; sigma* at
     which the loss is 1e-3 (and 1e-2) sets the naive bits/weight log2(1/sigma*).
  B. Rounding to b bits (per-tensor symmetric uniform), min b with accuracy loss < tol, in
     the raw basis and after dividing out the per-neuron ReLU rescaling symmetry (scale
     neuron i's incoming weights by c_i and outgoing by 1/c_i; balanced so incoming and
     outgoing max-abs agree; the c_i are not needed to decode, so they cost nothing).
  C. Quantisation-aware fine-tune (straight-through estimator, task cross-entropy) at each
     b: the best b found with re-tuning, which is the real upper bound on description length.
  D. Gauss-Newton Hessian of the KL to the original network in relative coordinates
     (432 x 432): spectrum (trace fractions, participation ratio) and the MDL floor over all
     linear re-parameterisations by reverse water-filling of the KL budget K = 1/2 sigma*^2
     tr(H) (same budget the isotropic noise spends): bits = sum_i 1/2 log2(tau^2 / d_i),
     d_i = min(tau^2, theta / lambda_i), tau = 1 (relative prior: a weight known to within
     its own size). The isotropic allocation d_i = sigma*^2 is the raw-basis cost; the ratio
     is how much a better basis could save. A few large eigenvalues and a flat tail mean a
     short explanation exists in some coordinates.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from torch.func import functional_call, jacrev, vmap
from ground_truth import forward, load_weights

NE = 1 << 21
XE = th.randn(NE, 10, generator=th.Generator().manual_seed(21)); LAB = XE.topk(2, -1).indices[:, 1]


def acc(W):
    with th.no_grad():
        return (forward(XE, *W).argmax(-1) == LAB).float().mean().item()


def quant(w, b):
    s = w.abs().max() / (2 ** (b - 1) - 1)
    return th.round(w / s) * s


def balance(W, iters=50):
    W_ih, W_hh, W_out = [w.clone() for w in W]
    for _ in range(iters):
        inn = th.maximum(W_ih.abs(), W_hh.abs().max(1).values)          # incoming max-abs per neuron (row i of W_hh, W_ih[i])
        out = th.maximum(W_hh.abs().max(0).values, W_out.abs().max(0).values)   # outgoing (column i of W_hh, W_out[:, i])
        c = th.sqrt(out / inn)
        W_ih = W_ih * c; W_hh = W_hh * c[:, None] / c[None, :]; W_out = W_out / c[None, :]
    return [W_ih, W_hh, W_out]


def min_bits(W, tol, a0):
    for b in range(2, 17):
        if a0 - acc([quant(w, b) for w in W]) < tol: return b
    return 99


def qat(W, b, steps=400, bs=4096, lr=3e-4):
    P = [w.clone().requires_grad_(True) for w in W]; opt = th.optim.Adam(P, lr=lr)
    g = th.Generator().manual_seed(b)
    for _ in range(steps):
        x = th.randn(bs, 10, generator=g); y = x.topk(2, -1).indices[:, 1]
        Q = [w + (quant(w.detach(), b) - w.detach()) for w in P]          # straight-through
        loss = th.nn.functional.cross_entropy(forward(x, *Q), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return [quant(w.detach(), b) for w in P]


def gauss_newton(W, n=8192):
    W0 = [w.clone() for w in W]; x = th.randn(n, 10, generator=th.Generator().manual_seed(5))
    def f(delta, xi):
        d_ih, d_hh, d_out = delta[:16], delta[16:272].view(16, 16), delta[272:].view(10, 16)
        return forward(xi[None], W0[0] * (1 + d_ih), W0[1] * (1 + d_hh), W0[2] * (1 + d_out))[0]
    J = vmap(jacrev(f), in_dims=(None, 0))(th.zeros(432), x)              # n, 10, 432
    with th.no_grad():
        p = th.softmax(forward(x, *W0), -1)
        F = th.diag_embed(p) - p[:, :, None] * p[:, None, :]              # softmax Fisher per sample
        H = th.einsum('nia,nij,njb->ab', J, F, J) / n
    return 0.5 * (H + H.T)


def waterfill(lam, K, tau=1.):
    lam = np.clip(lam, 1e-12, None)
    lo, hi = 1e-12, 1e6
    for _ in range(200):
        th_ = np.sqrt(lo * hi); d = np.minimum(tau ** 2, th_ / lam)
        if 0.5 * (lam * d).sum() > K: hi = th_
        else: lo = th_
    d = np.minimum(tau ** 2, th_ / lam)
    return (0.5 * np.log2(tau ** 2 / d)).sum(), (d < tau ** 2).sum()


def run(name, W):
    W = [th.tensor(a, dtype=th.float32) for a in W]; a0 = acc(W)
    print(f"\n{name}: accuracy {a0:.4f} ({NE} inputs, sd {np.sqrt(a0 * (1 - a0) / NE):.1e})")
    # A
    sig = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3]; loss = []
    for s in sig:
        g = th.Generator().manual_seed(1); r = []
        for _ in range(4):
            r.append(a0 - acc([w * (1 + s * th.randn(w.shape, generator=g)) for w in W]))
        loss.append(np.mean(r))
    print("  A. isotropic relative noise sigma -> accuracy loss: " + ', '.join(f"{s:g}: {l:.4f}" for s, l in zip(sig, loss)))
    ls, ll = np.log(sig), np.log(np.maximum(loss, 1e-6))
    sstar = {tol: float(np.exp(np.interp(np.log(tol), ll, ls))) for tol in (1e-3, 1e-2)}
    print(f"     sigma* for loss 1e-3: {sstar[1e-3]:.2e} -> naive {np.log2(1 / sstar[1e-3]):.1f} bits/weight, {432 * np.log2(1 / sstar[1e-3]):.0f} bits; "
          f"for 1e-2: {sstar[1e-2]:.2e} -> {np.log2(1 / sstar[1e-2]):.1f} bits/weight")
    # B
    Wb = balance(W); ab = acc(Wb)
    print(f"  B. rounding: min bits/weight for loss < 1e-3 / 1e-2: raw {min_bits(W, 1e-3, a0)} / {min_bits(W, 1e-2, a0)}, "
          f"balanced {min_bits(Wb, 1e-3, a0)} / {min_bits(Wb, 1e-2, a0)} (balanced accuracy {ab:.4f}; symmetry check)")
    print("     accuracy loss by bits, raw:      " + ' '.join(f"b{b}:{a0 - acc([quant(w, b) for w in W]):+.4f}" for b in range(3, 13)))
    print("     accuracy loss by bits, balanced: " + ' '.join(f"b{b}:{a0 - acc([quant(w, b) for w in Wb]):+.4f}" for b in range(3, 13)))
    # C
    row = []
    for b in range(2, 11):
        row.append(f"b{b}:{a0 - acc(qat(Wb, b)):+.4f}")
    print("     accuracy loss by bits after QAT (balanced start, 400 steps): " + ' '.join(row), flush=True)
    # D
    H = gauss_newton(W); lam = np.sort(np.linalg.eigvalsh(H.numpy()))[::-1]; lam = np.clip(lam, 0, None)
    f = lam / lam.sum(); pr = 1 / (f ** 2).sum()
    print(f"  D. Gauss-Newton Hessian (relative coords): trace {lam.sum():.3g}, top-1/5/20/50 trace {np.cumsum(f)[[0, 4, 19, 49]].round(3)}, "
          f"participation ratio {pr:.1f}, eigenvalues > 1e-3 max: {(lam > 1e-3 * lam[0]).sum()}")
    for tol in (1e-3, 1e-2):
        K = 0.5 * sstar[tol] ** 2 * lam.sum(); iso = 432 * 0.5 * np.log2(1 / sstar[tol] ** 2)
        wf, nd = waterfill(lam, K)
        print(f"     KL budget for loss {tol:g}: {K:.2e}; bits: isotropic (raw basis) {iso:.0f}, water-filled floor {wf:.0f} over {nd} directions ({wf / iso:.2f} of raw)")


if __name__ == '__main__':
    run('trained M(16,10)', load_weights('weights.npz'))
    s = np.load('seeds_h16.npz')
    run('seed0', [s[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
