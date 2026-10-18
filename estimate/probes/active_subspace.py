"""Active subspace of the margin on the tie wall: how many input directions does the
network's margin actually depend on? At sampled exact ties (boundary_layer.sample_ties,
no labels) take g = grad_x m, m = logit[target] - logit[comparator], split off the tie
component alpha (along e_hi - e_lo) and form C = E[g_perp g_perp^T] per ordered position
pair (9 wall directions). Eigenvalue decay of C (Constantine's active subspace): if a few
eigenvalues carry the trace, the width is a function of a few linear combinations of the
inputs and can be tabulated; a flat spectrum means no low-dimensional description of the
margin exists. Reports the median over pairs of the trace fraction in the top 1/2/3
eigenvectors and the effective dimension exp(H(lambda/trace)). Random-weight network for
reference.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from ground_truth import forward, load_weights
from boundary_layer import sample_ties

NT = 1 << 15


def grads(W, k, gen):
    x, a, b = sample_ties(k, NT, gen); ar = th.arange(NT); rows = []
    for hi, lo in ((a, b), (b, a)):
        tgt, cmp = (lo, hi) if k == 1 else (hi, lo)
        xx = x.clone().requires_grad_(True)
        l = forward(xx, *W); m = l[ar, tgt] - l[ar, cmp]
        g, = th.autograd.grad(m.sum(), xx)
        am = l.argmax(-1).detach()
        u = th.zeros_like(g); u[ar, hi] = 1; u[ar, lo] = -1; u = u / np.sqrt(2)
        al = (g * u).sum(-1, keepdim=True)
        gp = g - al * u
        keep = ((am == hi) | (am == lo)) & (al.squeeze() > 0)      # where a boundary layer exists
        rows.append((hi[keep].numpy(), lo[keep].numpy(), gp[keep].numpy(), m.detach()[keep].numpy(), al.squeeze()[keep].numpy()))
    return [np.concatenate(z) for z in zip(*rows)]


def spectrum(hi, lo, gp):
    out = []
    for i in range(10):
        for j in range(10):
            s = (hi == i) & (lo == j)
            if s.sum() < 200: continue
            C = gp[s].T @ gp[s] / s.sum()
            lam = np.sort(np.linalg.eigvalsh(C))[::-1]; f = lam / lam.sum()
            f = f[f > 1e-12]
            out.append((np.cumsum(f)[:3], np.exp(-(f * np.log(f)).sum())))
    top = np.median([o[0] for o in out], 0); eff = np.median([o[1] for o in out])
    return top, eff, len(out)


def run(name, W):
    W = [th.tensor(a, dtype=th.float32) for a in W]
    print(f"\n{name}")
    for k, lab in ((2, 'S/R'), (1, 'M/S')):
        hi, lo, gp, m, al = grads(W, k, th.Generator().manual_seed(k))
        top, eff, n = spectrum(hi, lo, gp)
        # pooled over pairs, in absolute position coordinates (mixes pairs; for reference)
        C = gp.T @ gp / len(gp); lam = np.sort(np.linalg.eigvalsh(C))[::-1]; f = lam / lam.sum()
        print(f"  {lab} ties: per-pair trace in top 1/2/3 eigvecs (median over {n} pairs) "
              f"{top[0]:.2f}/{top[1]:.2f}/{top[2]:.2f}, effective dim {eff:.1f}; "
              f"pooled top-3 {np.cumsum(f)[:3].round(2)}; wall-gradient rms / alpha rms "
              f"{np.sqrt((gp ** 2).sum(-1).mean()) / np.sqrt((al ** 2).mean()):.2f}", flush=True)


if __name__ == '__main__':
    run('trained M(16,10)', load_weights('weights.npz'))
    d = np.load('noisy_sigma0.005.npz'); s = np.load('seeds_h16.npz')
    run('noisy0', [d[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
    run('seed0', [s[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
    g = th.Generator().manual_seed(7)
    run('random h16', [th.randn(16, generator=g).numpy() * .3, th.randn(16, 16, generator=g).numpy() * .3, th.randn(10, 16, generator=g).numpy() * .3])
