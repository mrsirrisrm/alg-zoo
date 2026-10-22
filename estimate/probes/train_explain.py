"""PLAN J2: train-and-explain on the zoo checkpoints (h16, T10, seeds 0-4; n_seqs 0, 2^20..2^30).

ARC's "inductive step": advice that is updated incrementally as training proceeds. Does the
explanation of checkpoint k carry to k+1 with small updates (smooth drift), or do roles
reorganise? Per checkpoint:
  acc         accuracy (2^20 inputs)
  strip       share of errors inside the tie strip (top-two gap < 0.03); the tie-layer
              signature of the finished models
  r_next      mean matched activation correlation to the next checkpoint (Hungarian)
  r_final     same, to the final (2^30) model
  maxc        best |corr| of any neuron with the running max of x_<=t (max circuit present?)
  algR2       mean over neurons of R^2 on named variables (running max, 2nd max, x_t,
              relu(x_t - M_{t-1}), relu(M_{t-1} - x_t), M_t - S_t, steps since max / 9)
  bits        precision bits at tol 1e-3 (J1 method, per-tensor rms prior)
  N0.9        accuracy with the Schur coupling of W_hh scaled by 0.9 from the tie step (I6a)
Check: the 2^30 checkpoint reproduces seeds_h16.npz.

Usage: uv run python probes/train_explain.py [seeds...]
"""
import sys, math
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from scipy.optimize import linear_sum_assignment
from zoo_cache import zoo_weights
from seed_template import acts, corr
from schur_modes import real_schur_split, acc_fn
from opt_bits import Net, waterfill

T, H = 10, 16
CKPTS = [0] + [2 ** i for i in range(20, 31)]
XA = th.randn(1 << 13, T, dtype=th.float64, generator=th.Generator().manual_seed(7))


def named_vars(x):
    x = x.numpy(); N = len(x); F = []
    M = np.full(N, -np.inf); S = np.full(N, -np.inf); since = np.zeros(N)
    for t in range(T):
        xt = x[:, t]; Mp = np.where(np.isinf(M), 0.0, M)
        newmax = xt > M
        S = np.where(newmax, M, np.maximum(S, xt)); M = np.maximum(M, xt)
        since = np.where(newmax, 0, since + 1)
        Sf = np.where(np.isinf(S), 0.0, S)
        F.append(np.stack([M, Sf, xt, np.maximum(xt - Mp, 0), np.maximum(Mp - xt, 0), M - Sf, since / 9, np.ones(N)], 1))
    return np.concatenate(F)  # (T*N, 8), row order matches acts()


def alg_r2(A, F):
    beta, *_ = np.linalg.lstsq(F, A, rcond=None)
    res = A - F @ beta; v = A.var(0)
    r2 = 1 - res.var(0) / np.where(v > 0, v, 1)
    return float(np.mean(r2[v > 0])) if (v > 0).any() else float('nan')


def strip_share(W):
    Wt = [th.tensor(a, dtype=th.float32) for a in W]
    from ground_truth import forward
    x = th.randn(1 << 20, T, generator=th.Generator().manual_seed(3)); top = x.topk(2, -1)
    with th.no_grad():
        wrong = forward(x, *Wt).argmax(-1) != top.indices[:, 1]
    gap = top.values[:, 0] - top.values[:, 1]
    return wrong.float().mean().item(), (wrong & (gap < 0.03)).sum().item() / max(wrong.sum().item(), 1)


def precision_bits(W, tol=1e-3):
    net = Net(W); a0 = net.acc(net.p0); Hs = net.gn()
    svec = np.concatenate([np.full(w.size, math.sqrt((w ** 2).mean()) or 1.0) for w in W])
    G = svec[:, None] * Hs * svec[None, :]; mu = np.linalg.eigvalsh(G)[::-1].clip(0)
    rng = np.random.default_rng(1); pr = []
    for sig in np.geomspace(1e-4, 3e-1, 14):
        for _ in range(3):
            z = rng.standard_normal(len(svec)) * sig
            pr.append((0.5 * z @ G @ z, a0 - net.acc(net.p0 + th.tensor(z * svec, dtype=th.float32))))
    pr = np.array(pr); m = (pr[:, 1] > 3e-4) & (pr[:, 1] < 3e-2)
    if m.sum() < 3: return float('nan')
    c = np.median(pr[m, 1] / pr[m, 0]); d = waterfill(mu, tol / c)
    return 0.5 * (d - 1 - np.log(d)).sum() / math.log(2)


def coupling_09(W):
    Q, B, N = real_schur_split(W[1])
    return acc_fn(W, Wmod=Q @ (B + 0.9 * N) @ Q.T, from_tie=True)[0]


def main():
    seeds = [int(s) for s in sys.argv[1:]] or list(range(5))
    F = named_vars(XA[:, :].float())
    ref = np.load('seeds_h16.npz')
    for seed in seeds:
        Ws = [zoo_weights(16, n_seqs=n, seed=seed) for n in CKPTS]
        As = [acts(W, XA) for W in Ws]
        print(f'\nseed {seed} (stored final acc {ref["acc"][seed]:.4f}; final W_hh matches stored: '
              f'{np.abs(Ws[-1][1] - ref["W_hh"][seed]).max() == 0})')
        print('  n_seqs       acc     err   strip  r_next r_final  maxc  algR2   bits   N0.9')
        for k, (n, W, A) in enumerate(zip(CKPTS, Ws, As)):
            err, strip = strip_share(W)
            def mcorr(B):
                c = corr(A, B); r, j = linear_sum_assignment(-c); return c[r, j].mean()
            r_next = mcorr(As[k + 1]) if k + 1 < len(As) else float('nan')
            r_fin = mcorr(As[-1])
            Mcol = F[:, 0]; Ac = A - A.mean(0); Mc = Mcol - Mcol.mean()
            den = np.sqrt((Ac ** 2).sum(0) * (Mc ** 2).sum()); maxc = np.max(np.abs(Ac.T @ Mc) / np.where(den > 0, den, 1))
            bits = precision_bits(W) if k > 0 else float('nan')
            n09 = coupling_09(W)
            lab = '0' if n == 0 else f'2^{int(math.log2(n))}'
            print(f'  {lab:>6s}  {1 - err:8.4f}  {err:6.4f}  {strip:5.2f}  {r_next:6.3f}  {r_fin:6.3f}  {maxc:4.2f}  {alg_r2(A, F):5.2f}  {bits:6.0f}  {n09:5.3f}', flush=True)


if __name__ == '__main__':
    main()
