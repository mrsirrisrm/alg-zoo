"""PLAN J3: the sketching bound at a time cut, as the rank of the prefix x future correctness matrix.

At cut t (inputs x_0..x_{t-1} read), take N prefixes and M futures (x_t..x_9) and the 0/1
matrix C[i, j] = correct(prefix_i ++ future_j). ARC's tail kernel with the tail random over
futures is K = C C^T / M (Hilton's halfspace note, sketching theorem); its top-m eigenvectors
are the best m prefix features for an estimator that must work for every future.
Per held-out future j, the quantity estimated is its accuracy over prefixes, the column
mean <f_j, g> with g uniform. Two estimators of equal cost (m tail evaluations per future):
  KQ(m): kernel quadrature. m prefixes chosen by pivoted Cholesky on K and weights fitted by
         least squares to the column means of futures A (offline advice, like pi), scored
         on futures B: est_j = w . f_j(S).
  MC(m): m prefixes sampled; MSE Var_i(f_j) / m.
(A rank-m projection of g needs f_j on all N prefixes, so it is not a width-m estimator;
only its spectrum is reported.) Reported per cut: fraction of ||C||_F^2 outside the top m
singular values, and gain MC MSE / KQ MSE at m = 1..1024. KQ is the deterministic,
atom-based estimator with oracle advice: what the tail kernel allows at one cut.
Check: KQ at m atoms beats nearest-neighbour atom merging at equal m (FINDINGS "Effective
dimension": merged-state error ~ N^-0.12, slow).

Usage: uv run python probes/cut_rank.py [names...]
"""
import sys, math, time
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from opt_bits import loaders

T = 10
NP, MA, MB = 4096, 4096, 1024
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def corr_matrix(W, t, xp, xf):
    """C[i, j] for prefixes xp (N, t) and futures xf (M, T - t)."""
    W_ih, W_hh, W_out = (th.tensor(a, dtype=th.float32) for a in W)
    h = th.zeros(len(xp), len(W_ih))
    for k in range(t):
        h = th.relu(xp[:, k:k + 1] * W_ih + h @ W_hh.T)
    N, M = len(xp), len(xf); C = th.zeros(N, M, dtype=th.bool); B = max(1, (1 << 21) // M)
    with th.no_grad():
        for i in range(0, N, B):
            hb = h[i:i + B]; b = len(hb)
            g = hb.repeat_interleave(M, 0)
            for k in range(T - t):
                g = th.relu(xf[:, k].repeat(b)[:, None] * W_ih + g @ W_hh.T)
            pred = (g @ W_out.T).argmax(-1)
            full = th.cat([xp[i:i + B].repeat_interleave(M, 0), xf.repeat(b, 1)], 1)
            C[i:i + B] = (pred == full.topk(2, -1).indices[:, 1]).view(b, M)
    return C.double().numpy()


def pivoted_cholesky(F, k):
    """Greedy pivots for K = F F^T (rows of F = prefixes), largest residual diagonal first."""
    d = (F ** 2).sum(1).copy(); L = np.zeros((len(F), k)); piv = []
    for j in range(k):
        i = int(np.argmax(d)); piv.append(i)
        if d[i] <= 1e-12: break
        r = F @ F[i] - L[:, :j] @ L[i, :j]
        L[:, j] = r / math.sqrt(d[i]); d = np.maximum(d - L[:, j] ** 2, 0)
    return np.array(piv)


def run(name, W):
    t0 = time.time()
    print(f'\n{name}: h={len(W[0])}', flush=True)
    print('  cut t  acc    frob outside top-m (m = 1, 4, 16, 64, 256)    gain MC/KQ at m = ' + ' '.join(f'{m}' for m in MS))
    for t in range(1, T):
        gen = th.Generator().manual_seed(100 + t)
        xp = th.randn(NP, t, generator=gen); xf = th.randn(MA + MB, T - t, generator=gen)
        C = corr_matrix(W, t, xp, xf); CA, CB = C[:, :MA], C[:, MA:]
        U, s, _ = np.linalg.svd(CA / math.sqrt(MA), full_matrices=False)
        fr = np.cumsum(s ** 2) / (s ** 2).sum()
        piv = pivoted_cholesky(CA / math.sqrt(MA), max(MS))
        tgtA, tgtB = CA.mean(0), CB.mean(0)
        var = CB.var(0)
        gains, rms_s = [], []
        for m in MS:
            S = piv[:m]
            w, *_ = np.linalg.lstsq(CA[S].T, tgtA, rcond=None)
            err = CB[S].T @ w - tgtB
            mse_s = (err ** 2).mean(); mse_mc = (var / m).mean()
            gains.append(mse_mc / max(mse_s, 1e-30)); rms_s.append(math.sqrt(mse_s))
        print(f'  {t:5d}  {C.mean():.3f}  ' + ' '.join(f'{1 - fr[m - 1]:.1e}' for m in (1, 4, 16, 64, 256)) +
              '    ' + ' '.join(f'{g:7.1f}' for g in gains), flush=True)
        if t in (5, 8, 9):
            print(f'         KQ rms at m: ' + ' '.join(f'{m}:{r:.1e}' for m, r in zip(MS, rms_s)) +
                  f'; MC rms at m=1: {math.sqrt(var.mean()):.2e}')
    print(f'  ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    L = loaders(); names = sys.argv[1:] or ['trained', 'seed0', 'h8', 'h32', 'hand10']
    for n in names:
        run(n, L[n]())
