"""PLAN I6a: scattering-mode (Schur) view of W_hh, and a Capon/Bartlett view of the readout.

Carried over from tropical-pythia (paper/scattering_modes.md, experiments/adaptive-beamformer):
  W_hh = Q T Q^T (real Schur): the block-diagonal of T is transmission (eigenvalues, what
  eigen/SVD analyses see); the strictly upper part N is coupling ("backscatter"), visible
  only in a shared basis. On Pythia MLPs the coupling carried the work.

Questions here, on the six aligned h16/T10 models (trained example + zoo seeds 0-4), in the
unit-rms-activation canonical form (the exact rescaling symmetry fixes the basis):
  (a) Do the six models share a Schur shape (mode counts, coupling fraction and rank) even
      though per-entry weights do not agree (seed_template.py)?  -> candidate class for I6.
  (b) Is the comparator transmitted or backscattered? Accuracy with N dropped, scaled or
      shuffled, overall and from the tie step onward only (state exact up to the arrival
      of the later of the top two, then the modified operator).
  (c) Readout: whitening index omega = ||R^{-1/2} W_out^T||_F / ||W_out^T||_F against the
      step-10 state second moment R, and accuracy with the Capon-whitened readout
      W_out R_eps^{-1} at inference (the beamformer "experiment E").
The sampled R and accuracies are diagnostics, not part of an estimator.

Usage: uv run python probes/schur_modes.py
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from scipy.linalg import schur
from scipy.optimize import linear_sum_assignment
from seed_template import models, acts, canon, permute, corr
from ground_truth import forward

H, T = 16, 10


def real_schur_split(W):
    """W = Q T Q^T; return Q, block-diagonal part B of T (1x1 and 2x2 blocks), coupling N = T - B."""
    T_, Q = schur(W, output='real')
    B = np.zeros_like(T_); i = 0
    while i < H:
        if i + 1 < H and abs(T_[i + 1, i]) > 1e-12:
            B[i:i + 2, i:i + 2] = T_[i:i + 2, i:i + 2]; i += 2
        else:
            B[i, i] = T_[i, i]; i += 1
    return Q, B, T_ - B


def taxonomy(W):
    Q, B, N = real_schur_split(W)
    lam = np.linalg.eigvals(W); a = np.abs(lam)
    fro2 = (W ** 2).sum()
    rows = np.sort((N ** 2).sum(1))[::-1]; c = np.cumsum(rows) / max(rows.sum(), 1e-300)
    return dict(
        amp=int((np.isreal(lam) & (a > 1.05)).sum()), trans=int((np.isreal(lam) & (a <= 1.05) & (a >= 0.5)).sum()),
        att=int((np.isreal(lam) & (a < 0.5) & (a > 0.05)).sum()), null=int((a <= 0.05).sum()),
        cpairs=int((~np.isreal(lam)).sum() // 2), rho=float(a.max()),
        henrici=float(np.sqrt(max(fro2 - (a ** 2).sum(), 0) / fro2)),
        coup_frac=float((N ** 2).sum() / fro2), coup_rank90=int(np.searchsorted(c, 0.9) + 1), Q=Q, B=B, N=N)


def acc_fn(M, n_log2=18, seed=1, Wmod=None, from_tie=False):
    """Accuracy; with Wmod and from_tie, run W_hh exactly up to the arrival of the later of the
    top two inputs, then Wmod. Returns (acc overall, acc on the strip gap < 0.03)."""
    W_ih, W_hh, W_out = (th.tensor(a, dtype=th.float32) for a in M)
    Wm = W_hh if Wmod is None else th.tensor(Wmod, dtype=th.float32)
    x = th.randn(1 << n_log2, T, generator=th.Generator().manual_seed(seed))
    top = x.topk(2, -1); S = top.indices[:, 1]; b0 = top.indices.max(-1).values
    gap = top.values[:, 0] - top.values[:, 1]
    with th.no_grad():
        if Wmod is None or not from_tie:
            pred = forward(x, W_ih, Wm, W_out).argmax(-1)
        else:
            h = th.relu(x[:, 0:1] * W_ih)
            for t in range(1, T):
                # step t consumes x_t; the operator applied to h (state after x_{t-1}) is
                # modified once x_{b0} has arrived, i.e. for t > b0
                use_mod = (t > b0)[:, None]
                hm = th.where(use_mod, h @ Wm.T, h @ W_hh.T)
                h = th.relu(x[:, t:t + 1] * W_ih + hm)
            pred = (h @ W_out.T).argmax(-1)
    ok = (pred == S).float()
    return ok.mean().item(), ok[gap < 0.03].mean().item()


def main():
    Ms, names = models()
    x = th.randn(1 << 14, T, dtype=th.float64, generator=th.Generator().manual_seed(7))
    As = [acts(M, x) for M in Ms]
    Cs = [canon(M, A) for M, A in zip(Ms, As)]
    Al = []
    for C, A in zip(Cs, As):
        c = corr(As[0], A); r, k = linear_sum_assignment(-c); Al.append(permute(C, k))

    print('(a) Schur taxonomy of canonical W_hh (unit-rms neurons). amp: real |l|>1.05, trans: real 0.5-1.05,')
    print('    att: real 0.05-0.5, null: |l|<=0.05, cpairs: complex pairs; coup_frac = ||N||^2/||W||^2;')
    print('    coup_rank90 = rows of N for 90% of coupling energy.')
    print('  model     rho   amp trans att null cpairs  henrici coup_frac rank90')
    tax = []
    for n, M in zip(names, Al):
        t = taxonomy(M[1]); tax.append(t)
        print(f'  {n:8s} {t["rho"]:5.2f}   {t["amp"]:2d}   {t["trans"]:2d}   {t["att"]:2d}   {t["null"]:2d}    {t["cpairs"]:2d}     '
              f'{t["henrici"]:5.2f}    {t["coup_frac"]:5.2f}     {t["coup_rank90"]:2d}')
    print('  |eigenvalues| sorted, per model:')
    for n, M in zip(names, Al):
        print(f'  {n:8s}' + ' '.join(f'{v:5.2f}' for v in np.sort(np.abs(np.linalg.eigvals(M[1])))[::-1]))

    print('\n(b) comparator: accuracy (overall / strip gap<.03) with the coupling N modified; "from tie" = exact')
    print('    W_hh until the later of the top two arrives, modified operator after. 2^18 inputs.')
    rng = np.random.default_rng(0)
    print('  model     full          N=0 all       N=0 from tie  N*0.5 tie     N*0.9 tie     N*1.1 tie     N shuffled tie  B shuffled tie')
    for n, M, t in zip(names, Al, tax):
        Q, B, N = t['Q'], t['B'], t['N']
        mk = lambda Bm, Nm: Q @ (Bm + Nm) @ Q.T
        iu = np.triu_indices(H, 1); Ns = N.copy(); Ns[iu] = rng.permutation(N[iu])
        d = np.diag(np.diag(B)); Bs = B - d + np.diag(rng.permutation(np.diag(B)))
        rows = [acc_fn(M), acc_fn(M, Wmod=mk(B, 0 * N)), acc_fn(M, Wmod=mk(B, 0 * N), from_tie=True),
                acc_fn(M, Wmod=mk(B, 0.5 * N), from_tie=True), acc_fn(M, Wmod=mk(B, 0.9 * N), from_tie=True),
                acc_fn(M, Wmod=mk(B, 1.1 * N), from_tie=True), acc_fn(M, Wmod=mk(B, Ns), from_tie=True),
                acc_fn(M, Wmod=mk(Bs, N), from_tie=True)]
        print(f'  {n:8s}' + ' '.join(f'{a:.3f}/{s:.3f}' for a, s in rows))

    print('\n(c) readout: omega(eps) = ||(R+eps*tr(R)/16 I)^-1/2 W_out^T||_F/||W_out^T||_F, R = E[h_10 h_10^T];')
    print('    and accuracy with the Capon-whitened readout W_out (R+eps)^-1 at inference (2^18 inputs).')
    print('  model     omega(1e-2) omega(1e-4) omega(1e-6)  eff.rank(R)   acc: orig  capon 1e-1  1e-2   1e-3   1e-4')
    for n, M in zip(names, Al):
        A = acts(M, x)[-(1 << 14):]  # h_10 rows
        R = A.T @ A / len(A); ev = np.linalg.eigvalsh(R); er = ev.sum() ** 2 / (ev ** 2).sum()
        tr = np.trace(R) / H
        om = []
        for eps in (1e-2, 1e-4, 1e-6):
            w, V = np.linalg.eigh(R + eps * tr * np.eye(H)); Rih = V @ np.diag(w ** -0.5) @ V.T
            om.append(np.linalg.norm(Rih @ M[2].T) / np.linalg.norm(M[2].T))
        ac = [acc_fn(M)[0]]
        for eps in (1e-1, 1e-2, 1e-3, 1e-4):
            Wc = M[2] @ np.linalg.inv(R + eps * tr * np.eye(H))
            ac.append(acc_fn([M[0], M[1], Wc])[0])
        print(f'  {n:8s}  {om[0]:9.2f}   {om[1]:9.2f}   {om[2]:9.2f}     {er:5.2f}       ' + '  '.join(f'{a:.3f}' for a in ac))


if __name__ == '__main__':
    main()
