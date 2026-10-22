"""PLAN J5(a): every neuron against named algorithmic variables.

Features at step t (all positively homogeneous of degree 1, like the bias-free net):
  M, S          running max and 2nd max of x_<=t (S = 0 at t = 0)
  x_t
  relu(x_t - M_{t-1}), relu(M_{t-1} - x_t), relu(x_t - S_{t-1}), relu(S_{t-1} - x_t)
  M * 1[t - pM = k], S * 1[t - pS = k]   for lags k = 0..9 (position by time since each)
  --rich adds the joint lag table: (M, S, x_t) * 1[(t - pM, t - pS) = (k, l)] (300 more)
Fitted per neuron by least squares pooled over t = 0..9 (the variables are time-indexed,
so one map serves every step), on 2^14 inputs, scored on 2^14 held-out.
Reported per model:
  R2      per neuron, held-out (sorted); the mean
  surr    accuracy when the final state h_9 is replaced by its fit (logits = W_out h_fit)
  1-swap  worst accuracy with one neuron's final activation replaced by its fit
  strip   share of errors with top-two gap < 0.03, net vs surrogate (J2's statistic)
  err/g   error rate by S-R gap bin (S/R type) and M-S gap bin (M/S type), net vs surrogate
Check: Goldowsky-Dill's comparator fit on the trained model (n1/n6/n7/n8, R^2 ~0.99 on
clean two-impulse data) should appear as R^2 >= 0.95 on those neurons here.

Usage: uv run python probes/alg_vars.py [--rich] [names...]
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from opt_bits import loaders
from seed_template import acts

T = 10
RICH = '--rich' in sys.argv
GAPS = [0, 0.01, 0.03, 0.1, 0.3, 1, 10]


def features(x):
    x = x.numpy(); N = len(x); F = []
    M = np.full(N, -np.inf); S = np.full(N, -np.inf); pM = np.zeros(N, int); pS = np.zeros(N, int)
    for t in range(T):
        xt = x[:, t]; Mp = np.where(np.isinf(M), 0.0, M); Sp = np.where(np.isinf(S), 0.0, S)
        new = xt > M; sec = ~new & (xt > S)
        S = np.where(new, M, np.where(sec, xt, S)); pS = np.where(new, pM, np.where(sec, t, pS))
        M = np.where(new, xt, M); pM = np.where(new, t, pM)
        Sf = np.where(np.isinf(S), 0.0, S)
        oM = np.eye(T)[t - pM] * M[:, None]; oS = np.eye(T)[t - pS] * Sf[:, None] * (t > 0)
        cols = [np.stack([M, Sf, xt, np.maximum(xt - Mp, 0), np.maximum(Mp - xt, 0),
                          np.maximum(xt - Sp, 0), np.maximum(Sp - xt, 0)], 1), oM, oS]
        if RICH:
            o2 = np.eye(T * T)[(t - pM) * T + (t - pS)] * (t > 0)
            cols += [o2 * M[:, None], o2 * Sf[:, None], o2 * xt[:, None]]
        F.append(np.concatenate(cols, 1))
    return np.concatenate(F)  # (T*N, 27 or 327), t-major like acts()


def err_stats(x, out):
    x = x.numpy(); top = np.sort(x, 1)[:, ::-1]; idx = np.argsort(-x, 1)
    wrong = out != idx[:, 1]
    gSR, gMS = top[:, 1] - top[:, 2], top[:, 0] - top[:, 1]
    strip = (wrong & (np.minimum(gSR, gMS) < 0.03)).sum() / max(wrong.sum(), 1)
    sr, ms = out == idx[:, 2], out == idx[:, 0]
    bins = lambda g, e: [e[(g >= a) & (g < b)].mean() for a, b in zip(GAPS, GAPS[1:])]
    return 1 - wrong.mean(), strip, bins(gSR, sr), bins(gMS, ms)


def run(name, W):
    H = len(W[0]); g = th.Generator().manual_seed(21)
    xa, xb = th.randn(1 << 14, T, dtype=th.float64, generator=g), th.randn(1 << 14, T, dtype=th.float64, generator=g)
    Fa, Fb = features(xa), features(xb); Aa, Ab = acts(W, xa), acts(W, xb)
    beta, *_ = np.linalg.lstsq(Fa, Aa, rcond=None)
    res = Ab - Fb @ beta; v = Ab.var(0)
    r2 = np.where(v > 1e-12, 1 - res.var(0) / np.maximum(v, 1e-12), np.nan)
    N = len(xb); hT, fT = Ab[-N:], (Fb @ beta)[-N:]
    Wo = W[2]
    out_net, out_sur = (hT @ Wo.T).argmax(1), (fT @ Wo.T).argmax(1)
    swap = []
    for j in range(H):
        h = hT.copy(); h[:, j] = fT[:, j]; swap.append((h @ Wo.T).argmax(1))
    lab = np.argsort(-xb.numpy(), 1)[:, 1]
    sw = np.array([(o == lab).mean() for o in swap])
    a_n, s_n, sr_n, ms_n = err_stats(xb, out_net); a_s, s_s, sr_s, ms_s = err_stats(xb, out_sur)
    order = np.argsort(-np.nan_to_num(r2, nan=-9))
    print(f'\n{name}: h={H}  mean R2 {np.nanmean(r2):.3f}  acc net {a_n:.4f}  surrogate {a_s:.4f}  '
          f'worst 1-swap {sw.min():.4f} (n{sw.argmin()})  strip net {s_n:.2f} surr {s_s:.2f}')
    print('  R2 by neuron: ' + ' '.join(f'n{j}:{r2[j]:.2f}' for j in order))
    print('  1-swap acc:   ' + ' '.join(f'n{j}:{sw[j]:.3f}' for j in order))
    gl = ' '.join(f'[{a},{b})' for a, b in zip(GAPS, GAPS[1:]))
    print(f'  gap bins {gl}')
    print('  S/R err net  ' + ' '.join(f'{e:.3f}' for e in sr_n) + '   surr ' + ' '.join(f'{e:.3f}' for e in sr_s))
    print('  M/S err net  ' + ' '.join(f'{e:.3f}' for e in ms_n) + '   surr ' + ' '.join(f'{e:.3f}' for e in ms_s), flush=True)


if __name__ == '__main__':
    L = loaders(); names = [a for a in sys.argv[1:] if a != '--rich'] or ['trained', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4', 'h32']
    for n in names:
        run(n, L[n]())
