"""PLAN I1-I3: is there a shared structured weight distribution behind the h16/T10 models?

ARC's compression route to the MSP: advice pi = a structured weight distribution P_pi such
that the trained weights look like a typical draw, plus an average-case estimator for P_pi.
Here we ask the empirical prerequisite on the six h16/T10 models (trained example + zoo
seeds 0-4):
  I1. align neurons across models up to the exact symmetries (permutation, positive per-
      neuron rescaling D: W_ih -> D W_ih, W_hh -> D W_hh D^-1, W_out -> W_out D^-1, and
      per-output-column shift of W_out); match by activation correlation (Hungarian);
  I2. per-entry agreement of the aligned weights (sign agreement, |mean|/sd);
  I3. template distribution P_s = mean + s * sd * N(0,1) per entry (from the six aligned
      models); MC accuracy of draws (validation only) against the models' 0.88-0.95.

Usage: uv run python probes/seed_template.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np, torch as th
from scipy.optimize import linear_sum_assignment
from ground_truth import forward, load_weights

H, T = 16, 10
th.manual_seed(0)


def models():
    w = load_weights('weights.npz'); s = np.load('seeds_h16.npz')
    Ms = [list(w)] + [[s['W_ih'][i], s['W_hh'][i], s['W_out'][i]] for i in range(5)]
    return [[np.asarray(a, dtype=np.float64) for a in M] for M in Ms], ['trained'] + [f'seed{i}' for i in range(5)]


def acts(M, x):
    """h_t for t=1..T, shape (N*T, H)."""
    W_ih, W_hh, _ = (th.tensor(a) for a in M)
    hs, h = [], th.relu(x[:, 0:1] * W_ih)
    hs.append(h)
    for t in range(1, T):
        h = th.relu(x[:, t:t + 1] * W_ih + h @ W_hh.T); hs.append(h)
    return th.cat(hs).numpy()


def acc(M, n_log2=20, seed=1):
    Wt = [th.tensor(a, dtype=th.float32) for a in M]
    x = th.randn(1 << n_log2, T, generator=th.Generator().manual_seed(seed))
    with th.no_grad():
        return (forward(x, *Wt).argmax(-1) == x.topk(2, -1).indices[:, 1]).float().mean().item()


def canon(M, A):
    """Rescale neurons to unit rms activation; centre W_out columns. Exact symmetries."""
    r = np.sqrt((A ** 2).mean(0)); r[r == 0] = 1
    D = 1 / r
    W_ih, W_hh, W_out = M
    W_out = W_out * r
    W_out = W_out - W_out.mean(0, keepdims=True)
    return [D * W_ih, D[:, None] * W_hh * r[None, :], W_out]


def permute(M, p):
    W_ih, W_hh, W_out = M
    return [W_ih[p], W_hh[np.ix_(p, p)], W_out[:, p]]


def corr(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    sa = np.sqrt((A ** 2).sum(0)); sb = np.sqrt((B ** 2).sum(0))
    sa[sa == 0] = 1; sb[sb == 0] = 1
    return (A.T @ B) / np.outer(sa, sb)


def main():
    Ms, names = models()
    x = th.randn(1 << 14, T, dtype=th.float64, generator=th.Generator().manual_seed(7))
    As = [acts(M, x) for M in Ms]
    Cs = [canon(M, A) for M, A in zip(Ms, As)]

    print('I1. symmetry check (acc at 2^20, original vs canonical):')
    for n, M, C in zip(names, Ms, Cs):
        print(f'  {n:8s} {acc(M):.5f} {acc(C):.5f}')
    dead = [int(((A ** 2).mean(0) == 0).sum()) for A in As]
    print('  dead neurons per model:', dict(zip(names, dead)))

    print('\nI1. pairwise matched activation correlation (Hungarian on corr; mean over 16 neurons):')
    R = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            c = corr(As[i], As[j]); r, k = linear_sum_assignment(-c)
            R[i, j] = c[r, k].mean()
    print('         ' + ' '.join(f'{n:>8s}' for n in names))
    for n, row in zip(names, R):
        print(f'  {n:8s}' + ' '.join(f'{v:8.3f}' for v in row))

    # align every model to the trained example
    perms, mcorr = [], []
    for A in As:
        c = corr(As[0], A); r, k = linear_sum_assignment(-c)
        perms.append(k); mcorr.append(c[r, k])
    mcorr = np.array(mcorr)
    print('\nI1. per-neuron match to the trained example (rows = trained neuron n0..n15):')
    print('  neuron ' + ' '.join(f'{n:>8s}' for n in names[1:]) + '   mean')
    for h in range(H):
        print(f'  n{h:<5d} ' + ' '.join(f'{mcorr[m, h]:8.3f}' for m in range(1, 6)) + f'  {mcorr[1:, h].mean():6.3f}')
    Al = [permute(C, p) for C, p in zip(Cs, perms)]
    print('  aligned acc check:', ' '.join(f'{acc(M):.5f}' for M in Al))

    print('\nI2. aligned weights, per-entry agreement across the 6 models:')
    for j, nm in enumerate(('W_ih', 'W_hh', 'W_out')):
        S = np.stack([M[j] for M in Al])
        mu, sd = S.mean(0), S.std(0)
        sgn = np.abs(np.sign(S).sum(0)) == 6
        big = np.abs(S).max(0) > 0.1 * np.abs(S).max()
        snr = np.abs(mu) / (sd + 1e-12)
        print(f'  {nm}: {S[0].size} entries, large {big.sum()}, of those sign-consistent '
              f'{(sgn & big).sum()}, |mean|/sd>2 {((snr > 2) & big).sum()}, median |mean|/sd (large) {np.median(snr[big]):.2f}')
    print('  W_ih aligned (rows = models, cols = trained neuron order):')
    for n, M in zip(names, Al):
        print(f'  {n:8s}' + ' '.join(f'{v:6.2f}' for v in M[0]))

    print('\nI3. template distribution P_s = mean + s*sd*N(0,1) (acc at 2^18, 32 draws):')
    mean = [np.stack([M[j] for M in Al]).mean(0) for j in range(3)]
    sd = [np.stack([M[j] for M in Al]).std(0, ddof=1) for j in range(3)]
    print(f'  template mean itself: acc {acc(mean, 20):.5f}')
    rng = np.random.default_rng(0)
    for s in (0.1, 0.25, 0.5, 1.0):
        a = np.array([acc([m + s * d * rng.standard_normal(m.shape) for m, d in zip(mean, sd)], 18, seed=3 + i)
                      for i in range(32)])
        print(f'  s={s:4.2f}: acc mean {a.mean():.4f} sd {a.std():.4f} min {a.min():.4f} max {a.max():.4f}')
    print('  six models:', ' '.join(f'{acc(M):.4f}' for M in Ms))

    # I4: keep each model's core exact (trained-order n1, n2, n4, n6, n7, n8: the matched
    # max/comparator circuit), replace every weight touching a periphery neuron.
    core = np.array([1, 2, 4, 6, 7, 8]); per = np.setdiff1d(np.arange(H), core)
    print('\nI4. core exact, periphery weights replaced (acc at 2^18, 16 draws; aligned models):')
    print('  model     orig    zero   shuffle(mean/max)  gauss(mean/max)')
    for n, M in zip(names, Al):
        masks = [np.zeros(H, bool), np.zeros((H, H), bool), np.zeros((10, H), bool)]
        masks[0][per] = True; masks[1][per, :] = True; masks[1][:, per] = True; masks[2][:, per] = True

        def repl(mode, i):
            r = np.random.default_rng(100 + i); out = []
            for a, m in zip(M, masks):
                a = a.copy(); v = a[m]
                a[m] = 0 if mode == 'zero' else r.permutation(v) if mode == 'shuffle' else r.standard_normal(v.size) * np.sqrt((v ** 2).mean())
                out.append(a)
            return out
        sh = np.array([acc(repl('shuffle', i), 18, seed=50 + i) for i in range(16)])
        ga = np.array([acc(repl('gauss', i), 18, seed=50 + i) for i in range(16)])
        print(f'  {n:8s} {acc(M, 18):.4f}  {acc(repl("zero", 0), 18):.4f}   {sh.mean():.4f}/{sh.max():.4f}      {ga.mean():.4f}/{ga.max():.4f}')

    print('\nI5. which half carries it: periphery dynamics only (W_ih, W_hh rows/cols) vs readout only'
          ' (W_out cols), shuffled within the block (acc at 2^18, 16 draws: mean/max):')
    for n, M in zip(names, Al):
        res = []
        for which in ('dyn', 'read'):
            a = []
            for i in range(16):
                r = np.random.default_rng(200 + i); W_ih, W_hh, W_out = (w.copy() for w in M)
                if which == 'dyn':
                    W_ih[per] = r.permutation(W_ih[per])
                    m = np.zeros((H, H), bool); m[per, :] = True; m[:, per] = True
                    W_hh[m] = r.permutation(W_hh[m])
                else:
                    W_out[:, per] = r.permutation(W_out[:, per].ravel()).reshape(10, -1)
                a.append(acc([W_ih, W_hh, W_out], 18, seed=70 + i))
            res.append(np.array(a))
        print(f'  {n:8s} dyn {res[0].mean():.4f}/{res[0].max():.4f}   read {res[1].mean():.4f}/{res[1].max():.4f}')


if __name__ == '__main__':
    main()
