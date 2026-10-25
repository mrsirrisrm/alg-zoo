"""Is the teacher's pattern carried by a small part of the distilled student? (epiplexity K1)

The distilled h128 student has 17,792 parameters, 41x the teacher's 432, so as an account
of the teacher's correctness bit it is longer than the weights at any precision. The only
way it could compete is if the learned pattern sits in a small, coarse part of it. Test:
magnitude-prune the student (per matrix, keep the largest |w|) and quantise it (uniform per
matrix, b bits), and track its near-tie error, its near-tie shared fraction with the
teacher, its recovery of the teacher-unique errors, and the description length
(kept parameters x bits, plus log2 C(n, kept) for the mask). No retraining.

Usage: uv run python probes/epiplexity_prune.py
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from scipy.special import gammaln
from ground_truth import forward
from opt_bits import loaders
from surprise import feats
from epiplexity import regions, tens

STUDENT = 'probes/epiplexity_scratch_h128_k30_distil_trained_h128-distil.npz'


def log2_binom(n, k):
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) / np.log(2)


def prune(W, keep):
    out = []
    for w in W:
        k = max(1, int(round(keep * w.size))); thr = np.sort(np.abs(w).ravel())[-k]
        out.append(np.where(np.abs(w) >= thr, w, 0.0))
    return out


def quant(W, bits):
    out = []
    for w in W:
        s = np.abs(w).max() / (2 ** (bits - 1) - 1)
        out.append(np.round(w / s) * s)
    return out


def main():
    L = loaders(); xb = th.randn(1 << 20, 10, generator=th.Generator().manual_seed(2))
    F = feats(xb); nt = regions(F, xb)['near-tie']; lab = F['label']

    def err(W):
        with th.no_grad(): return (forward(xb, *tens(W)).argmax(-1) != lab).numpy()
    eT = err(L['trained']()); nseed = np.sum([err(L[f'seed{i}']()) for i in range(5)], 0)
    E = eT & nt; U = E & (nseed == 0)
    W = [np.load(STUDENT)[k] for k in ('W_ih', 'W_hh', 'W_out')]; n = sum(w.size for w in W)
    print(f'student {n} parameters; teacher 432 (2429 precision bits, opt_bits)')

    def row(tag, Wm, bits):
        eS = err(Wm); kept = sum((w != 0).sum() for w in Wm)
        desc = kept * bits + sum(log2_binom(w.size, (w != 0).sum()) for w in Wm if (w != 0).sum() < w.size)
        print(f'  {tag:22s} kept {kept:6d}  desc ~{desc:9.0f} bits  near-tie err {eS[nt].mean():.4f}  '
              f'shared {(eS & E).sum() / E.sum():.3f}  unique rec {(eS & U).sum() / U.sum():.3f}  acc {1 - eS.mean():.4f}')
    print('magnitude pruning (per matrix), float32 weights counted at 16 bits:')
    for keep in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02):
        row(f'keep {keep:.2f}', prune(W, keep), 16)
    print('uniform quantisation (per matrix), no pruning:')
    for b in (8, 6, 5, 4, 3, 2):
        row(f'{b} bits', quant(W, b), b)
    print('both:')
    for keep, b in ((0.3, 6), (0.2, 6), (0.2, 4), (0.1, 6), (0.1, 4), (0.05, 6)):
        row(f'keep {keep:.2f} x {b} bits', quant(prune(W, keep), b), b)


if __name__ == '__main__':
    main()
