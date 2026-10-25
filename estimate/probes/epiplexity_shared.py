"""Which of the teacher's near-tie errors do the distilled students recover? (epiplexity K1,
the saved students from epiplexity.py --save)

Splits the teacher's near-tie errors on the 2^20 held-out set into those no other h16 seed
makes (teacher-unique) and those at least one seed also makes, then reports for each saved
student and each seed the fraction of each part it also gets wrong, and its sharing with the
seeds' own near-tie errors. A student that learned only the task-hard part recovers the
shared part and none of the unique part (as the seeds do, by construction); a student that
learned the teacher's pattern recovers the unique part too.

Usage: uv run python probes/epiplexity_shared.py
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import forward
from opt_bits import loaders
from surprise import feats
from epiplexity import regions, tens

STUDENTS = [('h128 distil 2^30', 'probes/epiplexity_scratch_h128_k30_distil_trained_h128-distil.npz'),
            ('h128 label 2^30', 'probes/epiplexity_scratch_h128_k30_label_trained_h128-label.npz'),
            ('h128 distil s1 2^28', 'probes/epiplexity_scratch_h128_k28_s1_trained_h128-distil.npz'),
            ('h128 label s1 2^28', 'probes/epiplexity_scratch_h128_k28_s1_trained_h128-label.npz'),
            ('h64 distil 2^30', 'probes/epiplexity_scratch_h64_k30_distil_trained_h64-distil.npz'),
            ('h64 label 2^30', 'probes/epiplexity_scratch_h64_k30_label_trained_h64-label.npz')]


def main():
    L = loaders(); xb = th.randn(1 << 20, 10, generator=th.Generator().manual_seed(2))
    F = feats(xb); nt = regions(F, xb)['near-tie']; lab = F['label']

    def err(W):
        with th.no_grad(): return (forward(xb, *tens(W)).argmax(-1) != lab).numpy()
    eT = err(L['trained']()); seeds = {s: err(L[s]()) for s in ['seed0', 'seed1', 'seed2', 'seed3', 'seed4']}
    nseed = np.sum([seeds[s] for s in seeds], 0)
    uniq = eT & (nseed == 0) & nt; common = eT & (nseed >= 1) & nt; all5 = eT & (nseed == 5) & nt
    n = eT[nt].sum()
    print(f'teacher near-tie errors {n}: unique to the teacher {uniq.sum() / n:.3f}, shared with >= 1 seed '
          f'{common.sum() / n:.3f}, with all 5 seeds {all5.sum() / n:.3f}')
    print(f'{"model":20s} {"nt err":>7s}  recovers teacher errs: all / unique / shared>=1 / all5 | shares with seeds 0-4')
    for tag, f in STUDENTS + [(s, None) for s in seeds]:
        eS = seeds[tag] if f is None else err([np.load(f)[k] for k in ('W_ih', 'W_hh', 'W_out')])
        rec = lambda m: (eS & m).sum() / max(m.sum(), 1)
        print(f'{tag:20s} {eS[nt].mean():7.4f}  {rec(eT & nt):.3f} / {rec(uniq):.3f} / {rec(common):.3f} / {rec(all5):.3f} | '
              + ' '.join(f'{rec(seeds[s] & nt):.2f}' for s in seeds))


if __name__ == '__main__':
    main()
