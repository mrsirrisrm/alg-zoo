"""Which of the teacher's near-tie errors does the distilled student recover, and which not?
(epiplexity K1; saved student from epiplexity.py --save)

On the 2^20 held-out set, take the teacher's near-tie errors and split them by whether the
distilled student also errs. Compare the two groups on the input geometry (relative gaps g1,
g2, sign of the second, positions and lag of the top two, number of positive inputs), on the
teacher's own decision margin (its top logit minus the label logit: small = the input sits
near the teacher's decision boundary), on whether the error is unique to this teacher, and
on whether the student's wrong answer is the teacher's wrong answer. Then the recovery rate
binned by teacher margin and by min(g1, g2), to see whether the unrecovered third is the
thin-margin part (student resolves the teacher's function to finite resolution) or the
large-gap part (idiosyncratic errors away from the strips).

Usage: uv run python probes/epiplexity_which.py [--student probes/..._h128-distil.npz]
"""
import sys, argparse
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import forward
from opt_bits import loaders
from surprise import feats
from epiplexity import regions, tens

Q = (0.1, 0.25, 0.5, 0.75, 0.9)


def qs(v):
    return ' '.join(f'{x:.3f}' for x in np.quantile(v, Q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--student', default='probes/epiplexity_scratch_h128_k30_distil_trained_h128-distil.npz')
    ap.add_argument('--control', default='probes/epiplexity_scratch_h128_k30_label_trained_h128-label.npz')
    a = ap.parse_args()
    L = loaders(); xb = th.randn(1 << 20, 10, generator=th.Generator().manual_seed(2))
    F = feats(xb); nt = regions(F, xb)['near-tie']; lab = F['label']; x = xb.numpy()
    with th.no_grad():
        zT = forward(xb, *tens(L['trained']()))
        zS = forward(xb, *tens([np.load(a.student)[k] for k in ('W_ih', 'W_hh', 'W_out')]))
        zC = forward(xb, *tens([np.load(a.control)[k] for k in ('W_ih', 'W_hh', 'W_out')]))
        seeds = [(forward(xb, *tens(L[f'seed{i}']())).argmax(-1) != lab).numpy() for i in range(5)]
    yT, yS, yC = zT.argmax(-1), zS.argmax(-1), zC.argmax(-1)
    eT = (yT != lab).numpy(); eS = (yS != lab).numpy(); eC = (yC != lab).numpy()
    ar = th.arange(len(xb))
    mT = (zT[ar, yT] - zT[ar, lab]).numpy()                 # teacher margin for its wrong answer (> 0 on errors)
    zo = zS.clone(); zo[ar, lab] = -th.inf
    mS = (zS[ar, lab] - zo.max(-1).values).numpy()           # student margin for the label (< 0 where it errs)
    nseed = np.sum(seeds, 0)
    E = eT & nt; rec = E & eS; miss = E & ~eS
    print(f'teacher near-tie errors {E.sum()}: recovered by student {rec.sum()} ({rec.sum() / E.sum():.3f}), missed {miss.sum()}')
    print(f'student gives the teacher\'s wrong answer on {(rec & (yS == yT).numpy()).sum() / rec.sum():.3f} of recovered errors; '
          f'control (label-trained) also errs on {(E & eC).sum() / E.sum():.3f} of teacher errors')
    g1, g2 = F['g1'], F['g2']; gmin = np.minimum(g1, g2); lag = np.abs(F['pM'] - F['pS']); npos = (x > 0).sum(1)
    print(f'\nquantiles {Q} of each feature, recovered | missed')
    for name, v in [('teacher margin', mT), ('g1 (top1-top2)/|top1|', g1), ('g2 (top2-top3)/|top1|', g2), ('min(g1,g2)', gmin),
                    ('lag |pM-pS|', lag), ('pos of second', F['pS']), ('npos', npos), ('|top1|', np.abs(x.max(1)))]:
        print(f'  {name:24s} {qs(v[rec])} | {qs(v[miss])}')
    for name, v in [('second < 0', F['neg']), ('teacher-unique (no seed errs)', nseed == 0), ('all 5 seeds err', nseed == 5),
                    ('g1 < g2 (error is top pair)', g1 < g2), ('label at t=9', F['pS'] == 9), ('max at t=9', F['pM'] == 9)]:
        print(f'  {name:30s} fraction  recovered {v[rec].mean():.3f}  missed {v[miss].mean():.3f}')

    def binned(name, v, edges):
        print(f'\nrecovery rate by {name}:')
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = E & (v >= lo) & (v < hi)
            if m.sum() > 50:
                print(f'  [{lo:7.3f}, {hi:7.3f})  n {m.sum():6d}  teacher-unique {(nseed[m] == 0).mean():.3f}  '
                      f'recovered {eS[m].mean():.3f}  control {eC[m].mean():.3f}')
    binned('teacher margin', mT, np.quantile(mT[E], np.linspace(0, 1, 9)))
    binned('min(g1,g2)', gmin, np.array([0, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 1]))
    binned('g1', g1, np.array([0, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.3, 10]))
    # the student's own margin on the teacher's errors: how far is it from flipping?
    print(f'\nstudent margin (label logit minus its top non-label logit; < 0 where it errs), quantiles: '
          f'recovered {qs(mS[rec])} | missed {qs(mS[miss])}')


if __name__ == '__main__':
    main()
