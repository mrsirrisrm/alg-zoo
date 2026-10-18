"""Label-free Neyman allocation from the tie-layer explanation. Each stratum's error rate is
predicted as the stratum mean of e(x) = 1 - q_tie(x) (probes/surprise.py: boundary-layer
widths from the network's margins at NT sampled ties per tie kind, no correctness labels),
so no pilot is needed. Allocation n_h ∝ p_h sqrt(e_h(1-e_h)), with a 10% proportional floor
as in probes/strat_pos.py; variance from held-out split B. Compared with proportional
allocation, a pilot on split A, and the oracle. The widths cost 12*NT forward passes
(2 tie kinds x 2 orderings x 3 margins); with that charged, a design of gain G at n samples
beats MC once n > 12*NT*G/(G-1).
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import forward, load_weights
import surprise
from surprise import feats, widths, tie_q

N = 1 << 23
edges = [0, .005, .01, .02, .03, .05, .07, .1, .15, .2, .3, .5, 1, np.inf]
coarse = [0, .01, .03, .07, .15, .4, np.inf]
d, s = np.load('noisy_sigma0.005.npz'), np.load('seeds_h16.npz')
models = [('trained', load_weights('weights.npz'))]
models += [(f'noisy{i}', [d[k][i] for k in ('W_ih', 'W_hh', 'W_out')]) for i in range(4)]
models += [(f'seed{i}', [s[k][i] for k in ('W_ih', 'W_hh', 'W_out')]) for i in range(5)]


def keys(F):
    fine = (np.digitize(F['g1'], edges) * 14 + np.digitize(F['g2'], edges)) * 2 + F['neg']
    crs = (np.digitize(F['g1'], coarse) * 7 + np.digitize(F['g2'], coarse)) * 2 + F['neg']
    return {'gaps': fine, 'gaps+pos(M,S)': crs * 100 + F['pM'] * 10 + F['pS']}


def var(p, sb, sig):
    alloc = .9 * p * sig / max((p * sig).sum(), 1e-30) + .1 * p
    m = p > 0
    return (p[m] ** 2 * sb[m] ** 2 / alloc[m]).sum()


xa = th.randn(N, 10, generator=th.Generator().manual_seed(1)); FA = feats(xa)
xb = th.randn(N, 10, generator=th.Generator().manual_seed(2)); FB = feats(xb)
KA, KB = keys(FA), keys(FB)
if __name__ == "__main__":
    print(f"N = 2^23 per split; gains over MC at equal network samples, held-out (split B)")
    with th.no_grad():
        for name, W in models:
            W = [th.tensor(a, dtype=th.float32) for a in W]
            ea = (forward(xa, *W).argmax(-1) != FA['label']).numpy().astype(float)
            eb = (forward(xb, *W).argmax(-1) != FB['label']).numpy().astype(float)
            e = eb.mean(); out = [f"{name:8s} err {e:.4f}"]
            for lg in (12, 15, 18):
                surprise.NT = 1 << lg
                gen = th.Generator().manual_seed(lg)
                P1, P2 = widths(W, 1, gen), widths(W, 2, gen)
                et = tie_q(P1, P2, FB)          # predicted error from the inputs only (no labels, no network passes)
                for k in KB:
                    K = int(max(KA[k].max(), KB[k].max())) + 1
                    cb, rb = np.bincount(KB[k], minlength=K), np.bincount(KB[k], eb, minlength=K)
                    p = cb / N; qb = rb / np.maximum(cb, 1); sb = np.sqrt(qb * (1 - qb))
                    ph = np.bincount(KB[k], et, minlength=K) / np.maximum(cb, 1)
                    g = e * (1 - e) / var(p, sb, np.sqrt(ph * (1 - ph)))
                    if lg == 12:
                        ca, ra = np.bincount(KA[k], minlength=K), np.bincount(KA[k], ea, minlength=K)
                        qa = ra / np.maximum(ca, 1)
                        pil = e * (1 - e) / var(p, sb, np.sqrt(np.where(ca > 0, qa * (1 - qa), .25)))
                        prop = e * (1 - e) / (p * sb ** 2).sum(); orc = e * (1 - e) / (p * sb).sum() ** 2
                        out.append(f"| {k}: prop {prop:4.2f} pilot {pil:4.2f} oracle {orc:4.2f} tie@2^12 {g:4.2f}")
                    else:
                        out[-1 if k == 'gaps+pos(M,S)' else -2] += f" @2^{lg} {g:4.2f}"
            print(' '.join(out), flush=True)
