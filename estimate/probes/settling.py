"""Settling time of the comparator. probes/hazard.py found the commit hazard at EQUAL
relative gap rises ~100x from step 2 to step 9. Decoder- and completion-free check: the
final error rate by the arrival step of the later of the two confused values (S/R: later of
pS, pR; M/S: later of pM, pS) and the relative gap. Then the settling curve: for near-ties
(gap < .03) arriving at step t, the network's margin for the right answer after k benign
steps (inputs at the running min - .5, k = 0..9-t), normalised by |running max|, and the
fraction with the right sign. If the margin keeps growing with k, late ties are read out
before they have settled.
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import torch as th
from ground_truth import forward
from strat_tie import models

N = 1 << 21
BINS = ((0, .03), (.03, .1), (.1, .3), (.3, 99))


def run(name, W):
    W = [th.tensor(a, dtype=th.float32) for a in W]
    x = th.randn(N, 10, generator=th.Generator().manual_seed(9)); ar = th.arange(N)
    v, p = x.topk(3, -1); pM, pS, pR = p.T; vM, vS, vR = v.T
    with th.no_grad():
        pred = forward(x, *W).argmax(-1)
    err = pred != pS
    print(f"\n{name}: error {err.float().mean():.4f}")
    for kind, later, gap in (('S/R', th.maximum(pS, pR), (vS - vR) / vM.abs()), ('M/S', th.maximum(pM, pS), (vM - vS) / vM.abs())):
        print(f"  {kind}: final error rate by arrival step of the later value (cols t=1..9) and gap bin")
        for lo, hi in BINS:
            g = (gap >= lo) & (gap < hi)
            print(f"    gap {lo:.2f}-{hi:4.2f} " + ' '.join(f"{err[g & (later == t)].float().mean():.3f}" for t in range(1, 10)) + f"  (mass {g.float().mean():.3f})")
    # settling curves: near-ties (gap < .03) whose later value arrives at step t, read after k benign
    # steps with W_out rolled by 9-(t+k) (positions are encoded as steps ago; exact at k = 9-t).
    # S/R ties restricted to M already arrived (otherwise the running answer is R, not S).
    W_ih, W_hh, W_out = W
    for kind, m_all, a_all, b_all in (('S/R (M arrived before)', (pM < th.maximum(pS, pR)) & ((vS - vR) / vM.abs() < .03), pS, pR),
                                      ('M/S', (vM - vS) / vM.abs() < .03, pS, pM)):
        print(f"  {kind} near-ties: margin (logit right - logit wrong)/|max| after k benign steps from arrival step t; cols k=0..9-t  [mean | frac right sign]")
        with th.no_grad():
            for t in range(1, 10):
                m = m_all & (th.maximum(a_all, b_all) == t)
                xs = x[m]; a = a_all[m]; b = b_all[m]; n = len(xs); ar2 = th.arange(n)
                if n < 100: continue
                h = th.relu(xs[:, :1] * W_ih)
                for s_ in range(1, t + 1): h = th.relu(xs[:, s_:s_ + 1] * W_ih + h @ W_hh.T)
                c = xs[:, :t + 1].min(-1, keepdim=True).values - .5; row = []; g = h
                for k in range(0, 10 - t):
                    if k: g = th.relu(c * W_ih + g @ W_hh.T)
                    l = th.roll(g @ W_out.T, -(9 - (t + k)), 1); mg = (l[ar2, a] - l[ar2, b]) / xs[:, :t + 1].max(-1).values.abs()
                    row.append(f"{mg.mean():+.3f}|{(mg > 0).float().mean():.2f}")
                print(f"    t={t} (n={n:6d}): " + '  '.join(row), flush=True)

if __name__ == '__main__':
    for name, W in models[:1] + models[5:6]:
        run(name, W)
