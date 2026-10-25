"""What the flag channel does in the trained privileged-information student (epiplexity_flag.py --save).

Two questions on the 2^20 held-out set, near-tie region:
  1. Counterfactual f=+1 on inputs the teacher gets right: which ones flip? If the flag is a
     shift of a shared decision threshold, the flipped inputs are the thin-margin band of the
     teacher's own decision (small label-minus-runner-up logit gap); if the flag drives
     separate machinery, they are not margin-selected.
  2. Teacher errors that the student reproduces only when told (true f) and not with f=0:
     where are they in the teacher's margin? Expected: the thin band the f=0 student misses.

Usage: uv run python probes/epiplexity_flag_which.py [--student probes/epiplexity_flag_h128_k30_h128.npz]
"""
import sys, argparse
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import forward
from opt_bits import loaders
from surprise import feats
from epiplexity import regions, tens
from epiplexity_flag import fwd

Q = (0.1, 0.25, 0.5, 0.75, 0.9)
qs = lambda v: ' '.join(f'{x:.3f}' for x in np.quantile(v, Q))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--student', default='probes/epiplexity_flag_h128_k30_h128.npz')
    a = ap.parse_args(); L = loaders()
    xb = th.randn(1 << 20, 10, generator=th.Generator().manual_seed(2)); F = feats(xb); nt = regions(F, xb)['near-tie']; lab = F['label']
    S = np.load(a.student); P = tens([S[k] for k in ('W_ih', 'W_f', 'W_hh', 'W_out')]); ar = th.arange(len(xb))
    with th.no_grad():
        zT = forward(xb, *tens(L['trained']())); yT = zT.argmax(-1)
        y0 = fwd(xb, th.zeros(len(xb), 1), *P).argmax(-1)
        yp = fwd(xb, th.ones(len(xb), 1), *P).argmax(-1)
        ym = fwd(xb, -th.ones(len(xb), 1), *P).argmax(-1)
    eT = (yT != lab).numpy()
    # teacher's margin for its answer: top logit minus runner-up (small = near its own boundary)
    zs = zT.sort(-1, descending=True).values; mT = (zs[:, 0] - zs[:, 1]).numpy()
    top = xb.topk(3, -1).indices.numpy()
    right = ~eT & nt
    flip = right & (yp != lab).numpy()
    print(f'near-tie inputs the teacher gets right: {right.sum()}; flipped by f=+1: {flip.sum()} ({flip.mean() / right.mean():.3f})')
    print(f'  teacher margin quantiles {Q}: flipped {qs(mT[flip])} | not flipped {qs(mT[right & ~flip])}')
    print(f'  flipped answer is top1 {(yp.numpy()[flip] == top[flip, 0]).mean():.3f}, third {(yp.numpy()[flip] == top[flip, 2]).mean():.3f}')
    edges = np.quantile(mT[right], np.linspace(0, 1, 9))
    print('  flip rate by octile of the teacher margin on correct near-tie inputs:')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = right & (mT >= lo) & (mT < hi); print(f'    [{lo:6.3f}, {hi:6.3f})  n {m.sum():6d}  flipped {flip[m].mean():.3f}')
    # also f=-1 ("teacher is right") on inputs where the teacher errs: does the student then give the label?
    E = eT & nt
    print(f'\nteacher near-tie errors {E.sum()}: student f=0 reproduces {(y0 == yT).numpy()[E].mean():.3f}, '
          f'f=+1 {(yp == yT).numpy()[E].mean():.3f}, f=-1 (told it is right) says the label {(ym == lab).numpy()[E].mean():.3f}')
    only = E & (y0 != yT).numpy() & (yp == yT).numpy(); both = E & (y0 == yT).numpy(); never = E & (yp != yT).numpy()
    mE = (zT[ar, yT] - zT[ar, lab]).numpy()                     # teacher's margin for its wrong answer over the label
    print(f'  teacher wrong-answer margin quantiles: reproduced at f=0 {qs(mE[both])} | only with the flag {qs(mE[only])} | never {qs(mE[never])}')
    print(f'  counts: both {both.sum()}, only with flag {only.sum()}, never {never.sum()}')


if __name__ == '__main__':
    main()
