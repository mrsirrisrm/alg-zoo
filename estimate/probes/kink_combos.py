"""How many kink-neuron switching combinations carry the tie errors?
For each tie-layer error (network picks R: S/R tie; network picks M: M/S tie), project the
input onto the true wall (set the two tied values to their mean) and read the on/off
pattern of the kink neurons from the step the later tied value arrives to the end. Errors
sharing a pattern share one linear piece at the wall (all other gates as they are), so the
number of patterns covering most of the error mass measures how compact a
'one width per combination' explanation could be. Also: how many distinct patterns the
segment from the wall to the error input crosses (1 = no kink between wall and error).
"""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from ground_truth import forward, load_weights

KINK = [0, 4, 5, 6, 7, 8, 9, 11]
N = 1 << 22


def bits(x, W, neurons):
    W_ih, W_hh, _ = W; h = th.relu(x[:, :1] * W_ih); out = [h > 0]
    for t in range(1, 10):
        h = th.relu(x[:, t:t + 1] * W_ih + h @ W_hh.T); out.append(h > 0)
    return th.stack(out, 1)[:, :, neurons]          # n, 10, k


def key(b, t2):
    b = b.clone(); b[th.arange(10)[None, :] < t2[:, None]] = False    # only steps >= t2
    return np.packbits(b.reshape(len(b), -1).numpy(), axis=1)


def coverage(keys):
    _, cnt = np.unique(keys, axis=0, return_counts=True)
    c = np.cumsum(np.sort(cnt)[::-1]) / cnt.sum()
    return len(cnt), [int(np.searchsorted(c, q) + 1) for q in (.5, .9, .99)]


def run(name, W):
    W = [th.tensor(a, dtype=th.float32) for a in W]
    x = th.randn(N, 10, generator=th.Generator().manual_seed(4))
    t = x.topk(3, -1); pM, pS, pR = t.indices.T
    with th.no_grad():
        out = forward(x, *W).argmax(-1)
    print(f"\n{name}: error {(out != pS).float().mean():.4f}")
    for kind, mask, (a, b) in (('S/R', out == pR, (pS, pR)), ('M/S', out == pM, (pM, pS))):
        idx = th.nonzero(mask).flatten(); xe = x[idx]; a, b = a[idx], b[idx]; ar = th.arange(len(idx))
        x0 = xe.clone(); m = (xe[ar, a] + xe[ar, b]) / 2; x0[ar, a] = m; x0[ar, b] = m
        t2 = th.maximum(a, b)
        with th.no_grad():
            for lab, neu in (('kink 8', KINK), ('all 16', list(range(16)))):
                n_all, cov = coverage(key(bits(x0, W, neu), t2))
                print(f"  {kind} errors {len(idx)/N:.4f}: {lab:6s} patterns at the wall {n_all:6d}; cover 50/90/99% with {cov}")
            seg = [key(bits(x0 + f * (xe - x0), W, KINK), t2) for f in np.linspace(0, 1, 9)]
            ncross = np.array([len({s[i].tobytes() for s in seg}) for i in range(len(idx))])
            print(f"    wall->error segment: 1 piece {np.mean(ncross == 1):.2f}, 2 {np.mean(ncross == 2):.2f}, >2 {np.mean(ncross > 2):.2f}")
            pair = (a * 10 + b).numpy()
            per = [coverage(key(bits(x0[pair == p], W, KINK), t2[pair == p]))[1][1] for p in np.unique(pair) if (pair == p).sum() > 50]
            print(f"    per ordered position pair: patterns for 90% median {int(np.median(per))}, max {max(per)} ({len(per)} pairs)", flush=True)


if __name__ == '__main__':
    run('trained M(16,10)', load_weights('weights.npz'))
    d = np.load('noisy_sigma0.005.npz'); s = np.load('seeds_h16.npz')
    run('noisy0', [d[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
    run('seed0', [s[k][0] for k in ('W_ih', 'W_hh', 'W_out')])
