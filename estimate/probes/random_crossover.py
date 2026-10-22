"""PLAN J7: random-weight base case for the model-space map (survey/survey.py, crossover.py).

Is the region count, and with it the BnB-vs-MC crossover, set by the architecture or by
training? Per (h, T), h in 2/4/8/16 and T = 2..6, three weight sets:
  trained  the zoo best seed (n99 and best advantage from survey.json / crossover.json)
  init     PyTorch's default RNN/Linear init, uniform(+-1/sqrt(h)) (seed 0)
  shuffle  the trained entries permuted within each matrix (same magnitudes, no structure)
n99 = activation patterns covering 99% of mass (2^20 inputs, survey.measure); rho = spectral
radius of W_hh. BnB as in crossover.py, configs (eps, Q) = (1e-4, 256) and (1e-6, 1024),
lattice seeds 3-5, only where n99 <= 2500, 60 s cap per run; truth by 2^26 MC.
'>' marks an advantage where the BnB rms is below 2x the truth's own std (a lower bound).
Accuracy of a random net is near chance, but its estimation problem is as well posed.

Usage: uv run python probes/random_crossover.py
"""
import sys, json
sys.path[:0] = ['.', 'survey', 'probes']
import numpy as np, torch as th
from zoo_cache import zoo_weights
from bnb import BnB

# survey.py is a script (importing it reruns the survey and rewrites survey.json): copied.
N = 1 << 20


def run(x, W_ih, W_hh, W_out):
    h = th.relu(x[:, :1] * W_ih); pats = [h > 0]
    for t in range(1, x.shape[1]):
        h = th.relu(x[:, t:t + 1] * W_ih + h @ W_hh.T); pats.append(h > 0)
    return h @ W_out.T, th.cat(pats, 1)


def measure(W, T, gen):
    x = th.randn(N, T, generator=gen, dtype=th.float64)
    logits, pat = run(x, *W)
    acc = (logits.argmax(-1) == x.topk(2, -1).indices[:, 1]).double().mean().item()
    R = th.randint(-2**62, 2**62, (2, pat.shape[1]), generator=gen)
    key = th.stack([(pat.long() * R[0]).sum(1), (pat.long() * R[1]).sum(1)], 1)
    _, cnt = th.unique(key, dim=0, return_counts=True)
    c = cnt.sort(descending=True).values.double()
    n99 = int((c.cumsum(0) < 0.99 * N).sum().item()) + 1
    nc, K = 256, 4096
    g1, g2 = th.randn(nc, 1, T, generator=gen, dtype=th.float64), th.randn(nc, 1, T, generator=gen, dtype=th.float64)
    tt = th.linspace(0, 2 * np.pi, K + 1, dtype=th.float64)[:-1].view(1, K, 1)
    xc = (th.cos(tt) * g1 + th.sin(tt) * g2).reshape(-1, T)
    lc, pc = run(xc, *W)
    corr = (lc.argmax(-1) == xc.topk(2, -1).indices[:, 1]).view(nc, K)
    pc = pc.view(nc, K, -1)
    kinks = (pc != pc.roll(1, 1)).any(-1).double().sum().item()
    flips = (corr != corr.roll(1, 1)).double().sum().item()
    return dict(acc=acc, n_pat=int(len(c)), n99=n99, unseen=float((c == 1).sum().item() / N),
                kinks_per_circle=kinks / nc, flips_per_circle=flips / nc, kink_per_flip=kinks / max(flips, 1))


CFG = [(1e-4, 256), (1e-6, 1024)]


def truth(W, T):
    gen = th.Generator().manual_seed(1); c = 0; n = 1 << 26; B = 1 << 22
    W_ih, W_hh, W_out = W
    for _ in range(n // B):
        x = th.randn(B, T, generator=gen, dtype=th.float64)
        h = th.relu(x[:, :1] * W_ih)
        for t in range(1, T):
            h = th.relu(x[:, t:t + 1] * W_ih + h @ W_hh.T)
        c += ((h @ W_out.T).argmax(-1) == x.topk(2, -1).indices[:, 1]).sum().item()
    p = c / n; return p, (p * (1 - p) / n) ** .5


def best_adv(W, T, h):
    p, sd = truth(W, T); F = T * (2 * h * h + 2 * h) + 2 * T * h; best = None
    for eps, Q in CFG:
        try:
            rs = [BnB(W, T, Q=Q, eps=eps, seed=s, max_sec=60).run() for s in (3, 4, 5)]
        except RuntimeError:
            continue
        rms = float(np.sqrt(np.mean([(q['acc'] - p) ** 2 for q in rs]))); fl = float(np.mean([q['flops'] for q in rs]))
        lb = rms < 2 * sd; rms = max(rms, 2 * sd) if lb else rms
        adv = p * (1 - p) * F / max(rms, 1e-12) ** 2 / fl
        if best is None or adv > best[0]: best = (adv, lb)
    return best


def main():
    surv = {(r['h'], r['T']): r for r in json.load(open('survey/survey.json')) if r['model'] == 'zoo'}
    cross = json.load(open('survey/crossover.json'))
    print(' h  T |  trained: acc   n99  rho  adv     |  init: acc   n99  rho  adv     |  shuffle: acc   n99  rho  adv')
    for T in range(2, 7):
        for h in (2, 4, 8, 16):
            Wt = [th.tensor(a) for a in zoo_weights(h, T)]; g = th.Generator().manual_seed(0); k = h ** -.5
            Wi = [(th.rand(*w.shape, generator=g, dtype=th.float64) * 2 - 1) * k for w in Wt]
            Ws = [w.reshape(-1)[th.randperm(w.numel(), generator=g)].reshape(w.shape) for w in Wt]
            row = f'{h:2d} {T:2d} |'
            for name, W in (('trained', Wt), ('init', Wi), ('shuffle', Ws)):
                rho = float(np.abs(np.linalg.eigvals(W[1].numpy())).max())
                if name == 'trained':
                    r = surv[(h, T)]; acc, n99 = r['acc'], r['n99']; c = cross.get(f'zoo_h{h}_T{T}')
                    adv = None if c is None or c['best_adv'] is None else (c['best_adv'], False)
                else:
                    r = measure(W, T, th.Generator().manual_seed(2)); acc, n99 = r['acc'], r['n99']
                    adv = best_adv(W, T, h) if n99 <= 2500 else None
                a = '-' if adv is None else ('>' if adv[1] else ' ') + f'{adv[0]:.1e}'
                row += f'  {acc:.3f} {n99:6d} {rho:4.2f} {a:9s} |'
            print(row.rstrip('|'), flush=True)


if __name__ == '__main__':
    main()
