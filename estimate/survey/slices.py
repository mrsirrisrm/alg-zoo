"""Picture of the difficulty ramp: one 2-D slice of input space per sequence length, h = 16.

No biases => the network is positively homogeneous, so an affine slice x = c + u a + v b
(c a Gaussian point, a, b orthonormal and orthogonal to c) is a faithful window on the cone
structure (a gnomonic chart of the sphere). T = 2 is shown whole (the plane itself).
Per T, 16 random slices are drawn and the one with the median region count is shown.
Colour = linear region (full on/off pattern over all steps), red = network wrong,
black = true label boundary (x_i = x_j walls). Right: regions vs T, slice vs whole space.
"""
import json, sys
sys.path[:0] = ['.', 'survey']
import numpy as np, torch as th, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alg_zoo import zoo_2nd_argmax

Ts = [2, 3, 4, 5, 6, 10]; h = 16; R = 600; NS = 16


def render(W, T, seed, R):
    Wih, Whh, Wo = W
    g = th.Generator().manual_seed(seed)
    u = th.linspace(-1, 1, R, dtype=th.float64)
    U, V = [z.reshape(-1, 1) for z in th.meshgrid(u, u, indexing='xy')]
    if T == 2:
        x = th.cat([U, V], 1) * 2
    else:
        c = th.randn(T, generator=g, dtype=th.float64)
        a, b = th.randn(2, T, generator=g, dtype=th.float64)
        a -= (a @ c) / (c @ c) * c; a /= a.norm()
        b -= (b @ c) / (c @ c) * c + (b @ a) * a; b /= b.norm()
        x = c + (U * a + V * b) * c.norm()
    hs = th.zeros(x.shape[0], h, dtype=th.float64); key = th.zeros(x.shape[0], dtype=th.int64)
    mult = th.randint(1, 2**61, (T, h), generator=th.Generator().manual_seed(0))
    for t in range(T):
        pre = x[:, t:t + 1] * Wih[:, 0] + hs @ Whh.T
        key = key * 1000003 + ((pre > 0).long() * mult[t]).sum(1)
        hs = pre.clamp_min(0)
    lab = x.topk(2, 1).indices[:, 1]
    wrong = ((hs @ Wo.T).argmax(1) != lab).reshape(R, R).numpy()
    _, rid = np.unique(key.numpy(), return_inverse=True)
    return rid.reshape(R, R), lab.reshape(R, R).numpy(), wrong


def nbr(m):
    e = np.zeros(m.shape, bool); e[1:] |= m[1:] != m[:-1]; e[:, 1:] |= m[:, 1:] != m[:, :-1]
    return e


fig = plt.figure(figsize=(19, 10))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15])
mean_slice = {}
for k, T in enumerate(Ts):
    sd = zoo_2nd_argmax(h, T).state_dict()
    W = [sd[n].double() for n in ('rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight')]
    seeds = [0] if T == 2 else range(1, NS + 1)
    counts = {s: render(W, T, s, 300)[0].max() + 1 for s in seeds}
    mean_slice[T] = float(np.mean(list(counts.values())))
    s = sorted(counts, key=counts.get)[len(counts) // 2]
    rid, lab, wrong = render(W, T, s, R)
    nreg = rid.max() + 1
    img = np.random.default_rng(1).uniform(.45, .97, (nreg, 3))[rid]
    img[nbr(rid)] *= .6
    w = nbr(lab); w[1:] |= w[:-1]; w[:, 1:] |= w[:, :-1]
    img[w] = 0
    img[wrong] = [.9, 0, 0]
    ax = fig.add_subplot(gs[k // 3, k % 3])
    ax.imshow(img, origin='lower', interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"T = {T}:  {nreg:,} regions in this slice,  {wrong.mean():.2%} wrong", fontsize=11)
    print(T, nreg, mean_slice[T], wrong.mean(), flush=True)

sv = {r['T']: r for r in json.load(open('survey/survey.json')) if r['model'] == 'zoo' and r['h'] == h}
TT = sorted(sv)
ax = fig.add_subplot(gs[:, 3])
ax.semilogy(TT, [sv[t]['n99'] for t in TT], 'o-', c='k', label='whole input space (regions covering 99% of mass)')
ax.semilogy(Ts, [mean_slice[t] for t in Ts], 's--', c='tab:blue', label='one 2-D slice (mean of 16)')
ax.semilogy(TT, [sv[t]['kink_per_flip'] for t in TT], '^:', c='tab:red', label='region edges crossed per right/wrong flip')
for t in TT:
    if sv[t]['unseen'] > .1:
        ax.annotate('≥', (t, sv[t]['n99']), xytext=(-14, 2), textcoords='offset points')
ax.axvspan(3.5, 4.5, color='0.85', zorder=0); ax.text(4, 3e3, 'deduction /\nsampling\ncrossover', ha='center', fontsize=9, rotation=90)
ax.set_xlabel('sequence length T'); ax.set_title('h = 16: the growth a 2-D slice cannot show', fontsize=11)
ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=.3, which='both')
fig.suptitle("Same width (h = 16), longer sequences. Each panel: a typical 2-D slice through input space.\n"
             "colour = one linear region of the network    black = true label boundary (two inputs equal)    red = network wrong",
             fontsize=12)
fig.tight_layout(); fig.savefig('survey/slices.png', dpi=100)
