"""Model-space map figure: BnB advantage over MC (left) and linear-region count n99 (right) on h x T.
Advantage uses max(rms, truth s.e.) so cells where BnB beats the truth's own precision are lower bounds.
"""
import json
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cx = json.load(open('survey/crossover.json')); sv = json.load(open('survey/survey.json'))
H = [2, 3, 4, 5, 6, 8, 16, 32]; Ts = [2, 3, 4, 5, 6, 8, 10]
adv = np.full((len(Ts), len(H)), np.nan); n99 = np.full_like(adv, np.nan)
for k, v in cx.items():
    if not k.startswith('zoo'):
        continue
    best = max((v['acc'] * (1 - v['acc']) * v['F'] / max(r['rms'], v['truth_std']) ** 2 / r['flops']
                for r in v['rows'] if 'adv' in r), default=np.nan)
    adv[Ts.index(v['T']), H.index(v['h'])] = best
for r in sv:
    if r['model'] == 'zoo':
        n99[Ts.index(r['T']), H.index(r['h'])] = r['n99']
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for a, M, title, cmap, lim in [(ax[0], np.log10(adv), 'log10 advantage of deduction (BnB) over MC', 'RdBu', 4),
                                (ax[1], np.log10(n99), 'log10 linear regions covering 99% of mass', 'viridis', None)]:
    im = a.imshow(M, origin='lower', cmap=cmap, vmin=-lim if lim else None, vmax=lim if lim else None, aspect='auto')
    for i in range(len(Ts)):
        for j in range(len(H)):
            if not np.isnan(M[i, j]):
                a.text(j, i, f"{M[i, j]:.1f}", ha='center', va='center', fontsize=8)
    a.set_xticks(range(len(H)), H); a.set_yticks(range(len(Ts)), Ts)
    a.set_xlabel('hidden width h'); a.set_ylabel('sequence length T'); a.set_title(title, fontsize=10)
    fig.colorbar(im, ax=a)
ax[0].text(H.index(16), Ts.index(10), 'M(16,10)\nnot run', ha='center', va='center', fontsize=7)
ax[1].scatter([H.index(16)], [Ts.index(10)], s=300, facecolors='none', edgecolors='r')
fig.tight_layout(); fig.savefig('survey/crossover.png', dpi=130)
for i, T in enumerate(Ts):
    print(T, ' '.join(f"{x:8.2g}" for x in adv[i]))
