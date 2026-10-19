"""MSE vs compute for every estimator (CRITIQUE.md item 2), averaged over model sets.

Model sets: trained; 32 noisy copies (sigma = 0.005); zoo seeds 0-4.
Estimators (cost in FLOPs, one forward = 5760):
  MC            MSE = e(1-e)/N.
  strat-prop    strata on (g1, g2, sign vS), proportional allocation: stratum probabilities
                are input-law only, so this is a real estimator; MSE = MC / gain_prop.
  strat-Neyman  same strata, oracle Neyman allocation (ceiling, not an estimator).
  BL            1 - err_hat, err_hat = sum_k 1/2 f_Dk(0) mean(capped width); one width sample
                = 3 forwards for each of k = 1, 2. MSE(n) = bias^2 + var/n, bias vs the true
                total error (BL ignores the non-tie residual).
  BL+r          BL plus the trained model's residual (err_trained - BL_trained) as a fixed
                offset: offline advice from the trained model, no per-copy cost.
Writes mse_curves.json and mse_curves.png.
"""
import json
import numpy as np, torch as th
from boundary_layer import sample_ties, tie_density, widths
from ground_truth import forward, load_weights
from noisy_models import FLOPS_PER_SAMPLE as F

ERR_TRAINED = 1 - 0.9531186
f = {k: tie_density(k) for k in (1, 2)}
d, s = np.load('noisy_sigma0.005.npz'), np.load('seeds_h16.npz')
sets = {'trained': [([th.tensor(a) for a in load_weights('weights.npz')], ERR_TRAINED)],
        'noisy sigma=0.005 (32)': [([th.tensor(d[k][i]) for k in ('W_ih', 'W_hh', 'W_out')], 1 - d['acc'][i]) for i in range(32)],
        'zoo seeds 0-4': [([th.tensor(s[k][i]) for k in ('W_ih', 'W_hh', 'W_out')], 1 - s['acc'][i]) for i in range(5)]}

th.manual_seed(5); N = 1 << 22; xs = th.randn(N, 10)
top = xs.topk(3, -1); vM, vS, vR = top.values.T; S = top.indices[:, 1]
edges = [0, .005, .01, .02, .03, .05, .07, .1, .15, .2, .3, .5, 1, np.inf]
s1 = np.digitize(((vM - vS) / vM.abs()).numpy(), edges); s2 = np.digitize(((vS - vR) / vM.abs()).numpy(), edges)
strat = (s1 * len(edges) + s2) * 2 + (vS.numpy() < 0)


def bl_stats(W, gen, n=1 << 16):
    mean, var = 0.0, 0.0
    for k in (1, 2):
        x, a, b = sample_ties(k, n, gen)
        m0, al, ip = widths(x, a, b, W)
        sg = 1.0 if k == 2 else -1.0
        w = m0.abs() / (sg * al).clamp_min(1e-9)
        wc = th.where(ip & (sg * al > 0), w.clamp_max(0.25), th.zeros_like(w)) * 0.5 * f[k]
        mean += wc.mean().item(); var += wc.var().item()
    return mean, var


def strat_gains(W):
    e = (forward(xs, *W).argmax(-1) != S).numpy().astype(float)
    cnt = np.bincount(strat); err = np.bincount(strat, e)
    p = cnt[cnt > 0] / N; eh = err[cnt > 0] / cnt[cnt > 0]
    v = e.mean() * (1 - e.mean())
    return v / (p * eh * (1 - eh)).sum(), v / (p * np.sqrt(eh * (1 - eh))).sum() ** 2


gen = th.Generator().manual_seed(77)
Wt = [th.tensor(a) for a in load_weights('weights.npz')]
bl_tr, _ = bl_stats(Wt, gen)
r_tr = ERR_TRAINED - bl_tr
C = np.logspace(5, 13, 81)
res = {}
for name, models in sets.items():
    rows = []
    for W, e in models:
        bm, bv = bl_stats(W, gen)
        gp, gn = strat_gains(W)
        rows.append(dict(err=float(e), bl=bm, bl_var=bv, gain_prop=gp, gain_neyman=gn))
    E = np.array([r['err'] for r in rows])
    nmc = C / F; nbl = C / (6 * F)
    mse = {'MC': np.mean([e * (1 - e) / nmc for e in E], 0),
           'strat-prop': np.mean([r['err'] * (1 - r['err']) / nmc / r['gain_prop'] for r in rows], 0),
           'strat-Neyman (ceiling)': np.mean([r['err'] * (1 - r['err']) / nmc / r['gain_neyman'] for r in rows], 0),
           'BL': np.mean([(r['bl'] - r['err']) ** 2 + r['bl_var'] / nbl for r in rows], 0),
           'BL+r (advice)': np.mean([(r['bl'] + r_tr - r['err']) ** 2 + r['bl_var'] / nbl for r in rows], 0)}
    cross = {}
    for k in ('BL', 'BL+r (advice)'):
        better = mse[k] < mse['MC']
        cross[k] = float(C[better].max()) if better.any() else None
    res[name] = dict(rows=rows, rms_bias_bl=float(np.sqrt(np.mean([(r['bl'] - r['err']) ** 2 for r in rows]))),
                     rms_bias_blr=float(np.sqrt(np.mean([(r['bl'] + r_tr - r['err']) ** 2 for r in rows]))),
                     mean_gain_prop=float(np.mean([r['gain_prop'] for r in rows])),
                     mean_gain_neyman=float(np.mean([r['gain_neyman'] for r in rows])),
                     crossover_flops=cross, curves={k: np.sqrt(v).tolist() for k, v in mse.items()})
    print(name, json.dumps({k: v for k, v in res[name].items() if k not in ('rows', 'curves')}), flush=True)
res['C'] = C.tolist(); res['r_trained'] = r_tr; res['bl_trained'] = bl_tr
json.dump(res, open('mse_curves.json', 'w'), indent=1)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, name in zip(axs, sets):
    for k, v in res[name]['curves'].items():
        ax.loglog(C, v, label=k, ls='--' if 'ceiling' in k else '-')
    if name == 'trained':  # polyhedral S/R term alone, ~1e4 skeletons x 1e8 FLOPs, bias 3e-4..1e-3
        ax.errorbar([1e12], [np.sqrt(3e-4 * 1e-3)], yerr=[[np.sqrt(3e-4 * 1e-3) - 3e-4], [1e-3 - np.sqrt(3e-4 * 1e-3)]],
                    fmt='ks', label='polyhedral (S/R term only)')
    ax.set_xlim(1e5, 1e13); ax.set_title(name); ax.set_xlabel('FLOPs'); ax.grid(True, which='both', alpha=.3)
import os
if os.path.exists('probes/best_unbiased.json'):  # PLAN B5: measured stacked estimator (adaptive Neyman x exact last input), 12 reps
    bu = json.load(open('probes/best_unbiased.json'))
    groups = {'trained': ['trained'], 'noisy sigma=0.005 (32)': ['noisy0', 'noisy1'], 'zoo seeds 0-4': ['seed0', 'seed1']}
    for ax, name in zip(axs, sets):
        for key, mk, lab in (('adaptive/rb', 'o', 'adaptive Neyman x exact x9 (measured)'), ('oracle/rb', 'x', 'same, oracle allocation (ceiling)')):
            budgets = sorted(bu[groups[name][0]], key=int)
            xf = [np.mean([bu[m][b][key]['used'] for m in groups[name]]) * 1.15 * F for b in budgets]
            yr = [np.sqrt(np.mean([bu[m][b][key]['rms'] ** 2 for m in groups[name]])) for b in budgets]
            ax.loglog(xf, yr, mk, color='k', mfc='none' if mk == 'o' else None, label=lab + ('' if len(groups[name]) == 1 else f' ({len(groups[name])} models)'))
axs[0].set_ylabel('rms error of accuracy estimate'); axs[0].legend(fontsize=8, loc='upper right')
plt.tight_layout(); plt.savefig('mse_curves.png', dpi=120)
