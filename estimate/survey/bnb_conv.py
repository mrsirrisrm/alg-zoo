"""BnB convergence check on one model: is the rms floor quadrature variance or bias?
Many lattice scrambles per (Q, eps); report mean - truth (with its standard error), spread, lost.
Usage: python survey/bnb_conv.py zoo_h6_T3
"""
import json, sys
sys.path[:0] = ['.', 'survey']
import numpy as np
from alg_zoo import zoo_2nd_argmax
from bnb import BnB

key = sys.argv[1]; c = json.load(open('survey/crossover.json'))[key]
h, T, p, sd_t = c['h'], c['T'], c['acc'], c['truth_std']
sd = zoo_2nd_argmax(h, T).state_dict()
W = (sd['rnn.weight_ih_l0'][:, 0], sd['rnn.weight_hh_l0'], sd['linear.weight'])
print(f"{key} truth {p:.7f} +- {sd_t:.1e}", flush=True)
for eps in [1e-6, 1e-8]:
    for Q in [256, 1024, 4096, 16384, 65536]:
        rs = [BnB(W, T, Q=Q, eps=eps, seed=s, chunk=max(16, (1 << 22) // Q), max_sec=900).run() for s in range(3, 11)]
        a = np.array([r['acc'] for r in rs]); n = len(a)
        print(f"eps {eps:.0e} Q {Q:6d}  bias {a.mean() - p:+.2e} +- {a.std(ddof=1) / n**.5:.1e}  "
              f"spread {a.std(ddof=1):.1e}  rms {np.sqrt(np.mean((a - p)**2)):.1e}  lost {np.mean([r['lost'] for r in rs]):.1e}  "
              f"regions {np.mean([r['regions'] for r in rs]):.0f}  sec {np.mean([r['sec'] for r in rs]):.1f}", flush=True)
