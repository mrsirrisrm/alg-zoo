"""Does integrating more inputs deterministically (Gauss-Legendre product rule) beat the Sobol
lattice inside BnB regions? For each model and rule: rms vs truth over 4 scrambles, FLOPs,
advantage over MC (as in crossover.py). gl = number of lattice dims done by G-point GL.
Usage: python survey/exact_dims.py zoo_h6_T3 [...]
"""
import json, sys
sys.path[:0] = ['.', 'survey']
import numpy as np
from alg_zoo import zoo_2nd_argmax
from bnb import BnB

cx = json.load(open('survey/crossover.json')); out = {}
for key in sys.argv[1:]:
    c = cx[key]; h, T, p, F = c['h'], c['T'], c['acc'], c['F']
    sd = zoo_2nd_argmax(h, T).state_dict()
    W = (sd['rnn.weight_ih_l0'][:, 0], sd['rnn.weight_hh_l0'], sd['linear.weight'])
    d = T - 1; rows = []
    cfgs = [(0, 1, Q) for Q in (256, 1024, 4096)]
    for gl in range(1, d + 1):
        for G in (8, 16, 32):
            for Q in ((1,) if gl == d else (16, 64, 256)):
                if Q * G ** gl <= 1 << 16:
                    cfgs.append((gl, G, Q))
    print(f"{key} truth {p:.7f} +- {c['truth_std']:.1e}", flush=True)
    for gl, G, Q in cfgs:
        n = Q * G ** gl
        try:
            rs = [BnB(W, T, Q=Q, eps=1e-6, seed=s, gl=gl, G=G, chunk=max(16, (1 << 22) // n)).run()
                  for s in ((3,) if gl == d else (3, 4, 5, 6))]
        except RuntimeError as e:
            print(f"  gl {gl} G {G:2d} Q {Q:4d}  capped {e}", flush=True); continue
        a = np.array([r['acc'] for r in rs]); fl = float(np.mean([r['flops'] for r in rs]))
        rms = float(np.sqrt(np.mean((a - p) ** 2))); rr = max(rms, c['truth_std'])
        adv = p * (1 - p) * F / rr ** 2 / fl
        rows.append(dict(gl=gl, G=G, Q=Q, pts=n, rms=rms, flops=fl, adv=adv))
        print(f"  gl {gl} G {G:2d} Q {Q:4d} pts {n:6d}  rms {rms:.1e}  flops {fl:.1e}  adv {adv:.3g}", flush=True)
    out[key] = dict(h=h, T=T, acc=p, truth_std=c['truth_std'], rows=rows,
                    best={g: max([r['adv'] for r in rows if r['gl'] == g], default=None) for g in range(d + 1)})
    print(f"  best adv by gl: {out[key]['best']}  (crossover.py best {c['best_adv']:.3g})", flush=True)
    json.dump(out, open('survey/exact_dims.json', 'w'), indent=1)
