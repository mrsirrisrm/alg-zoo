"""Step 3 of the model-space map: deductive (BnB) error vs compute against MC, per model.

For each model: truth by MC (2^27, float64); BnB over eps x Q, 3 lattice scrambles each;
rms error vs truth (truth std reported). MC at budget C has rms sqrt(p(1-p) F / C),
F = T(2h^2 + 2h) + 2Th FLOPs per sample. advantage(config) = MC compute needed for the
same rms / BnB compute; > 1 means deduction beats sampling at that budget.
Usage: python survey/crossover.py [max_n99]
"""
import json, sys, time
sys.path[:0] = ['.', 'survey']
import numpy as np, torch as th
from alg_zoo import zoo_2nd_argmax, handcrafted
from bnb import BnB

max_n99 = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
surv = json.load(open('survey/survey.json'))
hand = {'hand2': handcrafted.handcrafted_2nd_argmax_2, 'hand3': handcrafted.handcrafted_2nd_argmax_3,
        'hand10': handcrafted.handcrafted_2nd_argmax_10}
try:
    out = json.load(open('survey/crossover.json'))
except FileNotFoundError:
    out = {}
for r in sorted(surv, key=lambda r: r['n99']):
    key = f"{r['model']}_h{r['h']}_T{r['T']}"
    if key in out or r['n99'] > max_n99:
        continue
    h, T = r['h'], r['T']
    m = hand[r['model']]() if r['model'] != 'zoo' else zoo_2nd_argmax(h, T)
    sd = m.state_dict(); W = (sd['rnn.weight_ih_l0'][:, 0], sd['rnn.weight_hh_l0'], sd['linear.weight'])
    m = m.double(); gen = th.Generator().manual_seed(1); c = 0; n = 1 << 27; B = 1 << 22
    with th.no_grad():
        for _ in range(n // B):
            x = th.randn(B, T, generator=gen, dtype=th.float64)
            c += (m(x).argmax(-1) == x.topk(2, -1).indices[:, 1]).sum().item()
    p = c / n; F = T * (2 * h * h + 2 * h) + 2 * T * h
    rows = []
    for eps in [1e-3, 1e-4, 1e-5, 1e-6]:
        for Q in [64, 256, 1024, 4096]:
            try:
                rs = [BnB(W, T, Q=Q, eps=eps, seed=s).run() for s in (3, 4, 5)]
            except RuntimeError as e:
                rows.append(dict(eps=eps, Q=Q, capped=str(e))); print(key, eps, Q, 'capped', e, flush=True); break
            est = np.array([q['acc'] for q in rs]); fl = float(np.mean([q['flops'] for q in rs]))
            rms = float(np.sqrt(np.mean((est - p) ** 2)))
            adv = p * (1 - p) * F / max(rms, 1e-12) ** 2 / fl
            rows.append(dict(eps=eps, Q=Q, flops=fl, rms=rms, lost=float(np.mean([q['lost'] for q in rs])),
                             regions=int(np.mean([q['regions'] for q in rs])), sec=float(np.mean([q['sec'] for q in rs])), adv=adv))
    ok = [q for q in rows if 'adv' in q]
    out[key] = dict(h=h, T=T, params=r['params'], acc=p, truth_std=(p * (1 - p) / n) ** .5, F=F, n99=r['n99'], rows=rows,
                    best_adv=max([q['adv'] for q in ok], default=None))
    b = max(ok, key=lambda q: q['adv']) if ok else None
    desc = f"eps {b['eps']:.0e} Q {b['Q']} flops {b['flops']:.1e} rms {b['rms']:.1e}" if b else '-'
    print(f"{key:14s} acc {p:.5f} n99 {r['n99']:6d}  best advantage {out[key]['best_adv']}  at {desc}", flush=True)
    json.dump(out, open('survey/crossover.json', 'w'), indent=1)
