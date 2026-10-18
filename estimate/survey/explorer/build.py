"""Build the interactive region explorer: embeds zoo weights (h x T), hand10, survey and
crossover stats into template.html -> explorer.html. The page runs the RNN in the browser."""
import json, sys
sys.path[:0] = ['.', 'survey']
from alg_zoo import zoo_2nd_argmax, handcrafted

sv = json.load(open('survey/survey.json')); cx = json.load(open('survey/crossover.json'))
r6 = lambda t: [float(f"{v:.6g}") for v in t.flatten().tolist()]
models = {}
for r in sv:
    h, T = r['h'], r['T']
    if r['model'] == 'zoo':
        m, key = zoo_2nd_argmax(h, T), f"zoo_{h}_{T}"
    elif r['model'] == 'hand10':
        m, key = handcrafted.handcrafted_2nd_argmax_10(), 'hand10'
    else:
        continue
    sd = m.state_dict()
    c = cx.get(f"{r['model']}_h{h}_T{T}")
    adv = None
    if c:
        ok = [q for q in c['rows'] if 'adv' in q]
        adv = max((c['acc'] * (1 - c['acc']) * c['F'] / max(q['rms'], c['truth_std']) ** 2 / q['flops'] for q in ok), default=None)
    models[key] = dict(h=h, T=T, params=r['params'], acc=r['acc'], n99=r['n99'], unseen=r['unseen'],
                       kpf=r['kink_per_flip'], adv=adv,
                       Wih=r6(sd['rnn.weight_ih_l0'][:, 0]), Whh=r6(sd['rnn.weight_hh_l0']), Wout=r6(sd['linear.weight']))
html = open('survey/explorer/template.html').read().replace('/*DATA*/null', json.dumps(models, separators=(',', ':')))
open('survey/explorer/explorer.html', 'w').write(html)
print(len(models), 'models,', len(html) // 1024, 'KB')
