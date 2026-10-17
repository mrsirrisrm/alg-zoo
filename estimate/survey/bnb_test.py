import sys, json; sys.path[:0] = [".", "survey"]
import torch as th
from alg_zoo import zoo_2nd_argmax, handcrafted
from bnb import BnB
def W(m):
    sd = m.state_dict(); return sd['rnn.weight_ih_l0'][:, 0], sd['rnn.weight_hh_l0'], sd['linear.weight']
for name, m, T in [('hand2', handcrafted.handcrafted_2nd_argmax_2(), 2), ('hand3', handcrafted.handcrafted_2nd_argmax_3(), 3),
                   ('zoo h3T3', zoo_2nd_argmax(3, 3), 3), ('zoo h2T3', zoo_2nd_argmax(2, 3), 3), ('zoo h4T4', zoo_2nd_argmax(4, 4), 4)]:
    x = th.randn(1 << 22, T, dtype=th.float64)
    with th.no_grad(): mc = (m.double()(x).argmax(-1) == x.topk(2, -1).indices[:, 1]).double().mean().item()
    r = BnB(W(m), T, Q=256, eps=1e-7).run()
    print(name, f'MC {mc:.5f}+-{(mc*(1-mc)/(1<<22))**.5:.1e}', json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
