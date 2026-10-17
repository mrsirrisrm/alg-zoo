"""Step 1 of the model-space map: how 'packed' is each 2nd-argmax RNN?

For every zoo model (hidden h, length T, best seed) and the handcrafted models:
  acc       accuracy, 2^20 Gaussian inputs
  n_pat     distinct activation patterns (h*T bits) among those inputs
  n99       patterns needed to cover 99% of the sampled mass
  unseen    Good-Turing unseen-mass estimate (singletons / N)
  kink/flip along random great circles x = cos(t) g1 + sin(t) g2 (exactly N(0,I)):
            linear-region changes per change of the correct/wrong indicator
Writes survey/survey.json.
"""
import json, sys
import numpy as np, torch as th
sys.path.insert(0, '.')
from alg_zoo import zoo_2nd_argmax, handcrafted

N = 1 << 20


def weights(m):
    sd = m.state_dict()
    return sd['rnn.weight_ih_l0'][:, 0].double(), sd['rnn.weight_hh_l0'].double(), sd['linear.weight'].double()


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


out = []
gen = th.Generator().manual_seed(0)
for name, T, fn in [('hand2', 2, handcrafted.handcrafted_2nd_argmax_2), ('hand3', 3, handcrafted.handcrafted_2nd_argmax_3),
                    ('hand10', 10, handcrafted.handcrafted_2nd_argmax_10)]:
    m = fn(); r = dict(model=name, h=m.rnn.hidden_size, T=T, params=sum(p.numel() for p in m.parameters()), **measure(weights(m), T, gen))
    out.append(r); print(json.dumps(r), flush=True)
for T in [2, 3, 4, 5, 6, 8, 10]:
    for h in [2, 3, 4, 5, 6, 8, 16, 32]:
        m = zoo_2nd_argmax(h, T)
        r = dict(model='zoo', h=h, T=T, params=sum(p.numel() for p in m.parameters()), **measure(weights(m), T, gen))
        out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open('survey/survey.json', 'w'), indent=1)
