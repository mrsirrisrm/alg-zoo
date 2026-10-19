"""Linear pieces vs decision crossings along great circles, across zoo sizes (H, T).

Per model (best-of-5 seed): accuracy on the circle points, activation pieces per circle,
crossings of the net's decision (predicted label changes) and of correctness (flips), and
the full-circle / best-arc efficiency vs MC under exact tracing ((pieces+1)*2 passes).
Run from estimate/.
"""
import sys, numpy as np, torch as th
from alg_zoo import zoo_2nd_argmax
th.manual_seed(0)

def stats(W_ih, W_hh, W_out, T, nc=256, K=4096):
    g1, g2 = th.randn(nc, 1, T), th.randn(nc, 1, T)
    t = th.linspace(0, 2*np.pi, K+1)[:-1].view(1, K, 1)
    x = (th.cos(t)*g1 + th.sin(t)*g2).reshape(-1, T)
    h = th.relu(x[:, :1]*W_ih); pats = [h > 0]
    for s in range(1, T):
        h = th.relu(x[:, s:s+1]*W_ih + h @ W_hh.T); pats.append(h > 0)
    pred = (h @ W_out.T).argmax(-1).view(nc, K)
    ok = (pred.flatten() == x.topk(2, -1).indices[:, 1]).float().view(nc, K)
    pat = th.cat(pats, 1).view(nc, K, -1)
    kink = (pat != pat.roll(1, 1)).any(-1).float()
    pieces = kink.sum(1).mean().item()
    dec = (pred != pred.roll(1, 1)).float().sum(1).mean().item()
    flips = (ok != ok.roll(1, 1)).float().sum(1).mean().item()
    p = ok.mean().item(); vb = p*(1-p)
    best = 0.
    for m in [1, 4, 16, 64, 256]:
        L = K//m
        f = ok.view(nc, m, L).mean(-1).flatten()
        c = 2*(kink.view(nc, m, L)[..., 1:].sum(-1).mean().item() + 1)
        v = f.var().item()
        if v > 0: best = max(best, vb/v/c)
    return p, pieces, dec, flips, best

print(f"{'H':>3} {'T':>3} {'acc':>6} {'pieces':>7} {'dec-x':>6} {'err-flips':>9} {'pieces/flip':>11} {'best eff':>8}")
for T in [3, 5, 10]:
    for H in [2, 4, 8, 16, 32]:
        try:
            sd = zoo_2nd_argmax(H, T).state_dict()
        except Exception as e:
            print(H, T, 'missing', e); continue
        W = [sd['rnn.weight_ih_l0'][:, 0].float(), sd['rnn.weight_hh_l0'].float(), sd['linear.weight'].float()]
        with th.no_grad():
            p, pc, dc, fl, be = stats(*W, T)
        print(f"{H:3d} {T:3d} {p:6.3f} {pc:7.1f} {dc:6.1f} {fl:9.1f} {pc/max(fl,1e-9):11.1f} {be:8.2f}", flush=True)
