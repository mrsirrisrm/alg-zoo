"""Kinks at the decision step vs error, across hidden size and training progress (CRITIQUE conjecture).

Conjecture: in a capacity-limited model the error set is where the piecewise-linear comparator is
coarse. Test: if kink density at the decision step tracks hidden size (and training) while the
error shrinks, the 'explanation is not a cheaper estimate' gap belongs to the model class.

Along great circles (as in pieces_vs_size.py), per model: accuracy, activation kinks per circle
in total and at the last step only, error flips per circle, last-step kinks per flip, and the
share of error flips with a last-step kink within +-w points, against the share expected if
kinks fell independently of flips (enrichment = local / chance).
Axis 1: T = 10, H in {4, 8, 16, 32}, seeds 0-4, fully trained.
Axis 2: T = 10, H in {8, 16}, best seed, n_seqs in {0, 2^20, 2^22, ..., 2^30}.
Run from estimate/.
"""
import numpy as np, torch as th
from alg_zoo import zoo_2nd_argmax
from alg_zoo.zoo import best_of_5_seed_2nd_argmax
th.manual_seed(0)
T, w = 10, 4

def stats(W_ih, W_hh, W_out, nc=256, K=4096):
    g1, g2 = th.randn(nc, 1, T), th.randn(nc, 1, T)
    t = th.linspace(0, 2*np.pi, K+1)[:-1].view(1, K, 1)
    x = (th.cos(t)*g1 + th.sin(t)*g2).reshape(-1, T)
    h = th.relu(x[:, :1]*W_ih); pats = [h > 0]
    for s in range(1, T):
        h = th.relu(x[:, s:s+1]*W_ih + h @ W_hh.T); pats.append(h > 0)
    pred = (h @ W_out.T).argmax(-1).view(nc, K)
    ok = (pred.flatten() == x.topk(2, -1).indices[:, 1]).float().view(nc, K)
    pat = th.cat(pats, 1).view(nc, K, -1)
    kink = (pat != pat.roll(1, 1)).any(-1)
    last = pats[-1].view(nc, K, -1)
    lkink = (last != last.roll(1, 1)).any(-1)
    flip = ok != ok.roll(1, 1)
    near = th.zeros_like(lkink)
    for d in range(-w, w+1):
        near |= lkink.roll(d, 1)
    dens = lkink.float().mean().item()
    local = near[flip].float().mean().item() if flip.any() else float('nan')
    chance = 1 - (1 - dens)**(2*w + 1)
    return dict(acc=ok.mean().item(), pieces=kink.float().sum(1).mean().item(),
                last=lkink.float().sum(1).mean().item(), flips=flip.float().sum(1).mean().item(),
                local=local, enrich=local/chance)

def weights(H, seed=None, n_seqs=None):
    sd = zoo_2nd_argmax(H, T, seed=seed, n_seqs=n_seqs).state_dict()
    return sd['rnn.weight_ih_l0'][:, 0].float(), sd['rnn.weight_hh_l0'].float(), sd['linear.weight'].float()

hdr = f"{'acc':>6} {'pieces':>7} {'last-kinks':>10} {'err-flips':>9} {'last/flip':>9} {'flip near last-kink':>19} {'enrich':>6}"
def row(s):
    return (f"{s['acc']:6.3f} {s['pieces']:7.1f} {s['last']:10.1f} {s['flips']:9.1f} "
            f"{s['last']/max(s['flips'], 1e-9):9.2f} {s['local']:19.2f} {s['enrich']:6.2f}")

print(f"Axis 1: fully trained, T = {T}; window +-{w} of 4096 points per circle")
print(f"{'H':>3} {'seed':>4} " + hdr)
with th.no_grad():
    for H in [4, 8, 16, 32]:
        for seed in range(5):
            print(f"{H:3d} {seed:4d} " + row(stats(*weights(H, seed=seed))), flush=True)
    print(f"\nAxis 2: training progress, best seed, T = {T}")
    print(f"{'H':>3} {'n_seqs':>6} " + hdr)
    for H in [8, 16]:
        seed = best_of_5_seed_2nd_argmax(hidden_size=H, seq_len=T)
        for n in [0] + [2**i for i in range(20, 31, 2)]:
            print(f"{H:3d} {'0' if n == 0 else '2^%d' % int(np.log2(n)):>6} " + row(stats(*weights(H, seed=seed, n_seqs=n))), flush=True)
