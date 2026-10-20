"""Control-flow view of the ramp: how many execution paths (full ReLU on/off patterns over all
T steps) the model takes, and how many of them carry its errors, across hidden size and training.

Per model, on N iid N(0,1) inputs: distinct paths seen, Good-Turing coverage (1 - singletons/N),
Chao-Shen entropy of the path distribution over all inputs and over the errors only, the number
of paths that cover 90% of the errors (a lower bound when error coverage < 0.9), and whether a
path decides correctness: over paths seen >= 8 times, the error-weighted within-path error
rate (1 = error paths are all-error regions, small = errors are slivers inside mostly-correct
paths) and the share of the errors those paths hold.
Prediction (from the decision-kink round): total path entropy falls then rises with training,
while the error-restricted path count keeps rising.
Paths are hashed to int64 (collision odds ~1e-6 at N = 2^22). Run from estimate/.
"""
import numpy as np, torch as th
from alg_zoo import zoo_2nd_argmax
from alg_zoo.zoo import best_of_5_seed_2nd_argmax
T, N, CH = 10, 2**22, 2**19

def weights(H, seed, n_seqs):
    sd = zoo_2nd_argmax(H, T, seed=seed, n_seqs=n_seqs).state_dict()
    return sd['rnn.weight_ih_l0'][:, 0].float(), sd['rnn.weight_hh_l0'].float(), sd['linear.weight'].float()

def run(W_ih, W_hh, W_out):
    H = W_ih.shape[0]
    gen = th.Generator().manual_seed(7)
    R = th.randint(-2**62, 2**62, (T, H), generator=th.Generator().manual_seed(11), dtype=th.int64)
    keys, oks = [], []
    for _ in range(N // CH):
        x = th.randn(CH, T, generator=gen)
        h = th.relu(x[:, :1]*W_ih); key = ((h > 0).long()*R[0]).sum(1)
        for s in range(1, T):
            h = th.relu(x[:, s:s+1]*W_ih + h @ W_hh.T); key += ((h > 0).long()*R[s]).sum(1)
        keys.append(key); oks.append((h @ W_out.T).argmax(-1) == x.topk(2, -1).indices[:, 1])
    return th.cat(keys), th.cat(oks)

def chao_shen(c):
    c = c[c > 0].double(); n = c.sum(); f1 = (c == 1).sum().clamp(max=n - 1)
    p = (1 - f1/n)*c/n
    return (-(p*p.log2())/(1 - th.exp(n*th.log1p(-p)))).sum().item(), (1 - (c == 1).sum()/n).item()

def stats(key, ok):
    u, inv, cnt = th.unique(key, return_inverse=True, return_counts=True)
    ecnt = th.bincount(inv[~ok], minlength=len(u))
    E = ecnt.sum().item()
    Ha, Ca = chao_shen(cnt); He, Ce = chao_shen(ecnt)
    cum = ecnt.sort(descending=True).values.cumsum(0).double()/max(E, 1)
    k90 = int((cum < 0.9).sum().item()) + 1
    big = cnt >= 8
    rate = ecnt[big].double()/cnt[big]
    pur = (ecnt[big]*rate).sum().item()/max(ecnt[big].sum().item(), 1)
    return dict(acc=ok.float().mean().item(), paths=len(u), Ca=Ca, Ha=Ha, He=He, Ce=Ce, k90=k90,
                lb=Ce < 0.9, pur=pur, held=ecnt[big].sum().item()/max(E, 1))

print(f"N = 2^{int(np.log2(N))} inputs per model, T = {T}, best seed; '>' marks k90 as a lower bound (error coverage < 0.9)")
print(f"{'H':>3} {'n_seqs':>6} {'acc':>6} {'paths':>8} {'cover':>6} {'H_all':>6} {'H_err':>6} {'err cover':>9} "
      f"{'paths for 90% err':>17} {'purity':>6} {'err held':>8}")
with th.no_grad():
    for H in [4, 8, 16, 32]:
        seed = best_of_5_seed_2nd_argmax(hidden_size=H, seq_len=T)
        for n in ([2**30] if H == 4 else [0] + [2**i for i in range(20, 31, 2)]):
            s = stats(*run(*weights(H, seed, n)))
            k = ('>' if s['lb'] else ' ') + str(s['k90'])
            print(f"{H:3d} {'0' if n == 0 else '2^%d' % int(np.log2(n)):>6} {s['acc']:6.3f} {s['paths']:8d} {s['Ca']:6.3f} "
                  f"{s['Ha']:6.1f} {s['He']:6.1f} {s['Ce']:9.3f} {k:>17} {s['pur']:6.2f} {s['held']:8.2f}", flush=True)
