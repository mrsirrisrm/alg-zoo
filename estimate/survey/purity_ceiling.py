"""Ceiling on verifier-style pruning (certify a node all-correct / all-wrong, stop splitting).
A BnB node at depth t = activation pattern of all neurons at steps <= t. Certification can only
stop a node whose correctness is constant on it. From samples: group by prefix pattern, call a
group pure if it has >= MINC samples and one outcome. Nodes a certifying BnB must still expand =
impure (or unresolved small) groups. Ceiling speedup = sum_t all nodes / sum_t nodes to expand.
Usage: python survey/purity_ceiling.py h T [h T ...]
"""
import sys
sys.path[:0] = ['.', 'survey']
import numpy as np, torch as th
from alg_zoo import zoo_2nd_argmax

MINC, N = 20, 1 << 21
args = list(map(int, sys.argv[1:]))
for h, T in zip(args[::2], args[1::2]):
    sd = zoo_2nd_argmax(h, T).state_dict()
    Wi, Wh, Wo = sd['rnn.weight_ih_l0'][:, 0].double(), sd['rnn.weight_hh_l0'].double(), sd['linear.weight'].double()
    x = th.randn(N, T, generator=th.Generator().manual_seed(0), dtype=th.float64)
    H = th.zeros(N, h, dtype=th.float64); bits = []
    for t in range(T):
        pre = x[:, t:t + 1] * Wi + H @ Wh.T; bits.append(pre > 0); H = pre.clamp_min(0)
    ok = ((H @ Wo.T).argmax(-1) == x.topk(2, -1).indices[:, 1]).numpy()
    print(f"h{h}T{T} acc {ok.mean():.4f}")
    tot_all = tot_exp = 0
    for t in range(T):
        B = th.cat(bits[:t + 1], 1).numpy()
        _, g, cnt = np.unique(np.packbits(B, 1), axis=0, return_inverse=True, return_counts=True)
        g = g.ravel(); nc = np.bincount(g, ok, len(cnt))
        pure = (cnt >= MINC) & ((nc == 0) | (nc == cnt))
        n99 = int(np.searchsorted(np.cumsum(np.sort(cnt)[::-1]), 0.99 * N) + 1)
        m_pure = cnt[pure].sum() / N
        # nodes to expand: groups (in 99% cover) not certified pure
        order = np.argsort(-cnt); cov = order[:n99]
        n_exp = int((~pure[cov]).sum())
        tot_all += n99; tot_exp += n_exp
        print(f"  depth t={t}: nodes99 {n99:7d}  pure mass {m_pure:.3f}  nodes still to expand {n_exp:7d}")
    print(f"  ceiling speedup (node count) {tot_all / max(tot_exp, 1):.2f}", flush=True)
