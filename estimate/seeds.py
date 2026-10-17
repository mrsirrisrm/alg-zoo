"""Zoo h16/T10 2nd-argmax models, seeds 0-4 (ARC's preferred robustness test), in the
weights.npz layout; MC accuracy at 2^N samples each."""
import json, sys
import numpy as np, torch as th
from alg_zoo import zoo_2nd_argmax
from ground_truth import forward

n_log2 = int(sys.argv[1]) if len(sys.argv) > 1 else 26
B = 1 << 22
Ws, accs = [], []
for seed in range(5):
    sd = zoo_2nd_argmax(16, 10, seed=seed).state_dict()
    W = [sd['rnn.weight_ih_l0'][:, 0].double().numpy(), sd['rnn.weight_hh_l0'].double().numpy(), sd['linear.weight'].double().numpy()]
    Wt = [th.tensor(a, dtype=th.float32) for a in W]
    gen = th.Generator().manual_seed(2000 + seed); c = 0
    with th.no_grad():
        for _ in range((1 << n_log2) // B):
            x = th.randn(B, 10, generator=gen)
            c += (forward(x, *Wt).argmax(-1) == x.topk(2, -1).indices[:, 1]).sum().item()
    acc = c / (1 << n_log2); Ws.append(W); accs.append(acc)
    print(f'seed {seed}: acc {acc:.5f}', flush=True)
np.savez('seeds_h16.npz', W_ih=np.stack([w[0] for w in Ws]), W_hh=np.stack([w[1] for w in Ws]),
         W_out=np.stack([w[2] for w in Ws]), acc=np.array(accs), n_log2=n_log2)
