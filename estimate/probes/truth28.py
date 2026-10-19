"""F1 (PLAN.md): 2^28-sample truths for the five B5 harness models (trained, noisy0-1, seed0-1),
so the z checks at 2^19-2^20 are not limited by the 2^26 seed truths (sd 3.5e-5).
sd of a 2^28 truth: sqrt(p(1-p)/2^28) ~ 1.3-2e-5. Run from estimate/: python probes/truth28.py
"""
import sys, json, time
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import forward
from strat_adaptive import models_list

names = sys.argv[1:] or ['trained', 'noisy0', 'noisy1', 'seed0', 'seed1']
out = {}; t0 = time.time()
for name, W, acc in models_list(names):
    g = th.Generator().manual_seed(2828); c = 0
    with th.no_grad():
        for _ in range(1 << 6):
            x = th.randn(1 << 22, 10, generator=g)
            c += (forward(x, *W).argmax(-1) == x.topk(2, -1).indices[:, 1]).sum().item()
    p = c / (1 << 28); sd = np.sqrt(p * (1 - p) / (1 << 28))
    out[name] = dict(acc28=p, sd=sd, acc_harness=acc, delta=p - acc)
    print(f"{name}: 2^28 acc {p:.6f} (sd {sd:.1e}); harness {acc:.6f}; delta {p - acc:+.1e} ({(p - acc) / sd:+.1f} sd of the 2^28 truth)  {time.time() - t0:.0f}s", flush=True)
json.dump(out, open('probes/truth28.json', 'w'), indent=1)
