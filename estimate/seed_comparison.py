"""Compare the example 432-param model with the zoo 2nd-argmax models
(hidden 16, seq 10, seeds 0-4; plus neighbouring sizes, best seed)."""
import numpy as np
import torch as th

from alg_zoo import example_2nd_argmax, zoo_2nd_argmax

th.manual_seed(0)
x = th.randn(1 << 21, 10)
y = x.argsort(-1)[:, -2]


def acc(m):
    with th.no_grad():
        return (m(x).argmax(-1) == y).float().mean().item()


ex = example_2nd_argmax()
print(f"example (432 params): {acc(ex):.4f}")
ws = {k: v for k, v in ex.state_dict().items()}
for seed in range(5):
    m = zoo_2nd_argmax(16, 10, seed=seed)
    same = all(th.equal(ws[k], v) for k, v in m.state_dict().items())
    print(f"zoo h16 T10 seed {seed}: {acc(m):.4f}{'  <- identical to example' if same else ''}")
for h in [8, 32]:
    for seed in range(5):
        print(f"zoo h{h} T10 seed {seed}: {acc(zoo_2nd_argmax(h, 10, seed=seed)):.4f}")
