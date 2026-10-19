"""A4 (PLAN.md): the copy-harness estimator that uses the trained model as advice.
Common random numbers: p(W') = p0 + E[y' - y0] exactly, with y0 the trained model's
correctness on the same input (p0 known, y0 and the trained signed logit margin m0 come from
a pool evaluated once for all copies). Only the difference d = y' - y0 in {-1, 0, 1} is
sampled; it is nonzero only on flips (8.4% of inputs), which concentrate at small |m0|.
Strata = |m0| bins; masses from the pool (advice, free); allocation proportional or adaptive
Neyman from the samples so far (per-stage estimates, planned weights, as strat_adaptive.py).
The quadratic form c Q of A1 is not needed: the CRN identity already carries p0.
Cost per sample: one copy pass (the trained pass is amortised over copies); the 2-pass cost
is also reported. Run from estimate/: python probes/wp_crn.py [reps]
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np, torch as th
from ground_truth import forward, load_weights

P0 = 0.9531186
EDGES = np.array([0, .1, .3, 1, 3, np.inf]); NS = len(EDGES) - 1
BUDGETS = [1 << k for k in range(14, 19)]
W0 = [th.tensor(a) for a in load_weights('weights.npz')]
d = np.load('noisy_sigma0.005.npz'); copies = [([th.tensor(d[k][i]) for k in ('W_ih', 'W_hh', 'W_out')], float(d['acc'][i])) for i in range(32)]


def pool(seed, n_log2=20):
    x = th.randn(1 << n_log2, 10, generator=th.Generator().manual_seed(seed)); lab = x.topk(2, -1).indices[:, 1]
    with th.no_grad():
        l0 = forward(x, *W0); y0 = (l0.argmax(-1) == lab).numpy().astype(float)
        o = l0.clone(); o[th.arange(len(x)), lab] = -th.inf; m0 = (l0[th.arange(len(x)), lab] - o.max(-1).values).abs().numpy()
    s = np.minimum(np.digitize(m0, EDGES) - 1, NS - 1)
    order = np.argsort(s, kind='stable'); starts = np.searchsorted(s[order], np.arange(NS + 1))
    return x, lab, y0, s, order, starts


def simulate(dv, order, starts, ps, scheme):
    n = np.zeros(NS, int); e = np.zeros(NS); e2 = np.zeros(NS); est, Ms = [], []; prev = 0; out = []
    for N in BUDGETS:
        M = N - prev; prev = N
        if scheme == 'prop':
            sig = np.ones(NS)
        else:
            vg = (e2.sum() - e.sum() ** 2 / max(n.sum(), 1)) / max(n.sum(), 1) if n.sum() else 0.1
            sig = np.sqrt((e2 - np.where(n > 0, e ** 2 / np.maximum(n, 1), 0) + 4 * vg) / (n + 4))
        w = ps * sig; tgt = M * (0.9 * w / w.sum() + 0.1 * ps) if scheme != 'prop' else M * ps
        tgt = np.maximum(np.ceil(tgt), 1).astype(int); ns = np.zeros(NS); es = np.zeros(NS)
        for s in range(NS):
            c = min(tgt[s], starts[s + 1] - starts[s] - n[s]); v = dv[order[starts[s] + n[s]: starts[s] + n[s] + c]]
            n[s] += c; e[s] += v.sum(); e2[s] += (v * v).sum(); ns[s] = c; es[s] = v.sum()
        est.append((ps * es / np.maximum(ns, 1)).sum()); Ms.append(ns.sum())
        out.append((sum(m * v for m, v in zip(Ms, est)) / sum(Ms), int(sum(Ms))))
    return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    res = {i: {sch: [[] for _ in BUDGETS] for sch in ('mc', 'prop', 'adaptive')} for i in range(32)}
    for rep in range(reps):
        x, lab, y0, s, order, starts = pool(4000 + rep); ps = np.bincount(s, minlength=NS) / len(s); rng = np.random.default_rng(rep)
        for i, (W, acc) in enumerate(copies):
            with th.no_grad():
                y1 = (forward(x, *W).argmax(-1) == lab).numpy().astype(float)
            dv = y1 - y0; perm = rng.permutation(len(x))
            for b, N in enumerate(BUDGETS):
                res[i]['mc'][b].append((y1[perm[:N]].mean() - acc, N))
            for sch in ('prop', 'adaptive'):
                for b, (v, used) in enumerate(simulate(dv, order, starts, ps, sch)):
                    res[i][sch][b].append((P0 + v - acc, used))
        print(f"rep {rep}: flip rate of the last copy {np.abs(dv).mean():.4f}; |m0| bin masses " + " ".join(f"{p:.3f}" for p in ps) + f"; flip rate by bin " + " ".join(f"{np.abs(dv)[s == k].mean():.3f}" for k in range(NS)), flush=True)

    print("\ngain = MC MSE / ours at equal copy passes (1 pass/sample; 2-pass cost in brackets), geometric mean over 32 copies;")
    print("rms = root mean square error over copies x reps; z = bias of the copy-averaged estimate over reps (copies share each rep's pool, so they are not independent)")
    print("budget    MC rms         prop                               adaptive")
    summary = {}
    for b, N in enumerate(BUDGETS):
        mc = np.array([[e for e, _ in res[i]['mc'][b]] for i in range(32)])
        row = {'mc_rms': float(np.sqrt((mc ** 2).mean()))}
        for sch in ('prop', 'adaptive'):
            a = np.array([[e for e, _ in res[i][sch][b]] for i in range(32)]); used = np.mean([u for i in range(32) for _, u in res[i][sch][b]])
            g = (mc ** 2).mean(1) * N / ((a ** 2).mean(1) * used); per_rep = a.mean(0)
            row[sch] = dict(gain=float(np.exp(np.mean(np.log(g)))), rms=float(np.sqrt((a ** 2).mean())), z=float(per_rep.mean() / (per_rep.std(ddof=1) / np.sqrt(len(per_rep)))))
        summary[N] = row
        print(f"2^{int(np.log2(N))}    {row['mc_rms']:.1e}   " + "   ".join(f"{row[s]['gain']:5.2f}x ({row[s]['gain'] / 2:4.2f}x) rms {row[s]['rms']:.1e} z{row[s]['z']:+.1f}" for s in ('prop', 'adaptive')))
    json.dump({str(k): v for k, v in summary.items()}, open('probes/wp_crn.json', 'w'), indent=1)
