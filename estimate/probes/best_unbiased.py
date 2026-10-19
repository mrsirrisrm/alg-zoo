"""B5 (PLAN.md): the stacked unbiased estimator.

  strata on the FIRST NINE inputs (gap cells of their top three x sign x positions pM, pS),
  + exact integration of x_9 (probes/exact_last.py: value = P(wrong | x_<9), cost 1.15 passes),
  + adaptive Neyman allocation with per-stage estimates and planned weights (strat_adaptive.py).

Strata must be functions of x_<9 so the x_9 integral is exact inside each stratum; masses are
known from the input law (gap-cell probabilities of 9 iid Gaussians from a 2^28 draw, / 72
position pairs by exchangeability). Values are continuous, so Neyman uses per-stratum sample
variances (shrunk to the coarse parent, then to the global variance).
Compared on the same pools: binary y with the same strata (isolates the x_9 integral), and
the oracle allocation (true per-cell variances from an independent pool).
Run from estimate/: python probes/best_unbiased.py [reps] [models...]
"""
import sys, json, os, time
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import forward, load_weights
from exact_last import cond_correct
from strat_adaptive import models_list, coarse

K = 32
BUDGETS = [1 << k for k in range(14, 20)]
COST = {'bin': 1.0, 'rb': 1.15}


def cells9(x):
    t = x[:, :9].topk(3, -1); vM, vS, vR = t.values.T; pM, pS = t.indices[:, 0].numpy(), t.indices[:, 1].numpy()
    den = vM.abs().clamp_min(1e-30); g1 = ((vM - vS) / den).numpy(); g2 = ((vS - vR) / den).numpy(); neg = (vS < 0).numpy()
    cid = (np.minimum(np.digitize(g1, coarse), 6) * 7 + np.minimum(np.digitize(g2, coarse), 6)) * 2 + neg
    return cid, cid * 100 + pM * 10 + pS


def gap_cell_probs9(path='probes/gap_cells9.json', n_log2=28, chunk_log2=22):
    if os.path.exists(path):
        return np.array(json.load(open(path))['p'])
    cnt = np.zeros(98); g = th.Generator().manual_seed(778)
    for _ in range(1 << (n_log2 - chunk_log2)):
        cid, _ = cells9(th.randn(1 << chunk_log2, 10, generator=g)); cnt += np.bincount(cid, minlength=98)
    p = cnt / cnt.sum(); json.dump(dict(n_log2=n_log2, p=p.tolist()), open(path, 'w')); return p


PC = gap_cell_probs9()
valid = np.array([(i % 100) // 10 < 9 and (i % 100) % 10 < 9 and (i % 100) // 10 != (i % 100) % 10 for i in range(9800)])
PF = np.repeat(PC, 100) / 72; PF[~valid] = 0
PAR = np.arange(9800) // 100


def make_pool(seed, n_log2):
    x = th.randn(1 << n_log2, 10, generator=th.Generator().manual_seed(seed))
    _, fid = cells9(x)
    order = np.argsort(fid, kind='stable'); starts = np.searchsorted(fid[order], np.arange(9801))
    return x, fid, order, starts


def simulate(val, order, starts, scheme, true_var=None, rng=None):
    n = np.zeros(9800, int); e = np.zeros(9800); e2 = np.zeros(9800)
    avail = starts[1:] - starts[:-1]
    out = []; stage_est = []; stage_M = []; prevN = 0
    for N in BUDGETS:
        M = N - prevN; prevN = N
        fine = (N * PF >= K) & valid
        sid = np.where(fine, np.arange(9800), 10000 + PAR)
        S, inv = np.unique(sid, return_inverse=True); Ms = len(S)
        ps = np.bincount(inv, PF, Ms); ns = np.bincount(inv, n, Ms); es = np.bincount(inv, e, Ms); e2s = np.bincount(inv, e2, Ms)
        if scheme == 'prop':
            sig = np.ones(Ms)
        elif scheme == 'oracle':
            sig = np.sqrt(np.bincount(inv, true_var * PF, Ms) / np.maximum(ps, 1e-300))
        else:
            SSg = e2.sum() - (e.sum() ** 2 / n.sum() if n.sum() > 0 else 0); vg = SSg / max(n.sum(), 1) if n.sum() > 0 else 0.05
            npar = np.bincount(PAR, n, 98); epar = np.bincount(PAR, e, 98); e2par = np.bincount(PAR, e2, 98)
            SSpar = e2par - np.where(npar > 0, epar ** 2 / np.maximum(npar, 1), 0); vpar = (SSpar + 4 * vg) / (npar + 4)
            SSs = e2s - np.where(ns > 0, es ** 2 / np.maximum(ns, 1), 0)
            vprior = np.bincount(inv, vpar[PAR] * PF, Ms) / np.maximum(ps, 1e-300)
            sig = np.sqrt((SSs + 4 * vprior) / (ns + 4))
        w = ps * sig; target = M * (0.9 * w / max(w.sum(), 1e-300) + 0.1 * ps) if scheme != 'prop' else M * ps
        target = np.where(ps > 0, np.maximum(np.ceil(target), 1), 0).astype(int)
        grp = np.argsort(inv, kind='stable'); bounds = np.searchsorted(inv[grp], np.arange(Ms + 1))
        n_st = np.zeros(Ms); e_st = np.zeros(Ms)
        for j in np.flatnonzero(target):
            k = target[j]; members = grp[bounds[j]:bounds[j + 1]]
            room = avail[members] - n[members]; k = min(k, room.sum())
            if k <= 0:
                continue
            cnts = np.array([k]) if len(members) == 1 else rng.multinomial(k, PF[members] / PF[members].sum())
            over = cnts - room
            while (over > 0).any():  # a member cell exhausted in the pool: redraw the shortfall among members with room (not skip it)
                short = over[over > 0].sum(); cnts = np.minimum(cnts, room); wfree = PF[members] * (room - cnts > 0)
                cnts = cnts + rng.multinomial(short, wfree / wfree.sum()); over = cnts - room
            for f, c in zip(members, cnts):
                if c <= 0:
                    continue
                v = val[order[starts[f] + n[f]: starts[f] + n[f] + c]]
                n[f] += c; e[f] += v.sum(); e2[f] += (v * v).sum(); n_st[j] += c; e_st[j] += v.sum()
        stage_est.append((ps * np.where(n_st > 0, e_st / np.maximum(n_st, 1), 0)).sum()); stage_M.append(n_st.sum())
        tot = sum(stage_M)
        out.append((sum(m * v for m, v in zip(stage_M, stage_est)) / tot, int(tot), int(((n_st == 0) & (ps > 0)).sum())))
    return out


def values(x, W):
    with th.no_grad():
        rb = 1 - cond_correct(x[:, :9], *W).numpy()
        bn = (forward(x, *W).argmax(-1) != x.topk(2, -1).indices[:, 1]).numpy().astype(float)
    return {'bin': bn, 'rb': rb}


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    names = sys.argv[2:] or ['trained', 'noisy0', 'noisy1', 'seed0', 'seed1', 'seed2']
    ms = models_list(names); pool_log2 = 22
    xo, fo, _, _ = make_pool(99993, 22)
    combos = [('prop', 'bin'), ('adaptive', 'bin'), ('prop', 'rb'), ('adaptive', 'rb'), ('oracle', 'rb')]
    res = {n: {c: [[] for _ in BUDGETS] for c in combos} for n, _, _ in ms}
    t0 = time.time()
    for rep in range(reps):
        x, fid, order, starts = make_pool(2000 + rep, pool_log2); rng = np.random.default_rng(rep)
        for name, W, truth in ms:
            vo = values(xo, W); v = values(x, W)
            cnt = np.bincount(fo, minlength=9800)
            tv = {k: np.bincount(fo, vo[k] ** 2, minlength=9800) / np.maximum(cnt, 1) - (np.bincount(fo, vo[k], minlength=9800) / np.maximum(cnt, 1)) ** 2 for k in vo}
            for scheme, kind in combos:
                r = simulate(v[kind], order, starts, scheme, tv[kind], rng)
                for b, q in enumerate(r):
                    res[name][(scheme, kind)][b].append(q)
        print(f"rep {rep} done {time.time() - t0:.0f}s", flush=True)
    print(f"\nreps {reps}; pool 2^{pool_log2}; K {K}; strata = gap cells of the first 9 inputs' top-3 (6x6) x sign x pos(M,S); rb = exact x_9 integral at cost 1.15")
    print("gain = p(1-p) / (N cost) / MSE vs truth; z = mean bias / se")
    summary = {}
    for name, W, truth in ms:
        e_true = 1 - truth; print(f"\n{name}  err {e_true:.4f}")
        print("budget   " + "  ".join(f"{s + '/' + k:>24s}" for s, k in combos)); summary[name] = {}
        for b, N in enumerate(BUDGETS):
            row = [f"2^{int(np.log2(N)):2d}    "]; summary[name][N] = {}
            for c in combos:
                a = np.array(res[name][c][b]); est, used = a[:, 0], a[:, 1]
                mse = ((est - e_true) ** 2).mean(); z = (est.mean() - e_true) / (est.std(ddof=1) / np.sqrt(len(est)))
                gain = e_true * (1 - e_true) / (used.mean() * COST[c[1]]) / mse
                row.append(f"{gain:6.2f}x rms {np.sqrt(mse):.1e} z{z:+4.1f}  ")
                summary[name][N]['/'.join(c)] = dict(gain=gain, rms=float(np.sqrt(mse)), z=float(z), used=float(used.mean()))
            print("  ".join(row))
    json.dump(summary, open('probes/best_unbiased.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
