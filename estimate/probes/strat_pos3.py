"""F3 (PLAN.md): a third refinement level, pos(M,S,R), under the B5 strata, and the shrinkage prior.

Strata hierarchy on the first nine inputs: L0 = gap cell x sign (98), L1 = L0 x pos(M,S) (/72),
L2 = L1 x pos(R) (/504). A cell is a stratum at the finest level with N * mass >= K. Masses from
the input law as in B5. Values = exact x_9 integral (rb). Allocation = adaptive Neyman with
per-stage estimates and planned weights; the variance of a stratum is shrunk to its L1 parent,
that to its L0 parent, that to the global variance, each with prior weight PRIOR (B5: 4).
Combos: levels 2 (= B5) and 3, prior 4 and 16, and the oracle at both depths.
Run from estimate/: python probes/strat_pos3.py [reps] [models...]
"""
import sys, json, time
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from exact_last import cond_correct
from strat_adaptive import models_list, coarse
from best_unbiased import gap_cell_probs9

K = 32
BUDGETS = [1 << k for k in range(14, 21)]
PC = gap_cell_probs9()
NC = 98000
ids = np.arange(NC); pM, pS, pR = (ids % 1000) // 100, (ids % 100) // 10, ids % 10
valid = (pM < 9) & (pS < 9) & (pR < 9) & (pM != pS) & (pM != pR) & (pS != pR)
PF2 = np.where(valid, np.repeat(PC, 1000) / 504, 0)
L1 = ids // 10; L0 = ids // 1000                       # parents: L1 id in 0..9799, L0 in 0..97
PF1 = np.bincount(L1, PF2, 9800); PF0 = np.bincount(L0, PF2, 98)


def cells9_3(x):
    t = x[:, :9].topk(3, -1); vM, vS, vR = t.values.T; p = t.indices.numpy()
    den = vM.abs().clamp_min(1e-30); g1 = ((vM - vS) / den).numpy(); g2 = ((vS - vR) / den).numpy(); neg = (vS < 0).numpy()
    cid = (np.minimum(np.digitize(g1, coarse), 6) * 7 + np.minimum(np.digitize(g2, coarse), 6)) * 2 + neg
    return cid * 1000 + p[:, 0] * 100 + p[:, 1] * 10 + p[:, 2]


def make_pool(seed, n_log2):
    x = th.randn(1 << n_log2, 10, generator=th.Generator().manual_seed(seed))
    fid = cells9_3(x); order = np.argsort(fid, kind='stable'); starts = np.searchsorted(fid[order], np.arange(NC + 1))
    return x, fid, order, starts


def simulate(val, order, starts, scheme, levels, prior, true_var=None, rng=None):
    n = np.zeros(NC, int); e = np.zeros(NC); e2 = np.zeros(NC); avail = starts[1:] - starts[:-1]
    out = []; stage_est = []; stage_M = []; prevN = 0
    for N in BUDGETS:
        M = N - prevN; prevN = N
        f2 = (levels >= 3) & (N * PF2 >= K) & valid; f1 = (N * PF1[L1] >= K) & valid
        sid = np.where(f2, ids, np.where(f1, 100000 + L1, 200000 + L0))
        S, inv = np.unique(sid, return_inverse=True); Ms = len(S)
        ps = np.bincount(inv, PF2, Ms); ns = np.bincount(inv, n, Ms); es = np.bincount(inv, e, Ms); e2s = np.bincount(inv, e2, Ms)
        if scheme == 'prop':
            sig = np.ones(Ms)
        elif scheme == 'oracle':
            sig = np.sqrt(np.bincount(inv, true_var * PF2, Ms) / np.maximum(ps, 1e-300))
        else:
            def ss(nn, ee, ee2): return ee2 - np.where(nn > 0, ee ** 2 / np.maximum(nn, 1), 0)
            vg = ss(n.sum(), e.sum(), e2.sum()) / max(n.sum(), 1) if n.sum() > 0 else 0.05
            n0, e0, e20 = (np.bincount(L0, a, 98) for a in (n, e, e2)); v0 = (ss(n0, e0, e20) + prior * vg) / (n0 + prior)
            n1, e1, e21 = (np.bincount(L1, a, 9800) for a in (n, e, e2)); v1 = (ss(n1, e1, e21) + prior * v0[np.arange(9800) // 100]) / (n1 + prior)
            vprior = np.bincount(inv, v1[L1] * PF2, Ms) / np.maximum(ps, 1e-300)
            sig = np.sqrt((ss(ns, es, e2s) + prior * vprior) / (ns + prior))
        w = ps * sig; target = M * (0.9 * w / max(w.sum(), 1e-300) + 0.1 * ps) if scheme != 'prop' else M * ps
        target = np.where(ps > 0, np.maximum(np.ceil(target), 1), 0).astype(int)
        grp = np.argsort(inv, kind='stable'); bounds = np.searchsorted(inv[grp], np.arange(Ms + 1))
        n_st = np.zeros(Ms); e_st = np.zeros(Ms)
        for j in np.flatnonzero(target):
            k = target[j]; members = grp[bounds[j]:bounds[j + 1]]; members = members[PF2[members] > 0]
            room = avail[members] - n[members]; k = min(k, room.sum())
            if k <= 0:
                continue
            cnts = np.array([k]) if len(members) == 1 else rng.multinomial(k, PF2[members] / PF2[members].sum())
            over = cnts - room
            while (over > 0).any():
                short = over[over > 0].sum(); cnts = np.minimum(cnts, room); wfree = PF2[members] * (room - cnts > 0)
                cnts = cnts + rng.multinomial(short, wfree / wfree.sum()); over = cnts - room
            for f, c in zip(members, cnts):
                if c <= 0:
                    continue
                v = val[order[starts[f] + n[f]: starts[f] + n[f] + c]]
                n[f] += c; e[f] += v.sum(); e2[f] += (v * v).sum(); n_st[j] += c; e_st[j] += v.sum()
        stage_est.append((ps * np.where(n_st > 0, e_st / np.maximum(n_st, 1), 0)).sum()); stage_M.append(n_st.sum())
        tot = sum(stage_M)
        out.append((sum(m * v for m, v in zip(stage_M, stage_est)) / tot, int(tot), int((f2 & (n > 0)).sum() > 0 and (sid < 100000).sum())))
    return out


def boot(err2, c, B=2000, seed=0):
    """16-84% interval of gain = c / mean(err2) under rep resampling; err2 (reps,) or (reps, 2) for a paired ratio."""
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(err2), (B, len(err2)))
    if err2.ndim == 1:
        g = c / err2[idx].mean(1)
    else:
        m = err2[idx].mean(1); g = m[:, 0] / m[:, 1]
    return np.percentile(g, [16, 84])


def rb(x, W):
    with th.no_grad():
        return 1 - cond_correct(x[:, :9], *W).numpy()


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    names = sys.argv[2:] or ['trained']
    ms = models_list(names); pool_log2 = 22
    xo, fo, _, _ = make_pool(99995, 22); cnt = np.bincount(fo, minlength=NC); tv = {}
    for name, W, _ in ms:
        vo = rb(xo, W); tv[name] = np.bincount(fo, vo ** 2, minlength=NC) / np.maximum(cnt, 1) - (np.bincount(fo, vo, minlength=NC) / np.maximum(cnt, 1)) ** 2
    combos = [('adaptive', 2, 4), ('adaptive', 2, 16), ('adaptive', 3, 4), ('adaptive', 3, 16), ('oracle', 2, 4), ('oracle', 3, 4)]
    res = {n: {c: [[] for _ in BUDGETS] for c in combos} for n, _, _ in ms}; t0 = time.time()
    for rep in range(reps):
        x, fid, order, starts = make_pool(4000 + rep, pool_log2); rng = np.random.default_rng(rep)
        for name, W, truth in ms:
            v = rb(x, W)
            for c in combos:
                for b, q in enumerate(simulate(v, order, starts, c[0], c[1], c[2], tv[name], rng)):
                    res[name][c][b].append(q)
        print(f"rep {rep} done {time.time() - t0:.0f}s", flush=True)
    print(f"\nreps {reps}; pool 2^{pool_log2}; K {K}; levels 2 = B5 strata (gap cell x sign x pos(M,S)), 3 = + pos(R); prior = shrinkage weight; rb values")
    print("gain = p(1-p) / (N 1.15) / MSE vs truth [16-84% bootstrap over reps]; z = mean bias / se; cells = L2 strata in use")
    summary = {}
    for name, W, truth in ms:
        e_true = 1 - truth; print(f"\n{name}  err {e_true:.4f}")
        print("budget   " + "  ".join(f"{c[0]}/L{c[1]}/p{c[2]:<2d}".rjust(38) for c in combos) + "   gain ratios [16-84% bootstrap]"); summary[name] = {}
        for b, N in enumerate(BUDGETS):
            row = [f"2^{int(np.log2(N)):2d}    "]; summary[name][N] = {}
            for c in combos:
                a = np.array(res[name][c][b]); est, used, nc = a[:, 0], a[:, 1], a[:, 2]
                mse = ((est - e_true) ** 2).mean(); z = (est.mean() - e_true) / (est.std(ddof=1) / np.sqrt(len(est)))
                cst = e_true * (1 - e_true) / (used.mean() * 1.15); gain = cst / mse; lo, hi = boot((est - e_true) ** 2, cst)
                row.append(f"{gain:6.2f}x [{lo:4.1f},{hi:4.1f}] rms {np.sqrt(mse):.1e} z{z:+4.1f} c{int(nc.mean()):5d}")
                summary[name][N]['/'.join(map(str, c))] = dict(gain=gain, ci=[float(lo), float(hi)], rms=float(np.sqrt(mse)), z=float(z), used=float(used.mean()), cells=float(nc.mean()), est=est.tolist(), used_reps=used.tolist())
            e2 = {c: (np.array(res[name][c][b])[:, 0] - e_true) ** 2 for c in combos}
            for a, bb, lab in ((('adaptive', 3, 4), ('adaptive', 2, 4), 'L3/L2 p4'), (('adaptive', 2, 16), ('adaptive', 2, 4), 'p16/p4 L2'), (('oracle', 3, 4), ('oracle', 2, 4), 'oracle L3/L2')):
                lo, hi = boot(np.stack([e2[bb], e2[a]], 1), 1.0); summary[name][N][lab] = dict(ratio=float(e2[bb].mean() / e2[a].mean()), ci=[float(lo), float(hi)])
                row.append(f"{lab} {e2[bb].mean() / e2[a].mean():4.2f} [{lo:4.2f},{hi:4.2f}]")
            print("  ".join(row))
    json.dump(summary, open('probes/strat_pos3.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
