"""B1/B2 (PLAN.md): adaptive Neyman allocation with NO separate pilot.

Strata: coarse gap cells (g1, g2 in 6 bins each) x sign(vS) x positions (pM, pS).
Stratum mass is known from the input law alone: by exchangeability the gap-cell
probability is the same for every position pair, so p(fine) = p(gap cell, sign) / 90;
the 98 gap-cell probabilities come from a 2^28 draw (no network).

Sequential scheme: budgets double 2^14 .. 2^20. A cell is a fine stratum once
N * p(fine) >= K (else its coarse parent is the stratum). At each stage the per-stratum
error rate is estimated from ALL samples so far (shrunk toward the running mean), the
target cumulative allocation is Neyman with a 10% proportional floor (>= 1 sample), and
only the shortfall is drawn. Every sample counts in the estimate; nothing is thrown away.
Inputs are drawn iid (cheap, not network cost) into a pool and bucketed; drawing from a
stratum = taking the next unused pool member of that cell, so the sequential estimator
is simulated exactly from one evaluated pool per rep. Network cost = evaluated samples.

Schemes: proportional (same refinement), adaptive Neyman, oracle Neyman (true rates from
an independent 2^23 pool), adaptive Neyman on a scrambled-Sobol order-statistic pool (B2).
Scored over reps against the truth: rms, gain = p(1-p)/N / MSE, z of the mean bias.
Run from estimate/: python probes/strat_adaptive.py [reps] [models...]
"""
import sys, json, os, time
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from torch.quasirandom import SobolEngine
from ground_truth import forward, load_weights
from rqmc import orderstat_permfirst

P0 = 0.9531186
coarse = [0, .01, .03, .07, .15, .4, np.inf]      # 6 bins -> digitize gives 1..6
K = 32
BUDGETS = [1 << k for k in range(14, 21)]


def models_list(names):
    d, s = np.load('noisy_sigma0.005.npz'), np.load('seeds_h16.npz')
    out = {}
    out['trained'] = ([th.tensor(a) for a in load_weights('weights.npz')], P0)
    for i in range(32):
        out[f'noisy{i}'] = ([th.tensor(d[k][i]) for k in ('W_ih', 'W_hh', 'W_out')], float(d['acc'][i]))
    for i in range(5):
        out[f'seed{i}'] = ([th.tensor(s[k][i]) for k in ('W_ih', 'W_hh', 'W_out')], float(s['acc'][i]))
    for n in names:
        if n.startswith('ft:'):                                   # fine-tuned variant (PLAN G): path to an npz
            f = np.load(n[3:]); out[n] = ([th.tensor(f[k]) for k in ('W_ih', 'W_hh', 'W_out')], float(f['acc28'] if 'acc28' in f else f['acc']))
    return [(n, *out[n]) for n in names]


def cells(x):
    """fine cell id = coarse_id * 100 + pM * 10 + pS, coarse_id = (b1 * 7 + b2) * 2 + neg in 0..97"""
    t = x.topk(3, -1); vM, vS, vR = t.values.T; pM, pS = t.indices[:, 0].numpy(), t.indices[:, 1].numpy()
    den = vM.abs().clamp_min(1e-30); g1 = ((vM - vS) / den).numpy(); g2 = ((vS - vR) / den).numpy(); neg = (vS < 0).numpy()
    cid = (np.minimum(np.digitize(g1, coarse), 6) * 7 + np.minimum(np.digitize(g2, coarse), 6)) * 2 + neg
    return cid, cid * 100 + pM * 10 + pS, t.indices[:, 1]


def gap_cell_probs(path='probes/gap_cells.json', n_log2=28, chunk_log2=22):
    if os.path.exists(path):
        return np.array(json.load(open(path))['p'])
    cnt = np.zeros(98)
    g = th.Generator().manual_seed(777)
    for _ in range(1 << (n_log2 - chunk_log2)):
        cid, _, _ = cells(th.randn(1 << chunk_log2, 10, generator=g))
        cnt += np.bincount(cid, minlength=98)
    p = cnt / cnt.sum()
    json.dump(dict(n_log2=n_log2, p=p.tolist()), open(path, 'w'))
    return p


PC = gap_cell_probs()                      # coarse cell mass, 98
PF = np.repeat(PC, 100) / 90               # fine mass indexed by fine id (coarse*100 + pM*10 + pS); pM==pS cells never occur
valid_fine = np.array([(i % 100) // 10 != (i % 100) % 10 for i in range(9800)])
PF[~valid_fine] = 0


def make_pool(seed, n_log2, sobol):
    if sobol:
        u = SobolEngine(20, scramble=True, seed=seed).draw(1 << n_log2)
        x = orderstat_permfirst(u).float()
    else:
        x = th.randn(1 << n_log2, 10, generator=th.Generator().manual_seed(seed))
    cid, fid, lab = cells(x)
    order = np.argsort(fid, kind='stable')                     # pool members grouped by fine cell, pool order within
    starts = np.searchsorted(fid[order], np.arange(9801))
    return x, fid, lab, order, starts


def simulate(err, order, starts, scheme, true_rate=None, rng=None):
    """err: error indicator per pool member. Stage r draws M_r = N_r - N_{r-1} new samples,
    allocated from the rates seen in stages < r (hierarchical shrinkage to the coarse parent),
    and has its OWN stratified estimate; stages are combined with planned weights M_r / N_r,
    so the estimator is unbiased (no sample's value influences its own stratum's weight).
    Returns per cumulative budget (estimate of error rate, samples used, empty strata)."""
    n = np.zeros(9800, int); e = np.zeros(9800)                # cumulative samples / errors per fine cell (for allocation)
    avail = starts[1:] - starts[:-1]
    out = []; stage_est = []; stage_M = []; prevN = 0
    for N in BUDGETS:
        M = N - prevN; prevN = N
        fine = (N * PF >= K) & valid_fine
        sid = np.where(fine, np.arange(9800), 10000 + np.arange(9800) // 100)
        S, inv = np.unique(sid, return_inverse=True); Ms = len(S)
        ps = np.bincount(inv, PF, Ms); ns = np.bincount(inv, n, Ms); es = np.bincount(inv, e, Ms)
        if scheme == 'prop':
            sig = np.ones(Ms)
        elif scheme == 'oracle':
            r = np.bincount(inv, true_rate * PF, Ms) / np.maximum(ps, 1e-300)
            sig = np.sqrt(r * (1 - r))
        else:
            par = np.arange(9800) // 100
            npar = np.bincount(par, n, 98); epar = np.bincount(par, e, 98)
            rbar = e.sum() / max(n.sum(), 1) if n.sum() > 0 else 0.05
            rpar = (epar + 4 * rbar) / (npar + 4)                       # coarse-cell rate, shrunk to global
            rs = (es + 4 * np.bincount(inv, rpar[par] * PF, Ms) / np.maximum(ps, 1e-300)) / (ns + 4)
            sig = np.sqrt(rs * (1 - rs))
        w = ps * sig; target = M * (0.9 * w / max(w.sum(), 1e-300) + 0.1 * ps) if scheme != 'prop' else M * ps
        target = np.where(ps > 0, np.maximum(np.ceil(target), 1), 0).astype(int)
        grp = np.argsort(inv, kind='stable'); bounds = np.searchsorted(inv[grp], np.arange(Ms + 1))
        n_st = np.zeros(Ms); e_st = np.zeros(Ms)
        for j in np.flatnonzero(target):
            k = target[j]; members = grp[bounds[j]:bounds[j + 1]]
            cnts = [k] if len(members) == 1 else rng.multinomial(k, PF[members] / PF[members].sum())
            for f, c in zip(members, cnts):
                c = min(c, avail[f] - n[f])
                if c <= 0:
                    continue
                idx = order[starts[f] + n[f]: starts[f] + n[f] + c]
                ee = err[idx].sum(); n[f] += c; e[f] += ee; n_st[j] += c; e_st[j] += ee
        est_r = (ps * np.where(n_st > 0, e_st / np.maximum(n_st, 1), 0)).sum()
        stage_est.append(est_r); stage_M.append(n_st.sum())
        tot = sum(stage_M)
        est = sum(m * v for m, v in zip(stage_M, stage_est)) / tot
        out.append((est, int(tot), int(((n_st == 0) & (ps > 0)).sum())))
    return out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    names = sys.argv[2:] or ['trained', 'noisy0', 'noisy1', 'noisy2', 'noisy3', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4']
    ms = models_list(names)
    pool_log2 = 23
    # oracle rates from an independent pool
    xo, fo, lo, _, _ = make_pool(99991, 23, False)
    schemes = ['prop', 'adaptive', 'oracle', 'adaptive-sobol']
    res = {n: {s: [[] for _ in BUDGETS] for s in schemes} for n, _, _ in ms}
    t0 = time.time()
    for rep in range(reps):
        pools = {False: make_pool(1000 + rep, pool_log2, False), True: make_pool(1000 + rep, pool_log2, True)}
        rng = np.random.default_rng(rep)
        for name, W, truth in ms:
            with th.no_grad():
                tr = np.zeros(9800)
                eo = (forward(xo, *W).argmax(-1) != lo).numpy()
                cnt = np.bincount(fo, minlength=9800); tr = np.bincount(fo, eo, minlength=9800) / np.maximum(cnt, 1)
                for sobol in (False, True):
                    x, fid, lab, order, starts = pools[sobol]
                    err = (forward(x, *W).argmax(-1) != lab).numpy().astype(float)
                    for s in (['adaptive-sobol'] if sobol else ['prop', 'adaptive', 'oracle']):
                        r = simulate(err, order, starts, s.replace('-sobol', ''), tr, rng)
                        for b, v in enumerate(r):
                            res[name][s][b].append(v)
        print(f"rep {rep} done {time.time() - t0:.0f}s", flush=True)
    print(f"\nreps {reps}; pool 2^{pool_log2}; K {K}; strata = gap cells (6x6) x sign x pos(M,S), refined where N p >= K")
    print("gain = p(1-p)/N / MSE vs truth; z = mean bias / se; empty = strata with no samples (mass uncounted)")
    summary = {}
    for name, W, truth in ms:
        e_true = 1 - truth
        print(f"\n{name}  err {e_true:.4f}")
        hdr = "budget   " + "  ".join(f"{s:>22s}" for s in schemes); print(hdr)
        summary[name] = {}
        for b, N in enumerate(BUDGETS):
            row = [f"2^{int(np.log2(N)):2d}    "]
            summary[name][N] = {}
            for s in schemes:
                a = np.array(res[name][s][b]); est, used, empty = a[:, 0], a[:, 1], a[:, 2]
                mse = ((est - e_true) ** 2).mean(); z = (est.mean() - e_true) / (est.std(ddof=1) / np.sqrt(len(est)))
                gain = e_true * (1 - e_true) / used.mean() / mse
                row.append(f"{gain:6.2f}x rms {np.sqrt(mse):.1e} z{z:+4.1f}")
                summary[name][N][s] = dict(gain=gain, rms=float(np.sqrt(mse)), z=float(z), used=float(used.mean()), empty=float(empty.mean()))
            print("  ".join(row))
    json.dump(summary, open('probes/strat_adaptive.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
