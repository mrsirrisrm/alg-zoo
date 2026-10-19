"""F1/F2 (PLAN.md): the B5 stack on a scrambled-Sobol pool.

Same estimator as probes/best_unbiased.py (strata = gap cells of the first nine inputs' top
three x sign x pos(M,S), adaptive Neyman with per-stage estimates and planned weights, exact
x_9 integral), but the pool is a scrambled Sobol stream in 9-input order-statistic coordinates:
dims 0-7 the Lehmer code of the rank permutation (position of the max first), dims 8-17 the
Renyi exponential spacings of the nine sorted values, dim 18 = x_9 (unused by the rb value).
Drawing from a stratum = the next unused pool member of that cell in sequence order, as in B2.
Each scrambled point is marginally uniform, so every stratum mean is unbiased; the question
F1 asks is whether the measured z agrees (B2 saw z +4.2 on seed0 at 2^20 with the 2^26 truth).
z is reported against the harness truth and, where probes/truth28.json exists, the 2^28 truth.
Run from estimate/: python probes/best_sobol.py [reps] [models...]
"""
import sys, json, os, time
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from torch.quasirandom import SobolEngine
from exact_last import cond_correct
from strat_adaptive import models_list
from best_unbiased import cells9, simulate, make_pool, BUDGETS, K, COST

icdf = th.distributions.Normal(0., 1.).icdf


def orderstat_d(u, d):
    """u: (n, 2d) uniforms -> (n, d) iid N(0,1) via rank permutation (dims 0..d-2) and sorted values (dims d-1..2d-2)."""
    n = u.shape[0]; u = u.double()                                               # float32 rounds the top uniform to 1.0 ~once per 2^22 points -> inf
    e = -th.log(u[:, d - 1:2 * d].clamp(1e-15, 1)); s = th.cumsum(e, 1)
    vals = icdf((1 - s[:, :d] / s[:, d:d + 1]).clamp(1e-15, 1 - 1e-15))          # descending
    rem = th.arange(d).repeat(n, 1); pos = th.empty(n, d, dtype=th.long)
    for k in range(d):
        m = d - k
        j = (u[:, k] * m).long().clamp(max=m - 1) if k < d - 1 else th.zeros(n, dtype=th.long)
        pos[:, k] = rem.gather(1, j[:, None])[:, 0]
        mask = th.ones_like(rem, dtype=th.bool); mask.scatter_(1, j[:, None], False)
        rem = rem[mask].view(n, m - 1) if m > 1 else rem
    x = th.empty(n, d); x.scatter_(1, pos, vals.float())
    return x


def make_pool_sobol(seed, n_log2):
    u = SobolEngine(19, scramble=True, seed=seed).draw(1 << n_log2)
    x = th.cat([orderstat_d(u[:, :18], 9), icdf(u[:, 18:19].double().clamp(1e-15, 1 - 1e-15)).float()], 1)
    assert th.isfinite(x).all()
    _, fid = cells9(x)
    order = np.argsort(fid, kind='stable'); starts = np.searchsorted(fid[order], np.arange(9801))
    return x, fid, order, starts


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
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    names = sys.argv[2:] or ['trained', 'noisy0', 'noisy1', 'seed0', 'seed1']
    ms = models_list(names); pool_log2 = 22
    t28 = json.load(open('probes/truth28.json')) if os.path.exists('probes/truth28.json') else {}
    xo, fo, _, _ = make_pool(99993, 22); cnt = np.bincount(fo, minlength=9800)
    tv = {}
    for name, W, _ in ms:
        vo = rb(xo, W); tv[name] = np.bincount(fo, vo ** 2, minlength=9800) / np.maximum(cnt, 1) - (np.bincount(fo, vo, minlength=9800) / np.maximum(cnt, 1)) ** 2
    combos = [('iid', 'adaptive'), ('sobol', 'prop'), ('sobol', 'adaptive'), ('sobol', 'oracle')]
    res = {n: {c: [[] for _ in BUDGETS] for c in combos} for n, _, _ in ms}
    t0 = time.time()
    for rep in range(reps):
        pools = {'iid': make_pool(3000 + rep, pool_log2), 'sobol': make_pool_sobol(3000 + rep, pool_log2)}; rng = np.random.default_rng(rep)
        for name, W, truth in ms:
            v = {k: rb(pools[k][0], W) for k in pools}
            for kind, scheme in combos:
                x, fid, order, starts = pools[kind]
                for b, q in enumerate(simulate(v[kind], order, starts, scheme, tv[name], rng)):
                    res[name][(kind, scheme)][b].append(q)
        print(f"rep {rep} done {time.time() - t0:.0f}s", flush=True)
    print(f"\nreps {reps}; pool 2^{pool_log2}; K {K}; B5 strata and estimator; sobol = scrambled Sobol pool in 9-input order-statistic coordinates")
    print("gain = p(1-p) / (N 1.15) / MSE vs harness truth [16-84% bootstrap over reps]; z = mean bias / se vs harness truth, z28 vs the 2^28 truth")
    summary = {}
    for name, W, truth in ms:
        e_true = 1 - truth; e28 = 1 - t28[name]['acc28'] if name in t28 else None
        print(f"\n{name}  err {e_true:.4f}" + (f"  (2^28: {e28:.4f})" if e28 else ""))
        print("budget   " + "  ".join(f"{k + '/' + s:>40s}" for k, s in combos) + "   sobol/iid ratio [16-84% bootstrap over reps]"); summary[name] = {}
        for b, N in enumerate(BUDGETS):
            row = [f"2^{int(np.log2(N)):2d}    "]; summary[name][N] = {}
            for c in combos:
                a = np.array(res[name][c][b]); est, used = a[:, 0], a[:, 1]
                mse = ((est - e_true) ** 2).mean(); se = est.std(ddof=1) / np.sqrt(len(est)); z = (est.mean() - e_true) / se
                z28 = (est.mean() - e28) / se if e28 else float('nan')
                cst = e_true * (1 - e_true) / (used.mean() * COST['rb']); gain = cst / mse; lo, hi = boot((est - e_true) ** 2, cst)
                row.append(f"{gain:6.2f}x [{lo:4.1f},{hi:4.1f}] rms {np.sqrt(mse):.1e} z{z:+4.1f} z28{z28:+4.1f}")
                summary[name][N]['/'.join(c)] = dict(gain=gain, ci=[float(lo), float(hi)], rms=float(np.sqrt(mse)), z=float(z), z28=float(z28), used=float(used.mean()), mean=float(est.mean()), se=float(se), est=est.tolist(), used_reps=used.tolist())
            e2 = np.stack([(np.array(res[name][c][b])[:, 0] - e_true) ** 2 for c in (('iid', 'adaptive'), ('sobol', 'adaptive'))], 1)
            r = summary[name][N]['sobol/adaptive']['gain'] / summary[name][N]['iid/adaptive']['gain']; lo, hi = boot(e2, 1.0)
            summary[name][N]['ratio'] = dict(ratio=r, ci=[float(lo), float(hi)])
            row.append(f"{r:5.2f} [{lo:4.2f},{hi:4.2f}]")
            print("  ".join(row))
    print("\ngeometric mean over models of gain: budget, iid/adaptive, sobol/adaptive, ratio")
    for N in BUDGETS:
        gi = np.exp(np.mean([np.log(summary[n][N]['iid/adaptive']['gain']) for n in summary])); gs = np.exp(np.mean([np.log(summary[n][N]['sobol/adaptive']['gain']) for n in summary]))
        print(f"2^{int(np.log2(N))}  {gi:5.2f}x  {gs:5.2f}x  {gs / gi:5.2f}")
    json.dump(summary, open('probes/best_sobol.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
