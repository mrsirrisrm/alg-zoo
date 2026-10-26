"""Can the cells be grouped by the layout of the error sliver inside them?

Per tie skeleton (tied pair a < b at value v; for S/R also M fixed), the background u (the
other inputs, iid N(0,1) below v) is split into cells by the on/off pattern of the kink
neurons (polyhedral.KSET) over all ten steps, read at the wall on one side (gap d = +-EPS;
at an M/S tie the comparator sits exactly on the wall). With sigma = +1 for S/R (the larger
of the pair is right) and -1 for M/S (the smaller), the input at gap d is right iff
sign(sigma (L_a - L_b)) = sign(d). Per background u the error set along d is an interval
next to the wall on one side; its true width w*(u) comes from the exact network (grid scan
then bisection for the zero of the margin, capped at W). Each input has two side-cells
(its wall cell for d > 0 and for d < 0), and on each side its one-sided width is w* if the
error is on that side, else 0. A side-cell's sliver: mean width w_c and tilt g_c (ridge
slope of w in u: which background inputs widen it). Side + / - is "earlier position
wrongly favoured" for M/S / S/R respectively.

The wall cell's own affine piece is not the sliver: moving along d crosses kinks, so the
far edge of the sliver often lies in a neighbour cell. Reported: how often the sliver is
contained in its wall cell, and the per-input wall-linear prediction (the cell-affine
model) against the true width.

Grouping: tilt in coordinates relative to the tie (inputs before a, between a and b, after
b, by offset) times the background sd, so cells from different skeletons compare. k-means
on (w_c, g_c) weighted by error mass (n_c w_c), K = 1..256; each side-cell's (w_c, g_c)
replaced by its centroid, width re-evaluated at each test input's own u. Scored on new
draws with gap d ~ U(-W, W) against the exact network's in-pair errors: predicted / true
error, shared fraction (network errors reproduced input by input), precision. Ladder: one
constant, constant and affine per skeleton-side, groups, per cell, per input. Then a rule:
a decision tree from the core gate bits at steps b..b+2 (on the input's side) plus the tie
layout to the group label, tested on held-out skeletons whose cells were never fitted.

Usage: uv run python probes/sliver_groups.py [--skeletons 64] [--holdout 16] [--weights weights.npz]
"""
import sys, argparse
sys.path.insert(0, '.')
import numpy as np, torch as th
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from polyhedral import Net, skeleton_draws, KSET
from boundary_layer import sample_ties
from ground_truth import load_weights

EPS = 1e-7
W = 0.1                                   # gap half-range (strip errors live at |d| < ~0.05)
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
NREL = 26                                 # relative offsets: before a 9, between 8, after b 9
RIDGE = 1.0                               # ridge on the within-cell slope, u in sd units


def jac(net, G):
    """Exact per-input linear map logits = A x from full gate patterns G [N, 10, 16]."""
    H = th.zeros(G.shape[0], 16, 10, dtype=th.float64)
    for t in range(10):
        P = net.W_hh @ H; P[:, :, t] += net.W_ih; H = G[:, t, :, None].double() * P
    return net.W_out @ H


def rel_index(j, a, b):
    return j - a + 8 if j < a else (j - a - 1 + 9 if j < b else j - b - 1 + 17)


def shift(x, a, b, d):
    y = x.clone(); y[:, a] += d / 2; y[:, b] -= d / 2; return y


def side_cell(net, x, a, b, sig, side):
    """Wall cell on one side: kink key (side-prefixed), gates, and the cell-affine width."""
    G, _ = net.bits(shift(x, a, b, side * EPS))
    A = jac(net, G); mr = A[:, a] - A[:, b]
    m0 = sig * (mr * x).sum(1); al = sig * 0.5 * (mr[:, a] - mr[:, b])
    root = -m0 / al                                     # zero of the cell-affine margin
    wlin = th.where((al > 0) & (th.sign(root) == side), root.abs(), th.zeros_like(root)).clamp(max=W)
    key = np.packbits(G[:, :, KSET].reshape(len(x), -1).numpy(), axis=1)
    key = np.c_[np.full(len(x), side > 0, dtype=np.uint8), key]
    return [bytes(r) for r in key], G, wlin.numpy()


def true_width(net, x, a, b, sig, ngrid=64, nbis=24):
    """Signed side and width of the error interval next to the wall, from the exact network."""
    m = lambda d: sig * (lambda L: L[:, a] - L[:, b])(net.bits(shift(x, a, b, d))[1])
    m0 = m(th.zeros(len(x), dtype=th.float64))
    side = -th.sign(m0)                                 # m0 > 0 acts as d > 0: errors at d < 0
    lo = th.zeros(len(x), dtype=th.float64); hi = th.full_like(lo, W); found = th.zeros(len(x), dtype=th.bool)
    prev = th.zeros_like(lo)
    for j in range(1, ngrid + 1):                       # first grid cell where the margin turns
        dj = W * j / ngrid; turned = (th.sign(m(side * dj)) != th.sign(m0)) & ~found
        lo = th.where(turned, prev, lo); hi = th.where(turned, th.full_like(lo, dj), hi); found |= turned
        prev = th.full_like(lo, dj)
    for _ in range(nbis):
        mid = (lo + hi) / 2; turned = th.sign(m(side * mid)) != th.sign(m0)
        hi = th.where(turned & found, mid, hi); lo = th.where(~turned & found, mid, lo)
    return side.numpy(), th.where(found, (lo + hi) / 2, th.full_like(lo, W)).numpy()


def contained(net, x, a, b, side, w, wallG):
    """Is the far edge of the sliver in the wall cell? (kink set, and all 16 neurons)"""
    G, _ = net.bits(shift(x, a, b, th.tensor(side * w * (1 - 1e-3))))
    return (G[:, :, KSET] == wallG[:, :, KSET]).all(-1).all(-1).numpy(), (G == wallG).all(-1).all(-1).numpy()


def core_bits(G, b):
    out = th.zeros(G.shape[0], 3, len(KSET), dtype=th.bool)
    for i in range(3):
        if b + i < 10: out[:, i] = G[:, b + i][:, KSET]
    return out.reshape(G.shape[0], -1).numpy()


def ridge(U, y):
    Uc = U - U.mean(0); return np.linalg.solve(Uc.T @ Uc + RIDGE * np.eye(U.shape[1]), Uc.T @ (y - y.mean()))


def skeleton(net, xi, k, gen, nfit, ntest):
    v = xi.topk(k + 1).values[k]
    pos = th.nonzero((xi - v).abs() < 1e-9).flatten(); a, b = int(pos[0]), int(pos[1])
    sig = 1.0 if k == 2 else -1.0
    S: dict = dict(a=a, b=b, sig=sig)

    def draws(n):
        x, bg = skeleton_draws(xi, v, n, gen); bgi = th.nonzero(bg).flatten().tolist()
        side, w = true_width(net, x, a, b, sig)
        D: dict = dict(u=x[:, bgi].numpy(), side=side, w=w)
        for sd in (1, -1):
            key, G, wlin = side_cell(net, x, a, b, sig, sd)
            D[sd] = dict(key=key, G=G, wlin=wlin, bits=core_bits(G, b), w=np.where(side == sd, w, 0.0))
        return x, bgi, D
    xf, bgi, F = draws(nfit)
    S['bg'] = bgi; S['sd'] = float(xf[:, bgi].std()); S['rel'] = [rel_index(j, a, b) for j in bgi]
    us = F['u'] / S['sd']
    # containment of the sliver in its wall cell, weighted by width
    e = F['w'] > 1e-9
    wallG = th.where(th.tensor(F['side'] > 0)[:, None, None], F[1]['G'], F[-1]['G'])
    ck, ca = contained(net, xf[e], a, b, th.tensor(F["side"][e]), th.tensor(F["w"][e]), wallG[e])
    S['contain'] = (np.sum(F['w'][e] * ck), np.sum(F['w'][e] * ca), np.sum(F['w'][e]))
    cells, S['skel'] = {}, {}
    for sd in (1, -1):
        Fs = F[sd]; keys = np.array([hash(kk) for kk in Fs['key']])
        for kh in np.unique(keys):
            m = keys == kh; n = int(m.sum()); ws = Fs['w'][m]
            g = ridge(us[m], ws) / S['sd'] if n > 1 and ws.any() else np.zeros(len(bgi))
            cells[Fs['key'][int(np.argmax(m))]] = dict(side=sd, n=n, w=float(ws.mean()), g=g, u=F['u'][m].mean(0),
                                                        wlin=float(Fs['wlin'][m].mean()))
        S['skel'][sd] = (float(Fs['w'].mean()), ridge(us, Fs['w']) / S['sd'], F['u'].mean(0))
    S['cells'] = cells
    _, _, S['test'] = draws(ntest)
    xt = S['test']; d = (th.rand(ntest, generator=gen, dtype=th.float64) * 2 - 1) * W
    # exact errors at the random gap, on fresh inputs built from the test backgrounds
    x = th.tensor(np.repeat(xi.numpy()[None], ntest, 0)); x[:, bgi] = th.tensor(xt['u'])
    out = net.bits(shift(x, a, b, d))[1].argmax(-1)
    lab = th.where(d > 0, a, b) if k == 2 else th.where(d > 0, b, a)
    xt['d'] = d.numpy(); xt['err'] = (out != lab).numpy(); xt['inpair'] = ((out == a) | (out == b)).numpy()
    return S


def predict(S, desc):
    """desc(S, side, key) -> (w, g, u) or None; width at each test input on its own side."""
    t = S['test']; out = np.empty(len(t['d']))
    for sd in (1, -1):
        idx = np.nonzero(np.sign(t['d']) == sd)[0]
        for i in idx:
            dd = desc(S, sd, t[sd]['key'][i])
            if dd is None: dd = S['skel'][sd]
            out[i] = dd[0] + dd[1] @ (t['u'][i] - dd[2])
    return out


def score(Ss, wfn):
    P = T = PT = 0
    for S in Ss:
        t = S['test']; pred = np.abs(t['d']) < wfn(S); m = t['inpair']
        P += (pred & m).sum(); T += (t['err'] & m).sum(); PT += (pred & t['err'] & m).sum()
    return P / max(T, 1), PT / max(T, 1), PT / max(P, 1), PT / max(P + T - PT, 1)


def feats(S, c):
    z = np.zeros(1 + NREL); z[0] = c['w']; z[1 + np.array(S['rel'])] = np.clip(c['g'] * S['sd'], -0.5, 0.5); return z


def from_feats(S, z):
    return z[0], z[1 + np.array(S['rel'])] / S['sd']


def run_kind(net, k, nsk, nho, gen, nfit, ntest):
    name = 'S/R' if k == 2 else 'M/S'
    early_side = -1 if k == 2 else 1
    sk = sample_ties(k, nsk, gen)[0].double()
    Ss = [skeleton(net, sk[i], k, gen, nfit, ntest) for i in range(nsk)]
    tr, ho = Ss[:nsk - nho], Ss[nsk - nho:]
    rows = [(S, kk, c) for S in tr for kk, c in S['cells'].items()]
    wt = np.array([c['n'] * c['w'] for _, _, c in rows]); wt = wt / wt.sum()
    terr = np.mean([(S['test']['err'] & S['test']['inpair']).mean() for S in tr])
    ncell = np.array([sum(1 for c in S['cells'].values() if c['w'] > 0) for S in tr])
    n90 = [int(np.searchsorted(np.cumsum(np.sort(v)[::-1]) / v.sum(), 0.9) + 1)
           for v in [np.array([c['n'] * c['w'] for c in S['cells'].values()]) for S in tr]]
    ck = sum(S['contain'][0] for S in tr) / sum(S['contain'][2] for S in tr)
    ca = sum(S['contain'][1] for S in tr) / sum(S['contain'][2] for S in tr)
    print(f'\n== {name} tie: {len(tr)} fit skeletons; side-cells with any error per skeleton median {np.median(ncell):.0f}, '
          f'covering 90% of error mass median {np.median(n90):.0f}; in-pair error at |d| < {W}: {terr:.4f}')
    big = np.array([c['n'] >= 16 for _, _, c in rows])
    print(f'  error mass in side-cells with >= 16 fit draws (tilt resolvable): {wt[big].sum():.3f}')
    print(f'  sliver contained in its wall cell (width-weighted): kink set {ck:.3f}, all 16 neurons {ca:.3f}')
    early = np.array([c['side'] == early_side for _, _, c in rows])
    tilt = np.array([np.sqrt(((c['g'] * S['sd']) ** 2).sum()) for S, _, c in rows])
    ww = np.array([c['w'] for _, _, c in rows])
    print(f'  error mass: earlier position favoured {wt[early].sum():.3f}, later {wt[~early].sum():.3f}')
    ratio = tilt / np.maximum(ww, 1e-9)
    print('  tilt (|g| x sd) / mean width: ' + ', '.join(f'{nm} {wt[(ratio >= lo) & (ratio < hi)].sum():.3f}'
          for lo, hi, nm in ((0, .5, 'slab < 0.5'), (.5, 2, 'wedge 0.5-2'), (2, np.inf, 'tilt-dominated > 2'))))
    gm = np.zeros(NREL)
    for (S, _, c), w in zip(rows, wt): gm[S['rel']] += w * np.abs(c['g'] * S['sd'])
    print('  mean |tilt| x sd by offset, before a (-9..-1): ' + ' '.join(f'{x:.3f}' for x in gm[:9]))
    print('                             between (a+1..a+8): ' + ' '.join(f'{x:.3f}' for x in gm[9:17]))
    print('                             after b (+1..+9):   ' + ' '.join(f'{x:.3f}' for x in gm[17:]))
    print('  model                                       pred/true  shared  precision  jaccard')
    pr = lambda nm, r: print(f'    {nm:42s} {r[0]:8.3f}  {r[1]:6.3f}  {r[2]:6.3f}  {r[3]:6.3f}')
    w0 = {sd: np.average([c['w'] for _, _, c in rows if c['side'] == sd], weights=[c['n'] for _, _, c in rows if c['side'] == sd]) for sd in (1, -1)}
    pr('one constant per side', score(tr, lambda S: np.where(S['test']['d'] > 0, w0[1], w0[-1])))
    pr('constant per skeleton-side', score(tr, lambda S: predict(S, lambda S, sd, kk: (S['skel'][sd][0], 0 * S['skel'][sd][1], S['skel'][sd][2]))))
    pr('affine per skeleton-side', score(tr, lambda S: predict(S, lambda S, sd, kk: None)))
    Z = np.array([feats(S, c) for S, _, c in rows]); wz = np.array([c['n'] * c['w'] for _, _, c in rows]) + 1e-9
    for K in KS:
        km = KMeans(K, n_init=4, random_state=0).fit(Z, sample_weight=wz); C = km.cluster_centers_
        lab = {(id(S), kk): l for (S, kk, _), l in zip(rows, km.labels_)}
        def mk(mode):
            def desc(S, sd, kk):
                c = S['cells'].get(kk)
                if c is None: return None
                wg, gg = from_feats(S, C[lab[(id(S), kk)]])
                return (c['w'] if mode == 'tilt' else wg, c['g'] if mode == 'offset' else gg, c['u'])
            return lambda S: predict(S, desc)
        rb, rt, ro = score(tr, mk('both')), score(tr, mk('tilt')), score(tr, mk('offset'))
        print(f'    groups K={K:3d}: both {rb[0]:.3f} {rb[1]:.3f} {rb[2]:.3f} {rb[3]:.3f} | own width, group tilt {rt[1]:.3f} {rt[2]:.3f} '
              f'| own tilt, group width {ro[1]:.3f} {ro[2]:.3f}')
        if K in (8, 32): rule(tr, ho, km, rows, wz, K, lab)
    cell = lambda mode: (lambda S: predict(S, lambda S, sd, kk: None if kk not in S['cells'] else
                         (S['cells'][kk]['w'], S['cells'][kk]['g'] * (mode == 'affine'), S['cells'][kk]['u'])))
    pr('per cell, width only', score(tr, cell('const')))
    pr('per cell, width + tilt', score(tr, cell('affine')))
    pr('per input, wall-cell affine (no kinks)', score(tr, lambda S: np.where(S['test']['d'] > 0, S['test'][1]['wlin'], S['test'][-1]['wlin'])))
    pr('per input, true width (ceiling)', score(tr, lambda S: np.where(np.sign(S['test']['d']) == S['test']['side'], S['test']['w'], 0)))
    pr('held-out skeletons: affine per skeleton-side', score(ho, lambda S: predict(S, lambda S, sd, kk: None)))
    pr('held-out skeletons: true width (ceiling)', score(ho, lambda S: np.where(np.sign(S['test']['d']) == S['test']['side'], S['test']['w'], 0)))


def rule(tr, ho, km, rows, wz, K, lab):
    """Decision tree: core gate bits at b..b+2 on the input's side + side + tie layout -> group."""
    lay = lambda S, sd: [sd, S['b'] - S['a'], S['a'], 9 - S['b']]
    X, y, wv = [], [], []
    for S in tr:
        for sd in (1, -1):
            t = S['test'][sd]; first = {}
            for i, kk in enumerate(t['key']): first.setdefault(kk, i)
            for kk, i in first.items():
                if (id(S), kk) in lab:
                    c = S['cells'][kk]; X.append(np.r_[t['bits'][i], lay(S, sd)]); y.append(lab[(id(S), kk)]); wv.append(c['n'] * c['w'] + 1e-9)
    for depth in (3, 6, 10):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(np.array(X), y, sample_weight=np.array(wv))
        def f(S):
            t = S['test']; out = np.zeros(len(t['d']))
            for sd in (1, -1):
                idx = np.nonzero(np.sign(t['d']) == sd)[0]
                if not len(idx): continue
                g = tree.predict(np.c_[t[sd]['bits'][idx], np.tile(lay(S, sd), (len(idx), 1))])
                keys = np.array([hash(t[sd]['key'][i]) for i in idx])
                for kh in np.unique(keys):                    # cell centre from unlabeled inputs sharing the pattern
                    m = keys == kh; uc = t['u'][idx[m]].mean(0)
                    for G in np.unique(g[m]):
                        mm = idx[m & (g == G)]; wg, gg = from_feats(S, km.cluster_centers_[G]); out[mm] = wg + (t['u'][mm] - uc) @ gg
            return out
        rt, rh = score(tr, f), score(ho, f)
        print(f'      rule K={K} tree depth {depth:2d} ({tree.get_n_leaves():3d} leaves): fit skeletons shared {rt[1]:.3f} '
              f'(pred/true {rt[0]:.3f}), held-out skeletons shared {rh[1]:.3f} (pred/true {rh[0]:.3f}, precision {rh[2]:.3f}, jaccard {rh[3]:.3f}); fit jaccard {rt[3]:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights.npz'); ap.add_argument('--skeletons', type=int, default=64)
    ap.add_argument('--holdout', type=int, default=16); ap.add_argument('--fit-log2', type=int, default=13)
    ap.add_argument('--test-log2', type=int, default=14)
    a = ap.parse_args(); net = Net(load_weights(a.weights)); gen = th.Generator().manual_seed(53)
    print(f'weights {a.weights}; KSET {KSET}; gap U(-{W}, {W}); fit 2^{a.fit_log2}, test 2^{a.test_log2} draws per skeleton')
    for k in (2, 1):
        run_kind(net, k, a.skeletons, a.holdout, gen, 1 << a.fit_log2, 1 << a.test_log2)


if __name__ == '__main__':
    main()
