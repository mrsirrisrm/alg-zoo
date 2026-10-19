"""B3 (PLAN.md): integrate the LAST input exactly (Rao-Blackwellise x_9).

Given x_0..x_8, h_8 is fixed and h_9 = relu(a + b x_9) with a = W_hh h_8, b = W_ih, so h_9 is
piecewise linear in the scalar x_9 with <= 16 breakpoints t_i = -a_i / b_i. The logits are
c + d x_9 on each piece. The LABEL also depends on x_9: with v1 >= v2 the top two of
x_0..x_8, label = pos(v1) for x_9 > v1, 9 for v2 < x_9 < v1, pos(v2) below. So on each of
<= 19 pieces correctness is an interval of x_9 (intersection of 9 half-lines), and
P(correct | x_<9) = sum of Phi-masses. Cost: 9-step forward (0.9 pass) + per piece an
incremental update of (c, d) and 9 bound checks (~0.25 pass) ~ 1.15 pass-equivalents.
Variance factor = Var(E[y | x_<9]) / Var(y) (= prefix R^2, ~0.5-0.6 from cond_ceiling).
Run from estimate/: python probes/exact_last.py
"""
import sys; sys.path.insert(0, '.')
import numpy as np, torch as th
from ground_truth import forward, load_weights

Phi = th.distributions.Normal(0., 1.).cdf


def cond_correct(x9, W_ih, W_hh, W_out):
    """x9: (n, 9) first nine inputs. Returns P(correct | x_<9), exact in x_9."""
    n = x9.shape[0]
    h = th.relu(x9[:, 0:1] * W_ih)
    for t in range(1, 9):
        h = th.relu(x9[:, t:t + 1] * W_ih + h @ W_hh.T)
    a = h @ W_hh.T; b = W_ih[None].expand(n, -1)                         # n,16
    top = x9.topk(2, -1); v1, v2 = top.values[:, 0], top.values[:, 1]; p1, p2 = top.indices[:, 0], top.indices[:, 1]
    bp = th.where(b.abs() > 1e-12, -a / b, th.full_like(a, float('inf')))  # neuron breakpoints
    pts = th.cat([bp, v1[:, None], v2[:, None]], 1)                        # n,18
    pts, _ = pts.sort(1)
    lo = th.cat([th.full((n, 1), -float('inf')), pts], 1); hi = th.cat([pts, th.full((n, 1), float('inf'))], 1)   # n,19 pieces
    mid = th.where(th.isfinite(lo) & th.isfinite(hi), 0.5 * (lo + hi), th.where(th.isfinite(lo), lo + 1, hi - 1))
    mid = th.where(th.isfinite(mid), mid, th.zeros_like(mid))
    total = th.zeros(n, dtype=th.float64)
    for k in range(19):
        m = mid[:, k]; L, H = lo[:, k], hi[:, k]
        on = (a + b * m[:, None]) > 0                                        # n,16 gate pattern on this piece
        c = (on * a) @ W_out.T; d = (on * b) @ W_out.T                     # n,10 logits = c + d x
        lab = th.where(m > v1, p1, th.where(m > v2, th.full_like(p1, 9), p2))
        cl = c.gather(1, lab[:, None]); dl = d.gather(1, lab[:, None])
        dd = dl - d; cc = c - cl                                             # need dd * x > cc for all j != lab
        dd = dd.scatter(1, lab[:, None], 1.0); cc = cc.scatter(1, lab[:, None], -float('inf'))
        # dd > 0: x > cc/dd ; dd < 0: x < cc/dd ; dd == 0: need cc < 0
        ratio = cc / th.where(dd == 0, th.ones_like(dd), dd)
        lb = th.where(dd > 0, ratio, th.full_like(ratio, -float('inf'))).max(1).values
        ub = th.where(dd < 0, ratio, th.full_like(ratio, float('inf'))).min(1).values
        ok = ((dd != 0) | (cc < 0)).all(1)
        A = th.maximum(lb, L); B = th.minimum(ub, H)
        mass = th.where(ok & (B > A), Phi(B.double()) - Phi(A.double()), th.zeros(n, dtype=th.float64))
        total += mass
    return total


def run(name, W, truth, n_log2=22, chunk=1 << 18):
    W = [th.tensor(w, dtype=th.float32) for w in W]
    g = th.Generator().manual_seed(11)
    rb, y = [], []
    with th.no_grad():
        for _ in range(1 << (n_log2 - 18)):
            x = th.randn(chunk, 10, generator=g)
            rb.append(cond_correct(x[:, :9], *W).numpy())
            y.append((forward(x, *W).argmax(-1) == x.topk(2, -1).indices[:, 1]).numpy())
    rb = np.concatenate(rb); y = np.concatenate(y).astype(float)
    N = len(rb); vy = y.var(); vr = rb.var()
    se_rb = np.sqrt(vr / N); se_y = np.sqrt(vy / N)
    cost = 1.15
    print(f"{name:8s} truth {truth:.5f} | MC {y.mean():.5f} (se {se_y:.1e}) | exact-last {rb.mean():.5f} (se {se_rb:.1e}, z vs truth {(rb.mean() - truth) / se_rb:+.1f}) | "
          f"var ratio {vr / vy:.3f} | gain at cost {cost} passes: {vy / vr / cost:.2f}x | frac of P(correct|x<9) in (0,1): {((rb > 1e-9) & (rb < 1 - 1e-9)).mean():.3f}")


if __name__ == '__main__':
    d, s = np.load('noisy_sigma0.005.npz'), np.load('seeds_h16.npz')
    run('trained', load_weights('weights.npz'), 0.9531186)
    for i in range(2):
        run(f'noisy{i}', [d[k][i] for k in ('W_ih', 'W_hh', 'W_out')], float(d['acc'][i]))
    for i in range(2):
        run(f'seed{i}', [s[k][i] for k in ('W_ih', 'W_hh', 'W_out')], float(s['acc'][i]))
