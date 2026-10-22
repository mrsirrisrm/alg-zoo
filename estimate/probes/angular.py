"""PLAN J4: harmonic scale in input space via angular decorrelation on the sphere S^(T-1).

The network is bias-free, so label, output and correctness depend only on x / |x|: functions
on the sphere, where Hilton's halfspace estimator works with harmonic polynomials. A direct
per-degree harmonic estimator is noise-dominated at high degree, so we measure the
equivalent angular structure function D(theta) = P(f(x) != f(y)), x uniform, y at angle
theta from x (uniform direction). For a function with sharp walls D rises linearly (slope
= wall area) and saturates at its independent-pair value. theta_half (D at half saturation)
sets the harmonic degree scale D* ~ pi / theta_half, and C(T + D*, T) is the polynomial
term count an input-space harmonic estimator would need at that scale.
Functions: label (2nd-argmax position), output (network argmax), correct (output == label).
Models: zoo h16 best seed at T = 2..10, M(16,10), zoo h32 T10, hand10.
Check: the term count against MC samples (p(1-p)/(5e-4)^2) should cross near the BnB
crossover (MAP: T ~ 3-4).

Usage: uv run python probes/angular.py
"""
import sys, math
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from zoo_cache import zoo_weights
from opt_bits import loaders
from ground_truth import forward

THETAS = np.geomspace(1e-4, math.pi / 2, 29)
NPAIR = 1 << 17


def funcs(W, x):
    Wt = [th.tensor(a, dtype=th.float32) for a in W]
    with th.no_grad():
        out = forward(x, *Wt).argmax(-1)
    lab = x.topk(2, -1).indices[:, 1]
    return {'label': lab, 'output': out, 'correct': out == lab}


def run(name, W, T):
    g = th.Generator().manual_seed(T)
    x = th.randn(NPAIR, T, generator=g); x = x / x.norm(dim=1, keepdim=True)
    u = th.randn(NPAIR, T, generator=g); u = u - (u * x).sum(1, keepdim=True) * x; u = u / u.norm(dim=1, keepdim=True)
    fx = funcs(W, x)
    ind = funcs(W, th.randn(NPAIR, T, generator=g))
    sat = {k: (fx[k] != ind[k]).float().mean().item() for k in fx}
    D = {k: [] for k in fx}
    for th_ in THETAS:
        y = math.cos(th_) * x + math.sin(th_) * u
        fy = funcs(W, y)
        for k in fx:
            D[k].append((fx[k] != fy[k]).float().mean().item())
    acc = fx['correct'].float().mean().item(); p = 1 - acc
    mc = acc * p / (5e-4) ** 2
    row = f'  {name:10s} T={T:2d} acc {acc:.4f}'
    for k in ('label', 'output', 'correct'):
        d = np.array(D[k]); half = 0.5 * sat[k]
        th_half = float(np.exp(np.interp(math.log(half), np.log(np.maximum(d, 1e-12)), np.log(THETAS)))) if d[-1] > half else float('nan')
        slope = np.median(d[:6] / THETAS[:6])
        Dstar = math.pi / th_half
        terms = math.comb(T + int(round(Dstar)), T)
        row += f' | {k}: sat {sat[k]:.3f} th1/2 {th_half:.2e} slope {slope:.2g} D* {Dstar:6.0f} terms {terms:.1e}'
    print(row + f' | MC samples for 5e-4 rms {mc:.1e}', flush=True)


def main():
    L = loaders()
    print('D(theta) = P(f(x) != f(y)) at angle theta; th1/2 at half the independent-pair value; D* = pi / th1/2;')
    print('terms = C(T + D*, T); slope = D(theta)/theta at the smallest angles (wall area).')
    for T in (2, 3, 4, 5, 6, 8, 10):
        run('h16 zoo', zoo_weights(16, T), T)
    for n in ('trained', 'h32', 'hand10'):
        run(n, L[n](), 10)


if __name__ == '__main__':
    main()
