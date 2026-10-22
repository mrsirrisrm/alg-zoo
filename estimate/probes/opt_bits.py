"""PLAN J1 (a, c): bits of optimisation as -log2 of the prior mass of the basin {acc >= acc* - tol}.

Laplace / Gaussian-channel form (the same machinery as precision.py, but in absolute
coordinates against a prior centred at zero, so the weights' values cost bits as well as
their precision):
  - GN Hessian H of the KL to the network's own softmax, absolute coordinates, 8192 inputs.
  - Whiten by the prior: u = w / s (s per tensor: (i) the trained tensor's rms; (ii) one
    global rms for all tensors). G = S H S, eigenvalues mu_i.
  - Calibrate accuracy loss = c * KL from isotropic whitened noise (draws with loss in
    [3e-4, 3e-2]); KL budget K = tol / c.
  - Water-fill posterior variances d_i = min(1, theta / mu_i) with 1/2 sum mu_i d_i = K;
    bits = KL(N(u*, diag d) || N(0, I)) / ln 2 = location 1/2 |u*|^2 + precision
    1/2 sum (d_i - 1 - ln d_i), both in bits. Exact symmetry directions have mu ~ 0 and cost 0.
  - log2(h!) permutation copies of the basin are reported, not subtracted.
(c) Direct check: step the top 50 eigendirections to their single-direction basin edge
    r_i = sqrt(2 K / mu_i) (both signs); measured loss / tol should be ~1.
hand10 is the built-in check: a ~100-bit integer construction. If its basin bits are large,
basin volume measures precision, not selection under a simplicity prior.

Usage: uv run python probes/opt_bits.py [names...]   (default: all)
"""
import sys, math, time
sys.path.insert(0, '.')
import numpy as np, torch as th
from torch.func import jacrev, vmap
from alg_zoo import handcrafted
from zoo_cache import zoo_weights
from ground_truth import forward, load_weights

T = 10
X = th.randn(1 << 20, T, generator=th.Generator().manual_seed(11)); LAB = X.topk(2, -1).indices[:, 1]
XH = th.randn(8192, T, generator=th.Generator().manual_seed(5))


def sd_w(m):
    sd = m.state_dict()
    return [sd['rnn.weight_ih_l0'][:, 0].double().numpy(), sd['rnn.weight_hh_l0'].double().numpy(), sd['linear.weight'].double().numpy()]


def loaders():
    s = np.load('seeds_h16.npz')
    L = {'trained': lambda: [np.asarray(a, float) for a in load_weights('weights.npz')]}
    for i in range(5):
        L[f'seed{i}'] = (lambda i=i: [s[k][i] for k in ('W_ih', 'W_hh', 'W_out')])
    L['h8'] = lambda: zoo_weights(8); L['h32'] = lambda: zoo_weights(32)
    L['hand10'] = lambda: sd_w(handcrafted.handcrafted_2nd_argmax_10())
    return L


class Net:
    def __init__(self, W):
        self.h = len(W[0]); self.sizes = [w.size for w in W]; self.shapes = [w.shape for w in W]
        self.p0 = th.tensor(np.concatenate([w.ravel() for w in W]), dtype=th.float32)
        self.o = np.cumsum([0] + self.sizes)

    def unpack(self, p):
        return [p[self.o[j]:self.o[j + 1]].view(self.shapes[j]) for j in range(3)]

    def acc(self, p):
        with th.no_grad():
            return sum((forward(X[i:i + (1 << 18)], *self.unpack(p)).argmax(-1) == LAB[i:i + (1 << 18)]).sum().item()
                       for i in range(0, len(X), 1 << 18)) / len(X)

    def gn(self):
        f = lambda p, xi: forward(xi[None], *self.unpack(p))[0]
        H = th.zeros(len(self.p0), len(self.p0), dtype=th.float64)
        for i in range(0, len(XH), 1024):
            x = XH[i:i + 1024]
            J = vmap(jacrev(f), in_dims=(None, 0))(self.p0, x).double()
            with th.no_grad():
                pr = th.softmax(forward(x, *self.unpack(self.p0)), -1).double()
                F = th.diag_embed(pr) - pr[:, :, None] * pr[:, None, :]
                H += th.einsum('nia,nij,njb->ab', J, F, J)
        H /= len(XH)
        return (0.5 * (H + H.T)).numpy()


def waterfill(mu, K):
    mu = np.clip(mu, 1e-30, None); lo, hi = 1e-30, 1e10
    for _ in range(300):
        t = math.sqrt(lo * hi); d = np.minimum(1.0, t / mu)
        if 0.5 * (mu * d).sum() > K: hi = t
        else: lo = t
    return np.minimum(1.0, t / mu)


def run(name, W, tols=(1e-3, 1e-2)):
    t0 = time.time(); net = Net(W); a0 = net.acc(net.p0); P = len(net.p0)
    H = net.gn()
    print(f'\n{name}: h={net.h}, {P} params, acc {a0:.5f}, log2(h!) {math.lgamma(net.h + 1) / math.log(2):.0f} bits', flush=True)
    rms_t = [math.sqrt((w ** 2).mean()) for w in W]
    glob = math.sqrt(sum((w ** 2).sum() for w in W) / P)
    for prior, svec in (('(i) per-tensor rms', np.concatenate([np.full(w.size, r) for w, r in zip(W, rms_t)])),
                        ('(ii) global rms', np.full(P, glob))):
        S = np.diag(svec); G = S @ H @ S
        mu, V = np.linalg.eigh(G); mu, V = mu[::-1].clip(0), V[:, ::-1]
        u = net.p0.double().numpy() / svec
        # calibrate accuracy loss = c * KL with isotropic whitened noise
        rng = np.random.default_rng(1); pairs = []
        for sig in np.geomspace(1e-4, 3e-1, 14):
            for _ in range(3):
                z = rng.standard_normal(P) * sig
                kl = 0.5 * z @ G @ z
                loss = a0 - net.acc(net.p0 + th.tensor(z * svec, dtype=th.float32))
                pairs.append((sig, kl, loss))
        pr = np.array(pairs); m = (pr[:, 2] > 3e-4) & (pr[:, 2] < 3e-2)
        c = np.median(pr[m, 2] / pr[m, 1]) if m.sum() >= 3 else float('nan')
        spread = np.percentile(pr[m, 2] / pr[m, 1] / c, [10, 90]) if m.sum() >= 3 else [np.nan, np.nan]
        print(f'  prior {prior}: scales {np.round(rms_t, 3).tolist() if prior.startswith("(i)") else round(glob, 3)}; '
              f'|u*|^2/2 = {0.5 * (u ** 2).sum() / math.log(2):.0f} bits (location); trace(G) {mu.sum():.3g}, '
              f'eigs > 1e-6 max: {(mu > 1e-6 * mu[0]).sum()}')
        print(f'    calibration: c = {c:.3g} (loss / KL), 10-90% of ratio/c {spread[0]:.2f}-{spread[1]:.2f}, {m.sum()} draws in range')
        for tol in tols:
            K = tol / c; d = waterfill(mu, K)
            prec = 0.5 * (d - 1 - np.log(d)).sum() / math.log(2)
            loc = 0.5 * (u ** 2).sum() / math.log(2)
            print(f'    tol {tol:g}: K {K:.3g}; precision {prec:.0f} bits over {(d < 1).sum()} directions; '
                  f'total {prec + loc:.0f} bits; precision per constrained direction {prec / max((d < 1).sum(), 1):.2f}')
        if prior.startswith('(i)'):
            tol = 1e-3; K = tol / c; ratios = []
            for i in range(50):
                r = math.sqrt(2 * K / mu[i]); step = th.tensor(V[:, i] * r * svec, dtype=th.float32)
                ratios.append([(a0 - net.acc(net.p0 + sgn * step)) / tol for sgn in (1, -1)])
            ratios = np.array(ratios); q = np.percentile(ratios.mean(1), [10, 50, 90])
            print(f'    (c) top-50 directions at their basin edge, loss/tol (mean of +-): median {q[1]:.2f}, 10-90% {q[0]:.2f}-{q[2]:.2f}; '
                  f'asymmetry |+ - -|/sum median {np.median(np.abs(ratios[:, 0] - ratios[:, 1]) / np.abs(ratios).sum(1).clip(1e-9)):.2f}; '
                  f'first 10: ' + ' '.join(f'{v:.2f}' for v in ratios.mean(1)[:10]))
    print(f'  ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    L = loaders(); names = sys.argv[1:] or list(L)
    for n in names:
        run(n, L[n]())
