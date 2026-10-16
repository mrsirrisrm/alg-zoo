"""Deterministic Gaussian primitives (torch, float64, batched).

- Phi, phi, Phi_inv
- Phi2: bivariate normal CDF via Gauss-Legendre in theta = asin(r)
- relu_moments: exact first/second moments of relu(z), z ~ N(mu, S)
- orthant_prob: P(Z < b), Z ~ N(0, C), by Genz separation of variables on a
  fixed Sobol lattice (numerical integration of a Gaussian, not model sampling)
"""
import math

import numpy as np
import torch as th

SQRT2 = math.sqrt(2.0)
INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def Phi(x):
    return 0.5 * th.erfc(-x / SQRT2)


def phi(x):
    return INV_SQRT2PI * th.exp(-0.5 * x * x)


def Phi_inv(p):
    return SQRT2 * th.erfinv(2.0 * p - 1.0)


_GL = {}


def _gl(n, device):
    key = (n, str(device))
    if key not in _GL:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL[key] = (th.tensor(x, device=device), th.tensor(w, device=device))
    return _GL[key]


def bvnu(h, k, r):
    """P(X > h, Y > k), corr(X, Y) = r. Genz (2004) BVN algorithm, 20-pt GL,
    with the |r| > 0.925 asymptotic branch. Broadcasts over h, k, r."""
    h, k, r = th.broadcast_tensors(h, k, r)
    xg, wg = _gl(20, h.device)
    x, w = xg[:10], wg[:10]  # one half of the symmetric rule; both (1-x), (1+x) are used
    tp = 2.0 * math.pi
    hk = h * k

    # |r| < 0.925
    hs = (h * h + k * k) / 2
    asr = th.asin(r) / 2
    def term(sgn):
        sn = th.sin(asr[..., None] * (1 + sgn * x))
        return (w * th.exp((sn * hk[..., None] - hs[..., None]) / (1 - sn * sn))).sum(-1)
    low = (term(-1) + term(1)) * asr / tp + Phi(-h) * Phi(-k)

    # |r| >= 0.925
    neg = r < 0
    k2 = th.where(neg, -k, k)
    hk2 = h * k2
    rr = r.abs().clamp_max(1 - 1e-15)
    as_ = 1 - rr * rr
    a = th.sqrt(as_)
    bs = (h - k2) ** 2
    c = (4 - hk2) / 8
    d = (12 - hk2) / 80
    asr2 = -(bs / as_ + hk2) / 2
    bvn = th.where(asr2 > -100, a * th.exp(asr2) * (1 - c * (bs - as_) * (1 - d * bs) / 3 + c * d * as_ * as_), th.zeros_like(a))
    b = th.sqrt(bs)
    sp = math.sqrt(tp) * Phi(-b / a)
    bvn = bvn - th.where(hk2 > -100, th.exp(-hk2.clamp_min(-100) / 2) * sp * b * (1 - c * bs * (1 - d * bs) / 3), th.zeros_like(a))
    a2 = a / 2
    def term2(sgn):
        xs = (a2[..., None] * (1 + sgn * x)) ** 2
        rs = th.sqrt(1 - xs)
        e = -(bs[..., None] / xs + hk2[..., None]) / 2
        return (w * th.exp(e) * (th.exp(-hk2[..., None] * xs / (2 * (1 + rs) ** 2)) / rs - (1 + c[..., None] * xs * (1 + d[..., None] * xs)))).sum(-1)
    bvn = bvn + a2 * (term2(-1) + term2(1))
    bvn = -bvn / tp
    L = th.where(h < 0, Phi(k2) - Phi(h), Phi(-h) - Phi(-k2))
    high = th.where(~neg, bvn + Phi(-th.maximum(h, k2)), th.where(h >= k2, -bvn, L - bvn))
    out = th.where(r.abs() < 0.925, low, high)
    return out.clamp(0.0, 1.0)


def Phi2(h, k, r):
    """P(X < h, Y < k), corr(X, Y) = r."""
    return bvnu(-h, -k, r)


def relu_moments(mu, S, jitter=1e-12):
    """mu [..., d], S [..., d, d] -> (E[relu z] [..., d], E[relu z relu z^T] [..., d, d])."""
    S = S.expand(mu.shape + mu.shape[-1:])
    var = th.diagonal(S, dim1=-2, dim2=-1).clamp_min(jitter)
    sd = var.sqrt()
    a = mu / sd
    Pa, pa = Phi(a), phi(a)
    m1 = mu * Pa + sd * pa
    m2_diag = (mu * mu + var) * Pa + mu * sd * pa

    # standardized thresholds: z_i > 0  <=>  X_i > h_i = -a_i
    hi, hj = (-a)[..., :, None], (-a)[..., None, :]
    si, sj = sd[..., :, None], sd[..., None, :]
    mi, mj = mu[..., :, None], mu[..., None, :]
    r = (S / (si * sj)).clamp(-1 + 1e-9, 1 - 1e-9)
    q = th.sqrt(1.0 - r * r)
    L = Phi2(-hi.expand_as(r), -hj.expand_as(r), r)  # P(X>hi, Y>hj)
    phi_i, phi_j = phi(hi), phi(hj)
    Qj_i = Phi(-(hj - r * hi) / q)  # P(Y > hj | X = hi)
    Qi_j = Phi(-(hi - r * hj) / q)
    EX = phi_i * Qj_i + r * phi_j * Qi_j
    EY = phi_j * Qi_j + r * phi_i * Qj_i
    rad = th.sqrt(((hi * hi - 2 * r * hi * hj + hj * hj) / (q * q)).clamp_min(0))
    EXY = r * L + r * hi * phi_i * Qj_i + r * hj * phi_j * Qi_j + q * INV_SQRT2PI * phi(rad)
    m2 = mi * mj * L + mi * sj * EY + mj * si * EX + si * sj * EXY
    d = mu.shape[-1]
    eye = th.eye(d, dtype=th.bool, device=mu.device)
    m2 = th.where(eye, th.diag_embed(m2_diag), m2)
    return m1, m2


_SOBOL = {}


def _sobol(n, dim, device):
    key = (n, dim, str(device))
    if key not in _SOBOL:
        eng = th.quasirandom.SobolEngine(dim, scramble=True, seed=7)
        _SOBOL[key] = eng.draw(n, dtype=th.float64).to(device).clamp(1e-12, 1 - 1e-12)
    return _SOBOL[key]


def orthant_prob(b, C, n=1024, jitter=1e-10):
    """P(Z < b) for Z ~ N(0, C). b [..., d], C [..., d, d]."""
    d = b.shape[-1]
    C = C + jitter * th.eye(d, dtype=C.dtype, device=C.device) * (1 + th.diagonal(C, dim1=-2, dim2=-1).mean(-1)[..., None, None])
    Lc = th.linalg.cholesky(C)
    w = _sobol(n, max(d - 1, 1), b.device)  # [n, d-1]
    bb = b[..., None, :]  # [..., 1, d]
    Lb = Lc[..., None, :, :]
    shape = b.shape[:-1] + (n,)
    f = th.ones(shape, dtype=b.dtype, device=b.device)
    ys = []
    for i in range(d):
        acc = sum(Lb[..., i, j] * ys[j] for j in range(i)) if i else 0.0
        e = Phi((bb[..., i] - acc) / Lb[..., i, i])
        f = f * e
        if i < d - 1:
            ys.append(Phi_inv((w[:, i] * e).clamp(1e-300, 1 - 1e-16)))
    return f.mean(-1)
