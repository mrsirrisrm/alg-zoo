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


def Phi2(h, k, r, n=24):
    """P(X < h, Y < k), corr(X, Y) = r. Broadcasts over h, k, r."""
    x, w = _gl(n, h.device)
    th_max = th.asin(r.clamp(-1.0, 1.0))
    theta = 0.5 * th_max[..., None] * (x + 1.0)  # [..., n]
    s, c = th.sin(theta), th.cos(theta).clamp_min(1e-300)
    hh, kk = h[..., None], k[..., None]
    f = th.exp(-(hh * hh - 2.0 * hh * kk * s + kk * kk) / (2.0 * c * c))
    integral = 0.5 * th_max * (f * w).sum(-1)
    return Phi(h) * Phi(k) + integral / (2.0 * math.pi)


def relu_moments(mu, S, jitter=1e-12):
    """mu [..., d], S [..., d, d] -> (E[relu z] [..., d], E[relu z relu z^T] [..., d, d])."""
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
