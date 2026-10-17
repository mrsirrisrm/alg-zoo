"""Generic deductive estimator: branch-and-bound over activation patterns.

No samples of the network are drawn. Regions of input space on which every ReLU gate is
fixed are polyhedra; the network is linear on each. Regions are found by splitting on
the sign of one neuron's pre-activation at a time (step t, neuron i). Each child's
Gaussian mass is computed by separation of variables: x_0..x_{T-2} on a fixed scrambled
Sobol lattice (a quadrature rule) and the last variable exactly through Phi. Children
lighter than eps are dropped and their mass is recorded as `lost`. At a full region,
P(correct) is exact in x_{T-1} given the lattice point: the logits and the input
comparisons are linear in x_{T-1}, so correctness is piecewise constant between at most
45 + (T-1) breakpoints.

Estimate: acc_hat = correct + lost * acc_fill (acc_fill = correct / kept mass).
Hard bracket: [correct, correct + lost], up to quadrature error.
FLOPs are counted per operation class (see `flops` updates).
"""
import math, time
import numpy as np, torch as th
from gauss import Phi, Phi_inv

DT = th.float64


class BnB:
    def __init__(self, W, T, Q=256, eps=1e-6, chunk=4096, seed=3, max_nodes=200000, max_sec=300):
        self.W_ih, self.W_hh, self.W_out = [th.as_tensor(a, dtype=DT) for a in W]
        self.h, self.T, self.Q, self.eps, self.chunk = self.W_hh.shape[0], T, Q, eps, chunk
        d = max(T - 1, 1)
        self.w = th.quasirandom.SobolEngine(d, scramble=True, seed=seed).draw(Q, dtype=DT).clamp(1e-12, 1 - 1e-12)
        self.flops = 0.0; self.max_nodes, self.max_sec = max_nodes, max_sec

    def _sov(self, rows, final=None):
        """rows [N, r, T] (constraint rows . x >= 0). Returns mass [N] and, if final is given
        (A [N, K, T] logits map), P(correct) [N]."""
        N, r, T = rows.shape; Q = self.Q
        nz = rows.abs() > 1e-12
        last = th.where(nz, th.arange(T).expand_as(rows), th.full_like(rows, -1, dtype=th.long)).amax(-1)  # [N, r]
        dead = (last < 0).any(-1)          # identically-zero row (pre-activation == 0): gate fixed off by convention
        x = th.zeros(N, Q, T, dtype=DT); wgt = th.ones(N, Q, dtype=DT)
        for j in range(T):
            act = (last == j)
            if not act.any() and j < T - 1:
                x[..., j] = Phi_inv(self.w[:, j]).expand(N, Q); continue
            val = (x[..., :j] @ rows[:, :, :j].transpose(1, 2)) if j else th.zeros(N, Q, r, dtype=DT)
            coef = rows[:, None, :, j]
            bnd = -val / th.where(coef.abs() > 1e-300, coef, th.ones_like(coef))
            lo = th.where(act[:, None] & (coef > 0), bnd, th.full_like(bnd, -th.inf)).amax(-1)
            hi = th.where(act[:, None] & (coef < 0), bnd, th.full_like(bnd, th.inf)).amin(-1)
            Plo, Phi_ = Phi(lo), Phi(hi); m = (Phi_ - Plo).clamp_min(0)
            self.flops += N * Q * (2 * r * j + 4 * r + 60)
            if j < T - 1:
                wgt = wgt * m
                x[..., j] = Phi_inv((Plo + self.w[:, j] * m).clamp(1e-300, 1 - 1e-16)).clamp(-12, 12)
            else:
                lo_T, hi_T, m_T = lo, hi, m
        mass = (wgt * m_T).mean(1) * (~dead)
        if final is None:
            return mass, None
        # correctness exact in z = x_{T-1} on [lo_T, hi_T]
        K = final.shape[1]
        a = x[..., :T - 1] @ final[:, :, :T - 1].transpose(1, 2)      # [N,Q,K]
        b = final[:, None, :, T - 1].expand(N, Q, K)
        iu, ju = th.triu_indices(K, K, 1)
        db = b[..., iu] - b[..., ju]
        bp = [(a[..., ju] - a[..., iu]) / th.where(db.abs() > 1e-300, db, th.full_like(db, 1e-300)), x[..., :T - 1]]
        z = th.cat([lo_T[..., None].clamp_min(-12), th.cat(bp, -1).clamp(-12, 12), hi_T[..., None].clamp_max(12)], -1)
        z = th.maximum(th.minimum(z, hi_T[..., None].clamp_max(12)), lo_T[..., None].clamp_min(-12)).sort(-1).values
        mid = 0.5 * (z[..., 1:] + z[..., :-1])                                    # [N,Q,P]
        L = a[..., None, :] + b[..., None, :] * mid[..., None]                   # [N,Q,P,K]
        xs = th.cat([x[..., None, :T - 1].expand(*mid.shape, T - 1), mid[..., None]], -1)
        ok = L.argmax(-1) == xs.topk(2, -1).indices[..., 1]
        pm = (Phi(z[..., 1:]) - Phi(z[..., :-1])).clamp_min(0)
        P = mid.shape[-1]
        self.flops += N * Q * P * (2 * K + 3 * T + 60) + N * Q * K * K * 3
        corr = (wgt * (pm * ok).sum(-1)).mean(1) * (~dead)
        return mass, corr

    def run(self, verbose=False):
        h, T = self.h, self.T
        # node state: rows [N, r, T]; H = dh_{t-1}/dx [N, h, T]
        rows = th.zeros(1, 0, T, dtype=DT); H = th.zeros(1, h, T, dtype=DT)
        lost = 0.0; t0 = time.time(); peak = 1
        for t in range(T):
            P = H.transpose(1, 2) @ self.W_hh.T                     # [N, T, h]: pre_t = x . P[:, :, i]
            P = P.transpose(1, 2).clone(); P[:, :, t] += self.W_ih  # [N, h, T]
            G = th.zeros(1, 0, dtype=th.bool).expand(rows.shape[0], 0)
            for i in range(h):
                N = rows.shape[0]
                cand = []
                for s in (1.0, -1.0):
                    cand.append(th.cat([rows, (s * P[:, i])[:, None]], 1))
                kids = th.cat(cand, 0)
                mass = th.cat([self._sov(kids[c:c + self.chunk])[0] for c in range(0, kids.shape[0], self.chunk)])
                keep = mass >= self.eps
                lost += mass[~keep].sum().item()
                gate = th.cat([th.ones(N, dtype=th.bool), th.zeros(N, dtype=th.bool)])
                rows = kids[keep]; P = P.repeat(2, 1, 1)[keep]
                G = th.cat([G.repeat(2, 1), gate[:, None]], 1)[keep]
                H = H.repeat(2, 1, 1)[keep]
                peak = max(peak, rows.shape[0])
                if peak > self.max_nodes or time.time() - t0 > self.max_sec:
                    raise RuntimeError(f'cap: nodes {peak}, {time.time() - t0:.0f}s at t={t} i={i}')
            H = G[:, :, None].to(DT) * P
            if verbose:
                print(f'  t={t} regions {rows.shape[0]} lost {lost:.2e} flops {self.flops:.2e} {time.time() - t0:.0f}s', flush=True)
        A = self.W_out @ H                                           # [N, K, T]
        mass, corr = [], []
        for c in range(0, rows.shape[0], self.chunk):
            m, k = self._sov(rows[c:c + self.chunk], A[c:c + self.chunk]); mass.append(m); corr.append(k)
        mass, corr = th.cat(mass).sum().item(), th.cat(corr).sum().item()
        fill = corr / max(mass, 1e-300)
        return dict(acc=corr + lost * fill, lo=corr, hi=corr + lost, kept=mass, lost=lost,
                    regions=int(rows.shape[0]), peak=peak, flops=self.flops, sec=time.time() - t0)
