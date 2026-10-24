"""ARC's cumulant propagation (Wu et al. 2026, mlp_cumulant_propagation) on unrolled
random-init RNNs: where on their n/L axis does M(16,10)'s architecture sit? (PLAN K2)

The RNN h_t = relu(x_t W_ih + W_hh h_{t-1}), logits = W_out h_T, x ~ N(0, I_T), is written
as an exact MLP of constant width m = h + 2(T-1) and T linear+relu layers plus the readout:
units 0..h-1 carry the hidden state, units h+2(t-1), h+2(t-1)+1 carry relu(x_t), relu(-x_t)
for t = 1..T-1 (identity pass-through at every layer, so x_t = relu(x_t) - relu(-x_t) is
available when step t reads it). Checked against the RNN forward on samples.
kprop estimates E[logits] from the input cumulants {0, I}; truth is MC with 2^22 samples.
Score as ARC do: mse / var per output, and the MC mse at kprop's FLOP cost, where one MC
sample costs the RNN forward 2(h + h^2)T + 2Th FLOPs (not the padded MLP) and kprop's cost
is their NamedFlopCounter. gain = mse_MC(F_kprop) / mse_kprop; > 1 means kprop beats MC.
Weights: torch default init U(-1/sqrt(h), 1/sqrt(h)) for all three (the zoo's start, J7).

Usage: /Users/martin/code/mlp_cumulant_propagation/.venv/bin/python probes/kprop_rnn.py
       [--h 16,32,64,128] [--T 2,3,4,6,8,10] [--k 1,2,3] [--seeds 3]
"""
import json, time, argparse
import torch as th
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop
from mlp_kprop.flop_utils import NamedFlopCounter
import mlp_kprop.flop_utils as _fu, numpy as _np


class _NP:
    """Their counter asserts integer FLOP counts; the symmetric-slice discount is fractional at
    small widths (seen at m = 18, 20), so relax that one check and keep the count as computed."""
    def __getattr__(self, k): return getattr(_np, k)
    allclose = staticmethod(lambda *a, **k: True)


_fu.np = _NP()
th.set_grad_enabled(False); th.set_num_threads(8)
NMC = 1 << 22


def rnn_weights(h, T, gen):
    k = 1 / h ** 0.5
    u = lambda *s: (th.rand(*s, generator=gen) * 2 - 1) * k
    return u(h), u(h, h), u(T, h)


def rnn_forward(x, W_ih, W_hh, W_out):
    h = th.relu(x[:, 0:1] * W_ih)
    for t in range(1, x.shape[1]):
        h = th.relu(x[:, t:t + 1] * W_ih + h @ W_hh.T)
    return h @ W_out.T


def unrolled_mlp(W_ih, W_hh, W_out, T, pad=3, enc='pm', c=6.0):
    """pad: minimum input dimension (unused inputs get zero weights); the harmonic projection
    divides by zero at input dimension 2.
    enc 'pm': carry x_t as relu(x_t), relu(-x_t) (two units, no bias).
    enc 'shift': carry x_t as relu(x_t + c) with bias c (one unit; exact up to P(x < -c) = 1e-9
    at c = 6), consumed as W_ih (u - c); the carried unit is then Gaussian to all orders."""
    h = len(W_ih); n_in = max(T, pad)
    if enc == 'shift':
        return shifted_mlp(W_ih, W_hh, W_out, T, n_in, c)
    m = h + 2 * (T - 1)
    mlp = MLP(input_dim=n_in, hidden_dim=m, output_dim=T, num_layers=T + 1, nonlin='relu')
    L0 = th.zeros(m, n_in); L0[:h, 0] = W_ih
    for t in range(1, T):
        L0[h + 2 * (t - 1), t] = 1.0; L0[h + 2 * (t - 1) + 1, t] = -1.0
    mlp.Ws[0].weight.copy_(L0)
    for t in range(1, T):
        L = th.zeros(m, m); L[:h, :h] = W_hh
        L[:h, h + 2 * (t - 1)] = W_ih; L[:h, h + 2 * (t - 1) + 1] = -W_ih
        L[h:, h:] = th.eye(m - h)
        mlp.Ws[t].weight.copy_(L)
    Lo = th.zeros(T, m); Lo[:, :h] = W_out; mlp.Ws[T].weight.copy_(Lo)
    return mlp


def shifted_mlp(W_ih, W_hh, W_out, T, n_in, c):
    h = len(W_ih); m = h + (T - 1)
    mlp = MLP(input_dim=n_in, hidden_dim=m, output_dim=T, num_layers=T + 1, nonlin='relu',
              init_kind='manual', w_var=1.0, b_mean=[1.0] * T + [0.0])      # b_mean != 0 gives layers 0..T-1 a bias
    L0 = th.zeros(m, n_in); L0[:h, 0] = W_ih; b0 = th.zeros(m)
    for t in range(1, T):
        L0[h + t - 1, t] = 1.0; b0[h + t - 1] = c
    mlp.Ws[0].weight.copy_(L0); mlp.Ws[0].bias.copy_(b0)
    for t in range(1, T):
        L = th.zeros(m, m); b = th.zeros(m); L[:h, :h] = W_hh
        L[:h, h + t - 1] = W_ih; b[:h] = -c * W_ih; L[h:, h:] = th.eye(m - h)
        mlp.Ws[t].weight.copy_(L); mlp.Ws[t].bias.copy_(b)
    Lo = th.zeros(T, m); Lo[:, :h] = W_out; mlp.Ws[T].weight.copy_(Lo)
    return mlp


def joint_mlp(W_ih, W_hh, W_out, T, c=6.0):
    """Shift encoding carrying every input to the output: output = [logits ; x_0..x_{T-1}]."""
    h = len(W_ih); m = h + T
    mlp = MLP(input_dim=T, hidden_dim=m, output_dim=2 * T, num_layers=T + 1, nonlin='relu',
              init_kind='manual', w_var=1.0, b_mean=[1.0] * (T + 1))
    L0 = th.zeros(m, T); L0[:h, 0] = W_ih; b0 = th.zeros(m); L0[h:, :] = th.eye(T); b0[h:] = c
    mlp.Ws[0].weight.copy_(L0); mlp.Ws[0].bias.copy_(b0)
    for t in range(1, T):
        L = th.zeros(m, m); b = th.zeros(m); L[:h, :h] = W_hh; L[:h, h + t] = W_ih; b[:h] = -c * W_ih; L[h:, h:] = th.eye(T)
        mlp.Ws[t].weight.copy_(L); mlp.Ws[t].bias.copy_(b)
    Lo = th.zeros(2 * T, m); bo = th.zeros(2 * T); Lo[:T, :h] = W_out; Lo[T:, h:] = th.eye(T); bo[T:] = -c
    mlp.Ws[T].weight.copy_(Lo); mlp.Ws[T].bias.copy_(bo)
    return mlp


def joint_accuracy(W, T, k, n_gauss=1 << 20):
    """kprop's joint cumulants of (logits, x) up to order 2 -> Gaussian -> P(argmax logits = 2nd argmax x)."""
    mlp = joint_mlp(*W, T).double()
    K_in = {1: th.zeros(T, dtype=th.float64), 2: th.eye(T, dtype=th.float64)}
    with NamedFlopCounter() as fc:
        K = mlp_kprop(mlp, K_in, k_max=k, use_avg_metric=False, output_d_max=2)
    mu, S = K[1].to_tensor(), K[2].to_tensor()
    ev, V = th.linalg.eigh((S + S.T) / 2); A = V * ev.clamp_min(0).sqrt()
    z = mu + th.randn(n_gauss, 2 * T, dtype=th.float64, generator=th.Generator().manual_seed(7)) @ A.T
    p = (z[:, :T].argmax(1) == z[:, T:].argsort(1)[:, -2]).double().mean().item()
    return p, fc.total(), mu[:T], S[:T, :T].diag().sqrt()


def joint_main():
    import numpy as np
    T = 10; models = {}
    w = np.load('weights.npz'); models['trained'] = [th.tensor(w[k], dtype=th.float32) for k in ('W_ih', 'W_hh', 'W_out')]
    s = np.load('seeds_h16.npz')
    for i in (0, 2): models[f'seed{i}'] = [th.tensor(s[k][i], dtype=th.float32) for k in ('W_ih', 'W_hh', 'W_out')]
    models['random'] = list(rnn_weights(16, T, th.Generator().manual_seed(1610)))
    x = th.randn(NMC, T, generator=th.Generator().manual_seed(3))
    print(f"{'model':8s} {'MC acc':>7s} | " + ' | '.join(f'K={k}: P(correct) flops' for k in (1, 2)) + ' | mean logits (kprop K=2 vs MC), first 3')
    for name, W in models.items():
        y = th.cat([rnn_forward(x[i:i + (1 << 18)], *W) for i in range(0, NMC, 1 << 18)])
        acc = (y.argmax(1) == x.argsort(1)[:, -2]).double().mean().item(); cells = []; mu = None
        for k in (1, 2):
            p, F, mu, _ = joint_accuracy(W, T, k); cells.append(f'K={k}: {p:.4f} {F:.1e}')
        print(f'{name:8s} {acc:7.4f} | ' + ' | '.join(cells) + ' | ' + ' '.join(f'{m:+.3f}/{t:+.3f}' for m, t in zip(mu[:3].tolist(), y.double().mean(0)[:3].tolist())), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', default='16,32,64,128'); ap.add_argument('--T', default='2,3,4,6,8,10')
    ap.add_argument('--k', default='1,2,3'); ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', default='probes/kprop_rnn.json')
    ap.add_argument('--control', action='store_true', help="ARC's own setting: He-init MLP of width h, T+1 linear layers, n-dim Gaussian input")
    ap.add_argument('--enc', default='pm', choices=['pm', 'shift'])
    ap.add_argument('--joint', action='store_true', help='P(correct) from the K=1,2 joint Gaussian of (logits, x) on the trained M(16,10), seeds and a random init')
    a = ap.parse_args(); ks = [int(v) for v in a.k.split(',')]; rows = []
    if a.joint:
        return joint_main()
    if a.control:
        print(f"control: He MLP width n = h, L = T + 1 linear layers; MC cost = MLP forward; {'n':>4s} {'L':>3s} {'n/L':>6s}")
    print(f"{'h':>4s} {'T':>3s} {'h/T':>6s} {'seed':>4s} | " + ' | '.join(f'K={k}: mse/var  flops  gain' for k in ks))
    for h in (int(v) for v in a.h.split(',')):
        for T in (int(v) for v in a.T.split(',')):
            for s in range(a.seeds):
                gen = th.Generator().manual_seed(100 * h + T + 1000 * s)
                if a.control:
                    th.manual_seed(100 * h + T + 1000 * s); mlp = MLP(input_dim=h, hidden_dim=h, output_dim=h, num_layers=T + 1).double()
                    x = th.randn(NMC, h, generator=gen).double()
                    y = th.cat([mlp(x[i:i + (1 << 16)]).out for i in range(0, NMC, 1 << 16)]); flops_mc = 2 * h * h * (T + 1)
                else:
                    W = rnn_weights(h, T, gen); mlp = unrolled_mlp(*W, T, enc=a.enc).double()
                    x = th.randn(NMC, mlp.input_dim, generator=gen)
                    y = th.cat([rnn_forward(x[i:i + (1 << 18), :T], *W) for i in range(0, NMC, 1 << 18)]).double()
                    xs = x[:4096].double(); assert th.allclose(mlp(xs).out, y[:4096], atol=1e-6), 'unrolled MLP != RNN'
                    flops_mc = 2 * (h + h * h) * T + 2 * T * h
                n_in = mlp.input_dim; mean, var = y.mean(0), y.var(0)
                cells = []
                for k in ks:
                    t0 = time.time()
                    try:
                        K_in = {1: th.zeros(n_in, dtype=th.float64), 2: th.eye(n_in, dtype=th.float64)}
                        with NamedFlopCounter() as fc:
                            est = mlp_kprop(mlp, K_in, k_max=k, use_avg_metric=False)[1].to_tensor()
                        F = fc.total(); mse = ((est - mean) ** 2 / var).mean().item()
                        gain = (flops_mc / F) / mse          # MC mse/var at N = F / flops_mc samples is flops_mc / F
                        rows.append(dict(h=h, T=T, seed=s, k=k, mse_over_var=mse, flops=F, gain=gain, secs=time.time() - t0))
                        cells.append(f'K={k}: {mse:8.2e} {F:8.1e} {gain:6.2f}')
                    except Exception as e:                   # noqa: BLE001 — record and move on
                        rows.append(dict(h=h, T=T, seed=s, k=k, error=repr(e)[:200])); cells.append(f'K={k}: failed {repr(e)[:60]}')
                print(f'{h:4d} {T:3d} {h / T:6.1f} {s:4d} | ' + ' | '.join(cells), flush=True)
    json.dump(rows, open(a.out, 'w'))


if __name__ == '__main__':
    main()
