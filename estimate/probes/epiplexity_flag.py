"""Privileged-information distillation: the student is told, at training time only, whether
the teacher errs on the sample (epiplexity K1 follow-up).

The teacher's near-tie behaviour has two parts: where it errs (the correctness bit c), and
what it says when it does (almost always the other member of the tie pair, so H(y_T | x, c)
is near zero). All the content is in c. Here the student gets c as an extra input channel
f, constant over the sequence: h_t = relu(x_t W_ih + f W_f + W_hh h_{t-1}), with
f = +1 (teacher errs), -1 (teacher right), 0 (not told). During training f is dropped to 0
with probability p_drop, so the student has to learn both routes; at test f = 0 everywhere,
the ordinary distillation setting. Question: does the flag make the student find the error
set sooner (the slowness is in finding it in input space) or not (the slowness is in
representing it)? Compare against the unflagged zoo-schedule students at the same budget.

Also reported at every mark: the student with the true flag (does it use it), and the
counterfactual f = +1 on inputs where the teacher is right (what it thinks the teacher's
wrong answer would be: the top-1 position, the third, or something else).

Usage: uv run python probes/epiplexity_flag.py --k 26 --h 128 --device cuda [--p-drop 0.5] [--save]
"""
import sys, json, time, argparse
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import forward
from opt_bits import loaders
from surprise import feats
import epiplexity as E
from epiplexity import regions, flops_per_sample

B = 1024; T = 10; NB = 1 << 20


def fwd(x, f, W_ih, W_f, W_hh, W_out):
    h = th.relu(x[:, 0:1] * W_ih + f * W_f)
    for t in range(1, x.shape[1]):
        h = th.relu(x[:, t:t + 1] * W_ih + f * W_f + h @ W_hh.T)
    return h @ W_out.T


def init(h, gen):
    k = 1 / h ** 0.5
    return [((th.rand(s, generator=gen) * 2 - 1) * k).to(E.DEV).requires_grad_() for s in ((h,), (h,), (h, h), (10, h))]


def ev(P, f, xb, yT, lab, reg):
    with th.no_grad():
        lp = th.log_softmax(fwd(xb, f, *P), -1).double()
    ar = th.arange(len(xb), device=E.DEV); y = lp.argmax(-1)
    disc = ((y == lab) != (yT == lab)).cpu().numpy()
    eT = (yT != lab).cpu().numpy(); eS = (y != lab).cpu().numpy()
    bits_c = (-(lp[ar, lab].exp().clamp(1e-9, 1 - 1e-9)).log() / np.log(2)).cpu().numpy()  # only meaningful on c=1
    out = {'acc': (y == lab).double().mean().item()}
    for k, m in reg.items():
        out[k] = dict(dis_y=float((y != yT).cpu().numpy()[m].mean()), dis_c=float(disc[m].mean()), err=float(eS[m].mean()),
                      shared=float((eT & eS)[m].sum() / max(eT[m].sum(), 1)), bits_c1=float(bits_c[m & ~eT].mean()))
    return out


def counterfactual(P, xb, yT, lab, nt):
    """f = +1 on inputs where the teacher is right (near-tie region): what would its wrong answer be?"""
    with th.no_grad():
        y = fwd(xb, th.ones(len(xb), 1, device=E.DEV), *P).argmax(-1)
    top = xb.topk(3, -1).indices; right = (yT == lab).cpu().numpy() & nt
    y, t0, t2 = y.cpu().numpy()[right], top[:, 0].cpu().numpy()[right], top[:, 2].cpu().numpy()[right]
    lb = lab.cpu().numpy()[right]
    return dict(n=int(right.sum()), still_label=float((y == lb).mean()), top1=float((y == t0).mean()),
                third=float((y == t2).mean()), other=float(((y != lb) & (y != t0) & (y != t2)).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=26); ap.add_argument('--h', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--p-drop', type=float, default=0.5)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--save', action='store_true')
    ap.add_argument('--out', default='probes/epiplexity_flag.json')
    a = ap.parse_args(); E.DEV = a.device; h = a.h; L = loaders()
    xb = th.randn(NB, T, generator=th.Generator().manual_seed(2)); F = feats(xb); reg = regions(F, xb)
    xb = xb.to(E.DEV); labb = F['label'].to(E.DEV); Wt = E.tens(L['trained']())
    with th.no_grad(): yTb = forward(xb, *Wt).argmax(-1)
    fT = (2 * (yTb != labb).float() - 1)[:, None]; f0 = th.zeros(NB, 1, device=E.DEV)
    print(f'trained: acc {(yTb == labb).float().mean():.4f}; student h{h}, p_drop {a.p_drop}, k {a.k}, zoo schedule')

    gen = th.Generator().manual_seed(1000 + a.seed); P = init(h, gen)
    opt = th.optim.AdamW(P, lr=0.04 / h ** 0.5, weight_decay=1e-3); nsteps = 1 << (a.k - 10)
    sched = th.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / (1 << 14), 1.0) * max(0.01 ** (2 * s / nsteps), 0.01))
    gx = th.Generator().manual_seed(a.seed); gf = th.Generator().manual_seed(7 + a.seed)
    tot_y = 0.0; marks = {}; nxt = 1 << 14; t0 = time.time()
    for step in range(nsteps):
        x = th.randn(B, T, generator=gx)
        keep = (th.rand(B, 1, generator=gf) >= a.p_drop).float()
        x = x.to(E.DEV); keep = keep.to(E.DEV)
        with th.no_grad():
            yT = forward(x, *Wt).argmax(-1); lab = x.topk(2, -1).indices[:, 1]
            f = (2 * (yT != lab).float() - 1)[:, None] * keep
        lp = th.log_softmax(fwd(x, f, *P), -1); ar = th.arange(B, device=E.DEV)
        with th.no_grad(): tot_y += (-lp[ar, yT]).sum().item() / np.log(2)
        loss = -lp[ar, yT].mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        n = (step + 1) * B
        if n == nxt:
            Pd = [p.detach() for p in P]
            marks[n] = dict(code_y=tot_y, noflag=ev(Pd, f0, xb, yTb, labb, reg), trueflag=ev(Pd, fT, xb, yTb, labb, reg),
                            cf=counterfactual(Pd, xb, yTb, labb, reg['near-tie'])); nxt *= 2
    dt = time.time() - t0; N = max(marks); m = marks[N]
    if a.save: np.savez(f'{a.out.rsplit(".", 1)[0]}_h{h}.npz', W_ih=P[0].detach().cpu().numpy(), W_f=P[1].detach().cpu().numpy(),
                        W_hh=P[2].detach().cpu().numpy(), W_out=P[3].detach().cpu().numpy())
    print(f'  student h{h} flag-distil ({dt:.0f}s, {3 * N * flops_per_sample(h):.2e} FLOPs): acc (f=0) {m["noflag"]["acc"]:.4f}, '
          f'acc (true f) {m["trueflag"]["acc"]:.4f}')
    for key in ('noflag', 'trueflag'):
        print(f'    {key:8s} student error / shared fraction of teacher errors: ' + '  '.join(
            f'{r} {m[key][r]["err"]:.4f}/{m[key][r]["shared"]:.3f}' for r in reg))
    print('    N: ' + ' '.join(f'2^{int(np.log2(n))}' for n in sorted(marks)))
    for key in ('noflag', 'trueflag'):
        print(f'    {key:8s} near-tie shared: ' + ' '.join(f'{marks[n][key]["near-tie"]["shared"]:.3f}' for n in sorted(marks)))
        print(f'    {key:8s} near-tie dis_c:  ' + ' '.join(f'{marks[n][key]["near-tie"]["dis_c"]:.3f}' for n in sorted(marks)))
    print('    counterfactual f=+1 on near-tie inputs the teacher gets right (still label / top1 / third / other), by mark:')
    for n in sorted(marks):
        c = marks[n]['cf']; print(f'      2^{int(np.log2(n))}: {c["still_label"]:.3f} / {c["top1"]:.3f} / {c["third"]:.3f} / {c["other"]:.3f}')
    print(f'    prequential code of y with the dropped flags: {m["code_y"]:.3e} bits')
    json.dump(marks, open(a.out, 'w'))


if __name__ == '__main__':
    main()
