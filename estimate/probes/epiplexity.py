"""Epiplexity of the network's behaviour by prequential coding (Finzi et al. 2026, arXiv
2601.03220, sec 4.1), as a follow-up to "Surprise accounting with the J5 family".

The observation is the stream (x_i, y_i), x ~ N(0, I_10), y_i = teacher's argmax, and the
derived correctness bit c_i = [y_i = label_i]. A compute-bounded observer is a student
RNN of the same architecture (or wider) trained online by AdamW on the stream; each batch
is coded before the update, so the code is pure prequential and charges no parameter bits.
  code(y) = sum_i -log2 p_i(y_i)              student's probability of the teacher's output
  code(c) = sum_i -log2 q_i(c_i),  q_i(c=1) = p_i(y = label_i)
Split as in the paper: H = held-out bits/input of the final student, S = total - N x H (the
area above the final loss: the structure a bounded learner extracted). Two students per
width: 'distil' trained on the teacher's outputs, 'label' trained on the true labels (the
task-only observer; its q(c=1) = p(label) is the learned analogue of gaps-logit).
References for the bottom line of the J5 surprise accounting:
  family account  116 + 5 x 11 bits + N x H_fam  (fam-logit bits/input, surprise_family.out)
  weights         PREC bits (J1 precision, opt_bits.out) + N x 0
  independent seed: disagreement of another trained seed with the teacher on c and y
    (what two independent learners share: the task-level part of the pattern)
Question: does the distilled student's held-out disagreement with the teacher on near-tie
inputs fall well below an independent seed's (the pattern is learnable structure, low H_T
at this budget), or does it sit at the seed level (time-bounded entropy for SGD at this
budget, though a 432-parameter program for it exists)?

Usage: uv run python probes/epiplexity.py [--k 24] [--h 16,64] [--names trained,seed0,seed2] [--device cuda]
"""
import sys, json, time, argparse
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import forward
from opt_bits import loaders
from surprise import feats

PREC = {'trained': 2429, 'seed0': 2516, 'seed2': 3247}
H_FAM = {'trained': 0.1902, 'seed0': 0.3738, 'seed2': 0.2085}    # fam-logit bits/input, surprise_family.out
FAM_DESC = 116 + 5 * 11
B = 1024; WARM = 1024; T = 10; NB = 1 << 20
DEV = 'cpu'                                    # set by --device; data generated on CPU (seeded), moved over
th.set_num_threads(8)


def tens(W):
    return [th.tensor(a, dtype=th.float32, device=DEV) for a in W]


def flops_per_sample(h):
    return 2 * (h + h * h) * T + 2 * 10 * h


def init(h, gen):
    k = 1 / h ** 0.5                           # torch.nn.RNN / Linear default: U(-1/sqrt(h), 1/sqrt(h))
    return [((th.rand(s, generator=gen) * 2 - 1) * k).to(DEV).requires_grad_() for s in ((h,), (h, h), (10, h))]


def regions(F, x):
    npos = (x.numpy() > 0).sum(1)
    sign = npos <= 1; near = (npos >= 2) & (np.minimum(F['g1'], F['g2']) < 0.1)
    return {'all': np.ones_like(sign), 'near-tie': near, 'sign': sign, 'far': ~sign & ~near}


def evaluate(P, xb, yT, lab, reg):
    with th.no_grad():
        lp = th.log_softmax(forward(xb, *P), -1).double()
    ar = th.arange(len(xb), device=DEV); y = lp.argmax(-1)
    bits_y = (-lp[ar, yT] / np.log(2)).cpu().numpy()
    q = lp[ar, lab].exp().clamp(1e-9, 1 - 1e-9); c = (yT == lab).double()
    bits_c = (-(c * q.log() + (1 - c) * (1 - q).log()) / np.log(2)).cpu().numpy()
    dis = (y != yT).cpu().numpy(); disc = ((y == lab) != (yT == lab)).cpu().numpy()
    out = {'acc': (y == lab).double().mean().item()}
    for k, m in reg.items():
        out[k] = dict(bits_y=float(bits_y[m].mean()), bits_c=float(bits_c[m].mean()),
                      dis_y=float(dis[m].mean()), dis_c=float(disc[m].mean()))
    return out


def run(Wt, h, target, k, seed, xb, yTb, labb, reg, W0=None, lr=None):
    """W0: student's initial weights (None: random init of width h); lr: None = zoo's 0.04/sqrt(h)."""
    gen = th.Generator().manual_seed(1000 + seed)
    P = init(h, gen) if W0 is None else [th.tensor(a, dtype=th.float32, device=DEV).requires_grad_() for a in W0]
    opt = th.optim.AdamW(P, lr=lr or 0.04 / h ** 0.5, weight_decay=1e-3)
    warm = WARM if W0 is None else 64
    sched = th.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 1.0))
    gx = th.Generator().manual_seed(seed)
    tot_y = tot_c = 0.0; marks = {}; nxt = 1 << 14; t0 = time.time()
    for step in range(1 << (k - 10)):
        x = th.randn(B, T, generator=gx).to(DEV)
        with th.no_grad():
            yT = forward(x, *Wt).argmax(-1); lab = x.topk(2, -1).indices[:, 1]
        lp = th.log_softmax(forward(x, *P), -1); ar = th.arange(B, device=DEV)
        with th.no_grad():                              # code the batch before the update
            tot_y += (-lp[ar, yT]).sum().item() / np.log(2)
            q = lp[ar, lab].double().exp().clamp(1e-9, 1 - 1e-9); c = (yT == lab).double()
            tot_c += (-(c * q.log() + (1 - c) * (1 - q).log())).sum().item() / np.log(2)
        tgt = yT if target == 'distil' else lab
        loss = -lp[ar, tgt].mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        n = (step + 1) * B
        if n == nxt:
            with th.no_grad(): Pd = [p.detach() for p in P]
            marks[n] = dict(code_y=tot_y, code_c=tot_c, **evaluate(Pd, xb, yTb, labb, reg)); nxt *= 2
    return marks, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=24); ap.add_argument('--h', default='16,64')
    ap.add_argument('--names', default='trained,seed0,seed2'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init', default='scratch', help="'scratch' or comma-separated model names to start the student from")
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--out', default='probes/epiplexity.json')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args(); L = loaders(); hs = [int(v) for v in a.h.split(',')]
    global DEV; DEV = a.device
    inits = [(None, h) for h in hs] if a.init == 'scratch' else [(nm, 16) for nm in a.init.split(',')]

    def init_weights(nm, W):                 # 'noisy:0.005' = teacher + sigma x rms per matrix (noisy_models.py)
        if not nm.startswith('noisy:'): return L[nm]()
        rng = np.random.default_rng(a.seed); s = float(nm.split(':')[1])
        return [w + s * np.sqrt((w ** 2).mean()) * rng.standard_normal(w.shape) for w in W]
    xb = th.randn(NB, T, generator=th.Generator().manual_seed(2)); F = feats(xb); reg = regions(F, xb)
    xb = xb.to(DEV); labb = F['label'].to(DEV); res = {}
    for name in a.names.split(','):
        Wt = tens(L[name]())
        with th.no_grad(): yTb = forward(xb, *Wt).argmax(-1)
        cT = (yTb == labb).cpu().numpy()
        print(f'\n{name}: acc {cT.mean():.4f}; held-out {NB} inputs; regions mass ' +
              ' '.join(f'{r} {m.mean():.3f}' for r, m in reg.items()))
        # independent-learner reference: the other trained seeds against this teacher
        for other in ('trained', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4'):
            if other == name: continue
            with th.no_grad(): yo = forward(xb, *tens(L[other]())).argmax(-1)
            co = (yo == labb).cpu().numpy(); dyo = (yo != yTb).cpu().numpy()
            print(f'  seed ref {other:8s}: disagree y / c  ' + '  '.join(
                f'{r} {dyo[m].mean():.4f}/{(co != cT)[m].mean():.4f}' for r, m in reg.items()))
        res[name] = {}
        for nm, h in inits:
            if nm == name: continue
            for target in ('distil', 'label'):
                marks, dt = run(Wt, h, target, a.k, a.seed, xb, yTb, labb, reg, None if nm is None else init_weights(nm, L[name]()), a.lr)
                tag = f'h{h}-{target}' if nm is None else f'init-{nm}-{target}'; res[name][tag] = marks
                N = max(marks); m = marks[N]; Hy, Hc = m['all']['bits_y'], m['all']['bits_c']
                print(f'  student {tag} ({dt:.0f}s, {3 * N * flops_per_sample(h):.2e} FLOPs): acc {m["acc"]:.4f}')
                print(f'    prequential code of y: {m["code_y"]:.3e} bits = S {m["code_y"] - N * Hy:.3e} + N x H {Hy:.4f}')
                print(f'    prequential code of c: {m["code_c"]:.3e} bits = S {m["code_c"] - N * Hc:.3e} + N x H {Hc:.4f}')
                print('    held-out disagreement with teacher y / c, bits_c: ' + '  '.join(
                    f'{r} {m[r]["dis_y"]:.4f}/{m[r]["dis_c"]:.4f} {m[r]["bits_c"]:.3f}' for r in reg))
                print('    N: ' + ' '.join(f'2^{int(np.log2(n))}' for n in sorted(marks)))
                print('    near-tie dis_c: ' + ' '.join(f'{marks[n]["near-tie"]["dis_c"]:.3f}' for n in sorted(marks)))
                print('    near-tie bits_c: ' + ' '.join(f'{marks[n]["near-tie"]["bits_c"]:.3f}' for n in sorted(marks)))
        if name in PREC:
            N = 1 << a.k
            print(f'  references at N = 2^{a.k}: weights {PREC[name]} bits; family {FAM_DESC} + N x {H_FAM[name]} = '
                  f'{FAM_DESC + N * H_FAM[name]:.3e} bits; constant {N * (-(cT.mean() * np.log2(cT.mean()) + (1 - cT.mean()) * np.log2(1 - cT.mean()))):.3e}')
    json.dump(res, open(a.out, 'w'))


if __name__ == '__main__':
    main()
