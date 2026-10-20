"""Prune the comparator by fine-tuning (PLAN G1); control arm (G3).

Loss = cross-entropy on the task + lambda * R, from the trained M(16,10) weights.
  --mode group    (G1) R = sum_j sqrt(g_j + eps), g_j the soft switching of neuron j's gates
                  at the steps after the first tied element, across background redraws
                  within exact M/S and S/R tie skeletons (ft_metrics.switch_soft). Few
                  neurons should be left switching, so that the post-tie computation has
                  few linear regions and the polyhedral route gets cheap.
  --mode linearise (G1b) R = sum_j sqrt(sum_t p_tj(1-p_tj)), p over random inputs: the periphery
                  neurons should be always on or always off at every step (linear), so the
                  regions over all ten steps that polyhedral.py needs become few.
  --mode lastkink (G3) R = -mean_j q_j(1-q_j) at the last step on random inputs: reward
                  kinks at the decision step (the h32 picture), as a Goodhart control.
Writes probes/finetune/<mode>_lam<lambda>.npz (W_ih, W_hh, W_out, acc) and a json of the
metrics in ft_metrics. Run from estimate/:
  python probes/finetune_prune.py --mode group --lams 0 0.001 0.003 0.01 0.03 0.1
"""
import sys, json, os, time, argparse
sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import load_weights
from ft_metrics import pre_acts, skeleton_batch, switch_soft, all_metrics

T = 10


def finetune(W0, lam, mode, n_log2, batch, lr, tau, seed, log_every=512):
    th.manual_seed(seed); gen = th.Generator().manual_seed(seed + 1)
    P = [th.nn.Parameter(w.clone()) for w in W0]; W_ih, W_hh, W_out = P
    opt = th.optim.AdamW(P, lr=lr, weight_decay=1e-3)
    steps = (1 << n_log2) // batch; warm = 100
    sched = th.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps))))
    B, m = 32, 64
    t0 = time.time(); acc_run = ce_run = r_run = 0.0; n_run = 0
    for step in range(steps):
        x = th.randn(batch, T)
        _, h = pre_acts(x, W_ih, W_hh); logits = h @ W_out.T
        y = x.topk(2, -1).indices[:, 1]
        ce = th.nn.functional.cross_entropy(logits, y)
        if mode == 'group':
            xs1, a01 = skeleton_batch(1, B, m, gen); xs2, a02 = skeleton_batch(2, B, m, gen)
            pre, _ = pre_acts(th.cat([xs1, xs2]), W_ih, W_hh)
            g = 0.5 * (switch_soft(pre[:B * m], a01, m, tau) + switch_soft(pre[B * m:], a02, m, tau))
            R = th.sqrt(g + 1e-6).sum()
        elif mode == 'linearise':
            pre, _ = pre_acts(th.randn(B * m, T), W_ih, W_hh)
            p = th.sigmoid(pre / tau).mean(0)                       # [10,16] P(gate on) over random inputs
            R = th.sqrt((p * (1 - p)).sum(0) + 1e-6).sum()           # few neurons nonlinear at any step
        elif mode == 'lastkink':
            pre, _ = pre_acts(th.randn(B * m, T), W_ih, W_hh)
            q = th.sigmoid(pre[:, -1] / tau)
            R = -(q * (1 - q)).mean(0).sum()
        else:
            raise ValueError(mode)
        loss = ce + lam * R
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with th.no_grad():
            acc_run += (logits.argmax(-1) == y).float().mean().item(); ce_run += ce.item(); r_run += R.item(); n_run += 1
        if (step + 1) % log_every == 0 or step == steps - 1:
            print(f"    step {step + 1:5d}/{steps}  ce {ce_run / n_run:.4f}  acc {acc_run / n_run:.4f}  R {r_run / n_run:8.4f}  lr {sched.get_last_lr()[0]:.1e}  {time.time() - t0:.0f}s", flush=True)
            acc_run = ce_run = r_run = 0.0; n_run = 0
    return [p.detach().clone() for p in P]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='group', choices=['group', 'linearise', 'lastkink'])
    ap.add_argument('--lams', type=float, nargs='+', default=[0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    ap.add_argument('--n-log2', type=int, default=25)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='probes/finetune')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W0 = [th.tensor(a) for a in load_weights('weights.npz')]
    res = {}
    base_path = f'{args.out}/trained_metrics.json'
    if os.path.exists(base_path):
        res['trained'] = json.load(open(base_path))
    else:
        res['trained'] = all_metrics(W0); json.dump(res['trained'], open(base_path, 'w'), indent=1)
    print('trained:', fmt(res['trained']), flush=True)
    for lam in args.lams:
        print(f"\n{args.mode} lambda {lam:g}", flush=True)
        W = finetune(W0, lam, args.mode, args.n_log2, args.batch, args.lr, args.tau, args.seed)
        mt = all_metrics(W); res[f'{args.mode}_lam{lam:g}'] = mt
        np.savez(f'{args.out}/{args.mode}_lam{lam:g}.npz', W_ih=W[0].numpy(), W_hh=W[1].numpy(), W_out=W[2].numpy(), acc=mt['acc'])
        json.dump(mt, open(f'{args.out}/{args.mode}_lam{lam:g}_metrics.json', 'w'), indent=1)
        print(f"  -> {fmt(mt)}", flush=True)
    print(f"\nSummary ({args.mode}; acc on 2^26, sd 2.6e-5; regions = patterns covering 99% of mass, median over 48 skeletons, all 16 neurons; switching = neurons with hard g_j > 0.05)")
    print(f"{'model':>16} {'acc':>7} {'d_acc':>7} {'reg M/S':>8} {'reg S/R':>8} {'sw M/S':>6} {'sw S/R':>6} {'last-kinks':>10} {'enrich':>6}")
    for k, v in res.items():
        print(f"{k:>16} {v['acc']:7.4f} {v['acc'] - res['trained']['acc']:+7.4f} {v['regions'][1][1] if 1 in v['regions'] else v['regions']['1'][1]:8.0f} {v['regions'][2][1] if 2 in v['regions'] else v['regions']['2'][1]:8.0f} "
              f"{v['n_switching'][1] if 1 in v['n_switching'] else v['n_switching']['1']:6d} {v['n_switching'][2] if 2 in v['n_switching'] else v['n_switching']['2']:6d} {v['kinks']['last']:10.1f} {v['kinks']['enrich']:6.2f}")
    for k, v in res.items():
        sw = v['switch']
        print(f"{k:>16} hard switching per neuron, M/S: " + ' '.join(f"{a:.2f}" for a in (sw[1] if 1 in sw else sw['1'])))
        print(f"{'':>16}                           S/R: " + ' '.join(f"{a:.2f}" for a in (sw[2] if 2 in sw else sw['2'])))


def fmt(mt):
    return (f"acc {mt['acc']:.4f}  regions99 M/S {mt['regions'][1][1] if 1 in mt['regions'] else mt['regions']['1'][1]:.0f} S/R {mt['regions'][2][1] if 2 in mt['regions'] else mt['regions']['2'][1]:.0f}  "
            f"switching neurons {mt['n_switching']}  last-kinks {mt['kinks']['last']:.1f} enrich {mt['kinks']['enrich']:.2f}")


if __name__ == '__main__':
    main()
