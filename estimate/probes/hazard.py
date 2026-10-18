"""When is an error committed? Decoder-free criterion: the state at step t is 'right' if
completing the sequence from h_t with benign inputs (a constant below the running minimum,
so the true answer cannot change) makes the network's own final readout give the running
2nd-argmax. (W_out applied directly to h_t with a 9-t index shift also decodes the running
answer, 0.69 -> 0.95 over t = 1..9, but is too noisy early on to define commit steps.)
For every final error, the commit step t* is the step after the last 'right' state; the
error is 'committed on arrival' if t* equals the arrival step of the later of the two
confused values (a comparison error), 'later' if the state was right once both had arrived
and was lost afterwards (overwritten by below-S inputs), 'earlier' otherwise. Also the
per-step hazard (first permanent loss at t) split by what x_t did (new max / new second /
below), and how often a correct final passed through a wrong intermediate state
(recoverable transients). Two completions (c = -1 and c = running min - 0.5) as a check.
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import torch as th
from strat_tie import models

N = 1 << 20


def decode(W, x, comp):
    """comp: None -> constant -1; 'min' -> running min - 0.5. Returns decodes and running
    2nd-argmax for steps 1..9 (n, 9)."""
    W_ih, W_hh, W_out = W; h = th.relu(x[:, :1] * W_ih); D, S = [], []
    for t in range(1, 10):
        h = th.relu(x[:, t:t + 1] * W_ih + h @ W_hh.T)
        c = th.full((len(x), 1), -1.) if comp is None else x[:, :t + 1].min(-1, keepdim=True).values - .5
        g = h
        for _ in range(t + 1, 10): g = th.relu(c * W_ih + g @ W_hh.T)
        D.append((g @ W_out.T).argmax(-1)); S.append(x[:, :t + 1].topk(2, -1).indices[:, 1])
    return th.stack(D, 1), th.stack(S, 1)


def run(name, W, comp):
    W = [th.tensor(a, dtype=th.float32) for a in W]
    x = th.randn(N, 10, generator=th.Generator().manual_seed(8))
    with th.no_grad():
        D, S = decode(W, x, comp)
    ok = D == S; err = ~ok[:, -1]
    print(f"\n{name} (completion {'c=-1' if comp is None else 'c=min-0.5'}): error {err.float().mean():.4f}; state-right fraction by step " + ' '.join(f"{a:.2f}" for a in ok.float().mean(0).tolist()))
    # commit step: 1 + last step (1..9) with a correct decode, 1 if never
    last_ok = th.where(ok.any(1), 9 - ok.flip(1).float().argmax(1), th.zeros(N, dtype=th.long))   # step index 1..9 of last ok, 0 if none
    tstar = last_ok + 1
    top = x.topk(3, -1).indices; pM, pS, pR = top.T; pred = D[:, -1]
    for kind, m, later in (('S/R', err & (pred == pR), th.maximum(pS, pR)), ('M/S', err & (pred == pM), th.maximum(pM, pS))):
        d = (tstar - later)[m]
        print(f"  {kind} errors ({m.float().mean():.4f}): committed on arrival of the later value {(d == 0).float().mean():.2f}, "
              f"later (state was right, then lost) {(d > 0).float().mean():.2f}, earlier {(d < 0).float().mean():.2f}; "
              f"of the 'later' ones, steps after arrival: " + ' '.join(f"{k}:{(d == k).float().mean() / max((d > 0).float().mean(), 1e-9):.2f}" for k in range(1, 5)))
    # hazard by step and by event at that step (steps 2..9; the running 2nd-argmax exists from step 1)
    first_bad = tstar.clone(); first_bad[~err] = 99
    rmax = th.cummax(x, 1).values; r2 = th.stack([x[:, :1] * 0 - th.inf] + [x[:, :t + 1].topk(2, -1).values[:, 1:2] for t in range(1, 10)], 1).squeeze(-1)
    newmax = x[:, 1:] > rmax[:, :-1]; new2 = (~newmax) & (x[:, 1:] > r2[:, :-1]); below = ~newmax & ~new2
    print("  hazard (final-error commit at step t | event at t), steps 2..9:")
    for lab, ev in (('new max', newmax), ('new 2nd', new2), ('below', below)):
        print(f"    {lab:8s} " + ' '.join(f"t{t}:{(first_bad[ev[:, t - 1]] == t).float().mean():.4f}" for t in range(2, 10)))
    # the same hazard conditioned on the comparison gap at t (relative to the running max), to
    # separate 'the task gets harder' (closer ties among more values) from 'the state degrades'
    gap_max = (x[:, 1:] - rmax[:, :-1]) / rmax[:, :-1].abs()                              # new max: vs old max
    gap_2nd = th.minimum(x[:, 1:] - r2[:, :-1], rmax[:, :-1] - x[:, 1:]) / rmax[:, :-1].abs()  # new 2nd: nearer of old 2nd / max
    print("  hazard | event at t and relative gap bin (<.03, .03-.1, >.1):")
    for lab, ev, gp in (('new max', newmax, gap_max), ('new 2nd', new2, gap_2nd)):
        for lo, hi in ((0, .03), (.03, .1), (.1, 99)):
            m = ev & (gp >= lo) & (gp < hi)
            print(f"    {lab} gap {lo:.2f}-{hi:4.2f} " + ' '.join(f"t{t}:{(first_bad[m[:, t - 1]] == t).float().mean():.4f}" for t in range(2, 10)) + f"  (share of event {m.float().sum() / ev.float().sum():.2f})")
    trans = (ok[:, -1] & ~ok[:, 1:].all(1)).float().mean() / ok[:, -1].float().mean()
    print(f"  correct finals that passed through a wrong intermediate decode (steps 2..9): {trans:.3f}")


if __name__ == '__main__':
    for name, W in models[:1] + models[1:2] + models[5:6]:
        run(name, W, None)
        if name == 'trained': run(name, W, 'min')
