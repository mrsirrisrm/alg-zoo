"""Are ReLU activation patterns associated with error, and at which step? (PLAN H)

For 2^20 iid inputs: the relevant tie is the closer of M/S and S/R (relative gap as the
B5 cells), b0 = the later position of that pair (from the input alone), the 16-bit pattern
at every step, the comparator margin at b0 = min_j |pre_j(b0)|. A predictor is scored by
the held-out log loss of error given a discrete feature (per-value error rates fitted on
one half, scored on the other, shrunk to the base rate with two pseudo-counts), as the
fraction of the constant predictor's log loss it removes ("explained"). Alignments:
absolute step t, and event step d = t - b0. The pattern alone cannot tell "went the wrong
way" from "went the right way", so the orientation o = (kind, whether x_b0 is the larger
of the pair), known from the input, is a second key: "given o" = explained(o x key) -
explained(o). Also per-neuron bits and margins |pre_j(b0)| at the aligned steps, the
error share of the top patterns at d = 0, and strip membership (gap < 0.03)."""
import sys, numpy as np, torch as th
sys.path[:0] = ['.', 'probes']
from alg_zoo import zoo_2nd_argmax
from alg_zoo.zoo import best_of_5_seed_2nd_argmax
from ft_metrics import pre_acts
from strat_adaptive import coarse
from ground_truth import load_weights

T, N_LOG2, CHUNK = 10, int(__import__("os").environ.get("NLOG2", 20)), 1 << 18
D = list(range(-2, 4))                                   # event steps t - b0


def weights(H, seed=None):
    sd = zoo_2nd_argmax(H, T, seed=seed).state_dict()
    return sd['rnn.weight_ih_l0'][:, 0].float(), sd['rnn.weight_hh_l0'].float(), sd['linear.weight'].float()


def explained(err, key, half):
    """1 - heldout logloss(err | key) / heldout logloss(err | const). key: int64 per sample."""
    e, k = err.astype(np.float64), key
    def ll(fit, test):
        u, inv = np.unique(k[fit], return_inverse=True)
        n = np.bincount(inv, minlength=len(u)); s = np.bincount(inv, weights=e[fit], minlength=len(u))
        p0 = e[fit].mean()
        pos = np.searchsorted(u, k[test]); pos = np.clip(pos, 0, len(u) - 1); seen = u[pos] == k[test]
        p = np.where(seen, (s[pos] + 2 * p0) / (n[pos] + 2), p0)
        return -np.mean(e[test] * np.log(p) + (1 - e[test]) * np.log(1 - p)), len(u)
    l1, nu = ll(half, ~half)
    p0 = e[half].mean(); l0 = -(e[~half] * np.log(p0) + (1 - e[~half]) * np.log(1 - p0)).mean()
    return 1 - l1 / l0, nu


def joint(*keys):
    """int64 label of the tuple of keys (np.unique on rows), for keys whose product overflows int64."""
    return np.unique(np.stack(keys, 1), axis=0, return_inverse=True)[1].reshape(-1).astype(np.int64)


def auc(score, y):
    r = np.argsort(np.argsort(score)) + 1.0; n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


@th.no_grad()
def run(name, W, core=None):
    W_ih, W_hh, W_out = W; H = W_ih.shape[0]; g = th.Generator().manual_seed(1234)
    codes, errs, b0s, kinds, gaps, margins, mbits, orients, absb0 = [], [], [], [], [], [], [], [], []
    for _ in range(1 << (N_LOG2 - 18)):
        x = th.randn(CHUNK, T, generator=g)
        pre, h = pre_acts(x, W_ih, W_hh)
        errs.append(((h @ W_out.T).argmax(-1) != x.topk(2, -1).indices[:, 1]).numpy())
        t3 = x.topk(3, -1); vM, vS, vR = t3.values.T; den = vM.abs().clamp_min(1e-30)
        g1, g2 = (vM - vS) / den, (vS - vR) / den
        kind = th.where(g1 < g2, 1, 2); gap = th.minimum(g1, g2)
        pa, pb, pb2 = t3.indices.T
        b0 = th.where(kind == 1, th.maximum(pa, pb), th.maximum(pb, pb2))
        orients.append(th.where(kind == 1, b0 == pa, b0 == pb).numpy())   # later arrival is the larger of the pair
        bits = (pre > 0).to(th.int64)                          # [n,T,H]
        codes.append((bits << th.arange(H)).sum(-1).numpy())  # [n,T]
        ar = th.arange(CHUNK); pb0 = pre[ar, b0]              # [n,H]
        sub = list(range(H)) if core is None else core
        margins.append(pb0[:, sub].abs().min(-1).values.numpy())
        mbits.append(bits[ar, b0].numpy().astype(np.int8)); absb0.append(pb0.abs().half().numpy())
        b0s.append(b0.numpy()); kinds.append(kind.numpy()); gaps.append(gap.numpy())
    codes, err, b0, kind, gap, margin, mbits, orient, absb0 = map(np.concatenate, (codes, errs, b0s, kinds, gaps, margins, mbits, orients, absb0))
    o = ((kind - 1) * 2 + orient).astype(np.int64); ex = lambda key: explained(err, key, half)[0]
    N = len(err); half = np.arange(N) < N // 2
    gcell = (np.minimum(np.digitize(gap, coarse), 6) * 2 + (kind - 1)).astype(np.int64)
    print(f"\n=== {name}: H {H}, err {err.mean():.4f}, b0 distribution " +
          " ".join(f"{(b0 == t).mean():.2f}" for t in range(1, T)))
    # 1. explained by the full pattern, absolute vs aligned
    row_abs = [explained(err, codes[:, t], half) for t in range(T)]
    aligned = {}
    for d in D:
        t = b0 + d; ok = (t >= 0) & (t < T)
        key = np.where(ok, codes[np.arange(N), np.clip(t, 0, T - 1)], -1)
        aligned[d] = explained(err, key, half)
    e_o = ex(o)
    print(f"explained by the {H}-bit pattern (held-out), and distinct patterns seen; orientation o alone {e_o:.3f}:")
    print("  absolute t : " + " ".join(f"t{t}={e:5.3f}({n})" for t, (e, n) in enumerate(row_abs)))
    print("  aligned d  : " + " ".join(f"d{d:+d}={aligned[d][0]:5.3f}({aligned[d][1]})" for d in D))
    keys = {}
    for d in D:
        t = b0 + d; ok = (t >= 0) & (t < T); keys[d] = np.where(ok, codes[np.arange(N), np.clip(t, 0, T - 1)], -1)
    print("  given o, absolute t: " + " ".join(f"t{t}={ex(o * (1 << H) + codes[:, t]) - e_o:5.3f}" for t in range(T)))
    print("  given o, aligned d : " + " ".join(f"d{d:+d}={ex(o * (1 << H) + keys[d] + 1) - e_o:5.3f}" for d in D))
    # 2. gap cell vs pattern vs both at d = 0, and the margin
    c0 = codes[np.arange(N), b0]; go = gcell * 4 + o
    mq = np.digitize(margin, np.quantile(margin, np.linspace(0, 1, 33)[1:-1])).astype(np.int64)
    print(f"  explained at b0: gap cell {ex(gcell):.3f} | gap x o {ex(go):.3f} | pattern {ex(c0):.3f} | o x pattern {ex(o * (1 << H) + c0):.3f} | "
          f"gap x o x pattern {ex(go * (1 << H) + c0):.3f}")
    print(f"             min-margin (32 bins) {ex(mq):.3f} | gap x o x margin {ex(go * 64 + mq):.3f}")
    sub = CORE16 if core is not None else list(range(H))
    cm = sum(1 << i for i in sub)
    k3 = [keys[0] + 1, keys[1] + 1, keys[2] + 1]                                   # -1 (off the end) -> 0
    j3, j3c = joint(*k3), joint(*[k & cm for k in k3])                              # three aligned steps jointly, all / core neurons
    print(f"  o x pattern over d=0..2 jointly: all neurons {ex(joint(o, j3)) - e_o:.3f}, core {len(sub)} neurons {ex(joint(o, j3c)) - e_o:.3f}; "
          f"gap x o x core-pattern d=0..2 {ex(joint(go, j3c)):.3f}")
    ins = gap < .03; hs = half[ins]
    exs = lambda key: explained(err[ins], key[ins], hs)[0]
    eos = exs(o)
    print(f"  inside the strip (gap<.03, err {err[ins].mean():.3f}): o {eos:.3f}; given o: pattern@b0 {exs(o * (1 << H) + c0) - eos:.3f}, "
          f"core d=0..2 {exs(joint(o, j3c)) - eos:.3f}, all d=0..2 {exs(joint(o, j3)) - eos:.3f}; gap x o {exs(go):.3f}, gap x o x core d=0..2 {exs(joint(go, j3c)):.3f}")
    print(f"  AUC for error: gap {1 - auc(gap, err.astype(bool)):.3f}, min-margin@b0 {1 - auc(margin, err.astype(bool)):.3f}")
    # 3. per-neuron explained at aligned steps
    print("  per-neuron bit explained (x1000) given o, at aligned d:")
    print("        " + " ".join(f"n{j:<4d}" for j in range(H)))
    for d in D:
        vals = [(ex(o * 3 + np.where(keys[d] < 0, 2, (keys[d] >> j) & 1)) - e_o) * 1000 for j in range(H)]
        print(f"  d{d:+d}   " + " ".join(f"{v:5.1f}" for v in vals))
    eb = err.astype(bool); strip = gap < .03
    print("  per-neuron AUC of small |pre_j(b0)| for error / for the strip:")
    print("  err    " + " ".join(f"{1 - auc(absb0[:, j].astype(np.float32), eb):5.2f}" for j in range(H)))
    print("  strip  " + " ".join(f"{1 - auc(absb0[:, j].astype(np.float32), strip):5.2f}" for j in range(H)))
    # 4. error share of the top patterns at d = 0, within kind
    for k in (1, 2):
        sel = kind == k; u, inv = np.unique(orient[sel] * (1 << H) + c0[sel], return_inverse=True)
        n = np.bincount(inv); s = np.bincount(inv, weights=err[sel])
        oi = np.argsort(-s); cs, cn = np.cumsum(s[oi]) / s.sum(), np.cumsum(n[oi]) / n.sum()
        print(f"  kind {'M/S' if k == 1 else 'S/R'}: {len(u)} (o, pattern) at b0; top 1/4/16 hold "
              f"{cs[0]:.2f}/{cs[min(3, len(u) - 1)]:.2f}/{cs[min(15, len(u) - 1)]:.2f} of the errors on "
              f"{cn[0]:.2f}/{cn[min(3, len(u) - 1)]:.2f}/{cn[min(15, len(u) - 1)]:.2f} of the mass; "
              f"top pattern error rate {s[oi][0] / n[oi][0]:.3f} vs {err[sel].mean():.3f}")
    # 5. strip membership (gap < 0.03): margin vs pattern vs sign-of-x pattern-free control
    es = lambda key: explained(strip, key, half)[0]
    print(f"  strip (gap<.03, mass {strip.mean():.3f}): AUC from min-margin@b0 {1 - auc(margin, strip):.3f}; explained by "
          f"pattern@b0 {es(c0):.3f}, min-margin bins {es(mq):.3f}, pattern x margin {es(joint(c0, mq)):.3f}, o x pattern x margin {es(joint(o, c0, mq)):.3f}")
    mb = margin[strip], margin[~strip]
    print(f"  median margin@b0: strip {np.median(mb[0]):.3f}, off-strip {np.median(mb[1]):.3f}; error-only {np.median(margin[err.astype(bool)]):.3f}")


CORE16 = [0, 1, 4, 5, 6, 7, 8, 9]
models = [('trained M(16,10)', [th.tensor(a) for a in load_weights('weights.npz')], CORE16),
          ('h16 seed0', weights(16, 0), None),
          ('h8 best', weights(8, best_of_5_seed_2nd_argmax(hidden_size=8, seq_len=T)), None),
          ('h32 best', weights(32, best_of_5_seed_2nd_argmax(hidden_size=32, seq_len=T)), None)]
print(f"2^{N_LOG2} inputs per model; relevant tie = closer of M/S, S/R; b0 = later position of that pair; "
      f"margin@b0 = min |pre| over core neurons (trained: n0,1,4-9; others: all)")
for name, W, core in models:
    run(name, W, core)
