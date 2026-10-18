# Mechanistic accuracy estimate for M(16,10) — running log

Target (ARC, AlgZoo post): a mechanistic estimate of accuracy that matches random
sampling in MSE vs compute, measured on average over noisy copies of the weights.

## Facts from the weights (no sampling)
- No biases ⇒ positively homogeneous: output depends only on x/‖x‖.
- Only n1, n4, n6, n7, n8 have large input weights (|W_ih| ≈ 10–13); n4 positive,
  the others negative. n14 −1.3; the rest |W_ih| < 0.3.
- n2: self-weight 0.97 (memory), fed by n4 (+1.73), feeds back on n4 (−0.49);
  n4 self-weight −0.99 ⇒ comparator structure.

## Ground truth (Monte Carlo, validation only)
- acc = 0.9531186 ± 1.1e-6 (N = 2^35, RTX 3090, 6.5 min). `gt_2e35.npz`.
- Failures: 63% predict the 3rd largest, 20% predict the max, 16% other.
- Accuracy falls when S − 3rd is small (0.58 below 0.01) AND when M − S is large
  (0.92 at 1.5, 0.83 at 2.5) — homogeneity: a large M makes S look like background.
- Worst cells (M pos, S pos): (0,7) 0.849, (7,6) 0.900, (9,8) 0.900, (9,0) 0.904.

## Measured structure
- n4 active ⇔ x_t is a new running max: 98.5% agreement for t ≥ 3.
  No neuron tracks "new 2nd max" directly.
- "Lucky seed" claim is wrong: all 5 zoo h16/T10 seeds reach 88–95%
  (example model 95.3%, best). h8 fails (35–43%), h32 ≈ 99%.

## Mechanism (derived from weights, then checked)
- Activation-weighted contributions E|W_ij h_j| (not raw weights — "small" weights
  from large periphery neurons matter; cutting periphery→core drops acc to 45%).
- n2 = memory: n2' = 0.97 n2 + 1.73 n4; on a record of value a with n4 = 10.16a − 0.49 n2,
  n2' = 17.58 a + 0.12 n2. Measured median n2/runmax = 17.6 (IQR 16.6–18.4). ✔
- n4 = soft comparator vs n2. One-term threshold 0.49·17.58/10.16 = 0.848·runmax is
  WRONG (prediction of a break at vS/vM ≈ 0.85 falsified: accuracy is smooth in the
  ratio). Full recurrent drive averages −9.53/runmax ⇒ mean threshold ≈ 0.94, but
  context-dependent: P(n4 on) = 0.25 at x = 0.90·max, 0.90 at 0.98, 0.994 at 1.00.
  Main threshold movers after n2: n4 self (−1.0), n6 (+0.93), n8 (−0.76).
- n6, n7 (and n1, n8 with mixing) = relu(−11..13·x + 0.5..0.7·n2): gap-below-max signals.
- Fragility lives in the W_hh rows of n1, n2, n6, n7, n8 (row noise ×4 → 0.75–0.86),
  not n4 or the periphery (≈0.94).
- Readout splits into two mechanisms:
  * S first (reverse, acc 0.9557): S is a record (n4 fires 98.9%), M is the last n4
    event (99.8%), S is the previous n4 event (95.6%); net outputs the previous event
    92.8%. Readout is amplitude-weighted, not a hard rule: when a false record
    intervenes the net still gets 71% right. Error budget: 0.013 from intervening
    false records, 0.031 with the event chain correct (decode errors).
  * M first (forward, acc 0.9508): S is not a record (n4 at S only 12%, and those are
    worse: 0.90). Errors: 65% pick R, 23% pick M, 29% pick something before M.
    R after M is harder (0.947) than R before M (0.959).
- **Unified decomposition** (`probes/ab_candidates.py`): S = argmax(A, B) with
  A = max before M (the penultimate record), B = max after M. Error budget, as a
  fraction of all inputs (total 0.0468):
  * within S's side, picks the wrong element: 0.0238 (51%) — equal on both sides
    (0.0121 before M, 0.0117 after M);
  * A/B swap: 0.0118 (25%) — a clean noisy comparator, P(swap) ≈ Φ(−d/σ),
    d = |vA − vB|/vM, σ ≈ 0.02 near 0, heavier tail beyond;
  * predicts M (M/S near-tie, vS/vM > 0.975): 0.0095 (20%);
  * other side, not its best: 0.0017 (4%).
  With only one candidate side (M at 0 or 9) accuracy is lower (0.93 vs 0.958), so
  within-side selection, not the A/B comparison, is the weak step.
- Weight-level origin of the A trace: after M's record, n2 = 17.58 vM + 0.122·n2_old,
  where 0.122 = 0.97 − 1.73·0.49 — n2 keeps a ~12% trace of the previous record.
- Offset rule argmax_{j≠M}(x_j − β(j,m)·x_M) fitted to the net's outputs explains
  nothing (96.39% vs 96.23% agreement) ⇒ errors are context-dependent measurement
  noise, not position biases.

## Boundary-layer estimator (`boundary_layer.py`)
- 77% of errors are within 0.1·vM of a tie (53% within 0.02).
- vS < 0 (P = 11/1024): net fails 78% ⇒ 0.0084 of inputs, 18% of errors (non-tie mode;
  record mechanism needs positive values).
- First-order formula err_k ≈ ½ f_Dk(0) · E_tie[|m0|/|α|], f_D analytic, m0, α from the
  network at exact tie points (finite differences, width capped at 0.25):
  * M/S tie: 0.0098 (observed pred==M 0.0095; near-tie only 0.0079), median width 0.0054
  * S/R tie: 0.0310 (observed pred==R 0.0297; near-tie only 0.0289), median width 0.0145
  Two layers ≈ 0.041 of 0.047 total error.
- Variance per FLOP on the tie part: 2.8× better than plain MC (width average still
  sampled over tie configurations — not yet deductive). Including the residual by MC,
  the combined estimator is ≈ MC (residual dominates).
- Noisy models (unchanged code, `bl_noisy_eval.py`): corr(BL, true err) = 0.77;
  residual 0.018 ± 0.017 (vs 0.006 on the trained model) — noise creates non-tie
  failures (wrong-sign α, widths beyond the cap, third-element resolutions).

## Making the width deductive: attempts (all negative)
- **Additive background model** (`probes/additivity.py`, `additivity2.py`): cut-HDMR of the
  tie margin m0 over the background inputs (anchor = conditional median). R^2 first order:
  -1.2 (M/S tie), 0.29-0.38 (S/R tie); second order -1.0 / 0.59. Other anchors (0, -1, -3)
  no better. m0 is a high-order function of the background.
- **Record skeleton** (`probes/record_skeleton.py`): keep only background inputs that are
  running-max records, anchor the rest: R^2(m0) <= 0.03 (M/S), <= 0.54 (S/R). Non-record
  inputs carry much of the margin (via the gap-below-max neurons n1, n6, n7, n8).
- **Total Sobol indices** (`probes/sobol_groups.py`): background share of Var(m0) is 0.76
  (M/S) and 0.35 (S/R); per-position indices sum to 1.28 / 0.45 ⇒ strong interactions for
  M/S. Influence decays with time: inputs early in the sequence matter most (t=0..3 each
  ~0.18 for M/S, t=9 0.03) — the noise is accumulated in the memory, not injected at readout.
- Conclusion: the comparator noise that sets the tie width is not a sum of per-input
  effects or a function of the record chain; a deductive width needs a model of the full
  distribution of the memory state, which is what the Gaussian closure got wrong.

## Noisy models: where the extra error is (`probes/noisy_residual.py`)
- Mean over the 32 copies (trained in brackets): error 0.104 (0.047); predicts M 0.031
  (0.010); predicts R 0.054 (0.030); picks 4th or lower 0.020 (0.006); vS<0 errors 0.009
  (0.008, unchanged — a property of the input law).
- Errors within 0.02·vM of a tie 0.040 (0.027), within 0.1 0.085 (0.040): the tie layer
  widens with noise, it does not get replaced by a new mechanism.
- BL estimate vs its own target (pred M + pred R): mean bias +0.0017 but rms per-model
  error 0.0136 (corr 0.74) — first-order linearisation fails when widths grow. An rms bias of
  0.0136 equals MC with ~500 samples, so BL as is loses to MC beyond ~3 MFLOP.

## Exact line-integral identity (`line_integral.py`)
- Writing the k-th/(k+1)-th pair as c ± d/2 (Jacobian 1, phi(c+d/2)phi(c-d/2) =
  phi(c)^2 e^{-d^2/4}) gives the TOTAL error exactly:
  err = f_Dk(0) · E_tie[ ∫_0^∞ e^{-d²/4} 1{order kept} 1{wrong at d} dd ].
  Verified unbiased on the trained model and 4 noisy copies (|z| < 1.4, both k).
- As a variance reducer it loses: 0.07-0.15× MC per forward pass (21 points per line;
  the variance is in where the line's error set sits, not in the d-integral).

## Stratification ceiling (`probes/strat_gain.py`)
- Strata on (g1, g2, sign vS), Neyman allocation with the true per-stratum error: gain
  over MC 3.1× (trained), 2.0× mean / 1.45× min (noisy, 13 bins per gap). Practical gain
  is lower (allocation needs a pilot). This is the honest "beats MC" result available:
  mechanism-guided sampling, not deduction.

## Effective dimension of the memory state (`probes/state_dim.py`, `state_merge.py`, `state_merge_skel.py`)
Test of the atom-and-merge design (law of the normalised state z_t = h_t / max|x_<=t|).
- Linear PCA compression of z_t: 95% of variance in 3-6 components, 99% in 5-10, but
  continuing from the compressed state destroys the tie estimate (12 components at t=4:
  BL 0.039 vs 0.031). Variance explained is the wrong criterion: the margin needs ~0.5%
  precision (tie widths 0.005-0.015 of the max).
- Merge by nearest neighbour (real reachable states, cone respected), 65k atoms:
  * history BEFORE the tied pair compresses: tie after t=4, 8 PCA features → BL 0.0330
    vs 0.0333 (S/R) and 0.0090 vs 0.0092 (M/S); 4 features within 5%; t <= 2 exact with 2.
  * once a tied element is in the state it does not: even with the skeleton (positions, v,
    values above v) held exact and only background merged, nearest neighbour in all 16 dims
    gives BL 0.028 vs 0.010 (M/S) and 0.052 vs 0.029 (S/R) at 4k atoms per skeleton.
    Convergence in atoms is very slow: M/S error 0.036 → 0.022 from 2^10 to 2^16 atoms
    (~N^-0.12). Merging after BOTH tied elements is biased the other way (0.004 vs 0.009).
- Conclusion: the context before the tie is low-dimensional (4-8 features); the comparison
  itself needs the background-induced part of the state to ~1e-3 relative precision, and
  generic merging cannot reach that at any affordable atom count. A deductive estimator
  must treat the post-tie dynamics analytically (e.g. propagate the margin's sensitivity
  directions, not the state), which is the open piece.

## Sensitivity propagation after the first tied element (`probes/sensitivity_merge.py`, `kink_diag.py`)
- Merge at the first tied element (nearest of 4k atoms, skeleton exact), then correct the
  margin by the forward-mode linear response along the target's own future:
  m(z) ≈ m(z*) + J(z*)·(z − z*).
  * S/R tie: merged 0.0533 → linear 0.0321 (true 0.0291).
  * M/S tie: merged 0.0293 → linear 0.0201 (true 0.0101).
  * Second order identical to first: the network is piecewise linear, so all remaining
    error is ReLU switching between z* and z. (Caveat: the in-pair flag uses the true
    argmax; small leak, same for all variants.)
- Where the switches are: 100% of the linearisation error comes from targets with a switch
  (26% of targets M/S, 16% S/R). Error-weighted, 85-90% involve a switch AT the step of the
  second tied element, 40-60% at +1/+2 steps, ~0 before it. Neurons: n0, n4, n7, n9
  (0.4-0.5 each), n11, n6 (0.25-0.4) — several at once (single-neuron switches carry only
  13-25% of the error).
- Interpretation: the comparison step is a multi-neuron kink event. Before it, the state
  is a low-dimensional context (4-8 features) and the response is linear; at it, the
  decision is which polyhedral region (over ~6 neurons) the context offset lands in.
- What a deductive estimator would then need: per context atom, the law of the offset
  (e.g. Gaussian cell moments), the pre-activations at the comparison step as affine
  functions of it, and polyhedral integration over the ~6-neuron kink set at steps b..b+2
  (orthant probabilities — `gauss.orthant_prob` exists). Untested; the cost estimate
  (skeleton quadrature × atoms × post-tie input quadrature) is the open question.

## Step 3: polyhedral integration (`polyhedral.py`, `probes/region_count.py`, `freeze_periphery.py`)
- Region counts per tie skeleton, post-tie steps, 99% of mass: all 16 neurons 2748 (M/S) /
  424 (S/R); kink neurons {0,4,6,7,9,11} only: 81 / 6; n4 alone: 4 / 1. The blow-up is the
  periphery.
- Freezing the periphery gates per kink-pattern class (fitted on independent draws):
  kink set {0,4,5,6,7,8,9,11} gives M/S 0.0102 vs 0.0098, S/R 0.0281 vs 0.0273; per-skeleton
  freezing fails (0.0251 vs 0.0097). Inside each region the network is exactly linear.
- Integrator: region = kink-sign constraints + box u < v (+ linearised in-pair rule); margin
  affine, alpha constant per region; constraints ordered in time, so u_1..u_6 by a fixed
  Sobol lattice through interval CDFs (Genz) and u_7 by Gauss-Legendre.
- S/R tie, 48 sampled skeletons, ~330 regions each: polyhedral 0.0208 / MC of the same
  linear model 0.0207 / MC of the exact net 0.0211. Per skeleton rms relative error: integrator
  vs linear model 8-10% (qmc 64-256; 15% at qmc 16), linear model vs exact net 15%;
  the aggregate agrees to ~1% (errors partly cancel; ±2% expected from 48 skeletons).
- Cost (estimate, FLOPs): ~1e8 per skeleton (qmc 64, ~330 regions) ≈ 17k plain-MC samples,
  plus region discovery (still sampled: 4k draws). A deductive estimate also needs quadrature
  over skeletons (tied positions x M position x v x vM ≈ 1e4 nodes) ⇒ ~1e12 FLOPs for the
  S/R term alone, where plain MC would reach ~1.5e-5 on the total accuracy. The polyhedral
  estimate's bias (~1-3% of the tie term ≈ 3e-4-1e-3) is 20-60x larger. Verdict: the
  explanation is validated (the error IS a few hundred linear regions per skeleton over 8
  kink neurons), but as an estimator it is 2-3 orders of magnitude behind sampling in compute
  at equal error, before adding the M/S term (~10x more regions).

## Protocol fixes and ceilings (response to CRITIQUE.md)
- **Noise level** (`sigma_sweep.py`, 8 copies each, 2^21 samples): mean acc 0.948 (σ=0.001),
  0.936 (0.0025), 0.900 (0.005), 0.827 (0.01), 0.53 (0.02), 0.34 (0.05), 0.19 (0.1). σ = 0.005
  is the largest level that keeps the mechanism (acc 0.85-0.93, error doubled) and so
  "significantly alters accuracy" in ARC's sense. At σ ≥ 0.02 the model is destroyed.
- **Seeds 0-4** (`seeds.py`, `seeds_h16.npz`, 2^26 each): acc 0.910, 0.880, 0.949, 0.932,
  0.890. None is the 0.9531 example model.
- **MSE vs compute** (`mse_curves.py`, `mse_curves.png`, `.json`). rms error averaged over
  each model set; crossover = largest budget at which the estimator still beats MC.

  | set | BL rms bias | BL crossover | BL+r (trained residual as advice) | strat-prop gain | Neyman ceiling |
  |---|---|---|---|---|---|
  | trained | 0.0066 | 3 MFLOP | (circular) | 1.43× | 3.1× |
  | noisy σ=0.005 (32) | 0.0248 | 0.4 MFLOP | 0.021, 0.6 MFLOP | 1.30× | 1.88× |
  | seeds 0-4 | 0.0244 | 0.4 MFLOP | 0.021, 0.5 MFLOP | 1.31× | 1.92× |

  The critique's "3 MFLOP on noisy" used the rms error vs the tie-error target only (0.0136).
  Against the total error, BL's bias is 0.025, so the crossover is ~0.4 MFLOP (~70 samples).
  Advice from the trained model does not transfer: the residual is model-specific.
  The polyhedral S/R term sits at 1e12 FLOPs with bias 3e-4-1e-3, where MC is at ~1.5e-5.
- **Only unbiased estimator ahead of MC at every budget**: proportional stratification on
  (g1, g2, sign vS). The stratum probabilities come from the input law only. Gain 1.3×
  (noisy, seeds) and 1.4× (trained); oracle Neyman allocation 1.9× / 3.1×.
- **Conditioning ceiling vs feature cost** (`probes/cond_ceiling.py`, `.json`; pick-freeze,
  2^21 pairs). R2 = Var(E[y|Z])/Var(y), y = 1{wrong}. A CV or proportional stratification
  using Z gains at most 1/(1-R2), and only if E[g(Z)] is free.

  | Z | cost | trained | noisy (8) | seeds 0-4 |
  |---|---|---|---|---|
  | top-1 (pos, value of max) | 0 | 0.04 | 0.03 | 0.04 |
  | top-2 | 0 | 0.25 | 0.25-0.45 | 0.19-0.24 |
  | top-3 | 0 | 0.63 (2.7×) | 0.63-0.76 (2.7-4×) | 0.53-0.58 (2.1-2.4×) |
  | top-5 | 0 | 0.83 | 0.79-0.87 | 0.73-0.80 |
  | top-9 | 0 | 0.96 | 0.95-0.97 | 0.94-0.96 |
  | prefix x_<5 (state h_5) | 0.5 pass | 0.045 | 0.05-0.09 | 0.05-0.08 |
  | prefix x_<9 (state h_9) | 0.9 pass | 0.49 | 0.49-0.60 | 0.48-0.56 |

  - Input order statistics carry most of the predictable variance. The positions and
    values of the top 3 explain 53-76%, against 19-45% for the 13-bin gap strata.
  - The hidden state explains almost nothing until the last input. Conditioning on the
    forward state (the atom design) cannot be the source of a variance gain.
  - The headroom left is the gap between strat-prop (1.3×) and the top-3 ceiling (2-4×).
    Closing it needs an exact quadrature of E[g(top-3)] over the order-statistic law for a
    g fitted per copy, e.g. the BL formula per (positions, gaps). That would be an unbiased
    mechanism-guided estimator capped at about 2-4× MC. It has not been tried.

- **RQMC in order-statistic coordinates** (ag-critic's `probes/rqmc.py`; scaling check
  `probes/rqmc_scale.py`, 64 reps, gains ±~18%). Scrambled Sobol; values from Rényi
  exponential spacings; positions from a Lehmer code given the leading dimensions. It is
  unbiased: all 40 (model, N) means lie within |z| < 2 of the truth.

  | N | trained | noisy 0-3 | seeds 0-4 |
  |---|---|---|---|
  | 2^12 | 1.2× | 1.5-2.3× | 1.3-1.8× |
  | 2^15 | 1.9× | 1.6-3.1× | 1.4-2.5× |
  | 2^18 | 1.9× | 2.5-5.0× | 1.7-2.5× |
  | 2^20 | 2.4× | 2.3-4.1× | 1.9-2.5× |

  - The gain grows from 2^12 to 2^18 and then roughly levels off. On raw x the same Sobol
    gives 1.0-1.6×. The transform costs ~5% of a forward pass.
  - This is the best unbiased estimator so far, ahead of strat-prop (1.3×). It is
    mechanism-chosen sampling, not deduction.
- **Cap revision** (critique follow-up): 2-4× is the ceiling for Z = top-3 only. Finer
  zero-cost order-statistic features have higher R2: top-5 0.83 (~6×), top-9 0.96 (~25×).
  For gain that grows with budget, the open question is how fast cells of those features
  can be resolved, which is a rate. The prefix-state R2 also does not test the atom design;
  the relevant number would be R2(top-3 ∪ h_t) − R2(top-3), not measured.
- **Position-aware Neyman allocation** (`probes/strat_pos.py`, `.out`). Strata are gaps ×
  sign vS, optionally × the positions of the top values. The allocation is learned on 2^23
  pilot samples and scored on an independent 2^23. Every stratum keeps a 10% proportional
  floor, so the estimator is unbiased.

  | strata | trained | noisy 0-3 | seeds 0-4 |
  |---|---|---|---|
  | gaps only (13 bins) | 3.1× | 1.5-2.3× | 1.3-2.5× |
  | gaps (6 bins) + pos(M,S) | 4.5× | 3.6-4.8× | 2.3-3.5× |
  | gaps (6 bins) + pos(M,S,R) | 4.8× | 4.7-6.3× | 3.2-4.0× |

  - The 1/(1-R2) cap (2-4× for top-3) binds proportional allocation and control
    variates only. Neyman allocation can beat it by starving strata with near-zero error.
    20-57% of the mass sits in (positions, gaps) strata with no pilot errors.
  - Hard trimming of those strata (zero samples) would miss ~3e-4 of error mass, about MC's
    own rms at practical budgets, so a floor is needed.
  - The error rate by (pos M, pos S) is irregular, not a smooth decay with distance. On the
    trained model: M0/S7 15%; S just before M late in the sequence ~9-10% (M9/S8, M7/S6,
    M6/S5); most cells 2-4%. The pattern differs across seeds.
  - The gain is a ceiling: the 2^23 pilot costs more than it saves. It becomes an estimator
    only if per-stratum error rates come from the weights, e.g. BL tie widths per position
    pair. Not tried.

## Model-space map (`MAP.md`, `survey/`)
- Generic deduction (BnB over activation patterns, `survey/bnb.py`) vs MC on the best zoo
  model per (h, T): advantage ≥1e4 at T=2, 1.3-25 at T=3, 0.01-25 at T=4 (falling with h),
  0.01-1.2 at T=5-6. Crossover T ≈ 3-4; ~1e3 per step at h16. Grid stopped at n99 ≈ 2000.
- n99 grows like (hT)^T: width is the base, T the degree. Spare-capacity h32 models have the
  most regions and kinks at every T, so region count is not a packing signature.
- BnB is unbiased (h6T3, 8 scrambles, Q to 65536); spread ~Q^-0.6, so its lattice behaves
  close to MC and its edge is the one exactly integrated input.
- Verifier-style pruning (certify a region all-correct or all-wrong, as in α,β-CROWN) has
  nothing to prune before the last step. `survey/purity_ceiling.py` finds pure mass 0.000 at
  every depth < T-1 on all models tested (T=3-6), because the label depends on inputs not yet
  read. Node-count ceiling at the last step is 1.0-3.7×.
- hand10 (h22, T10): ≥2 positive inputs (mass 1013/1024) gives no errors in sampling;
  sampling only the rest gives ~950x variance reduction. Explanation-guided deduction escapes the
  region count; generic deduction does not.

## Against ARC's ask: ablation and surprise accounting (`probes/ablation.py`, `probes/surprise.py`)
- Ablation: the same unbiased designs on models we never explained. Gap strata (Neyman,
  held-out): trained 3.0x, a network-free blurred comparator matched to the same error
  3.2x, seeds 1.3-2.5x. Gaps + positions: trained 3.4x, random-weight RNN 5.5x.
  Order-statistic RQMC: random-weight RNN 2.8x. So every gain over MC so far comes from
  knowing the task (or from a pilot), not from understanding M(16,10). Exception: hand10,
  163x on gap strata, because sign(vS) nearly coincides with its explained event.
- Surprise accounting of the correctness bit c(x) (held-out, 2^22 inputs, 11 bits/param):

  | predictor | trained | seed0 | noisy0 | params |
  |---|---|---|---|---|
  | constant accuracy | 0.274 bits/input | 0.437 | 0.562 | 1 |
  | gaps logistic (task) | 61.9% of const | 76.4% | 83.7% | 8 |
  | gaps table (task, fitted) | 58.7% | 72.2% | 80.4% | 303 |
  | tie-layer (mechanism, widths from network margins at ties, not fitted to c) | 57.3% | 71.3% | 71.6% | ~540 |
  | tie-layer + gaps logistic | 55.1% | 69.2% | 68.4% | ~550 |
  | gaps x pos(M,S) table (fitted) | 50.8% | 62.9% | 41.9% | 4440 |

  The explanation carries real information beyond the task: label-free, it beats a
  303-cell table fitted to the labels, most clearly on the noisy copy. But a fitted
  position table beats it, and ~55% of the surprise is left (the multi-neuron kink
  residue). Caveat: the observation is c(x) and parameters are charged MDL bits; on that
  rule the weights themselves (432 params, ~4.8k bits) would explain everything, and the
  tie-layer predictor partly replays the network, so only explanation-vs-explanation
  comparisons are meaningful here.

- Label-free Neyman from the tie layer (`probes/strat_tie.py`): stratum error rates
  predicted by the tie-layer q(x), no pilot. Held-out gains: gaps 1.25-2.3x, gaps+pos(M,S)
  1.5-2.85x (trained 2.65x; perturbed mean 2.14x), between proportional (1.2-2.3x) and a
  2^23 pilot (2.3-4.8x). Flat in the number of ties from 2^12 up (cost ~5e4 passes, so it
  beats MC above ~1e5 samples, ~0.5 GFLOP). Control (`strat_tie_blur.py`, `_cal.out`): a
  network-free blurred comparator as the prior gives 2.07x (perturbed mean, gaps+pos) once
  its one blur width is set to each model's error rate, 1.71x with the trained model's
  width. So the explanation's allocation is worth what a one-parameter task prior is
  worth; what it adds is that one parameter (the error level), without labels.

- Kink combinations (`probes/kink_combos.py`): tie errors projected onto the true wall,
  on/off pattern of the 8 kink neurons from the later tied step on. Trained: S/R errors
  (0.030) need 37 / 634 / 3711 patterns for 50 / 90 / 99% of the mass, M/S (0.0095)
  38 / 366 / 1668; all 16 neurons 190 / 7093 (S/R). Within one ordered position pair the
  median is 24 (S/R) and 12 (M/S) patterns for 90%. The wall-to-error segment stays in
  one linear piece for 74% (S/R) and 60% (M/S) of errors. Medium compactness: ~10^3
  combinations for 90% against ~6e5 regions overall, a few dozen per position pair. The
  kink set is the trained model's; for seed0 the neuron labels differ, so only its
  all-16 counts (560 for 90% S/R) compare.

- Position dependence of the tie widths (`probes/pair_widths.py`, `strat_tie_g1.py`;
  `pair_widths_bsweep.out`, `strat_tie_ceiling.out`, `tie_gap_diag.out` from inline
  snippets). Per ordered pair, the measured tie widths predict the actual tie-error rate
  well (log corr 0.88-0.95, rms log ratio 0.2-0.6). The pair dependence does not factor
  into per-position terms (additive fit corr 0.4-0.9). Constant-background skeletons, one
  pass per pair and no sampling, get only corr 0.5-0.8 and 2-3x per-pair errors for any
  background level: the width is a nonlinear average over the context state, which a
  point skeleton misses. Allocating with the TRUE tie-error rates per stratum would give
  4.5x (oracle 4.65x), so the shortfall to the pilot is prediction quality, not the
  non-tie errors. Conditioning S/R widths on g1 adds nothing (2.14x -> 2.17x). The miss
  is in the far field, where Neyman weights are set by large mass at low rates:
  S/R errors at g2 > 0.3 are 2-10x under-predicted, M/S errors at g1 > 0.5 (network picks
  M when M >> S, 24% of M/S errors, mass 0.23) are 100x under-predicted, and the
  lognormal tail over-predicts M/S at g1 0.03-0.3 by ~5x. The far-field failure is not a
  boundary layer; the tie-layer explanation does not cover it.

## Reliability-engineering tools: active subspace and FORM skeletons (`probes/active_subspace.py`, `probes/form_bg.py`, `probes/form_abs.py`)
- Active subspace of the wall margin (C = E[grad_perp m grad_perp m^T] per ordered position
  pair at sampled ties): trace in the top 1/2/3 directions 0.54/0.73/0.84 (S/R) and
  0.45/0.68/0.80 (M/S), effective dimension 4-5 of 9. Intermediate: tabulable in 3-D per
  pair with ~20% of the margin variance lost, not a collapse to 1-2 directions. The wall
  gradient is only 8-14% of alpha in rms on the trained model (random net: 4-5x alpha).
- FORM: per skeleton (tie value v at 5-15 quantile nodes, other top value at 3-7 nodes,
  all ordered pairs, background at its truncated-normal MEAN), one forward + one gradient;
  the margin is linearised in the background, m ~ N(m0, s^2), s^2 = sum (dm/dx_i)^2 var_i.
  Per-pair shape (same scoring as pair_widths.py): S/R trained corr 0.81, rms log 0.37
  (constant-b skeleton 0.52 / 0.90; measured widths 0.88 / 0.24); M/S still 0.5-0.8 rms.
- ABSOLUTE boundary-layer estimate 0.5 f_D(0) E[|m0|/alpha] from the mean-background
  skeleton with no noise term, no labels, no random inputs, ~1e4 skeleton evaluations
  (`form_abs.out`, `form_abs_nodes.out`, 15x7 nodes):
    trained S/R 0.0304 (actual 0.0297), M/S 0.0082 (0.0095): S/R+M/S 0.0386 vs 0.0392
    seed0   S/R 0.0524 (0.0525),        M/S 0.0432 (0.0321): 0.0956 vs 0.0846 (+13%)
    noisy0  S/R 0.0701 (0.0631),        M/S 0.0363 (0.0298): 0.1064 vs 0.0929 (+15%)
  The noise term (FORM proper) does not help the level: +5-10% on trained, +40-130% on
  seed0; the permuted-order-statistics background is 2x high. The 'other' error type
  (neither R nor M, 17% of trained errors) is not covered by any tie skeleton.
  This is the closest a deductive estimate has come: the level comes out within 2-15%
  with the per-pair distribution still 2x wrong, i.e. the mean background captures the
  comparator's average behaviour and the pair structure is in the background x position
  interaction. At ~3e4 forward-pass equivalents, MC has sd 0.0012; the skeleton bias is
  0.0006 (trained) but 0.011-0.014 on the perturbed models, so MC still wins on average.
- Across all 10 harness models (`probes/form_abs_all.out`): the trained model's -1.5% is the
  best of 10, not typical. Relative error vs S/R+M/S ranges -20% to +44%, rms 19%,
  unchanged from 5x3 to 15x7 nodes (a model error, not quadrature). Against the TOTAL
  error (missing 'other' errors) bias is -31% to +20%. MSE over the harness: deduced
  4.6e-4 vs MC at the same cost (33,750 passes) 2.4e-6, so MC is ~200x better. The
  mean-background skeleton is an order-of-magnitude-correct deduction (within ~20%),
  not a competitive estimator.

## Evaluation harness
- σ = 0.005 (relative RMS, per matrix) noisy copies: accuracy 0.83–0.94 (mean 0.896).
  Justification and the seeds 0-4 set: see "Protocol fixes and ceilings".
  `noisy_sigma0.005.npz`, 2^30 samples each.

## Dead ends (with reasons)
1. **Exact polyhedral enumeration.** 1.65M distinct 160-bit activation patterns in
   4M samples, ~30% unseen mass; 99% of mass needs ~all of them. Even conditioned on
   (sign pattern, record set) the dominant pattern holds only 6% of mass.
   Activation patterns are not the right explanatory variable on Gaussian inputs
   (impulse-on-zero-background stability does not transfer).
   Narrower follow-up (`probes/sig_hybrid.py`): the ERROR mass is no more concentrated.
   Trained: 139k distinct signatures among 196k errors; 62% of held-out errors fall in a
   signature never seen in 2^22 samples. A hybrid (exact cone integrals for the top K
   error signatures + MC for the rest) has oracle gain, with the integrals free, of 1.13
   at K=10^3 and 1.50 at K=10^5 (noisy0 2.26, seed0 2.84 at 10^5), below plain
   gap stratification (~3x) at a cost of 10^5 polytope integrals.
2. **Gaussian closure of h_t (E0, `moment_prop.py`).** Fails at step 1: h_0 lies on
   two rays (sign of x_0). A moment-matched Gaussian lets the ray parameter go
   negative; e.g. n9's next pre-activation is ≈ −80|x_0| (never positive) but the
   Gaussian makes it fire 9% of the time. Splitting on sign(x_t) (Gaussian-sum, 2^8
   components) helps forward cases but reverse cases stay badly wrong.
   Lesson: the state lives on a positive cone; closures must respect it.

## Numerics
- `gauss.py`: Genz BVN (needed: ray geometry gives correlations ≈ ±1, where naive
  quadrature for Φ₂ breaks and second-moment matrices go non-PSD).
