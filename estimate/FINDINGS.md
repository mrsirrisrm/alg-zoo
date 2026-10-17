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

## Evaluation harness
- σ = 0.005 (relative RMS, per matrix) noisy copies: accuracy 0.86–0.93.
  `noisy_sigma0.005.npz`, 2^30 samples each.

## Dead ends (with reasons)
1. **Exact polyhedral enumeration.** 1.65M distinct 160-bit activation patterns in
   4M samples, ~30% unseen mass; 99% of mass needs ~all of them. Even conditioned on
   (sign pattern, record set) the dominant pattern holds only 6% of mass.
   Activation patterns are not the right explanatory variable on Gaussian inputs
   (impulse-on-zero-background stability does not transfer).
2. **Gaussian closure of h_t (E0, `moment_prop.py`).** Fails at step 1: h_0 lies on
   two rays (sign of x_0). A moment-matched Gaussian lets the ray parameter go
   negative; e.g. n9's next pre-activation is ≈ −80|x_0| (never positive) but the
   Gaussian makes it fire 9% of the time. Splitting on sign(x_t) (Gaussian-sum, 2^8
   components) helps forward cases but reverse cases stay badly wrong.
   Lesson: the state lives on a positive cone; closures must respect it.

## Numerics
- `gauss.py`: Genz BVN (needed: ray geometry gives correlations ≈ ±1, where naive
  quadrature for Φ₂ breaks and second-moment matrices go non-PSD).
