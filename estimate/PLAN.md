# Plan: closing out the M(16,10) estimate (from 2026-10-19)

Goal: the best estimator we can build for M(16,10), an honest statement of why deduction
loses to sampling here, and a table placing the rest of the family on the same lenses.
Each step names its files, its check, and what it writes. Tick steps off in place.

Conventions: one commit per step (probe + .out + FINDINGS entry). Harness = trained,
32 noisy copies (σ = 0.005, `noisy_sigma0.005.npz`), zoo seeds 0-4 (`seeds_h16.npz`).
MC baseline cost 5,760 FLOPs per sample; report gain as MC compute for equal rms ÷ ours.

## A. Finish the perturbation result (copy harness) — started, uncommitted

- [x] A1. Commit `probes/weight_perturbation.py` + `.out` with a FINDINGS entry.
      Numbers to record: Q (GN-Fisher quadratic form) corr 0.947, LOO rms 0.0090,
      3.7e5 FLOPs/copy, crossover ~6 MFLOP; L (first-order logit shift, 65k pool)
      corr 0.996, rms 0.0024, 5.7e8 FLOPs/copy, loses to MC 2.4x there; flips 8.4% of
      inputs per copy, first-order criterion captures 83% in 15% of inputs, 95% in 43%.
- [x] A2. (out-of-sample c from fresh sigma 0.005 copies: harness rms 0.0126, not within 10% of LOO 0.0090; quadratic regime holds to 0.005, breaks by 0.01) Fit c out of sample. Regenerate copies at σ ∈ {0.0025, 0.005, 0.01}
      (reuse `noisy_models.py` with a different rng seed, 2^26 truth each is enough),
      fit c on those, score on the 32 harness copies. Check whether one c serves all
      three σ (quadratic regime) or drifts (cubic terms). → `probes/wp_calibrate.py`.
      Verify: harness rms with out-of-sample c within ~10% of the LOO 0.0090.
- [x] A3. (H is 99.7% W_hh; n7 0.32, n2 0.17, n8 0.13, n6 0.13, n1 0.09; top-5 eigvecs 0.79 of trace) Picture of H. Top 6 eigenvectors as weight maps; fraction of trace per neuron
      and per matrix; compare with `precision_blocks.out` (expect n2, n7, n8, n1, n6).
      Write-up line: "which copies are bad is set by the projection of the noise on
      ~5 stiff recurrent directions".
- [x] A4. (negative: CRN 1.0-1.3x, with margin strata 1.3-1.6x at one pass per sample; flips are one-directional so var(d) ~ var(y')) Q + residual by common random numbers. Estimate p(W') = p₀ − c·Q + E[y' − y]
      with the same inputs through both nets, and the optional trained-margin
      Neyman weighting (soft, floor 10%). Expect 1.2x from CRN alone, 2-3x with the
      weighting. Report its MSE-vs-compute curve next to MC. → `probes/wp_crn.py`.
- [x] A5. (`PROTOCOL_NOTE.md`, numbers updated after A2) Protocol note for ARC/Jacob: at σ = 0.005 the copies are perturbation theory
      around the trained model; anything with p₀ and second-order sensitivity wins at
      low budget. The seeds harness is the only honest test. Draft for Martin to send.

## B. Best unbiased estimator (the "as close as we can get" number)

- [x] B1. (32 reps: adaptive = oracle by 2^19-2^20; trained 5.5x, copies 3.4x, seeds 2.4x at 2^20; |z| <= 2.6) Adaptive Neyman, no separate pilot. Strata = gaps (6 bins) × sign vS ×
      pos(M,S) (and the pos(M,S,R) variant). Start proportional, re-allocate at each
      doubling from all samples so far, 10% proportional floor. Budgets 2^14 … 2^22.
      Score against truth over 64 reps; report gain vs MC at each budget and the gap to
      the 2^23-pilot ceiling (trained 4.5-4.8x, copies 3.6-6.3x, seeds 2.3-4.0x).
      → `probes/strat_adaptive.py`. Verify: unbiased (|z| < 2 across reps at every N).
- [x] B2. (6-rep smoke said negative; at 32 reps the Sobol stream inside strata is >= iid on 8/10 models, geomean 4.1x vs 3.0x at 2^20; seed0 z up to +4.2, so promising not established; B5 uses iid) RQMC-OS inside strata. Scrambled Sobol in order-statistic coordinates per
      stratum (reuse `probes/rqmc.py` transform). Report whether the gains compose
      (expect sub-multiplicative: strata already carry part of the gap structure).
- [x] B3. Exact last input. For each sampled x₀..₈, integrate x₉ exactly: h₉ is
      piecewise linear in the scalar x₉ (≤ 16 ReLU breakpoints), logits piecewise
      linear, argmax changes at their crossings; sum Φ-mass of correct intervals.
      Cost ~1.5 pass-equivalents. Expected variance factor R²(x<9) ≈ 0.5-0.6, net
      ~1.3x. Verify against MC on the trained model (unbiased, |z| < 2).
      → `probes/exact_last.py`.
- [x] B4. Prefix sharing. Tree-structured inputs sharing the first k ∈ {3, 5} steps
      across m ∈ {4, 8, 16} leaves. Measure design effect 1 + (m−1)·R²(prefix) and
      FLOP saving; pick the best (k, m). Expect ~1.3x. → `probes/prefix_tree.py`.
- [x] B5. (48 reps: 4.4-4.9x geomean at 2^17-2^19, 3-6x per model, intervals +-20%, unbiased; B4 not stacked) Stack B1-B4 into one estimator, `probes/best_unbiased.py`, and produce the
      MSE-vs-compute curve on all three harness sets (add to `mse_curves.png`).
      Target 5-10x MC. State plainly that it is task-level knowledge (ablation:
      random-weight RNN gets 5.5x from gaps+pos).

## C. Close the deductive route with a measured reason

- [x] C1. Copies' new failure modes. Census by mechanism (A/B swap, within-side pick,
      predicts M, "4th or lower", vS < 0) on 8 copies vs trained, as in
      `probes/ab_candidates.py`. Then classify the flips from A1's pool: which
      mechanism each first-order flip belongs to. Question: is the extra error one
      event? If yes, extend the tie-layer deduction and re-score its rms bias on
      copies (now 0.025; hoping ~0.01). → `probes/copy_failures.py`.
- [x] C2. (SKIPPED per the Order note: B5/C1/D1/D2 already support the closure; the level is deduced to ~20% by mean-background FORM and the per-input structure is shown to need 4-5 directions per pair) Active-subspace quadrature. Per ordered pair, Gauss-Hermite on the top-3
      margin directions (7³ nodes) × tie-value nodes × other-top nodes, S/R and M/S.
      ~1e7 passes. Compare level error with the mean-background FORM (rms 19% over 10
      models). Decision rule: if still > 10%, the deductive route is closed and the
      reason is the 20% of margin variance outside the subspace plus the far field.
      → `probes/as_quadrature.py`.
- [x] C3. Record the closure in EXPLAINER §16: the six-part argument (MC cost per
      sample; (hT)^T regions and no certified pruning before the last step; hidden
      state explains nothing until t=9; level is mean-field, per-input structure needs
      4-5 directions per pair and ~1e3 kink patterns at 1% precision; 1.3-1.9 kbit
      MDL floor vs ~100-bit algorithm; copies are perturbative, seeds are the test).

## D. The family table

Same four lenses on every T = 10 model: h8 (zoo seeds), h16 seeds 0-4 + trained,
h32, hand10. Lenses: (i) best unbiased gain vs MC (B5 estimator; gaps-only and
gaps+pos from `probes/ablation.py` already exist for most), (ii) FORM mean-background
level error (`probes/form_abs_all.py`), (iii) surprise explained by the tie layer
(`probes/surprise.py`), (iv) strip width proxy = background-gradient / α ratio
(`probes/bg_ratio.py`) and active-subspace effective dim.

- [x] D1. Missing cells: FORM and surprise on h8; FORM on h32; bg ratio on h8.
      Run with existing scripts; save `probes/family_*.out`.
- [x] D2. (seed0 yes, corr 0.93; h32 no, corr 0.19; hand10 no, loss has an offset: sigma 0.005 is non-perturbative for the near-perfect models) Q on h32 and hand10 copies (σ = 0.005 of their own weights, 16 copies each):
      does the quadratic-form prediction hold off M(16,10)? Also seed0 copies.
- [x] D3. `MAP.md` table: one row per model, columns = the four lenses + accuracy +
      n99 (where known). Narrative: strip width → 0 and error mass moving into an
      event of known mass (M(16,10) ratio 0.14, h32 0.04, hand10 0 and 950x); h8 as
      the failed end of the axis.

## E. Write-up and hygiene

- [x] E1. FINDINGS entries for the three uncommitted probes from 17-18 Oct
      (circle_rqmc, pieces_vs_size, exact_dims; committed in 8a1f3b4 without notes).
      Re-run the incomplete h3_T5 exact_dims sweep or mark it as not run.
- [x] E2. EXPLAINER §13 "against the original ask" refresh: headline numbers from B5,
      A1-A4, C2; retire superseded caps (2-4x) and the "pilot costs more than it
      saves" line.
- [x] E3. (`ARC_REPORT.md`) Final doc for ARC (reuse MAP.md + EXPLAINER): claim, best estimator curve,
      why deduction loses, family table, protocol note from A5.

## F. The tricks left after B5 (from 2026-10-20; expectations tempered: ~1.3x each at best)

- [x] F1. (clean: |z| <= 2.1 in 87 of 90 cells against 2^28 truths, no trend; the 2^26 seed0 truth is high by 4e-5, which was +1 of B2's +4.2) Settle the Sobol z. 2^28 truths for trained, noisy0-1, seed0-1 (`probes/truth28.py`,
      `.json`; the 2^26 seed truths have sd 3.5e-5, which is 1.5 z at 2^20). Then the
      scrambled-Sobol stream in 9-input order-statistic coordinates inside the B5 strata,
      24+ reps, prop and adaptive, z against both truths. Decide bias vs noise.
- [x] F2. (1.8-1.9x over iid adaptive at 2^18-2^19, 1.1-1.3x below; stacked geomean 6.7x at 2^18, trained 12x, copies 5.5-8.4x, seeds 4-6x; unbiased) Stack Sobol into B5 if F1 is clean: `probes/best_sobol.py` = best_unbiased with a
      Sobol pool (positions on the leading dims, spacings next, x_9 unused since it is
      integrated). Report gain vs iid adaptive/rb on the same reps; add to mse_curves.png.
- [x] F3. (pos(R) nothing for adaptive at any budget, oracle ceiling +1.3x at 2^18-2^19; prior 16 is 1.2-1.4x at 2^19-2^20; B5 trained is 6-8.6x at 48 reps, the 12-rep 3.7-4.4x was a low draw) Three-position strata (pos(M,S,R) as a third refinement level under pos(M,S)) on
      the trained model, where adaptive sits at 4.4x against a 7.2x oracle. Expect <= 1.1x;
      the oracle gap is allocation learning, not strata resolution, so also report whether
      a stronger shrinkage prior closes any of it. -> `probes/strat_pos3.py`.
- [x] F4. (left as a note: F3 did not land, and tree pools conflict with the Sobol stream as well as with per-stratum allocation; 1.2-1.6x standalone, ~1.1x composite) Prefix sharing stays a note unless F2 and F3 land: it needs tree-structured pools
      and conflicts with per-stratum allocation (realistic composite 1.1-1.2x).

## G. Training the explanation in (from 2026-10-20; a different question from the ARC one, reported as such)

Fine-tune M(16,10) so that a deductive estimator becomes cheap on it, keeping accuracy.
Not a claim about the original model: it tests whether the explanation-vs-estimate gap is
a property of how the model was trained. Pareto sweeps over the penalty weight, with the
gain of the relevant estimator as the only number that counts.

- [x] G0. (h32 is only modestly more estimable: 6.8x vs 3.4x on gap strata, 45% vs 57% surprise left, still a strip model with a boundary layer; so kink co-location at the decision step is not the property to train for, the pruning route is the one left) Cheap check before training: is h32, whose error flips sit on decision-step
      kinks (`decision_kinks.out`), markedly more estimable by the deductive routes than
      M(16,10)? Read off the family table in MAP.md.
- [x] G1. (survivor found on accuracy: lambda 0.07 at 2^25 and lambda 0.1 at 2^27 keep 0.9530 / 0.9527 with post-tie S/R regions 6.6x / 18x fewer and the periphery n3, n10-n15 frozen; the eight core neurons keep switching at every lambda, even at 85% accuracy; lambda 0 control lifts the zoo model to 0.9614) Prune the comparator by fine-tuning (`probes/finetune_prune.py`, `probes/finetune/`).
      Loss = task CE + lambda * switching penalty: within a tie skeleton (exact M/S or S/R
      tie, background redrawn below v as in `region_count.py`), each gate (t > first tied
      step, neuron j) should be deterministic; penalty per neuron g_j = sum_t p(1-p) with
      p = mean sigmoid(pre/tau) over backgrounds, grouped over neurons as sum_j sqrt(g_j)
      so that few neurons are left switching. Sweep lambda in {0, 1e-3, 3e-3, 1e-2, 3e-2,
      1e-1}; 2^25 sequences from the trained weights, AdamW, lr 2e-4 cosine.
      Per model: MC accuracy (2^26), hard per-neuron switching, region count at 99% mass
      (all 16 neurons, M/S and S/R, median over 48 skeletons), last-step kink stats.
      Verify: a lambda with accuracy within ~0.5 pt of 0.9531 and regions down >= 10x.
- [x] G2. (nothing: polyhedral regions span all ten steps and are the comparator's own history: learned kink set 402 vs 364 regions, exact regions 1101-1609 vs 3230 at 2^14 discovery draws with 72-85% of the mass found, poly vs MC-exact -6% vs -0.4%; sampling gain on group 0.1 is 4.7-6.2x iid, 7.6-9.4x Sobol, inside the trained intervals) Deductive gain on the survivors: `polyhedral.py --weights` with the learned
      kink set (S/R, 48 skeletons, same settings as the trained run) and the trained
      baseline re-run for a paired comparison: regions, poly vs MC-exact agreement, cost.
      Also 2^28 truth and the B5/Sobol sampling gain on the best survivor, to see whether
      the sampling route moves too.
- [x] G3. (Goodhart in the strong form: last-step kinks unmoved at lambda <= 0.3; at lambda 3-10 they reach h32's 54-58 per circle with accuracy 0.42 / 0.20 and enrichment 1.9-2.2) Control arm: the h32-style target (reward kinks at the last step,
      `--mode lastkink`), two lambdas. Expect the kink statistics to move and the gains not
      to (Goodhart); record either way.
- [ ] G4. (not attempted: G1 found a survivor on accuracy but no estimator got cheaper on it, so the premise is gone) Boundary-layer-aligned loss (errors only where the BL margin rule puts them):
      only if G1 fails to find a survivor, since it needs its own design.

## H. Activation patterns vs error, aligned to the tie step (from 2026-10-21; a description of the error set, not an estimator)

Are some ReLU patterns associated with error, and with the strip? The error is committed
early for many inputs (hazard entry: 60-66% at the arrival of the later confused value),
so a fixed-step table smears. Align to the event: b0 = the later position of the closer
tied pair, known from the input alone. Score every feature by held-out log loss of error
("explained" = fraction of the constant predictor's loss removed).

- [x] H1. (aligned d=0 explains 0.161 given o vs <= 0.08 at any fixed step; relay n4/n7/n8 -> n5/n9 -> n0 over d=0..2, n2 a state carrier; core 8 hold all of it; gap 0.26, gap x o x core d=0..2 0.42; inside the strip 0.16: the pattern is not the error) `probes/pattern_error.py`: 2^20 inputs on the trained model, h16 seed0, h8 best,
  h32 best. Explained by the full pattern at absolute t and at aligned d = t - b0; the
  orientation o (which of the pair arrived later) as a conditioning key; per-neuron bits
  and margins at the aligned steps; gap cell vs pattern vs both; the top patterns' error
  share; strip membership (gap < .03) from the margin vs the pattern.
- [x] H2. (entry written; h8 fails orientations not comparisons, o alone 0.18, gap 0.02; h32 peaks at t9 in absolute time and has an aligned ridge with n9 error-linked at every event step, capacity-limited at 29k patterns) FINDINGS entry and the reading: which neurons and which event steps carry the
  decision on M(16,10), whether the pattern adds anything beyond the gap, and the family
  contrast (h32 should peak at the last step, h8 nowhere).

## I. Advice as a structured weight distribution (from 2026-10-22; ARC's compression route)

ARC's suggested route from random to trained networks: advice = a structured weight
distribution the trained weights look typical of, plus an average-case estimator for it.
First, the empirical prerequisite on the trained example and seeds 0-4.

- [x] I1. (exact symmetries verified; core n4 0.94, n2 0.87, n1 0.86; n6/n7/n8 a shared role, not shared neurons; seeds 2/4 a second variant) `probes/seed_template.py`: align
  neurons up to permutation, positive rescaling and W_out shift by activation correlation.
  Check: accuracy unchanged by every transform.
- [x] I2. (sign-consistent large entries: W_hh 1/9, W_out 8/101) Per-entry agreement of the
  aligned weights.
- [x] I3. (template mean 0.114, draws 0.10-0.11: chance) Template distribution, mean +- s * sd.
  Check: do draws land in the models' 0.88-0.95?
- [x] I4/I5. (periphery replaced, shuffled or Gaussian: chance; dynamics alone and readout
  alone: chance) Core exact, periphery replaced.
- [ ] I6. Structured classes beyond independent entries: low-rank W_hh, or the hand10
  algorithm family with fitted parameters. Only worth it with a concrete class in hand.
- [x] I6a. (no shared Schur shape: rho 0.81-1.51, coupling 84-100% of the energy at rank 1 to 11;
  the comparator is backscattered, N*0.9 or *1.1 from the tie step costs 20-40 points; the
  readout is already more than Capon, omega 4-50, whitening at inference hurts) `probes/schur_modes.py`:
  Schur transmission/coupling split of W_hh and Capon/Bartlett view of the readout, from
  the tropical-pythia tooling.

## J. Insight tools after the ARC review (from 2026-10-22; insight first, FLOPs second)

Framing from the ARC material (sources and notes in FINDINGS "Is there a structured weight
distribution behind the seeds?" and the session review of 2026-10-22):
- "Mechanistic" = deductive: no random or pseudorandom sampling (Wu 2025 §1.2: a Sobol
  lattice is still inductive); low-dimensional numerical subroutines are fine (Remark 2.2).
  Our stratified/Sobol estimators are sampling, however well designed.
- Full understanding = total surprise of the explanation ~ bits of optimisation used to
  select the model (AlgZoo post; "Formal verification, heuristic explanations and surprise
  accounting"). This axis is FLOP-free and we have only measured one side of it.
- The sketching theorem (Hilton, halfspace note) bounds every deduction-projection
  estimator: projecting onto m modes costs at most the (m+1)th eigenvalue of the tail
  kernel E_f[f f^T]. Measured, it bounds a class, not one design.
- Train-and-explain is ARC's "inductive step"; the zoo ships checkpoints at 0, 2^20..2^29
  sequences (`n_seqs`) that no probe has used.
Dropped: further sampling designs, region enumeration, sampled covariances inside an
estimator (fail the deduction test or are at their ceiling: MAP, ARC_REPORT, I6a).

- [x] J1. (a, c done: basin bits 4-8 per parameter for every model, failed, trained or designed; hand10, a ~100-bit construction, costs 5.5 kbits, so basin volume measures precision, not selection; b not run, LLC is also local; precision-as-description framing withdrawn, FINDINGS) Bits of optimisation. -log2 of the prior mass of the basin {acc >= acc* - tol},
  tol 1e-3 and 1e-2, for M(16,10), seeds 0-4, h8, h32 (and hand10 as the zero-search
  reference). Three estimators, reported side by side:
  (a) Laplace: GN Hessian (reuse `weight_perturbation.py` / `precision.py`) with the exact
      symmetry directions (16 rescalings, 16 W_out column shifts) projected out and counted
      as free; basin widths sqrt(2 tol / (c lambda_i)) clipped at the prior width.
  (b) Local learning coefficient (singular learning theory, SGLD; devinterp-style): handles
      the degenerate directions the Laplace estimate clips by hand.
  (c) Direct check of (a) along the top 50 eigendirections: measured accuracy loss at the
      predicted half-width (the quadratic regime broke by sigma 0.01, A2).
  Prior: the zoo's training init is U(+-1/sqrt(h)) but trained W_ih sits at +-10, so the
  init has no mass on the basin. Report under (i) per-tensor Gaussian at the trained rms
  (scale-free, matches precision.py's tau = 1) and (ii) one global rms (used instead of the
  AdamW weight-decay prior, which has no clean scale under Adam).
  Check: (a) with prior (i) reproduces precision.py's eigenbasis floor (1.3-1.9 kbits)
  within the symmetry correction; (c) within 2x of (a) per direction.
  Reading to write: compare with the explanation side (algorithm ~100 bits; 8 numbers carry
  39 of 43 surprise points). If bits of optimisation ~ 1-2 kbits, the precision IS the
  explanation and M(16,10) is fully understood in ARC's sense with no estimator beating
  MC; if far larger, structure remains unfound. h32 and hand10 give the family contrast.
- [x] J2. (one reorganisation at 2^22-2^24 in every seed, then drift with consecutive correlation 0.94-1.00; precision 2.3-2.9 kbits by 2^24-2^26 then flat while accuracy gains 5-15 points; strip share of errors 0.05 -> 0.14-0.20) Train-and-explain on the checkpoints. M(16,10)'s zoo twin (seed of
  `zoo_2nd_argmax(16, 10)`, the trained example has no checkpoints) and seeds 0-4 at
  n_seqs 0, 2^20..2^29: accuracy, the I1 alignment of each checkpoint to its successor
  (role persistence: max circuit, comparator, periphery), the tie-layer FORM level and
  surprise left (existing probes), J1 bits per checkpoint, and where the coupling balance
  of I6a forms. Check: the final checkpoint reproduces the zoo model's accuracy.
  Reading: smooth drift with small per-step advice updates (ARC's inductive step is
  plausible here) vs role reorganisation (a concrete obstacle at 432 parameters).
- [ ] J3. Sketching bound in state space. At steps t = 3..9, the tail kernel
  K(z, z') = E_future[c(z, .) c(z', .)] over normalised states z = h_t / max|x_<=t| (the
  label-relevant past carried as the skeleton: positions and values of the running top
  two), eigen-decomposed on a reachable-state pool. Report eigenvalue decay and the m needed
  for 1e-3 relative error. Check: the bound must sit at or below the measured atom-merging
  error at equal m (FINDINGS "Effective dimension": N^-0.12), else the kernel is wrong.
  Family: h16 seeds, h32, hand10 (expect near rank one at the tie step).
- [ ] J4. Harmonic spectrum in input space. Energy of the correctness indicator on S^9 by
  spherical-harmonic degree (Gegenbauer projection, Hilton's halfspace basis), and the
  same for the label and the network-output indicators separately; degree D needed for
  1e-3 inner-product error and the term count C(10+D, 10). Across T = 2..10 at h16 and
  across h at T = 4. Check: predicts the BnB crossover (MAP: T ~ 3-4) without running BnB.
  After J1: J2's per-checkpoint bits measure how precision forms in training, not selection.
- [ ] J5. Algorithmic variables and a finite-gain hand family (the concrete I6 class).
  (a) Regress every neuron's activation on named variables (running max, 2nd max, time
  since each, current input, their ReLU differences); extend Goldowsky-Dill's n1/n6/n7/n8
  fit (R^2 0.992) to all 16. (b) hand10 with k finite comparator gains fitted to the
  per-(pair, gap) error curve, not the weights. Check: reproduces the strip statistics
  (tie widths, FORM level) of M(16,10) and the seeds within their harness spread.
  Count its total surprise for J1.
- [ ] J6 (only if J5 finds a small class). Approximate symmetry reduction: n4 against
  n1/n6/n7/n8 as abs-units; region count of the reduced net against the original.
- [ ] J7 (short note, alongside). Random-RNN base case for this architecture: the state
  after one step lies on two rays for any weights (scalar input), so ARC's Gaussian start
  fails at init. Run the BnB crossover (`survey/crossover.py`) on random-weight RNNs at
  T = 2..6 to separate architecture from training in MAP.

## Order

A1 → B1 → A2/A3 (parallel, cheap) → B3/B4 → B2 → B5 → C1 → C2 → D1/D2 → D3 → C3 → E.
Skip C2 if B5 and C1 leave no budget; it only sharpens a conclusion already supported.
G0 -> G1 -> G2 (-> G3 as control; G4 only if G1 fails).
H1 -> H2.
I1 -> I2 -> I3 -> I4/I5 -> I6a (I6 open).
J1 -> J2 -> J3/J4 -> J5 (-> J6); J7 alongside.
