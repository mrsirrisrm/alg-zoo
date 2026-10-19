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
- [x] B5. (4-5x geomean at 2^17-2^19, 2.5-8x per model, unbiased; B4 not stacked) Stack B1-B4 into one estimator, `probes/best_unbiased.py`, and produce the
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

- [ ] F1. Settle the Sobol z. 2^28 truths for trained, noisy0-1, seed0-1 (`probes/truth28.py`,
      `.json`; the 2^26 seed truths have sd 3.5e-5, which is 1.5 z at 2^20). Then the
      scrambled-Sobol stream in 9-input order-statistic coordinates inside the B5 strata,
      24+ reps, prop and adaptive, z against both truths. Decide bias vs noise.
- [ ] F2. Stack Sobol into B5 if F1 is clean: `probes/best_sobol.py` = best_unbiased with a
      Sobol pool (positions on the leading dims, spacings next, x_9 unused since it is
      integrated). Report gain vs iid adaptive/rb on the same reps; add to mse_curves.png.
- [x] F3. (pos(R) nothing for adaptive at any budget, oracle ceiling +1.3x at 2^18-2^19; prior 16 is 1.2-1.4x at 2^19-2^20; B5 trained is 6-8.6x at 48 reps, the 12-rep 3.7-4.4x was a low draw) Three-position strata (pos(M,S,R) as a third refinement level under pos(M,S)) on
      the trained model, where adaptive sits at 4.4x against a 7.2x oracle. Expect <= 1.1x;
      the oracle gap is allocation learning, not strata resolution, so also report whether
      a stronger shrinkage prior closes any of it. -> `probes/strat_pos3.py`.
- [ ] F4. Prefix sharing stays a note unless F2 and F3 land: it needs tree-structured pools
      and conflicts with per-stratum allocation (realistic composite 1.1-1.2x).

## Order

A1 → B1 → A2/A3 (parallel, cheap) → B3/B4 → B2 → B5 → C1 → C2 → D1/D2 → D3 → C3 → E.
Skip C2 if B5 and C1 leave no budget; it only sharpens a conclusion already supported.
