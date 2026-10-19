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
- [ ] A4. Q + residual by common random numbers. Estimate p(W') = p₀ − c·Q + E[y' − y]
      with the same inputs through both nets, and the optional trained-margin
      Neyman weighting (soft, floor 10%). Expect 1.2x from CRN alone, 2-3x with the
      weighting. Report its MSE-vs-compute curve next to MC. → `probes/wp_crn.py`.
- [ ] A5. Protocol note for ARC/Jacob: at σ = 0.005 the copies are perturbation theory
      around the trained model; anything with p₀ and second-order sensitivity wins at
      low budget. The seeds harness is the only honest test. Draft for Martin to send.

## B. Best unbiased estimator (the "as close as we can get" number)

- [ ] B1. Adaptive Neyman, no separate pilot. Strata = gaps (6 bins) × sign vS ×
      pos(M,S) (and the pos(M,S,R) variant). Start proportional, re-allocate at each
      doubling from all samples so far, 10% proportional floor. Budgets 2^14 … 2^22.
      Score against truth over 64 reps; report gain vs MC at each budget and the gap to
      the 2^23-pilot ceiling (trained 4.5-4.8x, copies 3.6-6.3x, seeds 2.3-4.0x).
      → `probes/strat_adaptive.py`. Verify: unbiased (|z| < 2 across reps at every N).
- [x] B2. (negative: a thinned Sobol stream inside strata is worse than iid, strat_adaptive.out) RQMC-OS inside strata. Scrambled Sobol in order-statistic coordinates per
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
- [ ] B5. Stack B1-B4 into one estimator, `probes/best_unbiased.py`, and produce the
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
- [ ] C2. Active-subspace quadrature. Per ordered pair, Gauss-Hermite on the top-3
      margin directions (7³ nodes) × tie-value nodes × other-top nodes, S/R and M/S.
      ~1e7 passes. Compare level error with the mean-background FORM (rms 19% over 10
      models). Decision rule: if still > 10%, the deductive route is closed and the
      reason is the 20% of margin variance outside the subspace plus the far field.
      → `probes/as_quadrature.py`.
- [ ] C3. Record the closure in EXPLAINER §16: the six-part argument (MC cost per
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

- [ ] D1. Missing cells: FORM and surprise on h8; FORM on h32; bg ratio on h8.
      Run with existing scripts; save `probes/family_*.out`.
- [ ] D2. Q on h32 and hand10 copies (σ = 0.005 of their own weights, 16 copies each):
      does the quadratic-form prediction hold off M(16,10)? Also seed0 copies.
- [ ] D3. `MAP.md` table: one row per model, columns = the four lenses + accuracy +
      n99 (where known). Narrative: strip width → 0 and error mass moving into an
      event of known mass (M(16,10) ratio 0.14, h32 0.04, hand10 0 and 950x); h8 as
      the failed end of the axis.

## E. Write-up and hygiene

- [x] E1. FINDINGS entries for the three uncommitted probes from 17-18 Oct
      (circle_rqmc, pieces_vs_size, exact_dims; committed in 8a1f3b4 without notes).
      Re-run the incomplete h3_T5 exact_dims sweep or mark it as not run.
- [ ] E2. EXPLAINER §13 "against the original ask" refresh: headline numbers from B5,
      A1-A4, C2; retire superseded caps (2-4x) and the "pilot costs more than it
      saves" line.
- [ ] E3. Final doc for ARC (reuse MAP.md + EXPLAINER): claim, best estimator curve,
      why deduction loses, family table, protocol note from A5.

## Order

A1 → B1 → A2/A3 (parallel, cheap) → B3/B4 → B2 → B5 → C1 → C2 → D1/D2 → D3 → C3 → E.
Skip C2 if B5 and C1 leave no budget; it only sharpens a conclusion already supported.
