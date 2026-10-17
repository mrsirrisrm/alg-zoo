# Toward a deductive accuracy estimate for M(16,10): where we are

## 1. Estimators
The unknown is one number, p = P(network output = position of the 2nd largest input), inputs
iid N(0,1). An estimator turns knowledge (weights, input law) plus compute into a guess p̂,
judged by MSE = bias² + variance as a function of compute.

Baseline: random sampling. Unbiased, variance p(1−p)/N, ~5,760 FLOPs per sample, treats the
network as a black box. At 1 GFLOP (~170k samples) its error is ~5e-4 (trained model) and
~7e-4 (noisy copies). ARC's bar: match that at equal compute, averaged over noisy copies
of the weights (so the answer cannot be hard-coded; the estimator must be a function of W).

## 2. Deductive estimators
Compute p from the weights and the input law without pushing random inputs through the
network. Analogy: output noise power of a filter from ∫|H(f)|² df instead of simulating.
For a ReLU network:
1. choose what to track (e.g. the law of a summary of the hidden state),
2. propagate its law forward deterministically (closed form or quadrature),
3. treat everything else as independent unless the explanation says otherwise,
4. add corrections for the dependencies the explanation identifies.
Error is pure bias; more compute = a finer explanation. The structure of the estimator IS
the explanation, which makes it falsifiable.

## 3. Shape of the estimator for this task
- Outer layer, exact: order statistics of 10 Gaussians are analytic, so
  p = Σ_(pos M, pos S) ∫∫ density(vM, vS) · P(correct | skeleton), with the other 8 inputs
  iid truncated below vS. Deterministic quadrature.
- Inner layer, hard: P(correct | skeleton) needs the law of the hidden state driven by the
  background inputs.
- Error terms: tie layers ≈ ½ f_D(0) · E[width] (~87% of errors; f_D(0) exact), and vS < 0
  (18% of errors; probability analytic, failure rate from the mechanism).

## 4. Why the Gaussian closure failed
- No biases ⇒ positive homogeneity: h₀ lies on two rays (sign of x₀), and every reachable
  state is a non-negative combination of a few directions — a union of polyhedral cones.
- A moment-matched Gaussian puts mass on unreachable states (negative ray amplitudes):
  a neuron whose pre-activation is ≈ −80|x₀| fires 9% of the time under the Gaussian.
- Accuracy is a threshold probability set by the 5% tail of the margin near zero, so even
  exact moments do not give it. The estimator must carry the shape near the boundary.
- Sign-split Gaussian mixtures (2^8 components) fixed forward cases, not reverse ones, and
  grow exponentially.

## 5. Cone-respecting design: atoms and merging
- Normalise by scale: z_t = h_t / max|x_≤t| (homogeneity ⇒ only ratios matter).
- Represent the law of z_t by atoms — real reachable states with weights — pushed through
  the exact RNN step with deterministic quadrature over the next input.
- Merge atoms that agree on a few features. The merging rule is the explanation.

## 6. What the dimension tests found
- Context before the tied pair is low-dimensional: merging into 4-8 features with 65k
  atoms reproduces the tie estimate (e.g. 0.0330 vs 0.0333).
- Once a tied input is in the state, merging fails at any affordable atom count (error
  shrinks ~N^-0.12): the comparison needs ~1e-3 relative precision (tie widths are
  0.5-1.5% of the max). Variance explained was the wrong criterion.

## 7. Sensitivity propagation
- Merge at the first tied input, then correct by the linear response along the future:
  m(z) ≈ m(z*) + J·(z − z*). S/R tie: 0.0533 → 0.0321 (true 0.0291); M/S: 0.0293 → 0.0201
  (true 0.0101). Second order adds nothing: the network is piecewise linear.
- All remaining error is ReLU switching: 85-90% at the step the second tied input arrives,
  40-60% at the next 1-2 steps, ~none before; mostly n0, n4, n7, n9, also n11, n6, usually
  several at once.

## 8. The structural claim
The 5% error is the probability that a 4-8-dimensional context state lands on the wrong
side of a multi-neuron kink set at the comparison step. Everything else is exact (input
order statistics) or smooth (linear response).

## 9. Proposed deductive estimator
1. Skeleton quadrature (tied positions and values) — exact.
2. Context atoms before the tie, each with a Gaussian offset — tested, works.
3. Comparison step: kink-neuron pre-activations are affine in the offset (and in later
   inputs), so each on/off combination's probability is a polyhedral Gaussian integral
   (`gauss.orthant_prob`). — tested for the S/R tie, see section 10.
4. Linear response within each combination for the remaining steps.
Open question: whether skeleton nodes × atoms × kink combinations stays under the sampling
budget at ~5e-4 accuracy.

## 10. Step 3 result
- The periphery causes the region blow-up; with 8 kink neurons {0,4,5,6,7,8,9,11} and the
  other gates frozen per region, the network is linear in each region and reproduces the
  tie widths (M/S 0.0102 vs 0.0098, S/R 0.0281 vs 0.0273).
- The polyhedral integrator reproduces the linear model (0.0208 vs 0.0207) and the exact
  net (0.0211) on 48 skeletons; per skeleton 8-15% rms errors that mostly cancel.
- Cost: ~1e8 FLOPs per skeleton, ~1e4 skeleton nodes ⇒ ~1e12 FLOPs for one tie term, with
  bias ~3e-4-1e-3. Sampling reaches ~1.5e-5 at that compute. As an explanation it holds; as
  an estimator it is 2-3 orders of magnitude behind sampling. Region discovery is still
  sampled.

## 11. After the critique: what the claim is
- Noise is σ = 0.005, the largest level that keeps the mechanism. At σ = 0.1 the
  accuracy is 0.19. ARC's preferred test, the zoo seeds 0-4 (acc 0.88-0.95), behaves
  like the noisy copies.
- MSE vs compute (`mse_curves.png`): the mechanistic BL estimator beats MC only below
  ~0.4 MFLOP (~70 samples) on noisy copies and seeds, and below 3 MFLOP on the trained
  model. The polyhedral step is far behind at its only budget.
- The one estimator ahead of MC at every budget is proportional stratification on the
  tie gaps: 1.3× (noisy, seeds), 1.4× (trained). Its stratum probabilities are deductive
  and its within-stratum error rates are sampled.
- Ceiling for any hybrid: the top-3 order statistics explain 53-76% of the error
  variance, so the gain is at most ~2-4×. The forward hidden state explains almost
  nothing until the last step, so state-conditioned hybrids gain less still.
- RQMC in order-statistic coordinates (Sobol, positions given the leading dimensions)
  is unbiased and gains ~2× (seeds, trained) to ~3× (noisy) over MC at 2^20 samples.
  The gain grows with N up to ~2^18 and then levels off.
- Sharpened claim: (a) no deductive or biased mechanism-derived estimator beats MC
  above ~0.4 MFLOP on perturbed models; (b) mechanism-chosen unbiased sampling beats MC
  by ~2-3× at every budget tested, and how far finer order-statistic structure can push
  that (top-5 R2 0.83, top-9 0.96) is a rate question still open; (c) full deduction would need ~1% relative
  precision on an error set that lives in multi-neuron kinks.

## 12. The model-space map (`MAP.md`)
- Generic deduction (enumerate linear regions, integrate each) beats sampling by ≥1e4 at
  T=2, narrowly at T=3, and loses from T≈4 on, by ~1e3 per extra step at h16.
- Regions grow like (hT)^T: sequence length compounds the nonlinearities; width only adds
  cuts. M(16,10) has ~6e5 regions covering 99% of the mass.
- A model whose explanation groups regions into events of known mass escapes this: hand10
  (h22, T10) gets ~950x by sampling only its 11/1024 degenerate stratum. M(16,10)'s error
  set (tie layers set by multi-neuron kinks) has not grouped that way under any attempt.
