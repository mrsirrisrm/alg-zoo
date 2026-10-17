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
   (`gauss.orthant_prob`). — being tested.
4. Linear response within each combination for the remaining steps.
Open question: whether skeleton nodes × atoms × kink combinations stays under the sampling
budget at ~5e-4 accuracy.
