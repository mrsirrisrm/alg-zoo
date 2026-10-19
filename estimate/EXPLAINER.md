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
- Outer layer, exact in principle: order statistics of 10 Gaussians are analytic, so
  p = Σ_(pos M, pos S) ∫∫ density(vM, vS) · P(correct | skeleton), with the other 8 inputs
  iid truncated below vS. Deterministic quadrature is possible. In the runs so far the
  skeletons were sampled from this exact law (`sample_ties`), not integrated on a grid.
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
1. Skeleton quadrature (tied positions and values): exact in principle, not built. Runs
   so far sample skeletons from the exact tie law (48 in section 10).
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
- The top-3 order statistics explain 53-76% of the error variance, so control variates
  and proportional stratification on them gain at most ~2-4×. Neyman allocation is not
  bound by that: strata on gaps plus the positions of the top three give 3-6× held-out
  (`probes/strat_pos.py`). The pilot is not needed: allocating each doubling from the
  samples so far, with planned stage weights, reaches the oracle allocation by 2^19-2^20
  (trained 5.5×, copies 3.4×, seeds 2.4× at 2^20, unbiased; `probes/strat_adaptive.py`).
  The forward hidden state explains almost nothing until the last step, and the last
  step can be integrated exactly (`probes/exact_last.py`, variance ratio 0.49-0.60).
- RQMC in order-statistic coordinates (Sobol, positions given the leading dimensions)
  is unbiased and gains ~2× (seeds, trained) to ~3× (noisy) over MC at 2^20 samples.
  The gain grows with N up to ~2^18 and then levels off.
- Sharpened claim: (a) no deductive or biased mechanism-derived estimator beats MC
  above ~0.4 MFLOP on perturbed models; (b) mechanism-chosen unbiased sampling beats MC
  by ~2-3× at every budget tested. Position-aware allocation shows ~3-6× is available if
  per-stratum error rates can be predicted from the weights, which has not been tried; (c) full deduction would need ~1% relative
  precision on an error set that lives in multi-neuron kinks.

## 12. The model-space map (`MAP.md`)
- Generic deduction (enumerate linear regions, integrate each) beats sampling by ≥1e4 at
  T=2, narrowly at T=3, and loses from T≈4 on, by ~1e3 per extra step at h16.
- Regions grow like (hT)^T: sequence length compounds the nonlinearities; width only adds
  cuts. M(16,10) has ~6e5 regions covering 99% of the mass.
- A model whose explanation groups regions into events of known mass escapes this: hand10
  (h22, T10) gets ~950x by sampling only its 11/1024 degenerate stratum. M(16,10)'s error
  set (tie layers set by multi-neuron kinks) has not grouped that way under any attempt.

## 13. Against the original ask
- ARC's bar (a mechanistic estimate at least as good as sampling at every budget) is not
  met deductively; the unbiased designs that do beat MC (adaptive Neyman on gap and
  position strata, no pilot, 2.4-5.5x at 2^20; stacked with the exact last-input
  integral 4.4-4.9x (geometric mean over the trained model, two copies and three seeds at 2^17-2^19, 48 reps; 3-6x per model), and 5-7x with a scrambled-Sobol stream inside the strata (24 reps; trained 10-12x)) are task knowledge: a network-free blurred comparator gains as much
  from gap strata, a random-weight RNN as much from position strata and RQMC
  (`probes/ablation.py`). The perturbed copies are a separate matter: their accuracy is
  predicted from the trained model's Hessian without evaluating them (§16, part 6).
- Surprise accounting (`probes/surprise.py`): the tie-layer explanation, with widths from
  the network's margins at ties and nothing fitted to correctness, removes 43% of the
  surprise of the correctness bit on the trained model (29% seed0, 28% noisy0), more than
  any fitted task-level gap model, less than a 4440-cell position table. That is the one
  place the explanation, rather than the task, measurably pays.
- Tie-layer q(x) as a label-free Neyman allocation (`probes/strat_tie.py`): 2.1x mean on
  the perturbed models (2.65x trained), no pilot, from ~5e4 network passes. A blurred
  comparator whose single width is set to each model's error rate does as well (2.07x), so
  the mechanism's contribution to variance reduction reduces to predicting the error level.


## 14. Tools from outside the field (reliability engineering, survival analysis)
Four probes after the standard interpretability toolbox had stalled. Two measure how
compressible the error problem is, one is the closest a deductive estimate has come,
one is new mechanism.
- **Activation-signature hybrid** (`probes/sig_hybrid.py`): exact cone integrals for the
  top K error-carrying 160-bit signatures plus MC for the rest. The error mass is spread
  over ~1.4e5 signatures among 2e5 errors; 62% of held-out errors fall in a signature
  never seen. Oracle gain 1.5x at K = 1e5 with the integrals free; dead.
- **Active subspace of the wall margin** (`probes/active_subspace.py`): per ordered
  position pair, C = E[grad_perp m grad_perp m^T] at sampled ties has effective dimension
  4-5 of 9 (top 3 directions carry ~80-84%). Intermediate: tabulable in 3-D per pair at
  the cost of ~20% of the margin variance, not a collapse to one or two directions. The
  background gradient is only 8-14% of the tie slope alpha on the trained model (random
  net: 4-5x alpha), which is why the tied pair dominates the margin.
- **FORM / mean-background skeletons** (`probes/form_bg.py`, `form_abs.py`,
  `form_abs_all.py`): per skeleton (tie value and other top value at quantile nodes,
  every ordered pair, background at its truncated-normal mean), one forward and one
  gradient; 0.5 f_D(0) E[|m0|/alpha] is an ABSOLUTE boundary-layer error estimate with no
  labels, no random inputs and no fitted scale, from ~3e4 pass-equivalents. Trained:
  S/R + M/S 0.0386 vs actual 0.0392. Across the 10 harness models that was the best of
  10: relative error -20% to +44%, rms 19%, unchanged with finer quadrature, and the
  'other' error type (17% of trained errors) is outside any tie skeleton. MSE over the
  harness 4.6e-4 vs MC at the same cost 2.4e-6. The background-noise term (FORM proper)
  improves the per-pair S/R shape (rms log ratio 0.90 -> 0.37; measured widths 0.24) but
  not the level. The level is deducible to ~20%; the distribution over pairs is not.
- **Commit step and settling time** (`probes/hazard.py`, `probes/settling.py`): with a
  decoder-free criterion (benign completion from h_t), ~60-66% of errors are committed at
  the arrival of the later confused value, the rest lost within 1-2 steps after. At
  equal relative gap, late close comparisons are resolved 2-4x worse on the trained model
  (near-tie error by arrival step: S/R 0.155 -> 0.307, M/S 0.056 -> 0.244 over t = 1..9;
  gap conditioning excludes the task-level 'closer ties among more values'). Settling
  curves show why: a near M/S tie is 59-82% right immediately and 84-98% one benign step
  later, margin still growing; comparisons arriving at the last steps are read out
  unsettled. seed0 has no settling dynamic (flat, 55-70%), just a weaker comparator.
  This is the position dependence the measured widths carry and the skeletons miss; the
  gaps+pos design already exploits it, so it is mechanism, not estimator.

Net of the round: the three compressibility measures (signatures, active subspace,
FORM) agree with everything before them — the error LEVEL is a mean-field quantity and
is deducible; what remains (per-pair distribution, far field, 'other' errors) needs
4-5 directions per pair and is where sampling wins. The settling-time result is the
first explanation of WHY position matters, and it is specific to the trained model.

## 15. Weight precision as description length
A different question from the estimator work: not 'can we beat sampling' but 'why is
deduction expensive here'. The trained model needs ~10.5 bits per weight (4.5 kbits) to
hold its accuracy to 1e-3 under isotropic relative noise, and 14 bits/weight by plain
rounding; the per-neuron rescaling symmetry buys nothing. The Gauss-Newton Hessian in
relative coordinates says where that precision sits: sparse, in the recurrent weights of
five neurons (n2, n7, n8, n1, n6 carry 89% of the stiffness; W_out is soft), with the top
5 directions holding 79% of the trace. But the MDL floor over all linear bases
(reverse water-filling of the KL budget) is still 1.9 kbits at 1e-3 (1.3 kbits at 1e-2),
spread over ~350 directions: the top 100 directions take about half, the rest need ~3
bits each. So a better basis saves a factor of ~2.4 (if it were free to specify, which
it is not), not an order of magnitude. Against this: the 2nd-argmax algorithm is ~100
bits, and 8 numbers (two tie-strip widths, two background rates) capture 39 of the 43
surprise points of the correctness bit. The behaviour is nearly mean-field; the function
needs 1-2 kbits of fine cancellation. That gap is the honest statement of the project:
the error set is carried by precision that has no short description in any linear
coordinates we can name, and sampling is cheaper than any description that long.
`probes/precision.py`, `precision_blocks.out`, `precision_alloc.out`.

## 16. Why sampling wins on M(16,10), in six measured parts (PLAN C3)
The question was whether a mechanistic estimate of p can match random sampling at every
compute budget, averaged over perturbed copies of the weights. For M(16,10) the answer is
no, and the reasons are now measured rather than argued.
1. **The sampler is absurdly cheap.** One sample costs 5,760 FLOPs; at 1 GFLOP MC has an
   rms error of 5e-4. Any deduction competes with that per-sample price, and the network
   has nothing cheaper inside it: a 16-neuron step is already the unit of work, so
   multilevel and surrogate tricks have no cost ratio to exploit.
2. **Generic deduction is six steps past its crossover.** Linear regions grow like (hT)^T;
   the branch-and-bound integrator beats MC at T = 2-3 and loses ~1e3 per extra step at
   h = 16 (§12). Certified pruning has nothing to prune before the last step, because the
   label depends on inputs not yet read. More exact quadrature inside regions
   (Gauss-Legendre dims) does not move this.
3. **The forward state carries no usable information until the end.** The hidden state
   after 5 inputs explains 5% of the correctness variance, after 9 inputs 50%. Propagating
   a law of the state cannot produce variance reduction; what is predictable sits in the
   inputs' order statistics (top-3: 53-76%), and that is task knowledge: a random-weight
   RNN gets the same 5.5x from gaps-plus-positions strata (§13).
4. **The level is mean-field, the structure is not.** Mean-background skeletons deduce
   the tie-error level to ~20% rms over the harness with no sampling, and 8 numbers carry
   39 of the 43 surprise points of the correctness bit. What remains (per-pair widths,
   far-field errors, 'other' errors) needs 4-5 directions per position pair and ~1e3 kink
   patterns, at 1% relative precision on a 5% error. The deductive rms error stalls at
   0.011-0.014 on perturbed models, where MC is at 0.0017 for the same compute.
5. **The function is long to describe.** Holding the accuracy to 1e-3 needs 1.3-1.9 kbits
   of weight precision in the best linear basis (§15), against ~100 bits for the
   algorithm. Sampling is cheaper than any description that long, and the precision sits
   in five neurons' recurrent weights where it cancels finely.
6. **The copies are perturbation theory, the seeds are the test.** At sigma = 0.005 the
   per-copy accuracy loss is c * delta^T H delta with H the trained model's Gauss-Newton
   Hessian (corr 0.95, rms 0.013 with c fitted on fresh copies, 65 samples' worth of
   compute; H is 99.7% recurrent weights of five neurons), so any estimator with the
   trained accuracy as advice beats MC on copies up to ~3 MFLOP without understanding
   anything. The same sigma is already non-perturbative on h32 and hand10. On
   independently trained seeds the best unbiased sampling gains 3.2-6.1x (stacked, no pilot, seeds 0-2 at 2^17-2^19, 48 reps; 3.7-5.9x with the Sobol stream) and the best
   deduction is 13-15% off (`PROTOCOL_NOTE.md`).

What we can do is sample better. The stacked unbiased estimator (adaptive Neyman on
gap-and-position strata of the first nine inputs, exact integration of the tenth; §PLAN B5)
gains 4.4-4.9x (geometric mean over the trained model, two copies and three seeds at 2^17-2^19, 48 reps; 3-6x per model), and 5-7x with a scrambled-Sobol stream inside the strata (24 reps; trained 10-12x) over MC on the harness with no pilot and no fitted model. That gain is
task structure, and we say so.

Across the family (`MAP.md` table) the axis is the background-gradient / tie-slope ratio:
1.0 for the failed h8 model, 0.14 for M(16,10), 0.04 for h32, 0 for hand10. The deducible
fraction of the error and the explained surprise rise along it; M(16,10) sits where the
level is deducible and the per-input structure is not.
