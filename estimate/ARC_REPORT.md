# M(16,10) second-argmax RNN: can a mechanistic estimate beat sampling? (report for ARC)

Everything below is measured on the harness of `PLAN.md`: the trained model (accuracy
0.9531), 32 copies with 0.5% rms weight noise (accuracy 0.83-0.94, truth 2^30 each) and
five independently trained seeds (accuracy 0.88-0.95, truth 2^26). One MC sample costs
5,760 FLOPs. Gains are MC compute for equal rms error divided by ours. Sources are the
probe files named in each paragraph; `FINDINGS.md` has the full entries.

## Claim

No deductive estimator of the accuracy beats Monte Carlo on M(16,10) above about 0.4
MFLOP, and we can say why in measured terms. The best estimator we have is unbiased
mechanism-guided sampling that needs no pilot and no fitted model; it gains 4.4-4.9x (geometric mean over the trained model, two copies and three seeds at 2^17-2^19, 48 reps; 3-6x per model), and 5-7x with a scrambled-Sobol stream inside the strata (24 reps; trained 10-12x) over
plain MC on the harness. That gain is task structure (gaps and positions of the top three
inputs), not the network's mechanism, and a random-weight RNN gets it too.

## The best estimator (`probes/best_unbiased.py`, curve in `mse_curves.png`)

Four unbiased parts stacked:

1. Strata = gap cells of the top three of the first nine inputs (6x6 bins of the two
   relative gaps) x sign of the second x positions of the top two. Stratum masses come
   from the input law, not from the model.
2. Adaptive Neyman allocation with no pilot: start proportional, at each doubling allocate
   the new samples from the variances seen so far (shrunk to the coarse parent), combine
   stage estimates with planned weights. It reaches the oracle allocation by 2^19-2^20
   samples and is unbiased (`probes/strat_adaptive.py`, 32 reps, |z| <= 2.6).
3. Exact integration of the tenth input: the hidden state is piecewise linear in it with
   at most 16 breakpoints, so P(correct | first nine) is a sum of Gaussian masses over
   at most 19 intervals, at 1.15 passes (`probes/exact_last.py`; variance ratio 0.49-0.60).
4. A scrambled-Sobol stream in order-statistic coordinates of the first nine inputs as the
   pool each stratum draws from (rank permutation on the leading dimensions, spacings next).
   Each scrambled point is marginally uniform, so every stratum mean stays unbiased.

Gains of parts 1-3 (48 reps, 16-84% intervals about +-20%) at 2^17-2^19: trained
6.1-6.3x, copies 3.1-5.4x, seeds 3.2-6.1x, geometric mean 4.4-4.9x against an
oracle-allocation ceiling of 5.1-5.2x. Part 4 (`probes/best_sobol.py`, 24 reps) is unbiased
against 2^28-sample truths and multiplies this by 1.8-1.9x at 2^18-2^19 (geometric mean 6.7x;
trained 12x, copies 5.5-8.4x, seeds 4-6x) and by 1.1-1.3x below, where strata hold too few
points for the stream to matter. Prefix sharing (`probes/prefix_tree.py`) is 1.2-1.6x on its
own and is not stacked: it needs tree-structured pools, which conflict with per-stratum
allocation and with the Sobol stream. A third position in the strata adds nothing for the
adaptive estimator and 1.3x to the oracle ceiling only (`probes/strat_pos3.py`).

## Why deduction loses here (EXPLAINER §16)

1. A sample is 5,760 FLOPs and nothing inside the network is cheaper than a 16-neuron
   step, so multilevel and surrogate tricks have no cost ratio to exploit.
2. Linear regions grow like (hT)^T. The branch-and-bound integrator beats MC at T = 2-3
   and loses ~1e3 per extra step; nothing can be pruned before the last step because the
   label depends on inputs not yet read.
3. The forward state explains 5% of the correctness variance after five inputs and 50%
   after nine. What is predictable sits in the inputs' order statistics, which is task
   knowledge.
4. The error level is mean-field (mean-background skeletons deduce the tie-error level to
   ~20% rms over the harness with no sampling, 8 numbers carry 39 of the 43 surprise
   points of the correctness bit), but the per-input structure needs 4-5 directions per
   position pair and ~1e3 kink patterns at 1% relative precision. The deductive rms
   stalls at 0.011-0.014 where MC is at 0.0017 for the same compute.
5. Holding the accuracy to 1e-3 needs 1.3-1.9 kbits of weight precision in the best
   linear basis, against ~100 bits for the algorithm; the precision sits in five neurons'
   recurrent weights where it cancels finely.
6. The noisy copies are perturbation theory around the trained model (next section).

## The copy protocol measures sensitivity, not estimation (`PROTOCOL_NOTE.md`)

At 0.5% weight noise a copy's accuracy loss is c * delta^T H delta with H the trained
model's Gauss-Newton Hessian: corr 0.95 with the truth, rms 0.013 with c fitted on 16
fresh copies, for 65 samples' worth of compute per copy. H is 99.7% recurrent weights,
five neurons carry 0.84 of its trace, and the top five eigen-directions carry 0.79. MC
needs ~3 MFLOP to match that rms, so anything holding the trained accuracy and a
second-order sensitivity beats MC on copies at low budget without understanding the
mechanism. The quadratic regime holds to sigma = 0.005 and breaks by 0.01. The
independently trained seeds are the honest test. The same sigma is non-perturbative on
h32 and hand10 (loss 0.09-0.11 from 0.99, not quadratic in delta), so sigma relative to
rms(W) is not a common regime across a family.

Suggestion: report independently trained models as the primary test; if copies are
kept, choose sigma where the quadratic prediction fails and report the
perturbation-theory baseline next to MC.

## The family under the same lenses (`MAP.md`)

| model | acc | best unbiased gain vs MC | FORM level (deduced / actual) | surprise left after tie layer | bg-gradient / tie slope |
|---|---|---|---|---|---|
| h8 (failed) | 0.427 | 2.7x | 0.22 / 0.48 (-54%) | 92% | 0.66, 1.02 |
| h16 seeds 0-4 | 0.88-0.95 | 3.2-6.1x stacked (seeds 0-2, 48 reps); 3.7-5.9x with the Sobol stream | rms 19% | 71% | 0.15-0.23 |
| M(16,10) | 0.953 | 6.1-6.3x stacked (48 reps); 10-12x with the Sobol stream | 0.0386 / 0.0392 (-2%) | 57% | 0.14, 0.08 |
| h32 | 0.994 | 6.8x (gaps) | 0.0048 / 0.0053 (-8%) | 45% | 0.04, 0.02 |
| hand10 | 0.990 | 163x (gaps), ~950x (its own event) | exact (one event) | 6% | 0.00 |

The ordering axis is how much the inputs other than the tied pair move the decision. At
1.0 the comparator is incoherent and nothing deduces; at 0 the error is one event of known
mass and deduction is exact. M(16,10) sits where the level is deducible and the per-input
structure is not, which is where the challenge was set.
