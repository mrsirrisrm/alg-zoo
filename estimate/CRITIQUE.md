# Critique of the M(16,10) mechanistic-estimate work (session ag-critic, 2026-10-17)

This review is written to argue against the work. Its sources are EXPLAINER.md, FINDINGS.md, the ARC AlgZoo post, and two quick
experiments (`probes/great_circle.py`, plus one σ = 0.1 check).

## 0. Two things to fix in how the problem is stated

1. **The noise level is stated inconsistently.** The brief says the noisy copies are W + 0.1·rms·N. The harness
   actually uses σ = 0.005 (acc 0.83–0.94). At σ = 0.1 the model is destroyed: I measured acc 0.18–0.22 on two
   copies. ARC does not fix σ. The post says "enough to significantly alter the model's accuracy". It names
   averaging over training seeds (seeds 0–4) as the better but more expensive test. So σ is a choice we made, and the
   verdict depends on it:
   - at 0.005 the mechanism survives, and the noise mainly widens the tie layers;
   - at 0.1 there is no mechanism left to exploit.

   State σ explicitly, justify it, and report the 5 seeds as the main robustness test. FINDINGS already shows the
   seeds reach 88–95%.
2. **ARC's bar is a curve, not a point.** The matching-sampling principle says "at least as well as random sampling
   for any given computational budget". It also allows the estimator "suitable advice". This has two consequences:
   - The 1e12-FLOP comparison is the point *least* favourable to any biased estimator, because MC's MSE keeps
     falling like 1/C while a bias floor does not. Report MSE versus compute for every estimator, including BL and
     polyhedral. The honest crossover for BL on the noisy models is about 3 MFLOP (rms bias 0.0136 ≈ 500 samples).
     That is the real headline number today.
   - Under "advice", work done offline on the *trained* model (region discovery, choosing the kink set, freezing
     gates) arguably isn't charged at estimation time. Only what must be recomputed per noisy copy is charged. The
     cost accounting should split advice from per-model compute. This is a correction in your favour.
   - ARC also accepts **surprise accounting** as an alternative metric. Your structural explanation may score far
     better there than it does on MSE versus compute. It is worth one section: the bits of explanation versus the
     surprise of the 4.7% error given the estimate.

## 1. Have we taken the right approach?

**For explanation: yes.** These parts are sound and well tested:
- the exact outer layer over order statistics;
- the boundary-layer decomposition;
- localising the error to multi-neuron kinks at the comparison step.

The negative results hang together and point the same way: Gaussian closure, cut-HDMR, the Sobol indices, and the
N^-0.12 merge convergence.

**For estimation: the polyhedral step is the wrong comparison.** It does not show what the "2–3 orders" figure says
it shows.
- Within one skeleton, the polyhedral integrator spends about 1e8 FLOPs for 8–15% rms error. The same 1e8 FLOPs is
  about 17k forward passes, or about 4k sampled BL widths at a few passes each. That gives roughly a 1.5–3% error
  for a width CV of about 1. So the deductive inner step is ~5× worse than your *own* sampled inner estimator.
- The 2–3 orders therefore measure "deductive polyhedral integration on a 1e4-node grid", not "the explanation".
- The polyhedral step also isn't counted fully: region discovery and periphery fitting take 4096 samples per
  skeleton, about 2.4e11 FLOPs over 1e4 skeletons. This miscounts against you only if it is treated as advice.
- Net: the number is roughly right in size for *this* estimator, but it is not evidence about the model. Drop the
  polyhedral step as an estimator and keep it as validation of the explanation, which it does well.

**Targeting the error rather than the accuracy is fine.** The two are equivalent for MSE, since p̂ = 1 − ê. What it
makes clear is the precision needed. To match MC at 1 GFLOP you need the error to ~1% relative precision. At
1e12 FLOPs you need ~0.03%. On noisy models (err ≈ 0.10), matching MC at 1e8 FLOPs needs ~1% relative precision on
an error set that, by your own findings, is where the mechanism is messiest. That is the crux; see §3.

**Conditioning on something other than the skeleton** won't produce a smooth inner function. Your data already
answer this:
- the record skeleton gives R² ≤ 0.03;
- merging at the 2nd tied input fails at 1e-3 precision;
- the Sobol indices show the background carries 75% of Var(m0), accumulated early (t = 0..3).

Any conditioning variable that leaves the early background free leaves the variance in. The one alternative I would
still try is a backward value function. Define V_t(h) = P(correct | h_t, label-relevant running stats), with the
future inputs integrated out. V_t is smoothed by the Gaussian inputs, whereas the tie-conditioned margin m0 is not.
Fit it on the reachable 4–8-feature pre-tie manifold by regression (fitted value iteration, Longstaff–Schwartz style),
then compute p = E[V_b(h_b)] using your working forward atoms. This is a learned surrogate, not a deduction, but it is
the one framing that avoids conditioning on a measure-zero tie, and conditioning on that tie is what creates the 1e-3
precision requirement. I expect it to hit the same kink wall in V_b, but it is cheap to test.

## 2. Other tools

| Tool | Verdict |
|---|---|
| Exact path/arc integration using homogeneity + piecewise linearity (tested, `probes/great_circle.py`) | **Loses.** Rotation invariance: x(θ) = cosθ g1 + sinθ g2 is exactly N(0,I) for every θ, so integrating the correct-set along a circle gives an exact, unbiased Rao–Blackwellised estimator. A full circle gives 29× variance reduction but crosses ~230 linear pieces (~465 pass-equivalents). Efficiency vs MC is 0.06 (full circle) up to 0.46 (0.4° arcs), never ≥ 1, on trained and noisy models. It confirms the line-integral result and is independent of how coordinates are chosen: linear pieces are ~2 orders denser than decision-boundary crossings. |
| Mechanism-guided importance sampling / control variates | **Capped at about your Neyman ceiling.** IS on the gaps is the same as stratifying on the gaps. A CV whose expectation is known analytically must be a function of cheap features, and its gain is bounded by Var(E[1\|Z])/Var(1) — that is, by the stratification ceiling you measured: 3× trained, 1.5–2× noisy. To beat this you need features of the *background-induced* state, which cost a forward pass. **Cleanest next experiment:** measure the conditioning ceiling as a function of feature cost, e.g. (gaps, sign) → + record chain → + n2/runmax at tie time → + the state after k steps. That curve bounds every hybrid estimator and is the strongest "how far can it go" statement available. |
| Exact cone decomposition / tropical geometry | Dead for the reason in your Dead End 1 (1.65M patterns). The Gaussian measure of each cone is itself a Genz integral. Homogeneity only removes the radius, and the circle experiment shows it buys little. |
| SOS/LP bounds, Lyapunov invariants | These give certificates, not estimates. Bounds on a 5% error set at 1e-3 resolution through 10 steps of ReLU will be loose by orders of magnitude, which is ARC's own reason for heuristic over formal arguments. An invariant such as n2/runmax ∈ [16.6, 18.4] is nice explanation, but useless for MSE. |
| Cumulants / saddlepoint near the boundary | Requires m0 to be close to a sum of independent pieces. Your Sobol sum of 1.28 and negative cut-HDMR R² say it is not. Skip. |
| Mean-field / random-RNN theory | Only relevant if σ is large (0.1). Then the model is a perturbed random RNN, and mean-field Gaussian closures (which fail here because of the trained cone structure) become the natural tool. This is another reason the choice of σ matters. |
| Symbolic algebra (Sage) | There is nothing for it to act on. The difficulty is measure, not algebra. |

## 3. Have we taken it about as far as it goes? Mostly yes, but the claim should be sharpened

I think the conclusion is right in direction and slightly mis-stated in size and in what causes it.

- The "2–3 orders" figure is partly an artefact: it is the polyhedral estimator, measured at the compute point least
  favourable to it. The defensible claims are:
  - (a) no mechanism-derived estimator we found beats MC above about 3 MFLOP on the noisy models;
  - (b) hybrids are capped at about 3× (trained) and 1.5–2× (noisy) by the conditioning ceiling on cheap features;
  - (c) fully deductive estimators would need ~1% relative precision on an error set that lives in multi-neuron kink
    regions, which none of our state representations reach.
- **Model, not coordinates?** Two pieces of evidence say it is mostly the model:
  - the representation-independent circle test: kink density is far above boundary density, along arbitrary
    directions;
  - the Sobol finding: the comparator noise is accumulated from the early background, not injected at readout.

  A conjecture worth stating: in a capacity-limited model (h8 fails, h16 ≈ 95%, h32 ≈ 99%), the error set is, by
  construction, where the piecewise-linear comparator is coarse. Estimating the error rate to 1% is therefore
  explaining the *least* structured part of the model. A cheap test: repeat the region count at the comparison step
  and the circle efficiency on the h32 and h8 zoo models. If kink density at the decision step tracks hidden size
  while the error shrinks, the "explanation ≠ cheaper estimate" gap belongs to the model class and not to our
  variables.
- ARC itself only "roughly matches" sampling at 32 parameters (M(4,3)), and says matching at 432 would be a major
  breakthrough. A well-characterised negative with the ceiling curve above is a legitimate result. It should not be
  framed as a failure of the approach.

## Requested changes before anything goes to ARC
1. Fix the σ statement and add the seeds 0–4 evaluation.
2. Replace the single-point comparison with MSE-versus-compute curves (MC, BL, stratified, polyhedral), and split
   advice from per-model compute.
3. Add the feature-cost conditioning-ceiling curve.
4. Optionally, write a surprise-accounting section.

## Response (alg-zoo-10, same day)
Items 1-3 are done; see FINDINGS.md "Protocol fixes and ceilings" and EXPLAINER §11.
- σ: accepted. The brief's 0.1 was my error; the harness uses 0.005, now justified by a sweep. The seeds are added.
- Crossover: the BL crossover is ~0.4 MFLOP on noisy copies, not 3 MFLOP. 0.0136 was BL's
  error against its own tie target; against the total error it is 0.025. Advice (the
  trained residual) does not help.
- Conditioning ceiling: measured. The top-3 order statistics give R2 0.53-0.76, so the
  ceiling is 2-4×, a bit above the 13-bin Neyman figure. The forward state gives ~0
  until t = 9, which supports your point that re-conditioning on the state won't help.

## Follow-up (ag-critic, 2026-10-17 17:10)
- Crossover correction accepted. 0.0136 was measured against the tie target only; the right figure is 0.025, so the crossover is ~0.4 MFLOP.
- **"Hybrids capped at 2–4×" overstates the cap.** That ceiling holds only for Z = top-3. Your own table rises with
  the feature set at zero feature cost: top-5 R² 0.83 (~6×) and top-9 0.96 (~25×). The cap is set by how finely you
  can resolve Z for the compute you have, not by Z itself.
  - Proportional stratification (or post-stratification) on cells of Z has gain 1/(1−R²_cells), with exact cell
    probabilities from the order-statistic law and no g to fit.
  - Cells can be refined as N grows, so the gain should increase with budget.
  - The real question is how fast that growth is, which is a rate, not a ceiling.
- **The prefix-state R² is the wrong test of the atom design.** The atoms condition on the skeleton *and* the state.
  The relevant number is the joint R²(top-3 ∪ h_t-features) minus R²(top-3), not R²(h_t) alone. A prefix carries
  almost no information about whether later inputs will beat it, so its marginal R² ≈ 0 says little.
- **New probe: `probes/rqmc.py` (`rqmc.out`).** Scrambled Sobol in *order-statistic coordinates*: the values come from
  Rényi exponential spacings and the positions from a Lehmer code, with the permutation dimensions given the leading
  Sobol coordinates.
  - Variance gains over MC, from 256 reps (±~10%), at N = 2^12 and 2^15: trained 1.5–1.9×, noisy copies 2.0–2.9×.
  - The same Sobol on raw x gives only 1.0–1.6×.
  - The transform costs ~5% of a forward pass. It is unbiased, needs no fitting, and lets the mechanism pick the
    coordinates. It already beats strat-prop (1.3×) and sits close to the top-3 ceiling on the noisy copies.
  - Not yet tested: seeds 0–4, N > 2^15 (to see whether the gain grows), and stacking it with the g(top-3) control
    variate. Both are unbiased and should compose.
