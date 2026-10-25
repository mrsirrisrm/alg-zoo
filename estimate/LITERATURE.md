# Literature survey, October 2026

Survey done after PLAN I/J (FINDINGS "Summary of PLAN I and J"). Its question: what has
appeared since the AlgZoo post (Jan 2026) that bears on our bottom line, which is that
which near-tie inputs fail is seed-specific, and nothing we tried compresses it below the
weights above ~10^4 inputs. Ordered by how directly each item bears on that.

## ARC's own work since AlgZoo

- **Wu, Lecomte, Winer, Robinson, Hilton, Christiano, "Estimating the expected output of
  wide random MLPs more efficiently than sampling"** (arXiv 2605.05179, May 2026; blog
  "Mechanistic estimation for wide random MLPs"). Cumulant and Hermite propagation, layer
  by layer, with no samples. Provably beats MC at large enough width for fixed depth. Worse
  in depth. 4x256 ReLU MLPs beat MC over 7 orders of FLOPs, more so in the tails. They
  state that large depth, narrow nets and trained nets are open, and that trained nets
  "seem likely to require access to the training trajectory". M(16,10) is narrow
  (h = 16) and deep (T = 10), so it sits in their open regime. Our Gaussian moment-prop
  failure at the h0 sign split (Oct 17) is the narrow-width failure of the same
  closure. Code: `mlp_cumulant_propagation`.
- **ARC White-Box Estimation Challenge** (AIcrowd, announced 2 Jun 2026; Phase 2 closed
  17 Oct; no official winners found yet). Expected activations of random ReLU MLPs under
  Gaussian input, CPU budget. Two public write-ups are relevant (self-reported, not
  verified):
  - Yan Su (#336537): exact unaugmented K=4 cumulants at 1024x16 with factored transport
    storage, 11.8x below ARC's augmented K=3. **The gain tracks n/L: 10-12x at n/L = 64
    and 2-4x at n/L = 8.** M(16,10) is at n/L = 1.6. Their "downstream readout
    observability" point (a carried cumulant is worth what later layers read from it)
    is our tail-guided projection and readout result (I6a) seen from the MLP side.
  - #330001: the 64 fitted constants in a mean propagator absorb **accumulated drift,
    not one-step skew** (the propagated residual correlates -0.05 with the Edgeworth
    term, the one-step residual 0.975). This matches what our stalled deductive rms
    showed: one-step error compounds over steps.
- **"Mechanistic estimation for expectations of random products"** (Hilton, Robinson,
  Neyman, Christiano, Winer, Lecomte, Wu, Wu; May 2026). Halfspace intersections, random
  #3-SAT and {0,1}-permanents by deduction-projection with L² sketching, plus a
  random-walk/Fourier method (Winer). They match sampling up to log factors in every case
  but the permanent, which is "further behind". The sketching needs symmetry (Gelfand
  pairs) to diagonalise E[ffᵀ]. We already used the sketching bound (J3) and the
  halfspace basis (J4). The new point is that the permanent, the case with S_n-valued
  inputs, is their hard case. Our task's structure is also order statistics.
- **Winer, "A Mike's-Eye View of ARC's Research"** (Jun 2026). Matching Sampling
  Principle, cumulant propagation as the MLP result. Structure is "best communicated
  through resource-bounded compression" (sophistication). Training-process monitoring is
  "necessary". He calls the structure-to-advice step "half-baked" and says good toy models
  would be a major win. That is what our negative result speaks to.
- **Christiano, "Returning to ARC"** (AF, 4 Aug 2026). Two bets: every phenomenon has an
  explanation, and finding explanations is not much harder than finding phenomena. He
  puts the second below 50-50. Our result is a data point on the second bet.
- **Matolcsi, "Obstacles in ARC's agenda"** (AF sequence, Apr 2025; ex-ARC Theory
  Oct 2023 to Jan 2025). Finding explanations: the obstacles are empirical regularities
  (brute facts versus structure that hides deception), capacity allocation, dependence on
  the outside world, teleology, and when to start explaining. His point 7 is directly
  about us: "explaining small networks on tasks like second argmax sidesteps empirical
  regularities, which are the main worry". LPE: the only results are an independence
  method that loses to importance sampling (Wu and Hilton 2024) and a toy theorem; the
  fallback "match what the model knows" he could not pin down in a year at ARC. Replies:
  Hilton (May 2025) agrees the best case is a formal framework with MAD/LPE as tests;
  **Christiano (Aug 2026): ARC now holds that purely computational systems can be fully
  explained, and capacity allocation is handled by a budget-indexed family of
  estimators, not by choosing what to explain.** M(16,10) is a purely computational
  system, so it is a clean test of the first claim with none of Matolcsi's caveats.
- **AlgZoo comment thread** (LW, Jan to Aug 2026). Goldowsky-Dill (28 Jan): neurons
  1, 6, 7, 8 are linear in running max, running second, and lag of the max, R² 0.992
  (gist NixGD/2651be79). That is a subset of our J5a named-variable fit, found
  independently. Hilton's ask to us (Feb): "turn this argument into a mechanistic
  estimate of the model's accuracy ... deducing correlations from the weights rather than
  only observing them". phoenix (6 Aug): accuracy is exactly computable from the weights
  so estimating it is a weaker goal (true at 2^35 FLOPs; the point of the exercise is the
  FLOP axis, which they miss). **Hilton (17 Aug 2026), asked about progress: knows of no
  effort on these models beyond the thread, progress "fairly marginal", "not much closer
  to succeeding at the challenge".** So as of August the field's state on M(16,10) was
  our Feb gist; ARC_REPORT.md is the only measured negative result.

## Separating structure from noise (most actionable)

- **Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson, "From Entropy to Epiplexity"** (arXiv
  2601.03220, Jan/Mar 2026; cited in Winer's post). Epiplexity S_T is the size of the
  best time-bounded program. Time-bounded entropy H_T is the residual per-sample bits.
  Pseudorandom content counts as entropy, not structure. Two ways to estimate it:
  prequential (area under the training loss curve above the final loss) and requential
  (a student trained on teacher-checkpoint samples, code = sum of KL). This is the formal
  version of our open question. If the seed-specific near-tie failure pattern is
  time-bounded entropy, our negative result says "that is not structure", not "we failed
  to find it". It is cheap here: 432 parameters, zoo checkpoints (J2).

## Compression and proofs as the measure of understanding (pre-2026, not cited in repo)

- **Gross, Agrawal, Chan et al., "Compact Proofs of Model Performance via Mechanistic
  Interpretability"** (NeurIPS 2024, arXiv 2406.11779). Max-of-K transformers. The named
  obstacle is "compounding structureless errors", which is our residual from the named
  variables (J5a).
- **Yip, Agrawal, Chan, Gross, "Modular addition without black-boxes"** (arXiv
  2412.03773). Infinite-width approximation of a ReLU MLP as quadrature gives a bound in
  time linear in parameters. Same move as our mean-background skeleton.
- **Wu, Jaburi, Drori, Gross, "Unifying and Verifying Mechanistic Interpretations"**
  (arXiv 2410.07476). No 2025-26 follow-ups from this group found.
- **Ayonrinde, Pearce, Sharkey, "Interpretability as Compression"** (arXiv 2410.11179).
  MDL for SAEs. Background for our surprise-accounting comparisons.

## Weight-space and training-dynamics tools

- **Baker, Wang, Hoogland, Murfet, "Structural Inference ... with Susceptibilities"**
  (arXiv 2504.18274, ICLR 2026) and **"Patterning: The Dual of Interpretability"** (arXiv
  2601.13548). Linear response of component observables to data shifts, via local SGLD.
  Patterning chooses which algorithm is learned by targeting each solution's LLC. We
  skipped the LLC in J1. Susceptibility to near-tie data is a direct probe of which
  weights carry the near-tie pattern.
- **Stochastic Parameter Decomposition** (Bushnaq, Braun, Sharkey et al., arXiv
  2506.20790). MDL-flavoured decomposition of weights into mechanisms. I-results suggest
  the comparator will not split cleanly. This would test that from the other side.

## Exact / probabilistic verification

- **Soltanalian, "Random Parameter Noise Does Not Make Exact ReLU Verification Easy"**
  (arXiv 2607.14375, Jul 2026). Under NP ⊄ BPP there is no smoothed-polynomial exact
  verifier, even for one hidden layer with Gaussian weight noise. This is the theory side
  of J7 (random weights are as hard as trained ones) and of the noisy copies.
- **Kofnov, Kapla, Bartocci, Bura** (arXiv 2502.11672). Exact, converging upper and
  lower CDF bounds on network output under Gaussian inputs. An alternative to our
  polyhedral BnB, and sound rather than estimated.
- **PREMAP** (arXiv 2408.09262). Preimage under-approximation with exact volume, used
  only when an MC estimate crosses the threshold.

## Not found

- No 2026 paper on mechanistic estimation for *trained* networks. ARC say it is the next
  step and have published nothing on it yet.
- Nobody else working on M(16,10) in public apart from our gist and Goldowsky-Dill's
  notebook, confirmed by Hilton in Aug 2026.
- MINAR (arXiv 2602.21442) is GNN circuit discovery, not relevant.

## Candidate next steps

1. **Epiplexity of the near-tie pattern** (done: PLAN K1, FINDINGS "Epiplexity of the
   near-tie pattern"; outcome: entropy for SGD on the same architecture outside the
   basin to 2^28, slow structure for a 4-8x wider student at 4-16x the training's
   FLOPs, which recovers two thirds of the teacher's errors from I/O pairs). Requential code for M(16,10) from zoo
   checkpoints. Compare S_T against the ~116-bit family and the 2.4-3.2 kbit weights.
   Then test whether the per-input near-tie correctness bit is predictable by any
   time-bounded learner given the family, or behaves like H_T. Decides whether our
   bottom line is "incompressible structure" or "noise".
2. **ARC's cumulant propagator on unrolled RNNs** (done: PLAN K2, FINDINGS "ARC's cumulant
   propagation on the unrolled RNN"; outcome: their base case holds here, 30x at T = 10 on
   random weights, and fails on trained weights). Run `mlp_cumulant_propagation`
   (K=3/K=4) on the random-init RNN (J7) at h16, T = 2..10. Widen h to find the crossover
   in h/T, and compare with Yan Su's n/L scaling. This puts our "architectural
   difficulty" claim on ARC's own axis.
3. **Write to Jacob.** Done reading. ARC_REPORT.md answers his Feb ask with a measured
   negative, lands in the regime (narrow, deep, trained) his May paper names as open,
   and is the kind of toy case Winer says would be "a major win" either way. Frame it
   against Christiano's Aug 2026 claim that purely computational systems can be fully
   explained: here is one with 432 parameters where the per-input error pattern has
   resisted every compression tried.
4. (Lower) Susceptibilities to near-tie data shifts. SPD on M(16,10).
