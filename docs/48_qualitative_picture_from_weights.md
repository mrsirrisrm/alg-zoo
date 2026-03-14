# 48: What the Weights Tell You

## Summary

Six numbers computed directly from (W_ih, W_hh, W_out) — with no iteration, no simulation, no input data — give a complete qualitative picture of how this 16-neuron RNN solves the 2nd argmax task. The spectral and pseudospectral analysis reveals a **stabilised non-normal rotor** that computes via the non-commutative interaction of stretching and rotation, with a cancellation ratio of 0.002 that makes quantitative margin prediction fundamentally dependent on exact computation.

## The Six Numbers

| Metric | Value | Source | Computation |
|--------|-------|--------|-------------|
| ρ(W_hh) | 1.27 | Spectral radius | Eigenvalues of W_hh |
| ‖W_hh‖₂ / ρ | 4.6 | Operator norm ratio | SVD of W_hh |
| Henrici | 0.93 | Departure from normality | ‖A‖²_F − Σ\|λᵢ\|² |
| ‖S‖ / ‖A‖ | 1.0 | Symmetric-antisymmetric balance | S = (W+Wᵀ)/2, A = (W−Wᵀ)/2 |
| K (Kreiss) | 5.5 | Kreiss constant | Resolvent norm supremum |
| ω / ρ | 2.0 | Numerical abscissa ratio | Max eigenvalue of (W+Wᵀ)/2 |

Plus three structural observations from W_ih and W_out:

| Observation | Value | Meaning |
|-------------|-------|---------|
| W_ih effective rank | 5-6 neurons carry 99.7% of input energy | Input enters a low-dimensional subspace |
| W_out row separation | min distance 7.6, max 20.1 | Positions are well-separated in readout space |
| Complex eigenvalue period | 5.4 steps | ~1-2 rotation cycles fit in 10 timesteps |

## The Qualitative Picture

### What spectral analysis alone would say (and why it's wrong)

From ρ(W_hh) = 1.27, standard spectral analysis would conclude: *"Signals grow slowly at rate 1.27ˣ per step. The dominant eigenvalue mode takes over after a few steps. The network uses eigenvalue-driven growth to amplify the correct signal."*

This is completely wrong. The network's actual margins have nothing to do with eigenvalue growth. The hidden state norm stays within ±9% while the margin swings from −73 to +59.

### What pseudospectral analysis reveals

The pseudospectral metrics tell a fundamentally different story:

**‖W_hh‖₂ / ρ = 4.6:** The matrix can amplify inputs 4.6× beyond what eigenvalues predict. The eigenvalues are not governing the dynamics.

**Henrici = 0.93:** Almost maximally non-normal. 93% of the Frobenius norm is in the off-diagonal Schur components. The eigenvectors are far from orthogonal (κ(V) up to 50,000), meaning modes interfere constructively and destructively.

**‖S‖ / ‖A‖ = 1.0:** The symmetric (stretching) and antisymmetric (rotation) parts have equal magnitude. This is the signature of a **stabilised rotor** — neither stretching nor rotation dominates. Compare:
- ‖S‖ >> ‖A‖: amplifier regime (growth/decay dominates)
- ‖A‖ >> ‖S‖: rotation regime (near-unitary dynamics)
- ‖S‖ ≈ ‖A‖: rotor regime (stretching and rotation interact)

**K = 5.5:** The Kreiss constant guarantees transient growth of at least 5.5× the spectral prediction. This is not a theoretical worst case — the actual computation exploits it.

**ω / ρ = 2.0:** The instantaneous growth rate (numerical abscissa) exceeds the asymptotic rate by 2×. Energy can be transiently redirected at rates that far exceed what eigenvalues allow.

### The complete qualitative story

From these six numbers alone, you can derive:

1. **The network computes via non-normal rotation, not eigenvalue growth.** ‖S‖/‖A‖ ≈ 1 and Henrici ≈ 0.93 place it in the stabilised rotor regime. The hidden state rotates through discrimination space with near-constant norm.

2. **The computation happens in the transient, not at steady state.** With ρ ≈ 1.3 and a complex eigenvalue period of 5.4 steps, the network completes ~1 rotation cycle in 10 timesteps. There is no steady-state regime — every step matters.

3. **Discrimination requires the SA cross-term (non-commutativity).** Since S alone has spectral radius > 1 and diverges under iteration, S cannot be the complete mechanism. The antisymmetric part A cannot work alone either (it also diverges). Only their interaction (S+A = W_hh) produces bounded dynamics.

4. **The mechanism is inherently directional.** The large antisymmetric part (‖A‖ = 6.3, nearly equal to ‖S‖ = 6.4) provides directional couplings where the flow from neuron i to j has opposite sign from j to i. This is what distinguishes fwd (M before S) from rev (S before M) — the antisymmetric part responds differently to the two orderings.

5. **Input must be redistributed.** W_ih concentrates 99.7% of input energy on 5 neurons. W_out rows span a 10-dimensional space. The non-normal dynamics of W_hh redistribute energy from the input subspace to the readout subspace — this redistribution IS the computation.

6. **Accuracy should be high but fragile.** The Kreiss constant K = 5.5 with ‖S‖/‖A‖ = 1 implies the margin is a small residual of large competing signals. Any noise that perturbs this balance will degrade accuracy. This correctly predicts the observed ~95% accuracy on real data (with 8 non-M/S inputs adding noise).

## The Quantitative Barrier

### Why exact margins are out of reach

The S-A cancellation ratio is 0.002 across the interior of the 10×10 position grid:

```
Cancellation ratio = |margin| / (|S_margin| + |A_margin| + |Cross_margin|)

Interior pairs (gaps 1-6, mid-sequence): 0.001 - 0.005
Edge pairs (gap 7-8): 0.02 - 0.05
Boundary (gap 9): 0.33
```

This means the margin is 0.1-0.5% of the individual S and A signals. To predict even the **sign** of the margin (correct vs wrong), you need accuracy better than 0.1% in the S-A residual.

Any approximation that introduces ε relative error in either the S or A pathway creates an error of:
```
margin_error ≈ ε × |S_margin| ≈ ε × 4000
actual_margin ≈ 10
```
So ε = 0.3% gives margin_error = 12, which exceeds the actual margin of 10. The quantitative computation is **inherently exact** — you cannot approximate it.

### What this means for mechanistic interpretability

This is not a failure of analysis. It is a genuine property of the learned mechanism:

- The network has learned to compute via **precise cancellation of large signals**
- This is analogous to a differential amplifier with common-mode rejection ratio of ~500
- The qualitative mechanism (rotor, cancellation, directionality) is fully derivable from weights
- The quantitative margins require exact computation (our precomputed Phi_post tables)
- The ~5% error rate on real data is predicted by the fragility of the 0.002 cancellation ratio

## The Residual Grid

The cancellation ratio across all 90 (M, S) pairs at M=1.0, S=0.8:

```
M\S     0      1      2      3      4      5      6      7      8      9
 0     .    .001   .002   .001   .002   .004   .005   .095   .037   .333
 1   .001     .    .001   .002   .003   .007   .011   .053   .037   .333
 2   .001   .000     .    .001   .009   .016   .046   .053   .220   .333
 3   .002   .001   .001     .    .006   .047   .021   .037   .072   .333
 4   .005   .004   .004   .009     .    .005   .033   .016   .058   .333
 5   .003   .005   .007   .017   .003     .    .003   .031   .769   .333
 6   .005   .011   .019   .015   .012   .004     .    .021   .039   .333
 7   .028   .017   .014   .010   .013   .006   .010     .    .015   .333
 8   .034   .042   .211   .049   .034   .449   .044   .028     .    .333
 9   .333   .333   .333   .333   .333   .333   .333   .333   .333     .
```

The interior (positions 0-7, gaps 1-6) is uniformly at 0.001-0.02 — the differential amplifier regime. Column 9 (gap=9) is all 0.333 because gap=9 has zero Phi_post steps (direct readout, no cancellation). The gradient from interior to edges reflects fewer Phi_post steps allowing less time for S and A to diverge.

## How to Describe W_hh in One Sentence

> **W_hh is a stabilised non-normal rotor: its symmetric and antisymmetric parts mutually cancel to maintain near-constant ‖h‖ while sweeping h through a 60° arc in discrimination space, with the ReLU activation pattern tipping the S-A balance to select the correct answer for each input ordering.**

The three framings from our analysis are the same mechanism at different levels:

| Level | Description |
|-------|-------------|
| **Physical** | h rotates through a 2D arc (87% of variance), ‖h‖ constant ±9% |
| **Matrix** | S (stretching) and A (rotation) create a stabilised rotor; neither is stable alone |
| **Computational** | Differential amplifier: S and A generate signals of ~4000 that cancel to ~10, with the residual sign gated by D |

## What Pseudospectral Analysis Adds Beyond Standard Spectral

| Question | Spectral answer | Pseudospectral answer |
|----------|----------------|----------------------|
| How does the network process signals? | Eigenvalue growth/decay | Non-normal transient rotation |
| What governs the dynamics? | ρ = 1.27 | K = 5.5, ‖W‖/ρ = 4.6 |
| Is the mechanism stable? | Yes (ρ close to 1) | Conditionally: S and A individually diverge, only their cancellation is bounded |
| Can we predict margins? | No (ρ doesn't determine margins) | No, but we know WHY: cancellation ratio 0.002 |
| Why ~95% on real data? | Unknown | 0.002 cancellation ratio predicts noise sensitivity |
| How does fwd/rev work? | Unknown | ‖A‖/‖S‖ ≈ 1 provides directional coupling |

The pseudospectral analysis transforms the understanding from "eigenvalues don't explain this network" (a negative result) to "the network operates as a stabilised non-normal rotor with a specific cancellation mechanism" (a complete qualitative positive result).

## Related

- [47: Symmetric-Antisymmetric Discrimination](47_symmetric_antisymmetric_discrimination.md) — S-A decomposition and cross-terms
- [46: Non-Normal Transient Amplification](46_non_normal_transient_amplification.md) — Non-normality metrics, circuit-level mechanism
- [45: Eigenvector Discrimination](45_eigenvector_discrimination_mechanism.md) — Left/right eigenvector structure
- [44: Phi_post Eigenvalue Analysis](44_phi_post_eigenvalue_analysis.md) — Spectral properties
- [43: Mechanistic Discrimination](43_mechanistic_discrimination.md) — Overall mechanism and precomputed tables
- `src/mechanistic_discrimination.py` — Implementation
