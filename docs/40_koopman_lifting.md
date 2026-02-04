# 40: Koopman Lifting — Linearizing the ReLU Dynamics

## Motivation

We have two well-developed views of M₁₆,₁₀:

- **Eigenvalue analysis**: W_hh has complex eigenvalues that create rotation, real eigenvalues > 1 that cause growth. This explains the phase wheel.
- **Tropical geometry**: 38 cells at t=9, discrete activation patterns, piecewise-affine regions. This explains how ReLU bounds the dynamics and creates the offset.

The first is linear but ignores ReLU. The second captures ReLU exactly but involves discrete combinatorics. Both are incomplete alone — linear theory gives 6.7% discrimination; tropical analysis requires enumerating all cell sequences.

**Koopman lifting** offers a way to absorb the nonlinearity of ReLU into a linear framework, without ignoring it.

## The Core Idea

A nonlinear dynamical system can become linear if you look at it in a higher-dimensional space.

Our RNN has nonlinear dynamics:

```
h[t+1] = ReLU(W_hh @ h[t] + W_ih * x[t])
```

Instead of tracking the 16D hidden state directly, we track **functions of h** — called observables:

```
g(h) = [h, h², ...]
```

The map `h → h'` is nonlinear. But `g(h) → g(h')` can be approximately **linear**:

```
g(h[t+1]) ≈ K @ g(h[t])
```

where K is a fixed matrix called the Koopman operator.

The price of linearity is dimensionality: the exact Koopman operator is infinite-dimensional, but finite approximations can be very good. In our case, including just (h, h²) — a 32D lifted space — captures **96.6% of the variance** of the dynamics.

## How It Connects to Our Existing Analysis

### Connection to PCA

PCA finds directions of maximum variance in the hidden state. These are **linear observables** — projections of h onto fixed directions.

Our finding that PC1–PC2 capture 60% of hidden state variance but only 58% of offset variance ([doc 35](35_tropical_geometry_lens.md)) shows that linear observables miss the discrimination signal.

Koopman extends PCA: instead of projecting onto linear subspaces, we add nonlinear features (h², products of components) so that more of the dynamics becomes visible to a linear analysis.

### Connection to the Phase Wheel

The phase wheel ([doc 32](32_phase_wheel_mechanism.md)) encodes position as **phase** in an oscillating system. Phase is fundamentally a ratio or angle between components:

```
phase ∝ atan2(h_a, h_b)
```

This is a nonlinear function of h — no linear observable can read it directly. But with quadratic observables:

```
cos(2θ) = (h_a² - h_b²) / (h_a² + h_b²)
sin(2θ) = 2 h_a h_b / (h_a² + h_b²)
```

The products and squares **linearize the phase**. The Koopman operator on the lifted space can track phase evolution as a linear rotation — which is exactly what the eigenvalues of W_hh would do if ReLU weren't in the way.

### Connection to Tropical Cells

Each tropical cell is a region where ReLU acts as identity (active neurons) or zero (dead neurons). Within a cell, dynamics are exactly linear. The nonlinearity is in *which cell you're in* — a discontinuous function of h.

The quadratic observables provide a continuous encoding of cell membership:

| h_i value | ReLU effect | h_i² value | Interpretation |
|-----------|-------------|------------|----------------|
| Large positive | Identity | Large | Deeply inside "active" region |
| Near zero positive | Identity | Small | Near the boundary |
| Would be negative | Clipped to 0 | 0 | In "dead" region |

The squared terms act as soft indicators of how far each neuron is from its ReLU boundary. The Koopman operator K implicitly tracks "how does cell membership evolve?" as part of its linear dynamics on the lifted state.

### Connection to the Offset

The discrimination offset ([doc 33](33_offset_discrimination_mechanism.md)) is separable: `offset(m, s) = f(m) - f(s)`. In the lifted space, the offset becomes:

```
g(h_fwd) - g(h_rev) = [h_fwd - h_rev, h_fwd² - h_rev², ...]
                     = [offset_linear, offset_quadratic, ...]
```

The linear part is the offset we already study. The quadratic part, `h_fwd² - h_rev²`, captures how the offset interacts with the ReLU boundaries. These are the terms that encode whether a neuron is clipped differently in the forward vs reverse trajectory.

Since the Koopman operator predicts the full lifted state linearly, it means **the nonlinear part of the discrimination signal has a linear structure when lifted**.

### Connection to the Eigenvalue Spectrum

The original W_hh eigenvalues are:

| Mode | λ | \|λ\| |
|------|---|-------|
| 0 | -1.27 | 1.27 |
| 1 | +1.17 | 1.17 |
| 2–3 | 0.44 ± 1.03j | 1.12 |
| 4–5 | -0.11 ± 0.90j | 0.90 |

Three modes have |λ| > 1 — these would cause unbounded growth without ReLU.

The approximate Koopman eigenvalues on the (h, h²) lifted space:

| Mode | λ | \|λ\| |
|------|---|-------|
| 0–1 | -0.67 ± 0.74j | 1.00 |
| 2–3 | -0.04 ± 0.96j | 0.96 |
| 4 | +0.95 | 0.95 |
| 5 | -0.94 | 0.94 |

All modes have |λ| ≤ 1. The unstable W_hh modes have been stabilized by the lifting — the quadratic observables absorb the bounding effect of ReLU, converting it from a hard nonlinearity into effective decay in the lifted linear system.

This is precisely what we observe empirically: the cell-specific eigenvalues ([doc 39](39_bridging_linear_tropical.md)) hover near 1.0 (range 0.99–1.02 after the first impulse). The Koopman spectrum formalizes this — the effective dynamics, with ReLU accounted for, are marginally stable.

## Why Previous Approaches Failed and Koopman Differs

| Approach | Problem | What was lost |
|----------|---------|---------------|
| Pure linear (W_hh^k) | Ignores ReLU | Stability (|λ| > 1 → explosion) |
| Cell-effective (D @ W_hh @ D) | Uses one cell's mask | Trajectory history |
| Manifold projection (PCA) | Projects to low-D subspace | Discrimination signal (high-PC components) |
| **Koopman** | Adds dimensions | Nothing — captures nonlinearity as extra features |

The key difference: previous methods **projected down** to a simpler space, losing information. Koopman **lifts up** to a richer space where nonlinear structure becomes linear.

## What 96.6% R² Means Concretely

We fitted a linear map K on all within-trajectory (h[t], h[t+1]) pairs:

```
[h[t+1]; h[t+1]²] ≈ K @ [h[t]; h[t]²]
```

96.6% of the variance in the full lifted next-state is explained by K.

The remaining 3.4% comes from:
- Cell boundary crossings where ReLU effects are not captured by quadratic terms
- Trajectory-dependent correlations (the lifted state at time t doesn't carry full information about the activation masks that will be applied)
- Higher-order observable terms (h³, cross products) that we haven't included

This could improve with a richer observable set, but even with just quadratics the fit is strong.

## Path to Mechanistic Estimate

### Step 1: Derive K from Weights

Within a tropical cell with activation mask D, the dynamics are:

```
h' = D @ (W_hh @ h + W_ih * x)
```

The quadratic terms evolve as:

```
(h')² = (D @ (W_hh @ h + W_ih * x))²
```

If we average over the distribution of activation patterns D, weighted by how often they occur, we get an effective K that depends only on weights and the activation statistics.

The question is whether this averaging can be done analytically rather than by running the model.

### Step 2: Propagate Through K

Instead of composing 10 cell-specific maps:

```
h[9] = D[9] @ W_hh @ D[8] @ W_hh @ ... @ D[0] @ W_hh @ h[0]
```

We propagate through K in the lifted space:

```
g(h[9]) ≈ K^9 @ g(h[0])
```

This is a single matrix power — no cell enumeration.

### Step 3: Verify Discrimination

The original readout is `logits = W_out @ h[9]`, which only needs the first 16 components of g(h[9]). If K^9 correctly predicts these components with margins that stay positive, we have a mechanistic accuracy proof.

## Open Questions

1. **Can K be derived from weights alone?** The fitted K depends on the distribution of hidden states. An analytical formula would need to characterize the typical activation patterns from W_hh structure.

2. **Is the 3.4% error correlated with discrimination?** If the Koopman error is random (uncorrelated with which pairs have small margins), the margins are safe. If the error concentrates on small-margin pairs, it's a problem.

3. **What observable set is optimal?** We used (h, h²). Including (h_i * h_j) cross-terms or (max(0, w @ h)) directional ReLU features could improve accuracy and potentially make K exactly derivable from weights.

4. **Does the Koopman spectrum explain the phase wheel period?** The dominant complex eigenvalue pair at |λ| ≈ 1.0 should have an angular frequency matching the ~36°/step rotation we observe. Checking this would validate the lifting.

## Related Documents

- [32: Phase Wheel Mechanism](32_phase_wheel_mechanism.md) — Phase encoding that Koopman linearizes
- [33: Offset Discrimination Mechanism](33_offset_discrimination_mechanism.md) — f(·) structure in lifted space
- [35: Tropical Geometry Lens](35_tropical_geometry_lens.md) — Cell structure that Koopman absorbs
- [38: Mechanistic Estimate Progress](38_mechanistic_estimate_progress.md) — Failed linear approaches
- [39: Bridging Linear and Tropical](39_bridging_linear_tropical.md) — Framework comparison and experimental results
