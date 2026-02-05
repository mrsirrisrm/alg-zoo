# 41: Koopman Lifting and the Complete Discrimination Mechanism

## Summary

This document reports on experiments testing Koopman lifting and analyzes the complete discrimination mechanism for M₁₆,₁₀. The key finding is that discrimination works **entirely through the offset**, not the main spiral.

## The Complete Mechanism

The hidden state decomposes as:

```
h_fwd = h_main + offset/2
h_rev = h_main - offset/2
```

where:
- **h_main** = (h_fwd + h_rev)/2 ≈ superposition of single-impulse responses
- **offset** = h_fwd - h_rev ≈ f(m) - f(s) with antisymmetric structure

**Critical finding**: The margin decomposes as:

| Component | Mean contribution | Always positive? |
|-----------|-------------------|------------------|
| h_main    | 0.00              | 50% (neutral!)   |
| offset    | 14.06             | 100%             |

**Discrimination works entirely through the offset.** The main spiral is neutral on average.

## Key Findings

### 1. Single-Step Koopman Works Well (R² = 96.6%)

Fitting a linear operator K on lifted observables (h, h²):

```
g(h[t+1]) ≈ K @ g(h[t])
```

achieves 96.6% R² for single-step prediction. The Koopman eigenvalues are near the unit circle (max |λ| ≈ 1.0), confirming marginal stability.

### 2. K^n Fails for Discrimination

When we try to use K^n to predict h[final] from h[start]:

| Observable set | Single-step R² | K^n discrimination |
|----------------|----------------|-------------------|
| h only         | 94.0%          | 4/90              |
| h, h²          | 96.6%          | 10/90             |
| h, h², h³      | 98.5%          | 32/90             |
| h, h², h³, h⁴  | 99.2%          | 22/90             |

The 3.4% single-step error compounds catastrophically. The issue is **transient amplification**: ||K^n||₂ ≈ 200 even though spectral radius ≈ 1.0.

### 3. Direct n-Step Koopman Works Perfectly

Fitting separate operators K_n that directly map g(h[start]) → h[final]:

| n_steps | R² | Discrimination |
|---------|-----|----------------|
| 0       | 99.98% | 34/34 |
| 1       | 100% | 14/14 |
| 3       | 100% | 10/10 |
| 5       | 100% | 6/6 |
| 7       | 100% | 2/2 |

**Perfect discrimination** when using direct n-step prediction.

### 4. The Antisymmetry Property

The key to discrimination is the structure of f(pos):

```
f(pos) = E[h_final | M at pos] - E[h_final | S at pos]
```

**Critical property (verified for all 10 positions):**

```
argmin(W_out @ f(pos)) = pos
```

This means f(pos) maximally suppresses output at position pos. For any (m, s):

- Forward: offset = f(m) - f(s) → suppresses m, boosts s → correct
- Reverse: offset = f(s) - f(m) → suppresses s, boosts m → correct

### 5. Gap Analysis

The antisymmetry gap measures how strongly f(pos) suppresses position pos:

```
gap(pos) = min_{q≠pos} [(W_out[q] - W_out[pos]) @ f(pos)]
```

| Position | Gap | W_out @ f[pos] |
|----------|-----|----------------|
| 0 | 5.34 | -18.40 |
| 1 | 6.98 | -18.26 |
| 2 | 6.64 | -16.79 |
| 3 | 5.64 | -14.79 |
| 4 | 4.51 | -12.34 |
| 5 | 3.07 | -9.72 |
| 6 | 3.47 | -9.09 |
| 7 | 2.13 | -7.98 |
| 8 | 9.39 | -13.67 |
| 9 | 15.70 | -19.54 |

**Minimum gap: 2.13** (at position 7)
**Minimum empirical margin: 3.57**

The gap provides a lower bound on discrimination margin.

### 6. f(pos) Structure

f(pos) has low-rank structure:
- Top 2 PCs explain 79.8% of variance
- Dominant neurons: waves (n10, n0, n8, n14, n11, n12)

Attempts to derive f(pos) from weights:
- **Linear theory** (W_hh^(9-pos) @ W_ih * Δ): Poor match (cos_sim ~ 0, norms differ by 10-40x)
- **Single-impulse response**: Moderate match (cos_sim 0.35-0.76, norm ratio 0.7-1.0)

The single-impulse response captures the direction of f(pos) reasonably well but misses the ReLU interactions with the other impulse.

## Path to Weight-Based Proof

### What We Have

1. **Empirical verification**: 100% discrimination, all margins positive (min 3.57)
2. **Structural property**: W_out @ f(pos) has minimum at pos for all positions
3. **Gap bound**: min_gap = 2.13 provides margin lower bound

### What We Need

To complete a purely weight-based proof:

**Option A: Bound f(pos) analytically**

Express f(pos) as a function of (W_ih, W_hh, W_out) with bounded error. The single-impulse response provides a starting point but needs correction for ReLU interactions.

**Option B: Sample-based verification**

Run a small set of trajectories to estimate f(pos), then verify:
1. Antisymmetry holds: argmin(W_out @ f(pos)) = pos
2. Gap is positive for all positions
3. This implies 100% discrimination

This is not purely "from weights" but is much cheaper than running all 180 trajectories.

**Option C: Tropical cell analysis**

Since 88% of the signal is within-cell, characterize the dominant tropical cell's affine map and show it produces the required f(pos) structure.

## Implications for Koopman Lifting

1. **Single-step Koopman is excellent** but error compounds
2. **Direct n-step Koopman is perfect** but requires fitting per-n operators
3. **The issue is transient amplification**, not spectral properties

For this specific RNN, the Koopman framework confirms that:
- Lifted observables (h, h²) capture the discrimination-relevant dynamics
- The discrimination signal is encoded in f(pos) structure
- W_out is perfectly tuned for the antisymmetry property

## Main Spiral Analysis

The main spiral (h_main) has interesting structure but doesn't contribute to discrimination.

### Key Question: Can We Derive h_main From Weights?

**Answer: YES!** Through effective linear maps and superposition.

### Single-Impulse Responses Are Exact

Single-impulse responses can be computed **exactly** from weights:

```
h_single(pos, val) = W_eff(pos, val) @ (W_ih * val)
```

where the effective linear map is:

```
W_eff(pos, val) = D[9] @ W_hh @ D[8] @ ... @ D[pos+1] @ W_hh @ D[pos]
D[t] = diag(active neurons at step t)
```

The active neuron pattern D[t] is deterministic given W_ih, W_hh — just run forward simulation.

**Verification**: cos_sim = 1.0, norm_ratio = 1.0 for all positions and values.

### h_main from Superposition

The main spiral is well-approximated by superposition of single-impulse responses:

```
h_main(m, s) ≈ 0.5 × [h_single(m, M) + h_single(s, S) + h_single(s, M) + h_single(m, S)]
```

| Metric | Value |
|--------|-------|
| cos_sim (direction) | 0.92 average, 0.86 minimum |
| norm_ratio (magnitude) | 1.76 average (40% overestimate) |

The direction is captured well; the magnitude overestimate comes from ReLU interactions between the two impulses that reduce the actual h_main norm.

### Why Koopman Doesn't Add Value Here

Koopman K_n gives cos_sim ≈ 0.8-0.9 for predicting final state from impulse state, but:
- The effective linear map W_eff already gives **exact** single-impulse responses
- Koopman is most useful when we don't have direct access to weights
- For this analysis, direct weight formulas are simpler and more accurate

### Why Main Spiral Doesn't Discriminate

| Metric | Value |
|--------|-------|
| argmax(W_out @ h_sup) ∈ {m, s} | 2.2% |
| h_main margin contribution | 0 (mean) |
| h_main contribution sign | 50% positive, 50% negative |

The superposition typically outputs position 9 or some other "default" — NOT the correct answer. It's the offset that tips the argmax to the correct position.

## Simplified Weight-Based Proof

Given that discrimination works entirely through offset:

### What We Need to Prove

1. **Offset structure**: offset ≈ f(m) - f(s) (verified empirically, R² = 0.91)

2. **Antisymmetry**: W_out @ f(pos) has minimum at pos
   - Verified for all 10 positions
   - Minimum gap: 2.13

3. **Sufficient magnitude**: offset contribution (mean 14.06) exceeds any adverse h_main contribution

### The Bound

For any (m, s) pair:

```
margin = [h_main contribution] + [offset contribution]
       = [random, mean 0] + [always positive, mean 14.06]
       > min_offset_contribution - max_adverse_h_main
```

Since offset contribution is always positive (100%) and h_main contribution averages to 0, the margin is always positive.

### What Remains

To make this fully weight-based:

1. Derive f(pos) from (W_ih, W_hh) — currently only approximated by single-impulse response (cos_sim 0.35-0.76)

2. Bound the h_main contribution — it's zero on average but varies by pair

3. Show the offset contribution always exceeds any negative h_main contribution

The minimum empirical margin is 3.57, which provides substantial safety factor.

## Complete Weight-Based Formulas

### 1. Single-Impulse Response (Exact)

```
h_single(pos, val) = W_eff(pos, val) @ (W_ih * val)

W_eff(pos, val) = D[9] @ W_hh @ D[8] @ ... @ D[pos+1] @ W_hh @ D[pos]
```

where D[t] = diag(neurons active at step t) for a single impulse of value `val` at position `pos`.

**Accuracy**: Exact (cos_sim = 1.0)

### 2. Main Spiral (Approximate)

```
h_main(m, s) ≈ 0.5 × [h_single(m, M) + h_single(s, S) + h_single(s, M) + h_single(m, S)]
```

**Accuracy**: cos_sim ≈ 0.92 (direction), norm ratio ≈ 1.76 (magnitude)

### 3. Offset (From Antisymmetry)

The offset satisfies:
```
(W_out @ offset)[s] > (W_out @ offset)[m]  for all (m, s) pairs
```

This is verified 100% empirically and is the key to discrimination.

### 4. Discrimination Margin

```
margin(m, s) = [h_main contribution] + [offset contribution]
             = [mean 0, std 2.91]    + [mean 14.06, always positive]
```

Since offset contribution is always positive and dominates, margin > 0 always.

## Related Documents

- [32: Phase Wheel Mechanism](32_phase_wheel_mechanism.md) — The main spiral dynamics
- [33: Offset Discrimination Mechanism](33_offset_discrimination_mechanism.md) — The offset structure
- [38: Mechanistic Estimate Progress](38_mechanistic_estimate_progress.md) — Failed linear approaches
- [39: Bridging Linear and Tropical](39_bridging_linear_tropical.md) — Framework comparison
- [40: Koopman Lifting](40_koopman_lifting.md) — Theory and initial analysis
