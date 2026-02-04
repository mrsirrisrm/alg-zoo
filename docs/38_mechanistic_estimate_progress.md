# 38: Progress Toward Mechanistic Accuracy Estimate

## Goal

Derive model accuracy purely from weights (W_ih, W_hh, W_out), without running the model on inputs.

## Current Understanding

### The Empirical f(·) Structure

The offset between forward and reverse trajectories is highly structured:

```
offset(m, s) = h_fwd - h_rev ≈ f(m) - f(s)
```

where f is a separable function with **antisymmetry**: f(pos) = -g(pos).

Properties:
- Separable model R² = 91% (explains 91% of offset variance)
- Remaining 9% is ReLU clipping effects
- f has effective rank ~2 (88% variance in top 2 PCs)

### Key Finding: f(pos) ≈ E[h | M at pos] - E[h | S at pos]

The position encoding f(pos) equals the conditional expectation difference:
- "What is the expected final hidden state when M is at position pos?"
- Minus "What is the expected final hidden state when S is at position pos?"

This averages over the uncertainty of where the other token is located.

Correlation between f_empirical and this conditional expectation difference:
- pos 3-5: cosine > 0.93
- pos 2,6-9: cosine 0.66-0.80
- Consistently high alignment

## Failed Approaches

### 1. Pure Linear Theory

**Attempt**: f(pos) = W_hh^(9-pos) @ W_ih × (M - S)

**Result**: Complete failure
- Linear offset prediction R² = -148.76 (worse than mean!)
- Discrimination accuracy: 6.7% (should be 100%)
- Linear margins are **wrong sign** for most pairs

**Why it fails**: W_hh has eigenvalues |λ| > 1 (max 1.27), causing 9-step growth factor ~9. Without ReLU, hidden states explode to ||h|| ~ 900 vs ~40 with ReLU. The ReLU isn't a small correction—it's essential for bounded operation.

### 2. Effective Cell Dynamics

**Attempt**: Use the dominant tropical cell's activation mask D, compute effective dynamics W_hh_eff = D @ W_hh @ D

**Result**: Also fails
- Effective eigenvalues are stable (max |λ| = 1.01)
- But discrimination: 0% (all margins negative!)

**Why it fails**: The discrimination information is encoded in the **trajectory through cells**, not just the final cell's linear dynamics. By the time we reach the final cell, the "damage" (discrimination signal) has already been done by the transient dynamics.

## The Core Challenge

The ReLU nonlinearity is **essential** for:
1. **Bounding**: Prevents exponential growth (||h|| stays ~40 vs ~900)
2. **Discrimination**: Different clipping patterns for fwd vs rev encode position
3. **Trajectory-dependent encoding**: The path matters, not just endpoint

For a mechanistic estimate, we need to either:
- Enumerate trajectories and their probabilities (hard)
- Find a closed-form approximation that captures ReLU effects (promising)
- Use bounds to prove margin positivity without exact computation

## Promising Direction: Trajectory Statistics

### Key Observations

1. **Offset grows through trajectory**:
   ```
   t | ||offset||
   0 |   0.0      (before M)
   M |   2.0      (just after M)
   ...|   7-9     (growing)
   S |   varies   (S impulse modifies)
   9 |   9-18     (final)
   ```

2. **ReLU clips same number of neurons** at M arrival (10 clips) regardless of position—the variation comes later

3. **Most pairs share similar trajectories**: 88% of activation patterns are identical between fwd/rev. The offset is 88% within-cell (continuous) and 12% boundary-crossing (categorical).

### Proposed Approach

**Semi-mechanistic estimate**:

1. **Empirically observe** that f(pos) ≈ E[h | M at pos] - E[h | S at pos]

2. **Derive from weights** that E[h | token at pos] follows a specific pattern based on:
   - W_ih injection at pos
   - W_hh propagation for (9-pos) steps
   - ReLU bounding (approximate as projection onto stable manifold)

3. **Prove** that W_out @ (f(m) - f(s)) > 0 for all valid (m,s) pairs

The key insight is that averaging over the other token's position **regularizes** the ReLU effects, making the conditional expectation more tractable than individual trajectories.

## What Counts as Success

According to the [AlgZoo challenge](https://www.lesswrong.com/posts/x8BbjZqooS4LFXS8Z):

> "Deduce correlations from the weights, rather than just observe them empirically"

For M₁₆,₁₀, a valid mechanistic estimate would:
1. Compute some function of (W_ih, W_hh, W_out) → predicted accuracy
2. Do so more efficiently than random sampling
3. Generalize to perturbed weights (for MSE evaluation)

Current status: We can **describe** the mechanism (phase wheel + offset discrimination) but cannot yet **derive** accuracy from weights alone.

## Next Steps

### Option A: Approximate ReLU as projection

Hypothesis: ReLU in this regime acts approximately as projection onto the stable eigenspace of W_hh. If we can show:
- The stable eigenspace captures the discrimination signal
- The projection is approximately linear
- W_out @ projected_offset has correct sign

This would give a weight-based formula with bounded error.

### Option B: Tropical cell enumeration

For each (m,s) pair:
1. Predict which tropical cells the trajectory visits (from weights)
2. Compose cell-specific affine maps
3. Compute margin from the composed map

This is exact but complex. Need to show cell membership is predictable.

### Option C: Perturbation analysis

Start from the empirical f(·), show it's a fixed point of some weight-derived iteration:
- f = Φ(W_ih, W_hh, f)  [f appears on both sides]
- If Φ is contractive, f is determined by weights

### Option D: Margin bounds

Instead of exact computation:
1. Lower bound the empirical margin (min = 7.91)
2. Upper bound the error from linear approximation
3. Show bound > error → accuracy = 100%

This may work since the minimum margin (7.91) is substantial.

## Summary Table

| Approach | Weight-based? | Accuracy | Status |
|----------|---------------|----------|--------|
| Empirical f(·) | No (runs model) | 100% | Baseline |
| Pure linear | Yes | 6.7% | Fails |
| Cell-effective linear | Partial | 0% | Fails |
| Conditional expectation | Partial | ~100% | Promising |
| Full tropical enum | Yes | Expected 100% | Complex |

The gap between "runs model" (100%) and "pure weight computation" (6.7%) shows ReLU is doing essential, non-trivial work. A successful mechanistic estimate must capture this.
