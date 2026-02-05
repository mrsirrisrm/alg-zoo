# 43: Mechanistic Discrimination from Weights

## Summary

We can predict the inference behavior of the M₁₆,₁₀ 2nd-argmax RNN entirely from its weights, achieving **100% accuracy on simplified inputs** and matching the model's ~95% accuracy on real data.

The key insight: the ReLU activation pattern is almost entirely determined by a **canonical sequence** that can be precomputed once from weights.

## The Problem

Given an RNN with weights (W_ih, W_hh, W_out), predict which position it will output for any input without running full simulation.

**Input**: Sequence of 10 values where M = max, S = 2nd max
**Output**: Position of S (the 2nd argmax)

## The Mechanism

### 1. Canonical D Sequence

The pattern of active neurons D[t] after a single positive impulse is **universal** - independent of position or magnitude:

```
D[0]:  6 neurons  (determined by sign(W_ih))
D[1]: 13 neurons  (determined by sign(W_hh @ ReLU(W_ih)))
D[2]: 12 neurons
...
```

This sequence is precomputed once from weights via forward iteration.

### 2. Three-Phase Computation

For input with M at position m, S at position s:

| Phase | Timesteps | D[t] computation |
|-------|-----------|------------------|
| 1. Before first impulse | t < min(m,s) | D[t] = ∅ |
| 2. After first, before second | first ≤ t < second | D[t] = canonical_D[t - first] |
| 3. At second impulse | t = second | D[t] = sign(M·pre[gap] + S·W_ih) |
| 4. After second | t > second | Iterate ~2 steps |

**Key insight**: Phase 2 is pure table lookup. Phase 3 is a single linear combination.

### 3. Discrimination via Offset

The final hidden state decomposes as h = h_main + offset/2, where:
- **h_main**: Superposition of single-impulse responses (neutral for discrimination)
- **offset**: Encodes position difference, always favors correct answer

The offset mechanism achieves 100% discrimination because W_out is tuned as a phase detector aligned with the offset structure.

## Results

| Input Type | Accuracy | Min Margin |
|------------|----------|------------|
| Simplified (M=1, S=0.8, others=0) | **100%** | +3.57 |
| Real data (all positions randn) | **~95%** | -11.2 |

The ~5% error on real data comes from:
1. Interference from other 8 inputs
2. Ambiguous cases where S ≈ 3rd largest value

When gap(S, 3rd) ≥ 0.2, accuracy rises to **99%**.

## Complexity

| Approach | Operations per input |
|----------|---------------------|
| Naive RNN simulation | 2560 |
| Mechanistic discriminator | ~850 |
| Precomputation (once) | 9 matrix-vector products |

## Usage

```python
from mechanistic_discrimination import MechanisticDiscriminator
import numpy as np

w = np.load('weights.npz')
disc = MechanisticDiscriminator(w['W_ih'], w['W_hh'], w['W_out'])

# Predict
pred, margin, D_seq = disc.predict(m=2, s=7)

# Detailed explanation
exp = disc.explain(m=2, s=7)
print(f"Correct: {exp['correct']}, Margin: {exp['margin']:.2f}")
```

## Connection to ARC Challenge

This answers the question: "Can you turn this into a mechanistic estimate of accuracy from weights?"

**Yes.** The accuracy decomposes as:
- 84% via linear phase wheel mechanism (W_hh eigenvalues + W_out correlations)
- 16% via ReLU differential clipping (canonical D sequence)
- = 100% on simplified inputs

All components are deterministic functions of (W_ih, W_hh, W_out).

## Related

- `src/mechanistic_discrimination.py` - Implementation
- `docs/41_koopman_discrimination_analysis.md` - Offset mechanism details
- `docs/42_mechanistic_accuracy_estimate.md` - Full accuracy decomposition
