# 09: Comparison with Prior Work

## Overview

This document compares our interference-based interpretation of M₁₆,₁₀ with prior work from ARC (Alignment Research Center), particularly:
- The "leave-one-out maximum" mechanism proposed for neurons 1, 6, 7
- Jacob Hilton's heuristic explanations for M₄,₃
- The surprise accounting framework

## Prior Work Summary

### Leave-One-Out Maximum (ARC Blog)

For M₁₆,₁₀, ARC identified that neurons 2 and 4 form an isolated subcircuit:
- **Neuron 2**: Computes approximately max(0, x₀, ..., x_{t-2})
- **Neuron 4**: Calculates the difference between consecutive maximums
- **Neuron 7**: Approximately max(0, x₀, ..., x_{t-1}) - x_{t-1}
- **Neuron 1**: Another leave-one-out-maximum feature

The interpretation is algebraic: neurons compute specific functions of running maximums with certain recent values excluded.

### Hilton's Analysis of M₄,₃

Hilton's paper analyzes the smaller M₄,₃ model through:
1. **Symmetric structure**: W matrices factor through sparse U, V matrices
2. **Region decomposition**: Input space partitioned by which neurons clip
3. **Piecewise linear analysis**: In each region, h is linear in inputs
4. **Covariance propagation**: Track noise through the network

Key finding: The 4 neurons form 2 symmetric pairs, reducing effective dimensionality.

### Surprise Accounting Framework

ARC proposes measuring understanding via:
```
Total surprise = surprise of explanation + surprise given explanation
```

A good explanation has low total surprise - it's compact but explains the accuracy well.

## Our Interpretation

### Interference-Based Encoding

We found that M₁₆,₁₀ uses wave interference for position encoding:

1. **Fourier-like components**: Comparators (n1, n6, n7, n8) respond at different frequencies based on self-recurrence values

2. **Anti-phase signals**: Argmax and 2nd_argmax create signals with **opposite phase**:
   | Neuron | Anti-phase correlation |
   |--------|----------------------|
   | n8 | -0.970 |
   | n1 | -0.895 |
   | n7 | -0.850 |
   | n6 | -0.538 |

3. **Destructive interference**: Position difference encoded through signal cancellation

4. **Two regimes**: Different mechanisms when 2nd_argmax comes before vs after argmax

## Reconciling the Interpretations

### Are They Compatible?

Yes - they describe different aspects of the same computation:

| Aspect | Leave-One-Out View | Interference View |
|--------|-------------------|-------------------|
| Focus | What is computed | How timing is encoded |
| Level | Algebraic functions | Signal dynamics |
| n2's role | Tracks running max | Sample-and-hold, threshold setting |
| n7's role | max - x_{t-1} | Fourier component at 3π/2 frequency |

### The Connection

The "leave-one-out maximum" can be reinterpreted through our lens:

**n7 ≈ max(0, ..., x_{t-1}) - x_{t-1}** means:
- n7 clips when x_t exceeds the running max
- After clipping, it rebuilds based on recurrence
- The value encodes "how long since the max occurred"
- This IS the Fourier-like recency encoding we identified

The algebraic view describes the **function computed**; the interference view describes the **mechanism** by which position emerges from that function.

### Unified Picture

```
INPUT SEQUENCE
     |
     v
[Leave-One-Out Computation]
  n2: running max (sample-and-hold)
  n7: max - x_{t-1} (comparison signal)
  n1, n6, n8: similar leave-one-out features
     |
     | These computations CREATE:
     v
[Fourier-like Encoding]
  - Clipping = impulse into resonant system
  - Different recurrence = different frequencies
  - h_final = superposition of impulse responses
     |
     v
[Interference Pattern]
  - Argmax impulse: positive phase
  - 2nd_argmax impulse: negative phase
  - Difference encoded in residual
     |
     v
[Matched Filter Decoding]
  - W_out templates match interference patterns
  - argmax(logits) = predicted 2nd_argmax
```

## Applying Surprise Accounting to Our Interpretation

### Explanation Bits (What We Claim)

| Claim | Bits |
|-------|------|
| 4 comparators with different frequencies | ~8 |
| Anti-phase relationship for 2nd_argmax | ~4 |
| Threshold cascade (W_ih ordering) | ~4 |
| Two operating regimes | ~2 |
| n2 as sample-and-hold normalizer | ~4 |
| Partial clipping encodes amplitude | ~4 |
| **Total explanation** | **~26 bits** |

### Predictions Made

Our interpretation predicts:
1. Accuracy ~89% ✓
2. Position 9 is special (single clip pattern) ✓
3. Errors concentrated at small gaps ✓
4. (9,0) is hardest case (n2 decay + late argmax) ✓
5. Accuracy similar for both regimes (~89% each) ✓
6. Confusion correlates with signature similarity (r=0.38) ✓

### Residual Surprise

What we **don't** fully explain:
- Exact role of neurons n0, n3, n5, n9-15 (~10 neurons)
- Why accuracy peaks at gap ~0.2-0.3, not monotonic
- Fine-grained error patterns for specific position pairs
- How the model generalizes to out-of-distribution inputs

### Comparison with Hilton's Approach

| Metric | Hilton (M₄,₃) | Ours (M₁₆,₁₀) |
|--------|---------------|---------------|
| Model size | 32 params | 432 params |
| Explanation type | Algebraic regions | Signal dynamics |
| Accuracy estimate | 3% RMS error | Qualitative |
| Key insight | Symmetric pairs | Anti-phase interference |
| Completeness | Near-complete | Partial (~8/16 neurons) |

## What's Missing

### From Our Analysis

1. **Quantitative accuracy prediction**: Hilton can estimate accuracy to 3% RMS; we haven't derived accuracy from first principles

2. **Role of other neurons**: We focused on comparators (1,6,7,8) and n2; neurons 0,3,5,9-15 need more analysis

3. **Formal verification**: Our claims are empirically supported but not mathematically proven

### From Prior Work

1. **The larger model**: ARC notes they "have not turned this understanding into an adequate mechanistic estimate" for M₁₆,₁₀

2. **Dynamical interpretation**: Prior work focuses on algebraic functions, missing the signal processing perspective

3. **Anti-phase discovery**: The opposite-phase relationship between argmax and 2nd_argmax signals appears to be novel

## Synthesis

The most complete picture combines both views:

**Algebraic level** (what):
- Neurons compute leave-one-out maximums
- These are the features being extracted

**Dynamical level** (how):
- Clipping creates impulses
- Recurrence creates Fourier-like responses
- Anti-phase encoding enables interference-based position decoding

**Information level** (why it works):
- Interference encodes timing difference, not absolute times
- This is more robust (common-mode rejection)
- Multiple frequencies provide multi-resolution encoding

## Future Work

1. **Derive accuracy analytically** using our interference model
2. **Complete the neuron inventory** - explain n0, n3, n5, n9-15
3. **Test on other models** - do M₈,₅ etc. use similar mechanisms?
4. **Formal surprise accounting** - quantify explanation quality rigorously

## References

- [AlgZoo: uninterpreted models](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/)
- [Formal verification and surprise accounting](https://www.alignment.org/blog/formal-verification-heuristic-explanations-and-surprise-accounting/)
- Hilton, J. (2024). Heuristic explanations for 2nd argmax models.
