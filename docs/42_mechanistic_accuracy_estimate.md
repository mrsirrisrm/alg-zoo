# 42: Mechanistic Accuracy Estimate from Weights

## Summary

This document provides a **mechanistic estimate of model accuracy** for M₁₆,₁₀ derived from examining the weights. This addresses the ARC challenge: "Can you turn this argument into a mechanistic estimate of the model's accuracy?"

**Result**: From weights alone, we can explain:
- **84% (76/90 pairs)** via linear phase wheel mechanism
- **16% (14/90 pairs)** via ReLU differential clipping
- **100% total** accuracy

## The Weight-Based Argument

### 1. Rotation from W_hh Eigenvalues

```
W_hh eigenvalues:
  λ₀ = -1.27 (real, growing)
  λ₁ = +1.17 (real, growing)
  λ₂,₃ = 0.43 ± 1.03j (complex, |λ| = 1.1, period = 5.4 steps)
```

**Prediction**: Hidden state rotates with period ~5-9 steps, slow growth bounded by ReLU.

### 2. Phase Wheel in W_out

W_out row correlations:
- Adjacent rows: +0.5 to +0.7 (similar phase)
- 4-apart rows: -0.4 to -0.9 (opposite phase, half period)

**Prediction**: W_out is arranged as a phase detector with ~5-step period.

### 3. Single-Impulse Response from W_eff

For an impulse at position `pos`:
```
r[pos] = W_eff(pos) @ ReLU(W_ih)
W_eff(pos) = D[9] @ W_hh @ D[8] @ ... @ D[pos+1] @ W_hh @ D[pos]
```
where D[t] = diag(active neurons at step t).

**Key insight**: D[t] is deterministic from weights — just iterate forward.

### 4. Linear Discrimination Condition

For the offset to favor S over M:
```
(W_out[s] - W_out[m]) @ (r[m] - r[s]) > 0
```

**From weights**: This holds for 76/90 pairs (84%).

The 14 failing pairs cluster at positions with largest phase misalignment:
- Position 0: 10 failures
- Position 1: 6 failures
- Position 7: 2 failures
- Positions 5, 6, 8, 9: 0 failures

### 5. ReLU Correction

For the 14 failing pairs, ReLU differential clipping provides correction:
```
actual_offset = linear_offset + Δ_ReLU
```

**Key finding**: Δ_ReLU is always aligned with discrimination direction.

| Pair | Linear | ReLU correction | Final |
|------|--------|-----------------|-------|
| (0,1) | -0.5 | +9.9 | +9.8 |
| (0,3) | -5.8 | +22.4 | +21.3 |
| (0,4) | -0.5 | +45.1 | +45.0 |

## The Complete Mechanism

```
┌─────────────────────────────────────────────────────────────────┐
│             MECHANISTIC ACCURACY ESTIMATE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FROM WEIGHTS:                                                   │
│  ├── W_hh eigenvalues → rotation (period 5.4)                   │
│  ├── W_out correlations → phase detector                         │
│  └── W_eff(pos) @ W_ih → single-impulse responses                │
│                                                                  │
│  LINEAR MECHANISM (84%):                                         │
│  ├── offset ≈ (M-S) × (r[m] - r[s])                             │
│  ├── Condition: (W_out[s]-W_out[m]) @ (r[m]-r[s]) > 0           │
│  └── Phase alignment → 76/90 pairs satisfy                       │
│                                                                  │
│  RELU CORRECTION (+16%):                                         │
│  ├── Different D[t] for fwd vs rev                               │
│  ├── Creates Δ_ReLU aligned with discrimination                  │
│  └── Rescues remaining 14/90 pairs                               │
│                                                                  │
│  TOTAL: 100% accuracy                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## What Can Be Computed From Weights

| Component | From Weights? | Method |
|-----------|--------------|--------|
| Rotation period | ✓ | W_hh eigenvalues |
| Growth bound | ✓ | max\|λ\| + ReLU clipping |
| Phase wheel | ✓ | W_out row correlations |
| Single-impulse r[pos] | ✓ | W_eff(pos) @ ReLU(W_ih) |
| Linear condition | ✓ | (W_out[s]-W_out[m]) @ (r[m]-r[s]) |
| Which pairs fail | ✓ | Evaluate condition for all 90 |
| ReLU correction sign | ✓* | Simulation or boundary analysis |

*ReLU correction can be computed exactly by simulation, which is deterministic from weights.

## Remaining Gap

To make this fully "just by looking at weights":

1. **Predict failing pairs without enumeration**: The failures occur at positions with large phase error. Could derive this from eigenvalue/W_out alignment.

2. **Bound ReLU correction without simulation**: Need to identify "boundary neurons" (near-zero pre-activation) from W_ih, W_hh structure and show their contribution is positive.

3. **Provide margin bounds**: Current minimum margin is 3.57. A weight-based bound would require tracking the linear + ReLU contributions.

## Comparison to Empirical

| Metric | Weight-based prediction | Empirical |
|--------|------------------------|-----------|
| Linear accuracy | 76/90 = 84% | — |
| ReLU correction | +14/90 = 16% | — |
| Total accuracy | 100% | 100% |
| Minimum margin | Need simulation | 3.57 |

## Conclusion

The ARC challenge asks for a mechanistic estimate from weights. We can now provide:

1. **Qualitative**: Rotation + phase wheel + antisymmetric readout → discrimination
2. **Quantitative**: 84% linear + 16% ReLU = 100%
3. **Computable**: All steps are deterministic functions of (W_ih, W_hh, W_out)

The only gap is predicting the ReLU correction's magnitude without running the full trajectory, but even this is deterministic from weights.

## Related Documents

- [32: Phase Wheel Mechanism](32_phase_wheel_mechanism.md)
- [33: Offset Discrimination Mechanism](33_offset_discrimination_mechanism.md)
- [41: Koopman Lifting Analysis](41_koopman_discrimination_analysis.md)
