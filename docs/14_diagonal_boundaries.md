# 14: Diagonal Decision Boundaries - RNN vs XGBoost

## Overview

We investigated whether XGBoost's space-partitioning approach would be well-suited for the 2nd argmax task. The results revealed a fundamental difference in how RNNs and tree-based models partition input space, and led to the discovery that the RNN implements **direct diagonal comparison boundaries**.

## XGBoost Baseline

### Performance Without Feature Engineering

| Config | Train Acc | Test Acc | Notes |
|--------|-----------|----------|-------|
| 100 trees, depth 6 | 91% | 75% | Heavy overfitting |
| 500 trees, depth 10 | 100% | 87% | Still overfitting |
| 500 trees, depth 12 | 100% | 88.5% | Best raw performance |

### Why XGBoost Struggles

XGBoost uses **axis-aligned splits**: each decision is "x[i] < threshold?"

For 2nd argmax, the model needs to answer: "Is x[i] > x[j]?" for many pairs. This comparison is a **diagonal boundary** in input space, not axis-aligned.

To approximate a diagonal with axis-aligned rectangles:

```
Diagonal boundary x[0] = x[1]:     XGBoost approximation:

┌──────────────────┐               ┌──────────────────┐
│ ╲                │               │        │ 0 │    │
│   ╲   region 1   │               │    0   ├───┤ 1  │
│     ╲            │      →        │        │ 0 │    │
│       ╲          │               │────────┼───┼────│
│ region 0 ╲       │               │    0   │ 1 │ 1  │
└──────────────────┘               └────────┴───┴────┘
                                   Many rectangles needed!
```

### With Feature Engineering (Cheating)

| Features | Config | Test Acc |
|----------|--------|----------|
| Raw + ranks | 1 tree, depth 2 | 100% |
| Raw + dist_from_max | 100 trees, depth 6 | 94% |

With rank features, XGBoost trivially achieves 100% - but computing ranks IS the hard part!

## The RNN's Advantage: Diagonal Boundaries

### Discovery: N7 Implements x[t] > running_max

We found that neuron n7 directly implements a comparison between the current input and the running maximum:

```
pre[n7] = h[t-1] @ W_hh[7,:] + x[t] * W_ih[7]
```

At t=1 (where running_max = x[0]):

```
Linear fit: pre[n7] = 13.41 * x[0] - 13.17 * x[1] + 0.00
```

The ratio of coefficients: **13.41 / 13.17 = 1.02 ≈ 1.0**

This means:
```
pre[n7] ≈ 13.2 * (x[0] - x[1])
        = 13.2 * (running_max - current_input)
```

N7 clips (pre < 0) when: **current_input > running_max**

### Empirical Verification

| Timestep | Match Rate (n7 clips ↔ x[t] > running_max) |
|----------|-------------------------------------------|
| t=1 | 99.1% |
| t=2 | 98.9% |
| t=3 | 99.2% |
| t=4 | 98.4% |
| t=5 | 97.5% |
| **Overall** | **97.8%** |

The match rate is nearly perfect, especially at early timesteps.

### Mismatch Analysis

At t=1, only 0.9% of samples mismatch:
- Type 1 (clips when shouldn't): 0.00%
- Type 2 (doesn't clip when should): 0.88%

Type 2 errors occur when x[1] is only slightly larger than x[0] (mean diff = 0.006). This is the "soft" threshold region.

## The Comparator Spectrum

All four comparator neurons implement different types of boundaries:

| Neuron | α (running_max) | β (current) | α/\|β\| | Type |
|--------|-----------------|-------------|---------|------|
| n7 | +13.41 | -13.17 | **1.02** | Pure comparison |
| n1 | +8.87 | -10.56 | **0.84** | Soft comparison |
| n6 | -0.46 | -11.00 | 0.04 | Mostly absolute threshold |
| n8 | -0.39 | -12.31 | 0.03 | Mostly absolute threshold |

Where:
- α = coefficient on running_max (via recurrence)
- β = coefficient on current input (W_ih)

### Interpretation

- **n7 (α/|β| = 1.02)**: Pure comparison "Is x[t] > running_max?"
- **n1 (α/|β| = 0.84)**: Soft comparison with bias toward current value
- **n6, n8 (α/|β| ≈ 0)**: Mostly absolute thresholds "Is x[t] > constant?"

This diversity allows the model to handle different scenarios:
- n7 detects new maxima precisely
- n6, n8 detect "large enough" values regardless of history

## How the Diagonal Boundary Emerges

### The Mechanism

```
pre[n7] = h[t-1] @ W_hh[7,:] + x[t] * W_ih[7]
          ^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^
          ≈ +13.4 * max        = -13.2 * x[t]
```

The recurrent state h[t-1] encodes the running maximum through the n2-n4 circuit (see doc 13). This combines with the negative input weight to create:

```
pre[n7] ≈ 13.4 * running_max - 13.2 * current
        = 13.3 * (running_max - current)
```

The clipping boundary pre[n7] = 0 is exactly:
```
running_max = current
```

This is a **diagonal hyperplane** in the input space!

### Visual Representation

```
        x[1] (current)
          ↑
        1 │        ╱
          │      ╱  n7 clips
          │    ╱    (current > max)
          │  ╱
        0 │╱─────────────→ x[0] (running_max)
          0              1

The boundary x[1] = x[0] is DIAGONAL, not axis-aligned!
```

## Efficiency Comparison

### Parameters vs Accuracy

| Model | Parameters/Splits | Test Accuracy | Overfitting |
|-------|-------------------|---------------|-------------|
| XGBoost (500×12) | ~10,000s | 88.5% | Severe (100% train) |
| RNN | 432 | 89% | Minimal |

The RNN is **~20x more parameter-efficient**.

### Why the Difference?

**XGBoost**: Each comparison requires many axis-aligned splits
- To check "x[0] > x[1]", need O(1/ε²) rectangles for precision ε
- 45 pairwise comparisons × many splits each = huge model

**RNN**: Each comparison is ONE hyperplane
- Linear combination directly computes x[i] - x[j]
- ReLU creates the decision boundary
- Recurrence maintains running statistics (max, etc.)

## Space Partitioning Analysis

### Unique Clipping Patterns

In 10,000 samples, the RNN creates **9,400 unique clipping patterns** (sequences of which neurons clip at which timesteps).

Theoretical maximum: 2^160 (16 neurons × 10 timesteps)
Actual: ~9,400 (constrained by input distribution and recurrence)

### Pattern Coverage

| Threshold | Patterns Needed |
|-----------|-----------------|
| 50% of samples | 4,400 patterns |
| 90% of samples | 8,400 patterns |
| 99% of samples | 9,300 patterns |

Each pattern corresponds to a region in input space with consistent model behavior.

## Connection to Prior Findings

### The Full Picture

1. **Amplitude encoding** (doc 13): n2-n4 noise cancellation encodes max value
2. **Position encoding** (docs 5, 8): Fourier-like oscillations encode position
3. **Diagonal boundaries** (this doc): Comparators implement x[t] vs running_max

The diagonal boundaries are HOW the model implements comparisons. The recurrent state (maintained by n2, n4) provides the "running_max" term, and the negative W_ih provides the "current" term.

### Why n7 is Special

n7 has the largest |W_ih| (-13.17), making it the most sensitive comparator. Its near-perfect α/|β| ratio of 1.02 means it implements almost exactly:

```
"Does current input exceed running maximum?"
```

This is the core operation needed for argmax detection!

## Summary

| Aspect | XGBoost | RNN |
|--------|---------|-----|
| Boundary type | Axis-aligned | Diagonal |
| Comparison x[i] > x[j] | Many rectangles | One hyperplane |
| Parameters for 89% | ~10,000s splits | 432 |
| Overfitting | Severe | Minimal |
| Natural fit for comparisons | No | Yes |

The RNN's linear+ReLU structure is naturally suited for comparison tasks because:
1. Linear combinations can directly compute differences (x[i] - x[j])
2. ReLU creates decision boundaries at the zero-crossing
3. Recurrence maintains running statistics for comparison

This explains why a tiny RNN (432 params) matches XGBoost (10,000+ splits) on this task - the RNN's inductive bias aligns perfectly with the comparison structure of 2nd argmax.
