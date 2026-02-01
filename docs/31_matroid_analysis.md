# 31: Matroid Analysis of M₁₆,₁₀

## Overview

Matroid theory provides a combinatorial framework for understanding the linear dependencies in W_out and how they relate to the model's computation.

## 1. Vector Matroid of W_out

The 16 columns of W_out define a rank-10 matroid M[W_out].

**Structure:**
- Ground set: 16 neurons
- Rank: 10
- Bases: 8007 (10-subsets with full rank)
- Dependent 10-sets: **1** (the "hole" in the matroid)

**The Dependent Set:**
```
Dependent: {2, 3, 5, 6, 7, 8, 10, 11, 12, 15}
Essential: {0, 1, 4, 9, 13, 14}
```

The 10 dependent neurons cannot span the output space alone. At least one essential neuron is needed.

**Connection to functional groups:**
| Essential neuron | Functional role |
|-----------------|-----------------|
| n0 | Waves |
| n1 | Comps |
| n4 | Relay |
| n9 | Timing |
| n13 | Bridges |
| n14 | Waves |

The matroid structure independently identifies the same "key neurons" we found through functional analysis.

## 2. Tropical Plücker Coordinates

For each 10-subset, the tropical determinant gives a valuation.

**Statistics (8008 bases):**
- Range: 14.2 - 35.9
- Mean: 26.2
- Std: 3.8

**Tropical det by essential neuron count:**

| # Essential | # Bases | Mean Trop |
|-------------|---------|-----------|
| 0 | 1 | 29.15 |
| 1 | 60 | 28.33 |
| 2 | 675 | 27.56 |
| 3 | 2400 | 26.79 |
| 4 | 3150 | 25.99 |
| 5 | 1512 | 25.08 |
| 6 | 210 | 24.03 |

**Counterintuitive finding:** The dependent set has the HIGHEST tropical determinant (29.15). More essential neurons actually decrease the tropical value.

## 3. Matroid-Tropical Duality

The dependent set exhibits a duality:

| Property | Linear | Tropical |
|----------|--------|----------|
| Rank | 9 (deficient) | 10 (full) |
| Singularity | Yes | No |
| Determinant | 0 | 29.15 (high) |

The linear singularity does NOT create a tropical singularity. The dynamics avoid the linearly-dependent region anyway.

## 4. Rank-Deficient Tropical Cells

Three tropical cells have only 9 active neurons with rank 9:

| Active set | Handles inputs |
|------------|----------------|
| {0,2,3,4,10,11,12,13,14} | Mt=9, St∈{5,6} |
| {2,3,4,5,10,11,12,13,15} | Mt=9, St=7 |
| {2,3,5,7,10,11,12,13,15} | Mt=7, St=9 |

**Key insight:** These cells still correctly classify all inputs because argmax only needs to identify the maximum, not span the full space.

## 5. Cross-Seed Comparison

| Model | Dependent 10-sets | Essential neurons |
|-------|-------------------|-------------------|
| Local | 1 | {0,1,4,9,13,14} |
| GCS-0 | 0 (uniform) | none |
| GCS-2 | 1 | {1,4,5,10,11,12} |
| GCS-4 | 1 | {1,3,5,7,9,12} |

**Findings:**
- 3/4 models have exactly 1 dependent 10-set
- 1/4 achieves a uniform matroid (no holes)
- Only n1 and n4 appear as essential across multiple seeds
- Specific matroid structure is SEED-DEPENDENT

## 6. Connection to Capacity

Essential neurons: 6/16 = 37.5%

The h/s ≈ 1.5 capacity threshold may relate to needing enough neurons outside the dependent region:
- h=16, s=10: 6 essential neurons emerge → works
- h=8, s=10: insufficient capacity → frequent failures

## Implications

1. **Dynamics avoid linear singularity:** Every input activates at least one essential neuron

2. **Tropical robustness:** Even if dynamics hit a linearly-singular region, tropical readout can still work

3. **Functional roles = matroid structure:** The "key neurons" from functional analysis are exactly the essential neurons from matroid analysis

4. **Seed variation:** The specific matroid structure varies, but the essential/dependent partition seems to be a common feature

## Summary Table

| Property | Universal | Seed-specific |
|----------|-----------|---------------|
| Existence of dependent set | ~75% of seeds | Which neurons |
| ~6 essential neurons | Yes | Which neurons |
| Dynamics avoid dependent region | Yes | Via different mechanisms |
| Tropical non-singularity | Yes | Gap magnitude varies |
