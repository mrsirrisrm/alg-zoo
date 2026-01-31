# 29: Tropical Geometry Analysis

## Overview

The RNN's computation can be viewed through the lens of tropical geometry. Each ReLU is a tropical operation: `relu(x) = max(x, 0) = x ⊕ 0` in the tropical semiring (ℝ ∪ {−∞}, max, +). The 160 ReLU activations (16 neurons × 10 timesteps) define a tropical hyperplane arrangement. Each input selects a **tropical cell** — a region where all activation patterns are fixed and the input→output map is affine.

## Key Findings

### 1. Cell Structure at t=9

The 90 clean pairs occupy **38 distinct tropical cells** at t=9 (based on which neurons are active). Many pairs share cells:

| Cell | Size | Active | Dead neurons | Targets |
|------|------|--------|--------------|---------|
| 0 | 23 | 13/16 | n4, n5, n9 | 0,1,2,3,4,5,6 |
| 1 | 8 | 12/16 | n4, n5, n9, n10 | 0,1,2,3,4,6 |
| 2 | 5 | 14/16 | n4, n9 | 0, 8 |
| ... | ... | ... | ... | ... |

The dominant cell contains **23 pairs** (12 forward, 11 reversed) mapping to 7 different targets — all handled by the same 13-neuron affine map.

### 2. ReLU Suppression is Critical

Bypassing ReLU at t=9 (using pre-activation instead of post-ReLU) drops accuracy from **100% to 17.8%**. The dead neurons carry enormous wrong-direction signals:

- Mean |dead neuron contribution to target| = 64.3
- Mean margin change without ReLU = −110.6

The ReLU isn't a mild nonlinearity — it's an essential filter that silences massive interference.

### 3. W_out is Tropically Full-Rank

The **tropical rank** measures discriminative capacity: the largest k such that some k×k submatrix has a unique optimal assignment (tropically non-singular).

| Matrix | Shape | Ordinary Rank | Tropical Rank |
|--------|-------|---------------|---------------|
| Full W_out | 10×16 | 10 | 10 |
| Cell 0 effective | 10×13 | 10 | 10 |
| Cell 7 effective | 10×10 | 10 | 10 |
| Cell 14 effective | 10×9 | 9 | 9 |

Every cell with ≥10 active neurons has **tropical rank 10** — maximum discriminative capacity.

### 4. Zero Tropical Singularity

For the dominant cell's 10×13 effective W_out, all **286** possible 10×10 submatrices are tropically non-singular. There are no ties in the optimal assignment problem — W_out has learned weights where every choice of 10 neurons gives a unique optimal readout.

This is the definition of **tropical genericity**: the matrix has no accidental structure that would cause assignment ambiguity.

### 5. Cross-Cell Robustness

Applying the wrong cell's activation mask (zeroing different neurons) breaks **61.6%** of predictions. W_out has not learned a universal readout — it relies on the correct neurons being active in each cell.

### 6. Forward vs Reversed in Tropical Terms

Forward and reversed pairs often share the same tropical cell at t=9:
- 23 pairs in the dominant cell include both orderings
- The "gap subtraction" for reversed happens through **continuous values within the shared cell**, not through different cells

The key tropical asymmetry is at the **second impulse**:
- Forward (S=0.8 arrives last): partial clip, some comps survive
- Reversed (M=1.0 arrives last): full clip, all comps zeroed

This creates different activation patterns during rebuild, but many pairs converge to the same cell by t=9.

### 7. The Rank-Deficient Cases

Four pairs have only 9 active neurons (tropical rank 9):
- (M9, S5), (M9, S6), (M9, S7): all 4 comps dead
- (M7, S9): 3 comps dead

These are Mt=9 forward pairs with minimal rebuild time. Despite all comps being dead, they still achieve correct predictions with margins 5.9–12.8. The waves, bridges, n2, and n4 carry sufficient signal.

## Interpretation

The model has achieved a tropically clean solution:

1. **W_out is tropically generic**: no singularities, unique assignments everywhere
2. **The ReLU layer does essential work**: suppressing neurons that would corrupt readout
3. **Cells share structure**: most pairs use 12–13 of 16 neurons, differing by 1–3 neurons
4. **Discrimination is within-cell**: 23 pairs in one cell, discriminated by continuous values

The tropical perspective reveals that the "holographic encoding" finding is about **selective silencing**: the dynamics route each input to a cell where the right neurons are suppressed, leaving a clean signal that W_out linearly decodes.

## Scripts

See `src/tropical_analysis.py`, `src/tropical_cells_readout.py`, `src/tropical_rank.py`, `src/tropical_heatmap.py`.
