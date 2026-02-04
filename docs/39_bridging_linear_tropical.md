# 39: Bridging Linear and Tropical Analysis

## The Question

How do we connect two seemingly disparate views of M₁₆,₁₀?

1. **Eigenvalue analysis** (linear): W_hh has eigenvalues, complex rotation, growth/decay modes
2. **Tropical skeleton** (nonlinear): 38 cells, discrete activation patterns, piecewise-affine regions

These aren't separate phenomena—they're two aspects of the same dynamics. This document explores frameworks that bridge them.

## Framework 1: Piecewise Linear Dynamical Systems (PLDS)

### Concept

View the RNN as a **hybrid automaton**:
- **Discrete states** = tropical cells (activation patterns)
- **Continuous dynamics** = affine map within each cell
- **Transitions** = crossing ReLU boundaries

### Key Finding

The 180 trajectories (90 pairs × 2 directions) produce **89 unique cell sequences**. This is nearly one sequence per trajectory—the paths are highly diverse.

Cell-specific eigenvalues evolve along the trajectory:

| Timestep | # Active | max\|λ\| | Trend |
|----------|----------|----------|-------|
| t=0 | 0 | 0.00 | — |
| t=1 | 6 | 0.69 | ↑ |
| t=2 | 13 | 1.00 | ↑ |
| t=3-7 | 12 | 0.99-1.02 | → stable |
| t=8-9 | 11-13 | 1.00-1.01 | → stable |

**Insight**: After the first impulse, the network enters a **marginally stable regime** (max |λ| ≈ 1.0). This explains why the hidden state doesn't explode or collapse—it's tuned to the edge of stability.

## Framework 2: Koopman Operator Lifting

### Concept

The Koopman operator lifts nonlinear dynamics to a (possibly infinite-dimensional) linear system on **observables**:

```
g(h[t+1]) = K @ g(h[t])
```

where g is a set of observable functions and K is linear.

For ReLU networks, natural observables include:
- h (the activations themselves)
- h² (quadratic terms)
- max(0, w @ h) for various directions w

### Key Finding

**Quadratic lifting (h, h²) achieves R² = 96.6%** for predicting the next lifted state.

This is much better than any linear approach we tried! The quadratic observables capture most of the ReLU nonlinearity.

**Koopman eigenvalues** (approximate, 32D lifted space):

| Mode | Eigenvalue | \|λ\| |
|------|------------|-------|
| 0-1 | -0.67 ± 0.74j | 1.00 |
| 2-3 | -0.04 ± 0.96j | 0.96 |
| 4 | +0.95 | 0.95 |
| 5 | -0.94 | 0.94 |

Compare to original W_hh:

| Mode | Eigenvalue | \|λ\| |
|------|------------|-------|
| 0 | -1.27 | 1.27 |
| 1 | +1.17 | 1.17 |
| 2-3 | 0.44 ± 1.03j | 1.12 |

**Insight**: The Koopman eigenvalues are **closer to unit circle** than W_hh eigenvalues. The lifting absorbs the ReLU's bounding effect, producing effective dynamics that are stable.

## Framework 3: Cell Transition Analysis

### Concept

View cell-to-cell transitions as a Markov chain:
- States = tropical cells
- Transitions = which cell follows which

### Key Finding

**74 unique cell transitions** across all trajectories.

Top transitions:

| From | To | Count |
|------|-----|-------|
| 0 active → 0 active | 336 | (before first impulse) |
| 6 active → 13 active | 162 | (first impulse expansion) |
| 0 active → 6 active | 144 | (first impulse entry) |
| 13 active → 12 active | 112 | (settling) |
| 12 active → 12 active | many | (stable regime) |

**Insight**: The transition structure has a clear pattern:
1. **Empty → 6 neurons** (first impulse)
2. **6 → 13 neurons** (expansion)
3. **13 → 12 → 12 → ...** (stable cycling with ~12 active)

This is the **phase wheel** viewed through the tropical lens.

## Framework 4: Trajectory Jacobian

### Concept

The **trajectory Jacobian** is the product of cell-specific transition matrices:

```
J = D[9] @ W_hh @ D[8] @ W_hh @ ... @ D[0] @ W_hh
```

where D[t] is the diagonal activation mask at timestep t.

This single matrix encodes:
- **Tropical structure**: Which D[t] masks are used
- **Linear dynamics**: Each D[t] @ W_hh has its own spectrum
- **Net sensitivity**: How perturbations propagate

### Key Finding

**Trajectory Jacobians have low-rank structure**: Only 4 PCs explain 90% of the variance across all 90 forward Jacobians.

| PC | Variance Explained | Cumulative |
|----|-------------------|------------|
| 1 | 61.0% | 61.0% |
| 2 | 19.3% | 80.3% |
| 3 | 9.7% | 89.9% |
| 4 | 3.1% | 93.0% |

However, the **mean Jacobian** has tiny eigenvalues (max |λ| = 0.026). This means:
- Individual Jacobians have structure
- But they point in different directions, canceling on average

**Insight**: The discrimination signal is in the **variation** of Jacobians, not their mean.

## Synthesis: The Three Perspectives

### View 1: Eigenvalue (Linear)

W_hh has eigenvalues > 1, which would cause exponential growth. But...

### View 2: Tropical (Nonlinear)

ReLU clips neurons, creating discrete activation patterns. The network visits different cells, each with different effective eigenvalues.

### View 3: Bridge (Koopman/Jacobian)

The **effective dynamics** (incorporating ReLU) have eigenvalues **near 1**:
- Koopman eigenvalues: max |λ| ≈ 1.00
- Cell-specific eigenvalues: max |λ| ≈ 0.99-1.02
- Trajectory Jacobians: vary, but structured

The network is **tuned to operate at marginal stability** within the ReLU-constrained regime.

## Implications for Mechanistic Estimates

### What the Koopman Finding Means

The 96.6% R² for quadratic lifting suggests:

```
h[t+1] ≈ K @ [h[t]; h[t]²]
```

where K is a **fixed linear operator**. This is almost a weight-based formula!

The remaining 3.4% error comes from:
- Higher-order nonlinearities
- Trajectory-dependent effects not captured by local observables

### Path to Mechanistic Estimate

1. **Derive K from weights**: The approximate Koopman operator K should be expressible in terms of W_ih, W_hh (with some averaging over ReLU patterns)

2. **Propagate through K**: Instead of D[t] @ W_hh @ ... @ D[0], use K^9

3. **Show discrimination**: The K-based propagation should preserve the discriminative offset

### Key Challenge

The Koopman approach still needs the **distribution of hidden states** to be meaningful. The observable g(h) = [h; h²] depends on what h values actually occur.

For a fully mechanistic estimate, we'd need to show that K captures the dynamics for **all** (m, s) pairs from weights alone.

## Summary Table

| Framework | Linear Aspect | Tropical Aspect | Key Insight |
|-----------|---------------|-----------------|-------------|
| PLDS | Cell-specific W_hh eigenvalues | Cell sequence | Marginal stability (|λ| ≈ 1) |
| Koopman | K is linear on observables | Observables encode ReLU | 96.6% R² with (h, h²) |
| Transitions | — | Cell-to-cell graph | Clear entry → expansion → stable pattern |
| Jacobian | Product of linear maps | Product of D[t] masks | Low-rank (4 PCs for 90%) |

## References

- [Tropical Geometry of Deep Neural Networks](https://arxiv.org/abs/1805.07091) — Zhang et al., foundational work
- [Deep learning for universal linear embeddings](https://www.nature.com/articles/s41467-018-07210-0) — Koopman + neural nets
- [Piecewise-linear dynamical systems](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1582080/full) — PLRNN analysis

## Next Steps

1. **Derive K explicitly**: Express the approximate Koopman operator in terms of model weights
2. **Test K-based discrimination**: Does K^9 @ [input; input²] give correct predictions?
3. **Bound the error**: Show the 3.4% Koopman error doesn't flip any margins
