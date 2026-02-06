# 44: Eigenvalue Properties of Phi_post

## Summary

The transition matrices `Phi_post[(gap, dir)]` that propagate the hidden state from the second impulse to the end of the sequence have sharply different spectral properties for **fwd** (M first) versus **rev** (S first) directions. Fwd is near-volume-preserving with spectral radius ~1.0; rev is exponentially amplifying with spectral radius up to ~2.9. This asymmetry is the core mechanism enabling position discrimination.

## Background

With the canonical D sequence known at every timestep, the post-second-impulse dynamics are piecewise-linear:

```
h[t] = diag(D[t]) @ W_hh @ h[t-1]
```

The cumulative transition matrix `Phi_post[(gap, dir)][steps]` is the product:

```
Phi = B[steps] @ B[steps-1] @ ... @ B[1]
where B[j] = diag(D_gap[j]) @ W_hh
```

This maps h_second (the hidden state at the second impulse) to h_final. Since logits = W_out @ h_final, the effective logit map is `L = W_out @ Phi_post @ h_second`.

## Spectral Radius: Fwd vs Rev

The dominant eigenvalue tells us whether Phi amplifies or contracts the hidden state.

```
gap | fwd rho | rev rho | rev/fwd
----|---------|---------|--------
  1 |  1.047  |  2.740  |  2.62x
  2 |  0.761  |  2.914  |  3.83x
  3 |  0.999  |  2.607  |  2.61x
  4 |  0.985  |  2.309  |  2.34x
  5 |  1.022  |  2.168  |  2.12x
  6 |  1.025  |  2.103  |  2.05x
  7 |  1.049  |  1.872  |  1.78x
  8 |  1.007  |  1.371  |  1.36x
```

**Fwd** (M arrives first): rho stays in the range 0.76-1.05 across all gaps. The hidden state is approximately preserved in magnitude. This is a near-unitary regime.

**Rev** (S arrives first): rho grows to 2.9 at gap=2, monotonically decreasing with gap since there are fewer steps to accumulate. Rev Phi_post exponentially amplifies one direction in hidden space.

### Rev growth rate

Fitting log(rho) vs steps gives a per-step growth factor of **~1.1x** (correlation r > 0.9):

```
gap=1 rev: 1.096x/step  (rho: 1.37 -> 1.91 -> 2.10 -> 2.16 -> 2.32 -> 2.61 -> 2.93 -> 2.74)
gap=2 rev: 1.113x/step  (rho: 1.37 -> 1.91 -> 2.09 -> 2.16 -> 2.31 -> 2.61 -> 2.91)
gap=3 rev: 1.117x/step  (rho: 1.36 -> 1.91 -> 2.09 -> 2.17 -> 2.31 -> 2.61)
gap=4 rev: 1.127x/step  (rho: 1.36 -> 1.91 -> 2.09 -> 2.17 -> 2.31)
```

The slight uptick in growth rate at larger gaps reflects the initial B[1] matrix (which varies by gap) having a larger spectral radius for certain gap values.

## Spectral Gap

The spectral gap (ratio of first to second eigenvalue magnitude) determines how quickly the matrix action converges to the dominant mode.

```
fwd: spectral gap = 1.04 - 1.34  (small, multiple modes contribute)
rev: spectral gap = 1.78 - 4.24  (large, dominant mode dominates)
```

**Interpretation**: Fwd Phi_post distributes information across several modes, making the final state depend on multiple components of h_second. Rev Phi_post concentrates information into a single amplified direction, making the output highly predictable from one component.

## Rank Collapse

Each per-step matrix B[j] has rank = |D[j]| (typically 11-14). The cumulative product's rank decreases:

```
Steps:   1    2    3    4    5    6    7    8
Rank:   12   11   10-11  10-11  10-11  10-11  10-11  11
```

Rank stabilizes at ~10-11 within 2-3 steps. Five to six dimensions of the 16-dimensional hidden space are systematically projected away. This nullspace is the subspace of neurons that are consistently deactivated by the canonical D sequence.

## Eigenvalue Structure

### Fwd: near-unitary with complex pairs

Representative eigenvalue spectrum (gap=3, fwd, 6 steps):

```
|lam| = 0.999  @ 0.000pi    (real, near unit circle)
|lam| = 0.747  @ +/-0.519pi (complex pair, rotating)
|lam| = 0.452  @ 0.000pi    (real, contracting)
|lam| = 0.176  @ pi         (real, sign-flipping)
|lam| = 0.124  @ 0.000pi
(+ 6 zero eigenvalues)
```

Multiple eigenvalues cluster near the unit circle (0.8 < |lam| < 1.2), with complex pairs indicating rotational dynamics. The determinant is zero due to the nullspace, but the non-zero eigenvalues are spread across magnitudes, giving a moderate condition number on the image.

### Rev: dominant real eigenvalue

All dominant eigenvalues are **real and positive** -- no oscillation in the leading mode. The dominant eigenvalue separates from the rest by a factor of 2-4x.

### Non-normality

Phi_post matrices are far from normal (`||Phi^T Phi - Phi Phi^T|| / ||Phi||^2 ~ 0.8-1.2`), meaning eigenvectors are non-orthogonal. The condition numbers on the image range from 10^4 to 10^9. Despite this, the dominant eigenvalue direction is well-separated enough that the spectral gap analysis remains meaningful.

## Amplification Geometry

### Rev: which direction is amplified?

SVD analysis of the final cumulative Phi_post (rev) reveals a consistent pattern:

| Component | Top neurons |
|-----------|------------|
| Input direction (v_1) | neurons 7, 4, 6, 8 |
| Output direction (u_1) | neurons 10, 12, 0, 14 |

The input direction that gets maximally amplified is dominated by neurons {7, 6, 4, 8} across all gaps. These are the neurons most sensitive to the ordering of the two impulses. The output direction, projected through W_out, creates logit patterns that differentiate between positions.

### What W_out sees: the effective logit map L

The matrix `L = W_out @ Phi_post` maps directly from h_second (16-dim) to logits (10-dim). Its singular value decomposition reveals:

```
gap=1 fwd: sv = [17.4, 6.5, 4.9, ...]  top-3 explain 97.1%
gap=1 rev: sv = [18.0, 9.4, 4.9, ...]  top-3 explain 94.0%
gap=3 fwd: sv = [13.2, 7.7, 4.5, ...]  top-3 explain 94.1%
gap=3 rev: sv = [37.3, 7.4, 5.2, ...]  top-3 explain 98.2%
gap=5 fwd: sv = [22.3, 10.4, 6.7, ...] top-3 explain 92.5%
gap=5 rev: sv = [42.2, 12.8, 6.9, ...] top-3 explain 96.7%
```

The logit map is effectively **2-3 dimensional**. For rev, the top singular value alone explains 70-93% of logit variance, consistent with the large spectral gap concentrating the output.

The effective dimension (singular values > 10% of sigma_1) is 4-5 for both fwd and rev.

## Why This Matters for Discrimination

The linear logit formula is:

```
logits = first_val * a + second_val * b
where a = W_out @ Phi @ D_mask @ w_first[gap]
      b = W_out @ Phi @ D_mask @ W_ih
```

The fwd/rev asymmetry in Phi_post means:

1. **Fwd case** (M first): Phi is near-unitary. Both `a` and `b` are moderate. The logit pattern depends comparably on both impulses.

2. **Rev case** (S first): Phi amplifies by ~2-3x. The `b` coefficient (second impulse = M in this case) gets amplified relative to `a`. The dominant singular direction concentrates the output.

Since the network must output position S regardless of whether M or S arrives first, this asymmetry is how it compensates for ordering. When the second impulse is larger (rev: M arrives second), Phi_post's amplification boosts its contribution, and the W_out phase wheel decodes the resulting pattern to the correct position.

## Convergence of Dominant Eigenvector

The dominant eigenvector of Phi_post does **not** converge quickly to a stable direction as steps increase. Cosine similarity between intermediate and final dominant eigenvectors:

```
gap=1 fwd:  steps 1-8: cos_sim = 0.11, 0.74, 0.53, 0.10, 0.82, 0.24, 0.65, 1.00
gap=1 rev:  steps 1-8: cos_sim = 0.59, 0.19, 0.40, 0.49, 0.67, 0.09, 0.66, 1.00
```

This non-monotonic convergence reflects the fact that individual B[j] matrices vary substantially (different D sets at each step), so the cumulative product doesn't settle into a fixed-point direction until enough steps accumulate. The dominant eigenvalue magnitude converges more smoothly than its direction.

## Related

- `src/mechanistic_discrimination.py` - Implementation with precomputed Phi_post
- [43: Mechanistic Discrimination](43_mechanistic_discrimination.md) - Overall mechanism
- [42: Mechanistic Accuracy Estimate](42_mechanistic_accuracy_estimate.md) - Accuracy prediction
- [34: Fwd/Rev Mechanism](34_fwd_rev_mechanism_explained.md) - Ordering-dependent dynamics
