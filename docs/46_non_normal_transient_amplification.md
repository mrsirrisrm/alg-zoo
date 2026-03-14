# 46: Non-Normal Transient Amplification

## Summary

The Phi_post transition matrices are **highly non-normal**, and this non-normality is not incidental — it is **essential** to the network's discrimination ability. Replacing Phi_post with a normal matrix having the same eigenvalues drops accuracy from **100% to 23%** (69/90 pairs broken). The non-normal transient amplification provides the mechanism by which the hidden state flips from an initially wrong discrimination signal to a correct one.

## Non-Normality Metrics

### Henrici Departure from Normality

The Henrici number measures how far a matrix is from normal: `sqrt(||A||_F^2 - sum|lambda_i|^2) / ||A||_F`. A value of 0 means normal; 1 means maximally non-normal.

```
Matrix              | Henrici (relative)
--------------------|-------------------
W_hh                |   0.932
Phi_post g1 fwd     |   0.976
Phi_post g1 rev     |   0.977
Phi_post g3 fwd     |   0.975
Phi_post g3 rev     |   0.978
Phi_post g5 fwd     |   0.964
Phi_post g5 rev     |   0.978
```

All matrices have Henrici departure > 0.93 — **extremely** non-normal. Almost all of the Frobenius norm is in the off-diagonal Schur components, not the eigenvalues.

### Operator Norm vs Spectral Radius

For a normal matrix, `||A||_2 = rho(A)`. The ratio `||A||_2 / rho` measures how much the matrix can amplify beyond what eigenvalues predict.

```
Matrix          |  rho  | ||A||_2 | ||A||/rho | Kreiss K | kappa(V)
----------------|-------|---------|-----------|----------|--------
W_hh            | 1.274 |   5.839 |     4.58  |    5.50  |    125
Phi g1 fwd      | 1.047 |   5.683 |     5.43  |    5.73  |    271
Phi g1 rev      | 2.740 |  13.224 |     4.83  |    5.24  |    594
Phi g3 fwd      | 0.999 |   5.298 |     5.30  |    4.39  | 50,836
Phi g3 rev      | 2.607 |  11.764 |     4.51  |    3.64  |  2,576
Phi g5 fwd      | 1.022 |   5.360 |     5.25  |    9.12  |    281
Phi g5 rev      | 2.168 |  13.282 |     6.13  |    3.87  |    989
```

Key observations:
- **||Phi||/rho ~ 4.5–6.1** for all cases — the matrices can amplify inputs 4-6x beyond what eigenvalues alone would predict
- **Kreiss constants K ~ 3.6–9.1** — significant potential for transient growth; the Kreiss theorem guarantees `sup_n ||A^n||/rho^n >= K`
- **kappa(V) ranges from 125 to 50,836** — eigenvectors are far from orthogonal, especially for gap=3 fwd

### Pseudospectral Radii

The epsilon-pseudospectrum extends significantly beyond eigenvalues:

```
Phi g3 fwd (rho = 0.999):
  eps = 0.1:  pseudospectral radius = 1.321  (+0.32 beyond rho)
  eps = 0.5:  pseudospectral radius = 2.113  (+1.11 beyond rho)
  eps = 1.0:  pseudospectral radius = 2.862  (+1.86 beyond rho)
```

Even a small perturbation (eps = 0.1) pushes the effective spectral radius to 1.32, far beyond the actual rho of 0.999. This means the matrix is highly sensitive and the eigenvalue analysis is misleading — the transient behavior is governed by the pseudospectrum, not the spectrum.

## The Smoking Gun: Margin Trajectory

The margin (logit[target] - logit[competitor]) evolves step by step as Phi_post propagates h_second. The trajectory reveals classic non-normal transient amplification:

### Gap=3 fwd (M at 0, S at 3)

```
step | margin   | ||h||
-----|----------|------
  0  |  -44.43  | 32.70    <-- WRONG DIRECTION
  1  |  +28.17  | 37.15    <-- flips to correct
  2  |  +58.82  | 33.69    <-- PEAK (5.4x final)
  3  |  +42.89  | 35.33
  4  |  +21.80  | 37.27
  5  |   +0.01  | 37.61
  6  |  +10.87  | 35.20    <-- final (correct)
```

The margin starts at -44 (predicting M's position, wrong) and must reach +11 (predicting S's position, correct). The non-normal dynamics create a transient overshoot to +59 at step 2 that then decays. Enough amplitude remains at step 6 for correct discrimination.

### Gap=3 rev (S at 0, M at 3)

```
step | margin   | ||h||
-----|----------|------
  0  |  +50.47  | 26.07    <-- starts correct (PEAK)
  1  |  -35.37  | 37.98    <-- goes wrong
  2  |  -73.09  | 32.01    <-- deeply wrong
  5  |   +9.55  | 36.39    <-- recovers
  6  |  +10.37  | 36.04    <-- final (correct)
```

The rev case starts correct, goes deeply wrong (margin -73), then recovers. Without non-normality, the recovery cannot happen.

Note: the hidden state norm `||h||` stays roughly constant (32-38) throughout — the transient amplification is not about growing the overall signal, but about **rotating it** through intermediate directions before settling into the correct discrimination pattern.

## The Counterfactual: Normal Phi_post

To prove non-normality is essential, we replace each Phi_post with a **normal matrix having the same eigenvalues** (by keeping only the diagonal of the Schur decomposition). This preserves the spectral radius, spectral gap, and all eigenvalue-based properties while removing non-normality.

```
Actual (non-normal Phi):  90/90 correct (100%)
Normal Phi (same eigs):   21/90 correct  (23%)

Non-normality accounts for: 69/90 pairs (77%)
```

**77% of the network's discrimination ability comes from non-normal transient dynamics, not from eigenvalues.** The 21 pairs that still work with normal Phi are the ones where Phi_post is close to identity (gap=8, gap=9 with 0-1 steps).

The failures span all gaps (1-8) and both directions (fwd and rev).

## How Non-Normality Creates Discrimination

### The mechanism in fwd

1. h_second is the hidden state at the second impulse
2. h_second projects strongly onto **non-eigenvector** directions (only 0.02–10.8% in the dominant eigenvector)
3. The non-normal Phi rotates h_second through intermediate directions where the margin temporarily overshoots
4. The transient amplitude decays, but enough remains at readout for correct discrimination
5. A normal matrix with the same eigenvalues would keep h_second near its initial projection, which gives the wrong answer

### The mechanism in rev

1. h_second initially has the correct margin (M's energy enters strongly)
2. Phi_post's non-normal dynamics rotate the state through a wrong-answer transient
3. The state then recovers to the correct answer at the final step
4. The Kreiss constant K ~ 4-5 bounds the maximum transient excursion

### Eigenvector conditioning

The huge kappa(V) values (up to 50,836 for gap=3 fwd) mean the eigenvector basis is nearly degenerate. Input vectors project with wildly different amplitudes onto different eigenvectors depending on small angle differences. This is the mathematical source of the transient: non-orthogonal eigenvectors create constructive/destructive interference patterns as different modes evolve at different rates.

## Kreiss Constants and Growth Bounds

The Kreiss constant K provides a lower bound on the maximum transient growth: `sup_n ||Phi^n x|| / ||x|| >= K * rho^n` for some initial x.

```
         | Kreiss K | K/rho
---------|----------|------
W_hh     |    5.50  | 4.32
g1 fwd   |    5.73  | 5.47
g3 fwd   |    4.39  | 4.39
g5 fwd   |    9.12  | 8.93
g1 rev   |    5.24  | 1.91
g3 rev   |    3.64  | 1.40
g5 rev   |    3.87  | 1.78
```

Fwd has higher K/rho ratios (4-9x) than rev (1.4-1.9x). This makes sense: fwd has rho ~ 1.0, so it needs more transient amplification to create discrimination. Rev already has rho ~ 2.6, so the eigenvalue-driven growth does more of the work, with non-normality providing a smaller (but still essential) correction.

## Implications

1. **Eigenvalue analysis is deeply misleading for this RNN.** The spectral radius alone explains almost nothing about discrimination. The operator norm, Kreiss constant, and pseudospectral radius are the relevant quantities.

2. **The network exploits non-normality as a computational resource.** Training has shaped the eigenvector geometry (not just eigenvalues) to create transient dynamics that flip discrimination signals from wrong to right.

3. **This is a 16-neuron network.** Non-normal transient amplification is typically studied in high-dimensional fluid dynamics (Navier-Stokes) or atmospheric science. Finding it as the dominant mechanism in a tiny RNN is surprising and suggests it may be widespread in small trained networks.

4. **The margin trajectory is the key observable.** Norm trajectories (||h[k]||) show only modest growth (~1.1-1.5x), but margin trajectories show dramatic 5x overshoots and sign flips. The non-normality preferentially amplifies the **discrimination-relevant** projection, not the overall signal magnitude.

## The Circuit: Neuron-Level Mechanism

Tracing the energy flows through W_hh reveals a specific circuit that implements the non-normal transient. The analysis below uses gap=3 fwd (M at 0, S at 3) as the canonical example, but the pattern is universal across gaps.

### Three classes of neurons

At step 0 (immediately after the second impulse), neurons partition into three functional classes:

| Class | Neurons | Properties |
|-------|---------|-----------|
| **Batteries** | n2 (h=18.4), n11 (h=11.1), n13 (h=8.0) | Large h, small \|dW\| — store energy without contributing to discrimination |
| **Discrimination outputs** | n7 (dW=+10.1), n1 (dW=-12.0), n14 (dW=+6.5), n8 (dW=+2.1), n10 (dW=-4.0) | Large \|dW\| — their activations directly determine the margin |
| **Decaying initial** | n10 (h=22, dW=-4.0) | Large h AND large negative dW — causes the initial wrong answer (margin contribution -89) |

### The energy redistribution

Neuron 2 is the primary battery. Via W_hh, it broadcasts to discrimination neurons in a single step:

```
                    W_hh[7,2] = +0.67
  n2 (battery) ---------> n7 (winner, dW = +10.1)
     h = 18.4                    |
                                 | W_hh[1,7] = -1.05 (inhibition)
                                 v
     W_hh[1,2] = +0.50 -> n1 (confounder, dW = -12.0)
```

n2 excites **both** n7 (the top winner) and n1 (the top confounder) simultaneously. But n7 gets more energy (0.67 vs 0.50) and crucially, n7 **inhibits n1** via W_hh[1,7] = -1.05.

### Winner-confounder competition

At step 1, the margin change decomposes as:

```
Winners:     n7 (+100), n14 (+64), n10 decay (+36), n8 (+21)  = +231
Confounders: n1 (-119), n6 (-32), n0 (-6)                     = -157
Net change: +73 (margin flips from -44 to +28)
```

n7 and n1 are in direct competition:
- n7 contributes +127 to the margin (via dW[7]=+10.1, h[7]=12.6)
- n1 contributes -119 to the margin (via dW[1]=-12.0, h[1]=10.0)

n7 wins because: (a) it receives more energy from n2, and (b) it actively suppresses n1 via cross-inhibition.

### The inhibition resolves the transient

Tracking the n7-to-n1 inhibition over time:

```
step | h[7] | h[1] | W[1,7]*h[7] | other input | net input to n1
-----|------|------|-------------|-------------|----------------
  0  |  2.7 |  0.0 |      -2.9  |     +12.8   |   +10.0
  1  | 12.6 | 10.0 |     -13.2  |     +24.8   |   +11.6
  2  | 12.0 | 11.6 |     -12.5  |     +24.3   |   +11.8
  3  | 11.5 | 11.8 |     -12.1  |     +22.9   |   +10.9
  4  | 11.6 | 10.9 |     -12.2  |     +21.7   |    +9.5
  5  | 12.0 |  9.5 |     -12.6  |     +20.8   |    +8.2
```

n7 stabilizes quickly (h ~ 12) while n1 gradually decays (h: 10 -> 8.2) because n7's inhibition (-12 to -13 per step) increasingly dominates n1's excitatory inputs.

### Universality across gaps

The three-class pattern holds across all gaps:

| Gap | Top winner | Top confounder | Winner sum | Confounder sum | Net |
|-----|-----------|----------------|------------|---------------|-----|
| 1 | n7 (+34) | n8 (-32) | +86 | -61 | +25 |
| 2 | n7 (+78) | n1 (-57) | +138 | -118 | +20 |
| 3 | n7 (+100) | n1 (-119) | +231 | -157 | +73 |
| 4 | n7 (+86) | n1 (-126) | +229 | -187 | +42 |
| 5 | n8 (+121) | n1 (-58) | +170 | -158 | +12 |
| 6 | n8 (+94) | n6 (-35) | +104 | -107 | -3 |
| 7 | n8 (+42) | n7 (-29) | +81 | -58 | +23 |

n7 is the dominant winner for gaps 1-4; n8 takes over for gaps 5-7 (where n7's dW flips sign). The competition between winners and confounders is always tight (ratio 1.1-1.5x), with the cross-inhibition circuit tipping the balance.

### The 180-degree rotation

In the (discrimination, orthogonal) projection plane, h[k] universally rotates ~170-190 degrees within the first 1-2 steps:

- **Fwd**: starts at ~-90 degrees (wrong), rotates to ~+80 degrees (correct)
- **Rev**: starts at ~-70 degrees (correct), rotates to ~+100 degrees (wrong), then slowly recovers

This rotation **is** the non-normal transient — the energy redistribution from batteries to discrimination neurons causes a rapid reorientation of the hidden state vector in the subspace that W_out reads out.

### Why normal matrices fail

A normal matrix has orthogonal eigenvectors, so modes evolve independently. The battery neurons (n2, n11, n13) occupy different eigenvector components than the discrimination neurons (n1, n7, n8, n14). With orthogonal modes, there is **no pathway** for energy to flow from batteries to discrimination outputs. The initial wrong answer (dominated by n10) persists.

The non-normal W_hh has the off-diagonal connections (W_hh[7,2]=+0.67, W_hh[1,2]=+0.50, etc.) that create cross-mode energy transfer — this is mathematically equivalent to the upper-triangular Schur entries, which have 6.1x the norm of the diagonal entries.

## The Margin Trajectory as a Phase Portrait

The complete margin trajectories for gap=3:

```
Fwd (M at 0, S at 3):                   Rev (S at 0, M at 3):
  step 0: -44  [wrong]                    step 0: +50  [correct, peak]
  step 1: +28  [flips!]                   step 1: -35  [wrong!]
  step 2: +59  [overshoot peak]           step 2: -73  [deep wrong]
  step 3: +43                             step 3: -60
  step 4: +22                             step 4: -21
  step 5:  +0  [nearly zero!]             step 5: +10  [recovers]
  step 6: +11  [final, correct]           step 6: +10  [final, correct]
```

Note: the hidden state **norms** barely change (32-38 throughout). The transient is entirely about rotation, not amplification.

## Related

- [44: Phi_post Eigenvalue Analysis](44_phi_post_eigenvalue_analysis.md) — Spectral properties
- [45: Eigenvector Discrimination](45_eigenvector_discrimination_mechanism.md) — Left/right eigenvector structure
- [43: Mechanistic Discrimination](43_mechanistic_discrimination.md) — Overall mechanism
- `src/mechanistic_discrimination.py` — Implementation

## References

- Trefethen & Embree, *Spectra and Pseudospectra* (2005) — Non-normal matrices and pseudospectral theory
- Kreiss (1962) — The Kreiss matrix theorem and resolvent bounds
- Henrici (1962) — Departure from normality metric
