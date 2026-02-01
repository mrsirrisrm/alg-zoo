# 30: Model Comparison — M₂,₂ vs M₄,₃ vs M₁₆,₁₀

## Overview

The 2nd argmax task admits different solutions at different scales. Comparing three models reveals how the computational strategy evolves with problem size.

| Model | Hidden | SeqLen | # Inputs | Parameters |
|-------|--------|--------|----------|------------|
| M₂,₂  | 2      | 2      | 1        | ~10        |
| M₄,₃  | 4      | 3      | 6        | ~30        |
| M₁₆,₁₀| 16     | 10     | 90       | ~450       |

## Weight Matrix Structure

### W_ih (Input → Hidden)

| Model | Structure | Interpretation |
|-------|-----------|----------------|
| M₂,₂  | [+1, −1] | One neuron tracks +, one tracks − |
| M₄,₃  | [+,+,−,−] | Two pairs: (n0,n1) excited by +, (n2,n3) by − |
| M₁₆,₁₀| Mixed signs | Complex distributed encoding |

### W_hh Eigenvalues

| Model | Eigenvalues | Interpretation |
|-------|-------------|----------------|
| M₂,₂  | 0, **−2** | Pure damping, no dynamics |
| M₄,₃  | −1±1.7j, −0.2, 0 | All negative real parts, damped oscillation |
| M₁₆,₁₀| 7 positive, 9 negative | Sustained oscillation + selective damping |

All models have **negative eigenvalues** for damping. M₁₆,₁₀ uniquely has positive eigenvalues that sustain oscillations during the "rebuild phase."

### W_out Structure

| Model | Rank | Tropical Rank | Tropical Gaps | vs Random |
|-------|------|---------------|---------------|-----------|
| M₂,₂  | 1    | 2             | —             | — |
| M₄,₃  | ~2 (σ₃=0.03) | 3 | **0.01** | 963× smaller |
| M₁₆,₁₀| 10   | 10            | **4.0** | 1.9× smaller |

M₄,₃'s W_out has paired columns: col0 ≈ −col3, col1 ≈ −col2. This creates structural near-singularity in tropical terms.

## Computational Strategies

### M₂,₂: Pure Comparator

```
h₀ = Σ xₜ        (sum of inputs)
h₁ = −Σ xₜ       (negated sum)
out = W_out @ h = [h₀−h₁, h₁−h₀]
```

- Rank-1 readout extracts `h₀ − h₁`
- No dynamics needed (eigenvalue 0 holds memory, −2 damps transients)
- 4 ReLUs, ~2 tropical cells

### M₄,₃: Binary Switch

```
if last input is largest:
    (n0, n1) active → one output pattern
else:
    (n2, n3) active → complementary output pattern
```

- Two activation patterns at final timestep
- W_out columns paired: switching halves flips outputs
- 12 ReLUs, 2 tropical cells at t=2
- Near-singular tropical structure (gap 0.005)

### M₁₆,₁₀: Holographic Encoding + Selective Suppression

```
Functional groups:
- n4 (Relay): passes first impulse
- n9: records second impulse timing
- Comps (n1,n6,n7,n8): rebuild from clipping
- Waves (n0,n10,n11,n12,n14): oscillate with mixed eigenvalues
- Bridges (n3,n5,n13,n15): couple groups
- n2: latch with hysteresis

Readout: All 16 neurons contribute, ReLU suppresses wrong-direction signals
```

- 38 tropical cells at t=9
- Most pairs share 12–13 active neurons
- Discrimination is within-cell via continuous values
- Tropically generic W_out with robust gaps (0.1–0.3)

## Tropical Geometry Perspective

| Aspect | M₂,₂ | M₄,₃ | M₁₆,₁₀ |
|--------|------|------|--------|
| Total ReLUs | 4 | 12 | 160 |
| Cells at final t | ~2 | 2 | 38 |
| Tropical genericity | Yes | **963× closer** to singularity | **1.9× closer** to singularity |

### Why M₄,₃ is Near-Singular

The column-pairing structure creates near-ties:
```
col0 + col3 ≈ 0   (pair 1 cancels)
col1 + col2 ≈ 0   (pair 2 cancels)
```

When selecting any 3×3 submatrix, paired columns create tropical determinants that nearly tie. The learned solution uses only 2 effective dimensions despite having 4 neurons.

### Why M₁₆,₁₀ is Less Singular

With 16 neurons and rich dynamics:
- Columns are more independent (no obvious pairing structure)
- Tropical gaps ~300× larger than M₄,₃
- All 286 possible 10×10 submatrices are non-singular
- Still 1.9× closer to singularity than random — but far less extreme than M₄,₃

## Key Insight: Dimension Determines Strategy

1. **M₂,₂**: Minimal parameters → pure comparison
2. **M₄,₃**: Limited parameters → binary switch with column pairing
3. **M₁₆,₁₀**: Ample parameters → rich dynamics with holographic encoding

The larger model doesn't just have more capacity — it finds a qualitatively different solution that is:
- More dynamically complex (oscillations, rebuild phases)
- Less tropically singular (gaps 300× larger than M₄,₃, though still 1.9× smaller than random)
- More distributed (all neurons contribute to readout)

**Both learned models are closer to tropical singularity than random** — suggesting that the optimization finds solutions near the boundary of the "generic" region. M₄,₃ is extremely close (963×), while M₁₆,₁₀ is moderately close (1.9×).

## Tropical Structure Across Scales

| Model | hidden | seq_len | W_out | Gap vs Random | Avg |Corr| |
|-------|--------|---------|-------|---------------|-------------|
| M₄,₃  | 4 | 3 | 3×4 | **756× smaller** | 0.99 (paired) |
| M₈,₆  | 8 | 6 | 6×8 | 4.9× smaller | 0.48 |
| M₈,₁₀ | 8 | 10 | 10×8 | **0.9× larger!** | 0.23 |
| M₁₆,₁₀| 16 | 10 | 10×16 | 1.6× smaller | 0.32 |

### Column Structure Determines Tropical Genericity

**M₄,₃: Binary switch creates column pairing**
- The (n0,n1) vs (n2,n3) mechanism means col0 ≈ -col3, col1 ≈ -col2
- Avg |correlation| = 0.99 — columns are almost perfectly paired
- This creates extreme tropical near-singularity (756×)
- Not forced by dimensionality (4 cols is enough for 3 outputs) — it's the learned mechanism

**M₈,₁₀: Learned orthogonality**
- Avg |correlation| = 0.23, lower than random's 0.27
- Achieves MORE tropical genericity than random
- The "cleanest" solution in tropical terms

**M₁₆,₁₀: Rich structure, moderate correlation**
- Avg |correlation| = 0.32
- Uses functional groups (waves, bridges, comps) rather than clean orthogonality
- Slightly closer to singularity than random (1.6×)

### Eigenvalue Evolution

| Model | Positive eigs | Negative eigs | Interpretation |
|-------|---------------|---------------|----------------|
| M₄,₃  | 0 | 3 | Pure damping, no oscillation |
| M₈,₆  | 5 | 3 | Oscillation appears |
| M₈,₁₀ | 5 | 3 | Sustained oscillation |
| M₁₆,₁₀| 8 | 8 | Rich oscillatory dynamics |

The appearance of positive eigenvalues correlates with the ability to maintain oscillations during the "rebuild phase" after impulse clipping.

## Cross-Seed Analysis (M₁₆,₁₀)

Comparing our local model against 5 GCS models (same architecture, different random seeds):

### Universal Properties (consistent across seeds)

| Property | Range | CV | Interpretation |
|----------|-------|-----|----------------|
| Positive eigenvalues | 6-9 | 12% | ~8 positive, ~8 negative |
| Spectral radius | 0.8-1.5 | 23% | Centered around 1.0 |
| W_out column correlation | 0.25-0.40 | 15% | Moderate, not paired |
| W_out effective rank | 6-9 | — | High rank, holographic |
| Tropical gap | 3.4-8.2 | 34% | All non-singular |

### Spurious Properties (vary dramatically)

| Property | Range | CV | Note |
|----------|-------|-----|------|
| **Tropical cells** | **10-38** | **40%** | 4× variation! |
| Specific neuron roles | — | — | n4, n9, etc. not preserved |
| Functional groups | — | — | Comps/Waves/Bridges are local |
| Individual W_ih values | — | — | Completely different |

### Implications

**Strong claims** (supported by cross-seed):
- Eigenvalue split enables oscillation + damping
- W_out uses all neurons (holographic, not sparse)
- Solutions avoid tropical singularity

**Weak claims** (specific to our model):
- "38 tropical cells" → actually ranges 10-38 across seeds
- "n4 is the relay" → different neurons in different seeds
- "Comps rebuild phase" → functional roles vary

The specific tropical cell structure and neuron role assignments are ONE valid solution, not THE solution.

## Bipartite Hypothesis Investigation

The consistent 8 positive / 8 negative eigenvalue split across seeds prompted investigation of a "bipartite" or "tick-tock" hypothesis: does W_hh have a bipartite structure where neurons alternate between two groups?

### Hypothesis

If W_hh were bipartite, the 16 neurons would partition into two groups where:
- Within-group connections are weak
- Between-group connections are strong
- Odd powers of W_hh map between groups, even powers map within groups

### Tests Performed

1. **Spectral partitioning**: Used Fiedler vector (graph Laplacian) to find optimal 2-partition
2. **Dominant eigenvector partitioning**: Used sign of dominant eigenvector
3. **Odd/even power analysis**: Compared energy distribution in W^1, W^2, ..., W^10
4. **Sorted weight matrix visualization**: Checked for checkerboard pattern

### Results: Hypothesis FALSIFIED

| Test | Expected (bipartite) | Observed |
|------|---------------------|----------|
| Between/Within ratio | >> 1 | 0.49-0.75 |
| Odd power between-energy | High | Mixed (0.5-1.2) |
| Even power within-energy | High | Mixed (0.8-1.0) |
| Sorted W_hh pattern | Checkerboard | No clear structure |

**Key finding**: The 8/8 sign split occurs in random matrices too!

```
Distribution of positive eigenvalues in random 16×16 matrices:
  6:   8.6%
  7:  23.9%
  8:  31.1%  ← most common
  9:  22.5%
  10: 10.0%
Mean: 8.02
```

### What the Eigenvalue Structure Actually Represents

Instead of bipartite structure, the eigenvalues show:

| Type | Count | Function |
|------|-------|----------|
| Growing modes (|λ| > 1) | 4 | Sustain signal during rebuild |
| Stable oscillators (|λ| < 1, complex) | 8 | Decay over ~2-10 steps |
| Fast-decaying real modes | 4 | Clear old information quickly |

The decay timescales match the sequence length (~10 steps):
- 2 modes decay in ~10 steps (memory of full sequence)
- 6 modes decay in 2-4 steps (short-term processing)
- 4 modes decay in < 1 step (instantaneous filtering)

### Conclusion

The 8/8 eigenvalue sign split is NOT a learned bipartite structure. It is:
1. A **statistical property of random matrices** (Central Limit Theorem applies)
2. **Preserved during training** because it doesn't harm the solution
3. **Not functionally meaningful** — the important structure is in magnitude and eigenvector direction

What IS learned:
- Spectral radius > 1 for some modes (sustains oscillation)
- Decay timescales matching sequence length
- Eigenvector directions (which neurons contribute to which dynamical modes)

## Eigenvector Structure: The Processing Pipeline

Analysis of which neurons participate in growing, stable, and damped eigenmodes reveals a clear signal processing pipeline:

```
INPUT ──→ GROWING MODES ──→ STABLE MODES ──→ DAMPED MODES ──→ OUTPUT
          (amplify)         (process)        (readout)

Input       8.2, 5.7          <2.1            2.4, 2.2       Output
coupling    (strong)          (weak)          (medium)       coupling
                                                             5.4, 4.0
                                                             (strong)
```

### Mode Coupling Summary

| Mode Type | Count | Input Coupling | Output Coupling | Function |
|-----------|-------|----------------|-----------------|----------|
| GROWING   | 4     | **Strong** (5.7-8.2) | Medium (2.6-3.9) | Amplify signal |
| STABLE    | 10    | Weak (<2.1) | Medium (2.5-3.3) | Process/transform |
| DAMPED    | 2     | Medium (2.2-2.4) | **Strong** (4.0-5.4) | Read out result |

### Neuron Participation by Mode Type

| Mode Type | Dominant Neurons | Essential % | Interpretation |
|-----------|------------------|-------------|----------------|
| GROWING   | n0*, n9*, n14*, n13*, n11, n12 | **54.6%** | Essential neurons amplify |
| STABLE    | n3, n14*, n15, n13*, n12 | 34.9% | Dependent neurons process |
| DAMPED    | n14*, n3, n8, n13*, n1* | **59.6%** | Essential neurons read out |

**Key insight**: Essential neurons (from matroid analysis) dominate both GROWING and DAMPED modes, while dependent neurons dominate STABLE modes. This explains why essential neurons are essential — they're the input/output interface!

### Phase Structure in Growing Oscillator

The complex growing mode (λ = 0.43 ± 1.03j, period = 5.4 steps) creates a "rotating wave" pattern:

```
Phase ≈   0°: n11              (leading)
Phase ≈  50°: n8, n12
Phase ≈ 130°: n9*, n14*
Phase ≈ 165°: n0*, n1*
Phase ≈-135°: n6, n13*
Phase ≈ -50°: n3, n10, n15     (lagging)
```

With period ~5 steps and sequence length 10, we get ~2 full oscillations per sequence. This may enable the network to "cycle through" different comparisons.

### Why This Architecture?

1. **Prevents output saturation**: Growing modes don't directly couple to output
2. **Enables non-linear processing**: Stable modes can transform the amplified signal
3. **Clean readout**: Damped modes project strongly to output without being overwhelmed

This is reminiscent of reservoir computing, where a rich dynamical system transforms input into a high-dimensional representation, and a simple linear readout extracts the answer.

## Phase Wheel Mechanism: How Position is Encoded

A key experiment reveals that the network does NOT store position explicitly. Instead, it uses a **phase wheel** mechanism.

### The Countdown Discovery

When we apply W_out readout at intermediate timesteps (not just the final one), we observe:

```
M@0, S@1: predictions over time = [4, 9, 8, 7, 6, 5, 4, 3, 2, 1]
```

The prediction **counts down from 9 to 1** as time progresses!

### The Formula

For any S position, after S arrives:

```
prediction(t) = s_pos + (9 - t)
              = s_pos + steps_remaining
```

At the final timestep t=9:
```
prediction(9) = s_pos + 0 = s_pos  ✓
```

### Mechanism

1. **When S arrives**: Hidden state enters a specific trajectory in phase space
2. **Each timestep**: State rotates via W_hh dynamics (period ≈ 6-8 steps)
3. **W_out decodes phase**: Each output row is tuned to a different phase
4. **At t=9**: Phase aligns with correct position

### W_out Structure

The W_out rows show a clear oscillatory pattern:

| Row pair | Correlation | Interpretation |
|----------|-------------|----------------|
| 0 ↔ 1 | +0.51 | Similar (adjacent) |
| 0 ↔ 4 | -0.73 | Opposite (half-period) |
| 0 ↔ 8 | +0.01 | Similar (full period) |

**Half-period ≈ 4 rows**, giving **full period ≈ 8 timesteps**.

### Implications

1. **Position is NOT stored**: The network doesn't learn "S is at position 3"
2. **Time encodes position**: Instead, it learns "S happened (9-t) steps ago"
3. **Dynamics do the work**: W_hh rotation + W_out decoding = position at readout
4. **Sequence length is baked in**: The period is calibrated for exactly 10 timesteps

This explains:
- Why oscillatory eigenvalues are essential (they create the rotation)
- Why readout only works at the final timestep (that's when phase = position)
- Why the network struggles with different sequence lengths (period mismatch)

### Connection to Eigenvector Analysis

The growing oscillator mode (λ = 0.43 ± 1.03j, period = 5.4 steps) creates the primary rotation. Combined with the real growing modes, this produces the countdown trajectory.

## Scripts

- `src/tropical_analysis.py`: Tropical cell analysis for M₁₆,₁₀
- `src/tropical_rank.py`: Tropical rank computation
- Model files: `/tmp/model_4_3.pth` (M₄,₃), `/tmp/model_8_*.pth`, `model.pth` (M₁₆,₁₀)
