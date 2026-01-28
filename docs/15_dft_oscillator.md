# 15: DFT-like Structure and the Nonlinear Oscillator

## Overview

We investigated whether the Fourier-like activity observed in the comparator neurons (docs 5, 8, 14) has a direct origin in the weight matrices. The answer: **W_out is a DFT-like decoder**, but the oscillation itself emerges from a **nonlinear oscillator** where ReLU clipping stabilizes an inherently unstable linear system and shifts its frequency to match the decoder.

## W_hh Eigenvalue Analysis

### The Linear System is Unstable

W_hh (16×16 recurrence matrix) has **three modes with |λ| > 1**:

| Eigenvalue | |λ| | Type | Frequency | Period |
|-----------|-----|------|-----------|--------|
| λ = -1.274 | 1.274 | Real (sign-flipper) | π rad | 2.0 |
| λ = +1.167 | 1.167 | Real (exponential) | 0 | ∞ |
| λ = 0.435 ± 1.027j | 1.115 | Complex (oscillatory) | 1.17 rad | 5.37 |

Without ReLU, the system **diverges** — energy grows exponentially. The dominant oscillatory mode has frequency 1.17 rad/step (period 5.37), which is **too fast** for position encoding across 10 timesteps.

### Other Notable Modes

| Eigenvalue | |λ| | Frequency | Period | Role |
|-----------|-----|-----------|--------|------|
| -0.106 ± 0.895j | 0.902 | 1.69 rad | 3.72 | Higher harmonic |
| +0.644 ± 0.400j | 0.758 | 0.56 rad | 11.30 | Slow mode |
| +0.406 ± 0.526j | 0.664 | 0.91 rad | 6.88 | Medium mode |

## W_out: A DFT-like Decoder

### Sinusoidal Column Structure

Each column of W_out (10×16) shows how one neuron contributes to each output position. The comparator columns are **strikingly sinusoidal**:

| Neuron | Dominant DFT k | % Energy | ω (rad/step) | Period | Sinusoidal quality |
|--------|---------------|----------|---------------|--------|-------------------|
| n7 | k=1 | 95% | 0.628 | 10.0 | 100% |
| n8 | k=1 | 85% | 0.628 | 10.0 | 98% |
| n9 | k=1 | 85% | 0.628 | 10.0 | 94% |
| n6 | k=1 | 67% | 0.628 | 10.0 | 94% |
| n1 | k=2 | 47% | 1.257 | 5.0 | 87% |
| **n13** | **k=5** | **34%** | **3.142** | **2.0** | **102%** (k=5+k=2) |

Most neurons have their W_out energy concentrated at k=1 (one full cycle over 10 positions). n1 encodes at k=2 (period 5), providing higher spatial resolution. n13 encodes at k=5 (Nyquist frequency).

### W_out as Matched Filter Bank

W_out acts as a **matched filter bank** analogous to a DFT decoder:
- Each column is a sinusoid at a specific frequency
- The columns at k=1 form a multi-phase system (different phases per neuron)
- Inner product `W_out @ h_final` is equivalent to computing a discrete Fourier coefficient

### SVD Decomposition

The SVD of W_out shows that 4 components capture 91% of variance:
- Mode 0 (σ=19.2): 72% sinusoidal at k=1
- Mode 1 (σ=17.9): 90% sinusoidal at k=1
- Mode 2 (σ=8.6): 65% sinusoidal at k=2
- Mode 3 (σ=6.1): 77% sinusoidal at k=4

The output layer is dominated by k=1 sinusoidal structure.

## The Nonlinear Oscillator Mechanism

### Why Not a Simple DFT?

If W_hh were an orthogonal rotation matrix, the system would implement a direct DFT: each step would rotate the state by a fixed angle, and position would be encoded as phase. But:

1. **W_hh is not orthogonal** — its 2×2 sub-blocks between comparators have rotation errors of 0.3-1.5
2. **W_hh has unstable eigenvalues** — the dominant oscillatory mode grows at |λ|=1.115 per step
3. **The eigenfrequency (1.17 rad) doesn't match W_out (0.63-0.93 rad)**

Instead, the system is a **nonlinear oscillator** where ReLU creates the useful dynamics.

### The Energy Balance

At each timestep, there's a battle between growth and damping:

| Step | W_hh growth | ReLU damping | Net | Neurons clipped |
|------|------------|-------------|-----|-----------------|
| t=1 | 18.3× | 0.70 | 12.8 | 3.8 |
| t=2 | 2.4× | 0.58 | 1.4 | 5.8 |
| t=3 | 2.1× | 0.96 | 2.0 | 2.6 |
| t=4 | 2.5× | 0.45 | 1.1 | 5.4 |
| t=5 | 3.3× | 0.43 | 1.4 | 3.8 |
| t=6-9 | 3.5-3.9× | 0.43-0.47 | 1.5-1.8 | 3-4 |

**W_hh amplifies energy 2-4× per step**, but **ReLU removes 40-60%** by clipping negative pre-activations. The net effect is slightly above 1 — the system is "barely unstable," which sustains the oscillation over 10 timesteps without diverging badly.

### The Frequency Shift

The critical insight: **ReLU shifts the oscillation frequency downward**.

| Level | Frequency | Period | Source |
|-------|-----------|--------|--------|
| Raw W_hh eigenvalue | 1.17 rad | 5.37 | Linear dynamics |
| Effective (with ReLU) | 0.7-0.9 rad | 7-9 | Measured from h_final |
| W_out decoder | 0.6-0.9 rad | 7-10 | Matched filters |
| Ideal for 10 positions | 0.628 rad | 10.0 | 2π/10 |

The mechanism:
1. W_hh drives oscillation at ω ≈ 1.17 rad (too fast)
2. ReLU clips different neurons at each step, changing the active subspace
3. The **effective subspace eigenvalues** have lower frequency (ω ≈ 0.7-0.8 rad)
4. W_out is tuned to this effective frequency, not the raw eigenfrequency

### The Varying Active Set

At each timestep after the max position, a different subset of neurons is active (not clipped). The effective W_hh sub-matrix changes accordingly, and its eigenvalues shift:

| Timestep | Clipped neurons | Dominant effective |λ| | Effective ω |
|----------|----------------|---------------------|-------------|
| t=4 | 6 neurons | 1.006 (≈marginal) | 0.74 |
| t=5 | 4 neurons | 0.999 | 0.81 |
| t=6 | 4 neurons | 1.069 (unstable) | 1.64 |
| t=7 | 4 neurons | 0.977 | 0.83 |
| t=8 | 4 neurons | 0.904 | 0.81 |
| t=9 | 3 neurons | 1.010 | 0.75 |

The effective frequency hovers around ω ≈ 0.7-0.8 rad, with occasional excursions when different neurons clip.

## Neuron Roles in the Oscillator

### Classification

Based on clipping behavior and recurrence structure:

| Role | Neurons | Clip rate | W_hh[n,n] | Function |
|------|---------|-----------|-----------|----------|
| **Backbone** | n2, n7, n12, n13, n14 | <5% | +0.08 to +0.97 | Always active, carry persistent state |
| **Comparator** | n1, n6, n8 | 7-38% | +0.36 to +0.62 | Encode position through clipping patterns |
| **Switch** | n3, n5, n15 | 22-59% | -0.42 to +0.23 | Alternate clipped/active, drive oscillation |
| **Usually clipped** | n4, n9 | 73-97% | -0.98 to -0.55 | Active only in specific circuits |
| **Mixed** | n0, n10, n11 | 15-18% | -0.00 to +0.20 | Variable roles |

### Switch Neurons Drive the Oscillation

The **switch neurons** (n3, n5, n15) are the engine of the oscillator:

- They flip between clipped and active at each timestep
- n3 has the highest switch count (4.6 transitions in 6 steps)
- Their negative self-recurrence (n3: -0.36, n5: -0.42) promotes alternation
- Phase relationships: n3-n15 are in-phase (r=+0.44), n5-n15 are anti-phase (r=-0.36)

When switch neurons clip, they change the effective W_hh subspace, which shifts the eigenvalues. This is how the ReLU-based frequency modification works mechanically — it's not a smooth frequency shift, but a **stroboscopic effect** from rapidly switching between different linear subsystems.

### Backbone Neurons Carry State

The **backbone neurons** (n2, n12, n13, n14) never clip and maintain continuous state:
- n2: Running accumulator (W_hh[2,2] = +0.97, near-perfect self-recurrence)
- n13: Nyquist oscillator (see below)
- n12, n14: Steady state carriers

### Negative Self-Recurrence Neurons

Five neurons have W_hh[n,n] < 0, meaning they naturally want to alternate sign:

| Neuron | W_hh[n,n] | Isolated period | Main feeders | Behavior |
|--------|-----------|-----------------|--------------|----------|
| n3 | -0.359 | 2 (fast decay) | n0(-1.23), n1(+1.20), n8(-1.07) | Switch neuron |
| n4 | -0.986 | 2 (slow decay) | n2(-0.49) | Usually clipped |
| n5 | -0.421 | 2 (fast decay) | n4(+1.19), n8(-0.85), n7(-0.59) | Switch neuron |
| n9 | -0.553 | 2 (medium decay) | n7(-5.03!), n8(-1.09) | Usually clipped |
| n13 | -0.605 | 2 (medium decay) | n15(+0.65), n4(+0.46), n3(+0.43) | Backbone/Nyquist |

In isolation, negative self-recurrence with ReLU would kill the signal (ReLU clips the negative phase). These neurons stay alive through positive inputs from other neurons.

## N13: The Nyquist Neuron

### The Nyquist Pattern in W_out

N13's W_out column has a striking alternating pattern:

```
W_out[:, 13] = [+0.24, +0.37, -2.24, +0.25, +0.18, +2.09, -1.39, +0.23, +0.07, +0.26]

Even positions mean: -0.626
Odd  positions mean: +0.639
Ratio: -0.979 ≈ -1.0  (near-perfect alternation)
```

DFT breakdown:
- **k=5 (Nyquist)**: 34% of energy
- **k=2 (period 5)**: 17% of energy
- Combined: **51%** — dominated by the highest spatial frequencies

### Self-Recurrence Creates Period-2 Oscillation

W_hh[13,13] = -0.605. This negative self-recurrence means n13 naturally wants to flip sign each step. Unlike the switch neurons, n13 **never clips** (0% clip rate post-max), so it actually sustains this oscillation through its feeders (n15: +0.65, n4: +0.46, n3: +0.43).

### The n9-n13 Rotation Pair

Among all 120 possible neuron pairs, n9-n13 has the most rotation-like 2×2 sub-block in W_hh:

```
W_hh[[9,13], [9,13]] = [[-0.553, -0.026],
                          [+0.039, -0.605]]

Rotation error: 0.066 (lowest of all pairs)
Scale: 0.554
Angle: 176° ≈ π (Nyquist!)
Period: 2.05 steps
```

This is a **near-perfect 180° rotation** scaled by 0.55 — essentially a Nyquist-frequency oscillator that decays by half each cycle. n9 is usually clipped (82%), so the rotation primarily manifests through n13's persistent activity.

### N13's Encoding

N13 h_final correlations:
- With max_val: r = +0.28 (moderate — encodes value magnitude)
- With sum(max_pos + 2nd_pos): r = -0.22 (encodes combined position)
- With max_pos: r = -0.11 (weak individual position)

Combined with the Nyquist W_out pattern, n13 provides **fine-grained position information** through its alternating contribution — it pushes toward even positions when h_final is negative and odd positions when positive. This supplements the coarser k=1 encoding from the comparators.

## The Full Picture

### How Fourier-like Activity Emerges

```
W_hh eigenvalues (ω = 1.17 rad, UNSTABLE)
         │
         ▼
  ┌─────────────────────┐
  │   ReLU at each step  │
  │   3-6 neurons clip   │
  │   Active set rotates │
  └──────────┬──────────┘
             │
             ▼
  Effective frequency: ω ≈ 0.7-0.9 rad
  (40% lower than raw eigenfrequency)
             │
             ▼
  W_out matched filters (ω ≈ 0.6-0.9 rad)
  Decode position from oscillation phase
```

The system is **not** a linear DFT. It is a **nonlinear oscillator** with three key properties:

1. **Self-sustaining**: Unstable eigenvalues provide energy; ReLU provides damping
2. **Frequency-modified**: ReLU clipping shifts the natural frequency down to match the decoder
3. **Input-dependent**: The clipping pattern depends on the input, so the oscillation encodes information about when events occurred

### Multi-Resolution Encoding

The four comparators encode at slightly different effective frequencies:

| Neuron | ω_effective | ω_wout | Period | Resolution |
|--------|------------|--------|--------|------------|
| n8 | 0.64 | 0.59 | ~10 | Coarsest |
| n7 | 0.72 | 0.70 | ~9 | Medium |
| n6 | 0.86 | 0.94 | ~7 | Finer |
| n1 | 0.96 | 0.92 | ~7 | Finest (k=2) |
| n13 | Nyquist | Nyquist | 2 | Even/odd parity |

This multi-frequency scheme is analogous to having multiple DFT bins — each comparator captures a different "spectral component" of the position information, and together they provide robust position encoding.

## Connection to Prior Findings

| Doc | Finding | Connection |
|-----|---------|------------|
| 5 | Fourier-like W_out patterns | These ARE the DFT-like decoder columns |
| 8 | Multi-frequency position encoding | Emerges from multi-frequency oscillator |
| 13 | N2-N4 noise cancellation | N2 is a backbone neuron that never clips |
| 14 | Diagonal boundaries | Comparators clip based on input comparisons; the post-clipping dynamics are the oscillator |

## Summary

| Aspect | Linear DFT | This RNN |
|--------|-----------|----------|
| Oscillation source | Rotation matrix | Unstable eigenvalues + ReLU damping |
| Frequency | Fixed by matrix structure | Emergent from growth/damping balance |
| Stability | Orthogonal → unit eigenvalues | Nonlinear limit cycle |
| Decoder | Inverse DFT matrix | Matched sinusoidal filters (W_out) |
| Freq shift | None | ReLU shifts ω down ~40% |
| Multi-resolution | Multiple DFT bins | Multiple comparators at different ω |

The RNN has learned a **nonlinear oscillator** that serves the same function as a DFT — encoding temporal position as phase — but through a fundamentally different mechanism. Rather than precise rotation, it uses the balance between unstable linear dynamics and ReLU clipping to generate sustained oscillations at frequencies the output layer can decode.
