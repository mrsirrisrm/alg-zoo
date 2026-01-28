# 15: DFT-like Structure and the Nonlinear Oscillator

## Overview

We investigated whether the Fourier-like activity observed in the comparator neurons (docs 5, 8, 14) has a direct origin in the weight matrices. The answer: **W_out is a DFT-like decoder**, and position is encoded through a **transient impulse response** whose shape is determined by the interplay of unstable linear dynamics and ReLU clipping.

**Correction (doc 16):** Deeper analysis revealed that the original "frequency shift" narrative was incorrect. ReLU does not shift a steady-state oscillation frequency. Instead, it performs **spectral reshaping** of the impulse response through mode-dependent damping, and position encoding relies on **two complementary mechanisms** — comparator amplitude discrimination and wave neuron phase readout.

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
| +0.644 ± 0.400j | 0.758 | 0.56 rad | 11.30 | Slow mode — close to k=1 (period 10) |
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

Instead, the system uses a **transient impulse response** shaped by ReLU dynamics.

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

**W_hh amplifies energy 2-4× per step**, but **ReLU removes 40-60%** by clipping negative pre-activations. The net effect is slightly above 1 — the system is "barely unstable," which sustains activity over 10 timesteps without diverging badly.

### ~~The Frequency Shift~~ Spectral Reshaping (Corrected)

**Original claim (incorrect):** ReLU shifts the oscillation frequency downward from 1.17 rad to ~0.7 rad, a ~40% reduction.

**Corrected understanding (doc 16):** ReLU does not shift a continuous oscillation frequency. Instead, it performs **spectral reshaping** of the transient impulse response through **mode-dependent damping**:

| Mode | ω | |λ| | Linear growth (5 steps) | Nonlinear growth | Suppression ratio |
|------|---|-----|------------------------|------------------|-------------------|
| Sign-flipper | π | 1.274 | 3.36× | 0.36× | 0.11 (10× suppressed) |
| Complex oscillatory | 1.17 | 1.115 | 1.72× | 1.55× | 0.90 (mild suppression) |
| High harmonic | 1.69 | 0.902 | 0.60× | 0.86× | 1.44 (boosted) |
| Slow mode | 0.56 | 0.758 | 0.25× | 0.62× | 2.47 (strongly boosted) |
| Medium mode | 0.91 | 0.664 | 0.13× | 1.08× | 8.35 (massively boosted) |

The mechanism:
- **High-frequency modes** (ω ≈ π) alternate sign rapidly, creating negative pre-activations that ReLU clips → heavy damping
- **Low-frequency modes** (ω < 1) vary slowly, rarely triggering negative pre-activations → relative preservation
- Net spectral centroid shift: only **7.3%** (from 1.46 to 1.35 rad), not 40%
- The k=1 (ω=0.63 rad) pattern in h_final comes from a **different source** — see "Wave Neurons" below

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
| **Wave** | n10, n11, n12 | 15-18% | -0.00 to +0.61 | Carry traveling wave for position encoding |

### Switch Neurons Drive the Oscillation

The **switch neurons** (n3, n5, n15) are the engine of the oscillator:

- They flip between clipped and active at each timestep
- n3 has the highest switch count (4.6 transitions in 6 steps)
- Their negative self-recurrence (n3: -0.36, n5: -0.42) promotes alternation
- Phase relationships: n3-n15 are in-phase (r=+0.44), n5-n15 are anti-phase (r=-0.36)

When switch neurons clip, they change the effective W_hh subspace, which shifts the eigenvalues.

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

## The Full Picture (Revised)

### Two Encoding Mechanisms

Position encoding relies on two complementary systems:

```
                      ┌──────────────────────────────────────┐
                      │        COMPARATORS (n1, n6, n7, n8)  │
                      │                                      │
  Max input arrives → │  h_final varies ~1-2 units by pos    │
                      │  W_out columns: 85-95% sinusoidal    │
                      │  → Amplitude × sinusoidal filter     │
                      │  → Strong max-position SUPPRESSION   │
                      │  Gap: +9.96 between 2nd and max      │
                      └──────────────────────────────────────┘

                      ┌──────────────────────────────────────┐
                      │      WAVE NEURONS (n10, n11, n12)    │
                      │                                      │
  n4 burst →          │  Traveling wave through feedback     │
                      │  Natural freq: ω ≈ 0.51 rad (T≈12)  │
                      │  h_final: 74% sinusoidal at k=1      │
                      │  W_out: also sinusoidal at k=1       │
                      │  → Phase readout encodes position    │
                      └──────────────────────────────────────┘
```

### Comparator Mechanism (Amplitude-Based)

The comparators encode position through **how much** they activate, not through oscillation:
- h_final for n7 ranges from 5.2 (max at pos 2) to 6.3 (max at pos 8), with 0 at pos 9
- This ~1-unit variation is decoded by the sinusoidal W_out column (amplitude range ±7.3)
- The sinusoidal decoder acts as a **position-dependent amplifier**: small h_final differences become large output differences at specific positions
- Key role: **suppress the max position** in the output (gap = +9.96 logits between 2nd_pos and max_pos)

### Wave Neuron Mechanism (Phase-Based)

The wave neurons encode position through the **phase** of a decaying transient:
- n4 (W_ih = +10.16) fires a burst when a large input arrives
- This burst propagates through n10 ↔ n11 ↔ n12 feedback loops
- The {n10, n11, n12} sub-block has a complex eigenvalue pair at ω = 0.51 rad (T ≈ 12)
- The wave's phase at readout time (t=9) encodes when the impulse occurred
- h_final traces out the impulse response curve, which is 74% sinusoidal at k=1

### Wave Propagation Chain

```
Max input → n4 (W_ih=+10.16, burst)
             ↓ W_hh[10,4]=+1.76     ↓ W_hh[12,4]=+1.28
             n10 ──→ n11 ──→ n12
              ↑  W_hh[10,11]=+1.10   ↑ W_hh[11,12]=+0.66
              └───── feedback ────────┘

Sub-block eigenvalues: λ = 0.529 ± 0.297j
  |λ| = 0.606, ω = 0.511 rad, T = 12.3 steps
```

### How "Fourier-like Activity" Actually Emerges

The original description of a "nonlinear oscillator with frequency shift" was partly wrong. The corrected picture:

1. **W_out IS a DFT-like decoder** (correct) — columns are genuinely sinusoidal
2. **Comparators encode via amplitude** (revised) — not via oscillation frequency
3. **Wave neurons carry a genuine oscillation** (new finding) — the n10-n11-n12 subsystem has a natural frequency near k=1
4. **ReLU performs spectral reshaping** (corrected) — suppresses high-frequency modes through differential damping, but the spectral centroid shift is only 7.3%, not 40%
5. **The k=1 pattern has a specific origin** (new finding) — it comes from the ω ≈ 0.51 rad eigenmode of the n10-n11-n12 sub-block, not from frequency-shifting the whole-network ω = 1.17 rad mode

### ReLU's Role (Corrected)

ReLU does **not** shift a continuous oscillation frequency. Its actual roles:

| Function | Mechanism |
|----------|-----------|
| **Stabilization** | Clips negative pre-activations, preventing divergence from |λ|>1 modes |
| **Spectral reshaping** | Differentially damps modes: high-freq → 10× suppressed, low-freq → 2-8× boosted |
| **Event detection** | Creates clipping events that mark "new max found" (doc 14 diagonal boundaries) |
| **Impulse shaping** | Shapes the transient decay curve to match the W_out decoder's expected profile |

## Connection to Prior Findings

| Doc | Finding | Connection |
|-----|---------|------------|
| 5 | Fourier-like W_out patterns | These ARE the DFT-like decoder columns |
| 8 | Multi-frequency position encoding | Two sources: comparator amplitude + wave phase |
| 13 | N2-N4 noise cancellation | N2 is a backbone neuron that never clips |
| 14 | Diagonal boundaries | Comparators clip based on input comparisons; post-clipping dynamics feed the wave |
| 16 | Corrected frequency mechanism | Spectral reshaping, not frequency shift |

## Summary

| Aspect | Linear DFT | This RNN |
|--------|-----------|----------|
| Position encoding | Phase of rotation | Amplitude (comparators) + phase (wave neurons) |
| Oscillation source | Rotation matrix | Sub-block eigenvalues (ω ≈ 0.51) + transient decay |
| Frequency | Fixed by matrix structure | Emergent from sub-network dynamics |
| Stability | Orthogonal → unit eigenvalues | ReLU clips unstable modes |
| Decoder | Inverse DFT matrix | Matched sinusoidal filters (W_out) |
| ReLU role | N/A | Spectral reshaping (mode-dependent damping) |
| Multi-resolution | Multiple DFT bins | Multiple comparators + wave neurons at different ω |

The RNN has learned a **dual encoding system** that serves the same function as a DFT — encoding temporal position for output decoding — but through two complementary mechanisms: comparator neurons that discriminate max from non-max positions through amplitude differences decoded by sinusoidal filters, and wave neurons that carry a genuine low-frequency oscillation seeded by the n4 impulse burst and sustained through n10-n11-n12 feedback.
