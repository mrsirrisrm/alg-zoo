# 33: The Offset Discrimination Mechanism

## Overview

The M₁₆,₁₀ network distinguishes forward (M before S) from reverse (S before M) sequences through an **offset mechanism**. The forward and reverse hidden state trajectories are not independent spirals — they are related by an offset that W_out reads as "boost s_pos, suppress m_pos."

This document details how the offset is created, structured, and amplified to produce correct discrimination.

## The Core Relationship

At any timestep t after the 2nd impulse:

```
h_fwd(t) = h_rev(t) + offset(t)
```

The scale is exactly **1.0** — this is not an approximation. The trajectories differ only by an additive offset, not a multiplicative scaling.

## Offset Structure: Separable and Antisymmetric

### Separability

The offset can be decomposed into position-dependent components:

```
offset(m_pos, s_pos) ≈ f(m_pos) + g(s_pos)
```

This separable model achieves **R² = 0.89** across all 90 (M, S) position pairs.

### Perfect Antisymmetry Through W_out

The f and g components have a remarkable property:

```
W_out @ f(m) = -W_out @ g(m)    (exactly, correlation = 1.0000)
```

This means:
- **f(m)** contributes a **negative** value at readout position m
- **g(s)** contributes a **positive** value at readout position s
- They are perfect mirror images through W_out

| Position | W_out @ f(pos) | W_out @ g(pos) | Sum |
|----------|----------------|----------------|-----|
| 0 | -84.5 | +84.5 | 0.0 |
| 1 | -54.4 | +54.4 | 0.0 |
| 2 | -13.3 | +13.3 | 0.0 |
| 3 | +23.4 | -23.4 | 0.0 |
| 4 | +36.3 | -36.3 | 0.0 |
| 5 | +32.1 | -32.1 | 0.0 |
| 6 | -1.2 | +1.2 | 0.0 |
| 7 | -11.3 | +11.3 | 0.0 |
| 8 | -14.6 | +14.6 | 0.0 |
| 9 | -39.8 | +39.8 | 0.0 |

This antisymmetry guarantees that the offset always favors s_pos over m_pos:

```
(W_out @ offset)[s_pos] > (W_out @ offset)[m_pos]
```

Verified for **100%** of the 90 position pairs.

## Offset Evolution Through W_hh

The offset is not constant — it evolves at each timestep:

```
offset(t+1) ≈ W_hh @ offset(t)    (approximately, ReLU modifies this)
```

### Offset Growth

| Timestep | ||offset|| | Margin |
|----------|------------|--------|
| t=7 | 9.8 | 22.6 |
| t=8 | 15.3 | 37.3 |
| t=9 | 17.6 | 53.1 |

The offset grows because W_hh has **growing eigenmodes** (|λ| > 1):
- Mode 0: λ = -1.27
- Mode 1: λ = +1.17
- Modes 2-3: λ = 0.44 ± 1.03j (|λ| = 1.12)

### W_hh Does NOT Amplify the Discriminative Direction

A key finding: W_hh actually **shrinks** the discriminative direction:

```
||W_out[s] - W_out[m]||         = 15.62
||W_hh @ (W_out[s] - W_out[m])|| = 15.91  (growth factor: 1.02)
||W_hh² @ (W_out[s] - W_out[m])|| = 14.90  (growth factor: 0.94)
```

The margin grows not because W_hh amplifies the discriminative direction, but because:
1. The offset has a positive projection onto the discriminative direction
2. The offset norm grows (due to growing eigenmodes)
3. ReLU provides an additional boost (see below)

## The ReLU Boost Mechanism

ReLU doesn't just bound the dynamics — it **selectively clips neurons that would hurt discrimination**.

### Linear vs Actual Margin

| Timestep | Linear prediction | Actual (with ReLU) | ReLU bonus |
|----------|-------------------|--------------------| -----------|
| t=7 | 22.6 | 22.6 | — |
| t=8 | 33.3 | 37.3 | +4.0 |
| t=9 | 43.0 | 53.1 | +10.1 |

ReLU contributes **+10 points** of margin by t=9.

### How ReLU Helps

At each timestep, some neurons would contribute negatively to the margin. ReLU clips these neurons differently for forward vs reverse:

**t=7 → t=8:**
- Neuron 10 would contribute -28.1 to margin (hurts discrimination)
- ReLU clips it differently, reducing harm to -24.8
- Net bonus: **+3.3**

**t=8 → t=9:**
- Neuron 0 would contribute -11.1 to margin
- ReLU reduces this to -6.8
- Net bonus: **+4.3**

The network learned to place harmful contributions on neurons that get **differentially clipped** by ReLU between forward and reverse trajectories.

## Complete Mechanism Summary

```
┌─────────────────────────────────────────────────────────────────┐
│              THE OFFSET DISCRIMINATION MECHANISM                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. INITIAL OFFSET (at 2nd impulse)                              │
│     ├── ReLU gating differs for fwd vs rev                       │
│     ├── Creates offset with positive discriminative projection   │
│     └── Initial margin = 22.6                                    │
│                                                                  │
│  2. OFFSET STRUCTURE                                             │
│     ├── Separable: offset ≈ f(m_pos) + g(s_pos)                  │
│     ├── Antisymmetric: W_out @ f = -W_out @ g                    │
│     └── Result: always boosts s_pos, suppresses m_pos            │
│                                                                  │
│  3. W_hh EVOLUTION                                               │
│     ├── Growing eigenmodes amplify offset norm                   │
│     ├── Does NOT specifically amplify discriminative direction   │
│     └── Offset grows: 9.8 → 15.3 → 17.6                          │
│                                                                  │
│  4. ReLU GATING                                                  │
│     ├── Clips neurons that would hurt margin                     │
│     ├── Asymmetric clipping for fwd vs rev                       │
│     └── Bonus margin: +10.1 by t=9                               │
│                                                                  │
│  5. W_out READOUT                                                │
│     ├── Discriminative direction: W_out[s] - W_out[m]            │
│     ├── Offset has positive projection onto this direction       │
│     └── Final margin at t=9: 53.1                                │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  h_fwd = h_rev + offset                                          │
│                                                                  │
│  margin = (W_out[s] - W_out[m]) @ offset                         │
│         = ||offset|| × ||discrim|| × cos(θ) + ReLU_bonus         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Why This Works

The network solves a fundamentally nonlinear problem (distinguishing which of two impulses is larger) using three interacting mechanisms:

1. **Separable encoding**: The offset factorizes as f(m) + g(s), allowing independent encoding of M and S positions

2. **Antisymmetric readout**: W_out is structured so that f suppresses and g boosts, guaranteeing correct discrimination regardless of specific positions

3. **ReLU cooperation**: The network learned to place "harmful" signal on neurons that get differentially clipped, turning ReLU from a mere nonlinearity into an active discrimination aid

## Implications

1. **The spiral picture is incomplete**: Forward and reverse don't follow the same spiral with different phases — they follow offset trajectories where the offset carries all discriminative information

2. **ReLU is not just for bounding**: It actively participates in discrimination by selectively filtering harmful contributions

3. **The antisymmetry is exact**: The f/g decomposition with W_out @ f = -W_out @ g is not approximate — it's a precise structural property the network learned

4. **Position encoding is distributed**: No single neuron encodes position; it emerges from the collective offset structure across all 16 neurons

## Related Documents

- [32: Phase Wheel Mechanism](32_phase_wheel_mechanism.md) — The countdown dynamics
- [30: Model Comparison](30_model_comparison.md) — Cross-model analysis
- Visualizations: `offset_mechanism_explained.png`, `offset_antisymmetry.png`, `offset_separable_structure.png`
