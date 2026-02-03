# Unified Readout Mechanism for M₁₆,₁₀

## Overview

The 2nd-argmax RNN (16 hidden neurons, sequence length 10) solves the task of identifying the position of the second-largest input. Given two impulses M (magnitude 1.0) and S (magnitude < 1.0) at positions m and s, the network must output s.

The readout mechanism combines two interleaved computations in a shared low-dimensional subspace:

1. **Phase Wheel** — Encodes position as phase in a recurrent spiral
2. **Discrimination Offset** — Separates M from S using a single learned function f(·)

## The Phase Wheel

The hidden state follows a spiral trajectory through time. After both impulses have occurred, the network executes a **countdown** to the S position:

```
prediction(t) = s_pos + (10 - t)
```

At t=10, the countdown reaches exactly s_pos, yielding the correct output.

This countdown is implemented by a rotating phase in the hidden state. The W_out projection converts phase angle to position logits, with the argmax tracking the countdown.

**Key property**: The spiral shape is identical for forward (M first) and reverse (S first) cases — only the scale differs by a factor of M/S ≈ 1.25.

## The Discrimination Offset

When M precedes S (forward) vs S precedes M (reverse), the network must predict different positions. The final hidden states differ by an **offset**:

```
h_forward = h_reverse + offset(m, s)
```

This offset is **separable** and **antisymmetric**:

```
offset(m, s) = f(m) + g(s)  where  g(s) = -f(s)
```

Equivalently:
```
offset(m, s) = f(m) - f(s)
```

There is a single learned encoding f: {0,...,9} → R¹⁶ that maps position to a 16-dimensional vector. The network applies +f for the M position and -f for the S position.

### Properties of f(·)

| Property | Value |
|----------|-------|
| Separable model R² | 91% |
| Antisymmetry | Perfect (cosine = 1.0000) |
| Dimensionality | ~88% in rank-2 subspace |
| Functional form | Roughly cubic/sinusoidal, not simply linear |

The remaining 9% of offset variance comes from ReLU clipping effects at tropical boundaries.

## Shared Subspace

Both mechanisms operate in the **same low-dimensional subspace** of the hidden state:

| Subspace | Variance in PC1-2 |
|----------|-------------------|
| Main phase wheel | 60% |
| f(m) discrimination | 80% |
| f(m) in Main PC1-4 | 72% |

The principal angles between the 2D subspaces are 49° and 20° — partially overlapping but encoding different information.

**Alignment structure**:
- f(m) PC1 ≈ Main PC2 (cosine = 0.92)
- f(m) PC2 ≈ Main PC1 (cosine = 0.67)

Despite sharing the same subspace, the position-by-position correlation is only 0.04 — the spirals encode **orthogonal information** within a **shared geometric space**.

**Relative magnitudes**:

| Spiral | Mean ‖·‖ | Range |
|--------|----------|-------|
| Main (hidden state) | 38.3 | 24 - 59 |
| Offset (f(m) - f(s)) | 10.5 | 5 - 18 |
| **Ratio** | **27.5%** | |

The discrimination offset is a small perturbation (~1/4 magnitude) on top of the main phase wheel. The main spiral carries the bulk of the position encoding, while the offset provides a modest correction to shift the readout between M and S.

## How Discrimination Works

The offset f(m) - f(s) projects through W_out to create discriminative logits:

```
W_out @ offset = W_out @ f(m) - W_out @ f(s)
```

This creates:
- **Positive bias** at positions 0-4 (early positions)
- **Negative bias** at positions 5-9 (late positions)

For a forward case (M at early position m, S at late position s):
- The offset suppresses logit[m] and boosts logit[s]
- Combined with the phase wheel countdown, this yields prediction = s

For reverse (M at late position, S at early position):
- The offset sign flips (since offset = f(m) - f(s))
- Now the offset boosts the early position and suppresses the late position
- The countdown still targets s (the smaller input)

## Tropical Geometry Perspective

The offset arises from **linear dynamics within shared tropical cells**:

- Forward and reverse trajectories have identical ReLU activation patterns 88% of the time
- Only 1.3 neurons per pair are differentially clipped on average
- The offset is 88% linear, 12% from clipping effects

This means the offset is primarily computed by the same linear map applied to different initial conditions, not by crossing different ReLU boundaries.

## Summary

The M₁₆,₁₀ network solves 2nd-argmax through two interleaved mechanisms in a shared ~2D subspace:

```
┌─────────────────────────────────────────────────────────────────┐
│                     HIDDEN STATE SUBSPACE                       │
│                                                                 │
│   Phase Wheel (Main Spiral)     Discrimination (f spiral)       │
│   ─────────────────────────     ────────────────────────────    │
│   • Encodes position pair       • Encodes M vs S distinction    │
│   • Drives countdown to S       • Creates offset = f(m) - f(s)  │
│   • Same shape for fwd/rev      • Opposite sign for fwd/rev     │
│                                                                 │
│              Both share PC1-PC2, encode orthogonal info         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │        W_out          │
                    │  (16 → 10 projection) │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Logits + Argmax     │
                    │   → Output position   │
                    └───────────────────────┘
```

The elegance is in the reuse: a single recurrent oscillatory system encodes both "where" (phase wheel) and "which" (discrimination offset), with the linear readout W_out extracting both signals to produce the correct answer.
