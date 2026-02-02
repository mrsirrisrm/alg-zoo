# 32: The Phase Wheel Mechanism

## Overview

The M₁₆,₁₀ network solves the 2nd argmax task using a **phase wheel mechanism** — a remarkably elegant solution where position is not stored explicitly but instead encoded as phase in an oscillating dynamical system.

## The Mechanism in Five Steps

### 1. Impulse Response

When M (max value, 1.0) and S (second max, 0.8) tokens arrive, they inject energy into the hidden state via W_ih:

```
h_t = ReLU(W_ih · x_t + W_hh · h_{t-1})
```

Each non-zero input creates an "impulse response" that kicks the hidden state into motion.

### 2. Rotation

W_hh has complex eigenvalues that cause the hidden state to **rotate**:

| Eigenvalue | |λ| | Period | Role |
|------------|-----|--------|------|
| 0.43 ± 1.03j | 1.115 | 5.4 steps | Primary oscillator (growing) |
| -0.11 ± 0.90j | 0.902 | 3.7 steps | Secondary oscillator (stable) |
| +1.17 | 1.167 | ∞ | Amplifier |

The complex eigenvalues create rotation; the real growing eigenvalue sustains energy.

### 3. Bounded Spiral (ReLU's Role)

Without ReLU, the growing eigenvalues (|λ| > 1) would cause the hidden state to explode:

| Timestep | With ReLU | Without ReLU |
|----------|-----------|--------------|
| t=2 | ||h|| = 37.7 | ||h|| = 98.4 |
| t=9 | ||h|| = 37.6 | ||h|| = 509.2 |

ReLU clips negative activations, creating a **bounded, decaying spiral** that preserves phase while preventing explosion.

### 4. Phase = Position

The hidden state trajectory can be visualized in PCA space as a spiral. The key insight: **phase encodes position**.

At any timestep t after S arrives:
```
predicted_position(t) = S_pos + (9 - t)
                      = S_pos + steps_remaining
```

At the final timestep t=9:
```
predicted_position(9) = S_pos + 0 = S_pos  ✓
```

### 5. Readout via W_out

W_out rows are arranged around the "phase wheel" — adjacent rows have positive correlation, rows 4 apart have negative correlation (half-period):

| Row pair | Correlation | Interpretation |
|----------|-------------|----------------|
| 0 ↔ 1 | +0.51 | Adjacent (similar phase) |
| 0 ↔ 4 | -0.73 | Half-period (opposite phase) |
| 0 ↔ 8 | +0.01 | Full period (similar phase) |

The argmax of W_out @ h extracts the position corresponding to the current phase.

## The Complete Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE PHASE WHEEL MECHANISM                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    1st IMPULSE (M or S)                                          │
│         │                                                        │
│         ▼                                                        │
│    ┌─────────┐     complex          ┌─────────┐                  │
│    │   W_ih  │ ──→ eigenvalues ──→  │ SPIRAL  │  (stalls alone)  │
│    └─────────┘     create           │ in h(t) │                  │
│                    rotation         └────┬────┘                  │
│                                          │                       │
│    2nd IMPULSE (S or M)                  │                       │
│         │         ReLU nonlinear         │                       │
│         └──────── interaction ───────────┘                       │
│                                          │                       │
│                                     ReLU bounds                  │
│                                     amplitude                    │
│                                          │                       │
│                                          ▼                       │
│                                    ┌──────────┐                  │
│                                    │ COUNTDOWN│                  │
│                                    │  SPIRAL  │                  │
│                                    └────┬─────┘                  │
│                                         │                        │
│                                    W_out decodes                 │
│                                    phase → position              │
│                                         │                        │
│                                         ▼                        │
│                                    ┌──────────┐                  │
│                                    │  OUTPUT  │                  │
│                                    │ position │                  │
│                                    └──────────┘                  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   pred(t) = S_pos + (9 - t)    ←── The countdown formula         │
│                                                                  │
│   Countdown starts at t = max(m_pos, s_pos)                      │
│   At t=9: pred = S_pos ✓                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Critical Refinement: Two Impulses Required

The clean countdown does **not** emerge from a single impulse. Both M and S must arrive.

### Single impulse behaviour

A sequence with only one non-zero token (M or S alone) produces:

```
Single impulse at t=0: predictions = [4, 6, 9, 8, 0, 9, 9, 9, 9, 9]
                                      transient → stalls at 9
```

The hidden state enters a fixed-point attractor after ~5 steps (6 neurons active), and the readout locks onto position 9.

### Why both impulses matter

The 2nd impulse arrives into a hidden state already shaped by the 1st impulse's spiral. ReLU's nonlinear interaction between the existing state and the new input creates a qualitatively different trajectory:

| Condition | Active neurons at 2nd impulse | Trajectory |
|-----------|-------------------------------|------------|
| Single impulse | 6 | Stalls at prediction 9 |
| Both impulses | 10 | Clean countdown begins |

The 1st impulse sets the spiral in motion; the 2nd impulse's ReLU interaction reshapes the activation pattern, launching the countdown from that timestep onward.

### Forward / Reverse Symmetry

The mechanism is symmetric with respect to M/S arrival order:

- **Forward** (M before S): countdown starts from S arrival
- **Reverse** (S before M): countdown starts from M arrival
- **General rule**: countdown starts from the 2nd impulse, i.e. from `max(m_pos, s_pos)`

Verified for all 90 (M, S) position pairs at 100% accuracy in both directions.

## Key Insight

**The network doesn't "remember" that S is at position 3.**

Instead, it:
1. The 1st impulse kicks the hidden state into a spiral
2. The 2nd impulse's nonlinear (ReLU) interaction with the existing spiral launches the countdown
3. Oscillatory dynamics count down the remaining steps
4. W_out converts phase to position at readout time

Time itself encodes position through the phase wheel.

## Evidence

### The Countdown Experiment

Reading out at intermediate timesteps reveals the countdown:

```
M@0, S@1: predictions = [4, 9, 8, 7, 6, 5, 4, 3, 2, 1]
                              ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
                        counts down from 9 to 1
```

### Spiral Visualization

The hidden state traces a decaying spiral in PCA space:
- Starts with large radius after impulse
- Decays as ReLU clips growth
- Rotates ~36° per timestep
- At t=9, phase points to correct position

### W_out Structure

W_out row correlations show wave pattern with half-period ≈ 4 positions, confirming phase wheel arrangement.

## Prediction Scorecard

Five falsifiable predictions were derived from the phase wheel model and tested:

| # | Prediction | Result | Notes |
|---|-----------|--------|-------|
| P1 | Countdown formula holds for all 90 (M,S) pairs | **CONFIRMED** (100%) | pred(t) = S_pos + (9 - t) |
| P2 | Impulse superposition: h(M+S) ≈ h(M) + h(S) | **PARTIALLY FALSE** | cos_sim 0.89–0.97; ReLU breaks linearity |
| P3 | Reverse direction uses same countdown | **CONFIRMED** (100%) | Symmetric: countdown from 2nd impulse |
| P4 | Single impulse stalls (no countdown) | **CONFIRMED** | Stalls at prediction 9 after ~5 steps |
| P5 | Value invariance (argmax independent of input magnitude) | **CONFIRMED** | Same positions for any non-zero value |

**Key refinement from P2/P4:** The mechanism is fundamentally nonlinear. Superposition fails because ReLU's interaction between the two impulses is what creates the countdown — not the sum of two independent spirals.

## Implications

1. **Sequence length is baked in**: The oscillation period is calibrated for exactly 10 timesteps. Different sequence lengths would require retuning.

2. **Eigenvalue structure is essential**: The complex eigenvalues create the rotation; the growing eigenvalues sustain energy.

3. **ReLU is critical**: Without it, the system would explode. ReLU creates bounded oscillation — and ReLU's nonlinear mixing of two impulses is what launches the countdown.

4. **Position = time encoding**: This is a fundamentally temporal solution, not a lookup table.

5. **Two impulses are necessary**: A single impulse produces a transient that stalls. The countdown requires the nonlinear interaction of both impulses.

6. **Forward/reverse symmetric**: The mechanism doesn't care which token (M or S) arrives first. The 2nd impulse always initiates the countdown.

## Related Documents

- [33: Offset Discrimination Mechanism](33_offset_discrimination_mechanism.md) — How forward/reverse are distinguished
- [30: Model Comparison](30_model_comparison.md) — Cross-model analysis
- [31: Matroid Analysis](31_matroid_analysis.md) — Essential neurons
- Visualizations: `phase_wheel_summary.png`, `spiral_analysis.png`, `countdown_pattern.png`, `impulse_interaction.png`
