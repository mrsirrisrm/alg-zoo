# 18: Impulse Response and Zero-Crossing Analysis

## Overview

We investigated the network's response to impulse inputs ([1,0,0,...], [0,1,0,...], etc.) and double-impulse inputs ([1,0,0,0.8,0,...]). This revealed: (1) a deterministic zero-crossing cascade that classifies neurons into three dynamical tiers, (2) that n0 and n14 are a tightly coupled secondary wave pair, and (3) that superposition fails badly — the second impulse disrupts the first through clipping-induced nonlinear interactions, yet the model correctly tracks both positions.

## Single Impulse: Time-Translation Invariance

Every impulse produces the **same propagation pattern shifted in time**. An impulse at t=0 triggers: boot-up at t=0, ring-up at t=1, then a characteristic decay/oscillation from t=2–9. An impulse at t=k produces the identical pattern starting at t=k. This confirms the bell-ringing model (doc 17).

Full trajectory for impulse at t=0 (selected neurons):

```
t    n0     n4     n7     n10    n11    n12    n14    n15
0    0.1   10.2    0.0C    0.1    0.2    0.3    0.0C   0.0C
1    6.2    0.0C  13.4    18.0   17.7   13.2    8.9    1.9
2    0.0C   0.0C  13.3    23.1   19.3   11.3    0.0C   4.2
3    4.8    0.0C  13.3    21.9   11.0    2.7    4.6    0.0C
4   13.7    0.0C  11.6    13.7    3.4    0.4   13.6    0.0C
5   21.0    0.0C   9.7     4.9    0.0C   6.2   20.7    5.1
6   23.2    0.0C   8.6     0.0C   2.6   16.7   23.1   12.2
7   20.1    0.0C   8.8     0.3    9.4   26.1   19.6   17.0
8   13.6    0.0C  10.2     6.4   16.5   29.7   12.6   18.3
9    7.8    0.0C  11.9    14.3   20.5   26.8    6.6   14.4
```

## Zero-Crossing Cascade

Each neuron that clips after the impulse does so at an **exact, fixed delay** — zero variance across impulse positions:

| Delay | Neurons | Mechanism |
|-------|---------|-----------|
| **+1** | n4 | Self-destruction: W_hh[4,4] = -0.99 |
| **+2** | n0, n9, n14 | First oscillation dip |
| **+3** | n5, n15 | Second dip (driven by n4 cascade) |
| **+5** | n11 | Wave neuron trough |
| **+6** | n10 | Wave neuron trough (one step later) |
| **never** | n1, n2, n3, n6, n7, n8, n12, n13 | No post-impulse clipping |

The cascade is entirely deterministic and position-independent — it is a property of the recurrent dynamics.

### Three Dynamical Tiers

**Tier 1 — Transient (clip and stay clipped):** n4, n5, n9
- Fire once and die (n4 at +1) or briefly activate then permanently clip (n5 at +3, n9 at +2)
- Carry one-shot information about the impulse event

**Tier 2 — Oscillating (clip once, recover):** n0, n10, n11, n14, n15
- Single zero-crossing creates a non-monotonic impulse response
- {n0, n14}: dip at +2, recover at +3
- {n10, n11}: dip at +5/+6, recover one step later
- n15: dip at +3, recover at +5
- These carry phase-encoded position information

**Tier 3 — Monotone (never clip post-impulse):** n1, n2, n3, n6, n7, n8, n12, n13
- Comparators (n1, n6, n7, n8): amplitude-encoded position
- Backbone (n2, n12, n13): always active, persistent state

### Clip Rate by Delay (all impulse positions aggregated)

```
delay  n0  n1  n2  n3  n4  n5  n6  n7  n8  n9 n10 n11 n12 n13 n14 n15
  0     .   C   .   C   .   C   C   C   C   C   .   .   .   C   C   C
  1     .   .   .   .   C   .   C   .   C   .   .   .   .   .   .   .
  2     C   .   .   .   C   .   .   .   .   C   .   .   .   .   C   .
  3     .   .   .   .   C   C   .   .   .   C   .   .   .   .   .   C
  4     .   .   .   .   C   C   .   .   .   C   .   .   .   .   .   C
  5     .   .   .   .   C   C   .   .   .   C   .   C   .   .   .   .
  6     .   .   .   .   C   C   .   .   .   C   C   .   .   .   .   .
  7+    .   .   .   .   C   C   .   .   .   C   .   .   .   .   .   .
```

(C = always clipped at this delay, . = always active)

At delay 0, exactly the 10 neurons with W_ih < 0 clip — the universal boot-up pattern from doc 17. The subsequent cascade proceeds deterministically from there.

## N0 and N14: A Secondary Wave Pair

### Near-Identical Dynamics

n0 and n14 correlate at **r = 0.992** across all 10 impulse response values:

| Impulse pos | n0 h_final | n14 h_final | Ratio |
|-------------|-----------|------------|-------|
| t=0 | 7.77 | 6.55 | 1.19 |
| t=1 | 13.63 | 12.56 | 1.09 |
| t=2 | 20.09 | 19.57 | 1.03 |
| t=3 (peak) | 23.16 | 23.12 | 1.00 |
| t=4 | 20.98 | 20.67 | 1.01 |
| t=5 | 13.67 | 13.56 | 1.01 |
| t=6 | 4.76 | 4.61 | 1.03 |
| t=7 (zero) | 0.00 | 0.00 | — |
| t=8 | 6.16 | 8.89 | 0.69 |
| t=9 | 0.07 | 0.00 | — |

Both trace a bell-shaped envelope peaking when the impulse was 6 steps before readout, with a zero-crossing at 7 steps. The only divergence is at t=8 (n14 is larger, likely due to its W_ih = -1.32 vs n0's +0.07 creating different initial conditions at the impulse step).

### Wave-Like Impulse Responses

DFT of h_final across impulse positions:

| Neuron | k=1 energy | k=2 energy | Peak | Phase |
|--------|-----------|-----------|------|-------|
| **n0** | **90.1%** | 3.1% | k=1 | -1.84 rad |
| **n14** | **82.3%** | 6.6% | k=1 | -1.88 rad |
| n10 | 60.8% | 17.4% | k=1 | +1.94 rad |
| n11 | 31.0% | 32.5% | k=2 | -1.80 rad |
| n12 | 61.6% | 18.6% | k=1 | -0.66 rad |

n0 and n14 have the **highest k=1 concentration** of any neurons — purer sinusoidal impulse responses than the primary wave neurons (n10, n11, n12).

### Different W_out Decoders

Despite near-identical dynamics, their W_out columns decode at different frequencies:

| Neuron | k=1 % | k=2 % | k=4 % | k=5 % | Primary decoder |
|--------|-------|-------|-------|-------|----------------|
| n0 | 6% | **54%** | 11% | 11% | **k=2 (period 5)** |
| n14 | 9% | **35%** | **22%** | **27%** | **k=2 + higher harmonics** |

Compare to primary wave neuron decoders (k=1 dominated, ~85% energy at k=1). The secondary pair provides **higher spatial frequency** discrimination.

### N14's Position-0 Suppression

n14's W_out has a standout feature:

```
W_out[:, 14] = [-5.14, +0.61, +1.11, +1.31, +1.23, -0.62, -1.01, +1.22, +0.09, +1.18]
```

Position 0 gets -5.14 while the mean of other positions is +0.57. With mean h_final ≈ 5.0, this contributes **-25.7 to position 0's logit** — a dedicated suppression mechanism preventing over-prediction of position 0. This complements n2's -14.7 contribution (doc 05).

### N0's Period-5 Pattern

n0's W_out shows a k=2 (period-5) structure:

```
W_out[:, 0] = [+1.96, -2.69, -1.65, +0.77, +0.68, +1.96, +1.24, -2.90, -0.30, +0.69]

Positions {0,5}: mean = +1.96
Positions {2,7}: mean = -2.28
```

This discriminates between positions that are 5 apart — the pairs that k=1 encoding cannot distinguish.

### Shared Connectivity

Both receive from comparators and timing neurons in the same direction:

| Source | → n0 | → n14 |
|--------|------|-------|
| n6 (comparator) | +1.13 | +0.77 |
| n7 (comparator) | -0.78 | -0.71 |
| n4 (burst) | +0.59 | +0.87 |
| n9 (integrator) | -1.42 | -0.65 |
| n5 (timing) | -0.91 | -0.42 |

Mutual coupling: n0 → n14: +0.42, n14 → n0: +0.40 (positive feedback pair).

Both feed into **n3** with opposite signs: n0 → n3: **-1.23**, n14 → n3: **+0.59**. Since n0 ≈ n14, n3 receives ≈ -0.64 × (their shared signal), making n3 an inverter of the secondary wave.

### Impulse Response Correlation Clusters

Cross-correlations between impulse responses reveal three clusters:

| Cluster | Neurons | Mutual r |
|---------|---------|----------|
| **Secondary wave** | n0, n14 | +0.99 |
| **Wave + followers** | n12, n3, n15 | +0.84 to +0.96 |
| **Primary wave** | n10, n11 | +0.63 |

n0/n14 anti-correlate with n10 (r ≈ -0.60), confirming they operate in anti-phase with the primary wave system.

## Full Impulse Response DFT Classification

| Neuron | k=1 energy | Zero-cross delay | Classification |
|--------|-----------|-----------------|----------------|
| n0 | 90% | 2 | Secondary wave (k=2 decoder) |
| n14 | 82% | 2 | Secondary wave (k=2+ decoder, pos-0 suppressor) |
| n15 | 79% | 3 | Tertiary wave |
| n8 | 79% | never | Comparator (amplitude) |
| n6 | 76% | never | Comparator (amplitude) |
| n13 | 65% | never | Backbone (Nyquist decoder) |
| n1 | 63% | never | Comparator (amplitude) |
| n12 | 62% | never | Wave (primary, backbone) |
| n10 | 61% | 6 | Wave (primary, oscillating) |
| n3 | 37% | never | Integration (broadband) |
| n2 | 33% | never | Accumulator (broadband) |
| n11 | 31% | 5 | Wave (primary, oscillating) |
| n5 | 23% | 3 | Transient (one-shot) |
| n4 | 20% | 1 | Impulse source (one-shot) |
| n7 | 15% | never | Comparator (broadband amplitude) |
| n9 | 20% | 2 | Transient (one-shot) |

n7 — the primary comparator — has the **lowest** k=1 energy (15%). Its impulse response is nearly flat across positions 0–8, then drops to 0 at position 9. It encodes position primarily through the binary clip/no-clip at the last timestep, not through smooth variation.

## Double Impulse: Superposition Fails

### The Test

Compare single [1,0,0,...], double [1,0,0,0.8,0,...], and isolated [0,0,0,0.8,0,...]. If the system were linear, double = single + isolated.

### h_final Comparison

| Neuron | Single | Double | Isolated (0.8) | Sum | Double − Sum |
|--------|--------|--------|---------------|-----|-------------|
| n0 | 7.77 | 4.03 | 18.53 | 26.29 | **-22.27** |
| n7 | 11.89 | 12.64 | 6.88 | 18.76 | -6.12 |
| n10 | 14.30 | 2.66 | 0.00 | 14.30 | **-11.63** |
| n11 | 20.51 | 8.01 | 2.10 | 22.62 | **-14.60** |
| n12 | 26.81 | 15.23 | 13.33 | 40.14 | **-24.92** |
| n14 | 6.55 | 3.79 | 18.50 | 25.05 | **-21.26** |

**Superposition fails catastrophically.** The double response is typically 15–85% of the linear sum. The L2 superposition error is 50.28 (for A=0.8 at t=3).

### Failure Direction: Always Subadditive

Every deviation is **negative** — the double is always less than the sum. The second impulse doesn't add to the first; it **suppresses** activity by causing additional clipping events.

### New Clipping Events from the Second Impulse

The 0.8 at t=3 creates 6 clipping changes relative to the single impulse:

| Neuron | Timestep | Change | Mechanism |
|--------|----------|--------|-----------|
| n1 | t=3 | Now clips | Comparator: 0.8 exceeds threshold |
| n8 | t=3 | Now clips | Comparator: same |
| n3 | t=4 | Now clips | Integration: cascade from clipping |
| n3 | t=5 | Now clips | Continued suppression |
| n5 | t=4 | Now active | Released from clipping by perturbation |
| n15 | t=5 | Now clips | Cascade from n5 change |

The second impulse triggers comparators (n1, n8 clip at t=3), which cascades through integration neurons (n3 clips at t=4–5) and timing neurons (n5, n15 flip state). These clipping changes alter the effective W_hh sub-matrix for all subsequent timesteps, causing the wave neurons to follow entirely different trajectories.

### Amplitude Sweep: [1, 0, 0, A, 0, ...]

| A | Prediction | n10 | n11 | n12 | Superpos. error |
|---|-----------|-----|-----|-----|----------------|
| 0.0 | pos 9 | 14.30 | 20.51 | 26.81 | 0.0 |
| 0.1 | pos 3 | 13.21 | 20.15 | 26.65 | 5.8 |
| 0.3 | pos 3 | 9.26 | 17.54 | 25.06 | 18.0 |
| 0.5 | pos 3 | 6.23 | 13.73 | 21.64 | 30.7 |
| 0.8 | pos 3 | 2.66 | 8.01 | 15.23 | 50.3 |
| 1.0 | pos 3 | 0.00 | 3.06 | 11.06 | 63.0 |

Three observations:
1. **Even A=0.1 switches the prediction** from pos 9 to pos 3. The model is extremely sensitive to secondary inputs.
2. The superposition error grows roughly linearly with A.
3. Wave neurons (n10, n11, n12) are suppressed monotonically as A increases — the second impulse progressively disrupts the first impulse's wave pattern.

### Position Sweep: [1, 0, ..., 0.8, ..., 0]

| 2nd impulse position | Prediction | Correct? |
|---------------------|-----------|----------|
| t=1 | pos 1 | ✓ |
| t=2 | pos 2 | ✓ |
| t=3 | pos 3 | ✓ |
| t=4 | pos 4 | ✓ |
| t=5 | pos 5 | ✓ |
| t=6 | pos 6 | ✓ |
| t=7 | pos 7 | ✓ |
| t=8 | pos 8 | ✓ |
| t=9 | pos 9 | ✓ |

The model correctly identifies the 2nd argmax at every position, despite massive nonlinearity in the hidden state. The superposition error varies by position (18–55), but prediction accuracy is perfect.

## The Nonlinear Encoding Mechanism

### Why Subadditivity Helps

The subadditive interaction is not a bug — it's the encoding mechanism. In the single-impulse case, only the max position is encoded (via wave pattern + comparator state). When a second value arrives:

1. **Comparators clip at the second value** → marks its position in the clipping pattern
2. **This disrupts the first impulse's wave** → changes the interference pattern
3. **The disruption pattern depends on relative timing** → encodes position difference
4. **W_out matched filters decode the interference** → predict 2nd argmax

The model doesn't need superposition to work. It needs the second impulse to **perturb** the first impulse's representation in a position-dependent way. The perturbation IS the signal.

### Trajectory Divergence

Looking at n0 for impulse at t=0, with and without second impulse at t=3:

```
t:     0     1     2     3     4     5     6     7     8     9
single: 0.1   6.2   0.0   4.8  13.7  21.0  23.2  20.1  13.6   7.8
double: 0.1   6.2   0.0   4.8   9.9   5.5   9.1   8.0   5.4   4.0
```

Trajectories are identical through t=3 (before the second impulse arrives). At t=4, the second impulse's clipping cascade changes the effective dynamics, and trajectories diverge permanently. By t=9, double is ~50% of single.

### Consistent Clipping Changes

The second impulse at different positions produces a consistent pattern of clipping changes:

- **Comparators always clip at the second impulse** (n1 and/or n8 at the impulse timestep)
- **n3 clips 1–2 steps later** (integration cascade)
- **n5 flips to active 1–2 steps later** (perturbation releases it)
- **n4 reactivates at the impulse** (when the second value is large enough, n4 fires briefly again)

This cascade creates the position-specific perturbation that encodes the 2nd argmax location.

## Connection to Prior Findings

| Doc | Finding | Connection |
|-----|---------|------------|
| 7 | Dual role mechanism | Zero-crossing tiers explain which neurons encode via amplitude vs phase |
| 8 | Interference encoding | Superposition failure shows interference is nonlinear, not wave-like addition |
| 13 | N2-N4 noise cancellation | n4's +1 delay self-destruction is the first step of the zero-crossing cascade |
| 15 | DFT-like structure | n0/n14 are secondary wave neurons, providing k=2 spatial resolution |
| 17 | Bell-ringing model | Single impulses obey time-translation invariance; double impulses break it through clipping interactions |

## Summary

| Finding | Detail |
|---------|--------|
| **Zero-crossing cascade** | Deterministic: n4 at +1, {n0,n9,n14} at +2, {n5,n15} at +3, {n11} at +5, {n10} at +6 |
| **Three tiers** | Transient (clip-and-die), Oscillating (clip-and-recover), Monotone (never clip) |
| **n0 and n14** | Secondary wave pair (r=0.99), k=1 impulse response decoded by k=2+ W_out columns |
| **n14 special role** | Position-0 suppression: -25.7 logit contribution |
| **n0 special role** | Period-5 (k=2) decoder: discriminates positions 5 apart |
| **Superposition** | Fails badly (50+ L2 error). Double response is always subadditive. |
| **Subadditivity is functional** | Second impulse perturbs first via clipping cascade; perturbation encodes 2nd argmax position |
| **Prediction accuracy** | Perfect on all double-impulse tests despite nonlinear hidden states |

## Scripts

See `src/impulse_and_connectivity.py` for the experimental code.
