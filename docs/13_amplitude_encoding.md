# 13: Amplitude Encoding - The N2-N4 Noise Cancellation Circuit

## Overview

While investigating how "clipping magnitude" might encode information, we discovered a sophisticated noise-cancellation circuit. The model doesn't just encode the *position* of the max value - it also encodes the *amplitude* with near-perfect accuracy (r=0.98) through a differential signaling mechanism between neurons n2 and n4.

## The Problem

The model needs to remember how large the max value was, not just where it occurred. But at any timestep, the hidden state encodes a mix of:
- **Signal**: information about max value
- **Noise**: information about previous inputs (history)

How does the model extract the signal from this noisy representation?

## The Discovery

### Two Neurons with Different Roles

**N2 (Running Accumulator)**:
- W_ih[2] = +0.015 (tiny input weight)
- W_hh[2,2] = +0.97 (strong self-recurrence)
- Accumulates history, minimal response to current input

**N4 (Max Value Sensor)**:
- W_ih[4] = +10.16 (large input weight!)
- Jumps strongly when large value arrives

### What Happens at Max Position

When the max value arrives:

| Neuron | Before Max | At Max | Change |
|--------|------------|--------|--------|
| h2 | ~10 | ~12 | Small (just self-recurrence) |
| h4 | ~3 | ~9 | Large jump (∝ max_val) |

The key transformation:
- **Before max**: h2 and h4 are weakly correlated (r ≈ -0.50)
- **After max**: h2 and h4 become **strongly negatively correlated** (r = -0.85)

Both still positively correlate with max_val:
- r(h2, max_val) = +0.42
- r(h4, max_val) = +0.13

## The Noise Cancellation Mechanism

At t=max+1, n2's pre-activation receives contributions from both:

```
pre[n2] = h2 * W_hh[2,2] + h4 * W_hh[2,4] + other
        = h2 * 0.97      + h4 * 1.73      + small
          ↑                ↑
          n2_contrib       n4_contrib
```

### Individual vs Combined Correlations

| Component | r with max_val |
|-----------|----------------|
| n2_contrib | +0.42 |
| n4_contrib | +0.13 |
| **SUM** | **+0.98** |

How can the sum have much higher correlation than either component?

### The Math: Variance Reduction

```
Var(n2_contrib) = 9.01
Var(n4_contrib) = 6.82
Cov(n2, n4)     = -6.70  ← Negative!

Var(sum) = Var(n2) + Var(n4) + 2*Cov(n2, n4)
         = 9.01 + 6.82 + 2*(-6.70)
         = 9.01 + 6.82 - 13.40
         = 2.43
```

**84.6% variance reduction** from the negative covariance term!

The negative correlation between n2 and n4 contributions causes their "noise" (history-dependent variance) to cancel, while their "signal" (max_val correlation) adds constructively.

## Circuit Diagram

```
    Input x_max ──────────────────────────────────────┐
         │                                            │
         │ (+0.01)                                    │ (+10.16)
         ▼                                            ▼
    ┌─────────┐                                  ┌─────────┐
    │   n2    │                                  │   n4    │
    │ history │◄─────── -0.85 correlation ──────►│  signal │
    │ (noise) │                                  │ (max)   │
    └────┬────┘                                  └────┬────┘
         │                                            │
         │ (*0.97)                                    │ (*1.73)
         │                                            │
         └──────────────────┬─────────────────────────┘
                            │
                            ▼ SUM
                     ┌──────────────┐
                     │ Noise cancels │
                     │ Signal sums   │
                     │ r = 0.98 !!   │
                     └──────────────┘
```

## Analogy: Differential Signaling

This mechanism is analogous to **differential signaling** in electronics:

- In electronics: Two wires carry inverted signals; subtracting them cancels common-mode noise
- In the RNN: Two neurons carry correlated "noise" (history); summing their negatively-correlated contributions cancels the noise

The model has learned a form of **common-mode rejection**!

## Trajectory Analysis

Tracking h2's correlation with max_val over time (for samples with max at position 4):

| Time | Mean h2 | r(h2, max_val) |
|------|---------|----------------|
| t=0 | 0.01 | +0.15 |
| t=1 | 7.98 | +0.15 |
| t=2 | 10.78 | +0.25 |
| t=3 | 12.44 | +0.33 |
| t=4 (max) | 13.50 | +0.42 |
| **t=5** | **16.96** | **+0.98** |
| t=6 | 16.79 | +0.99 |
| t=7 | 17.26 | +0.99 |

The correlation **jumps from 0.42 to 0.98** at t=max+1 when n4's contribution arrives!

## Why This Matters

### For Position Encoding
The comparator neurons (n1, n6, n7, n8) encode position through clipping patterns and Fourier-like oscillations.

### For Amplitude Encoding
The n2-n4 circuit encodes the max value's amplitude through:
1. n4 sensing the input magnitude
2. n2 providing a "reference" signal (history)
3. Differential combination canceling noise

### For the 2nd Argmax Task
The model needs to distinguish the 2nd largest value from:
- The max (position encoded by comparators)
- The 3rd largest (amplitude helps discriminate)

The n2-n4 amplitude encoding likely helps the model:
- Determine if a value is "close to max" (small gap)
- Distinguish between similar-valued candidates

## Connection to Partial Clipping

This explains why partial clipping patterns encode amplitude:

When 2nd value arrives:
- The comparators clip based on their thresholds
- But n2 already encodes "how big was max?"
- The comparison between n2's accumulated value and the new input helps discriminate

## Neurons That Don't Clip

At the max position, these neurons rarely clip and preserve continuous information:

| Neuron | Clip Rate | r(h, max_val) | Role |
|--------|-----------|---------------|------|
| n2 | 0.0% | +0.20 | Running accumulator |
| n4 | 0.1% | +0.10 | Max value sensor |
| n10 | 6.5% | +0.12 | Unknown |
| n11 | 6.4% | +0.11 | Unknown |
| n12 | 1.1% | +0.11 | Unknown |

These non-clipping neurons form an "analog backbone" that maintains continuous amplitude information while the comparators handle discrete position encoding.

## Summary

| Aspect | Position Encoding | Amplitude Encoding |
|--------|-------------------|-------------------|
| Primary neurons | n1, n6, n7, n8 (comparators) | n2, n4 |
| Mechanism | Clipping + Fourier oscillation | Differential noise cancellation |
| Key weight | W_ih negative (threshold) | W_ih[4] = +10.16 (sensor) |
| Correlation with target | Via W_out Fourier patterns | r = 0.98 with max_val |
| Information type | Discrete (clipped/not) | Continuous (magnitude) |

The model has learned to separate position encoding (clipping-based) from amplitude encoding (continuous differential signaling) - a remarkably sophisticated solution to the 2nd argmax problem.
