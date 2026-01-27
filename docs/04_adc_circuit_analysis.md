# 04: ADC-like Circuit Analysis

## Overview

The 2nd argmax model combines analog and digital-like neurons in a hybrid architecture reminiscent of an Analog-to-Digital Converter (ADC). This document analyzes the circuit structure and neuron roles.

## Neuron Classification

### Binary Neurons (Comparators)

| Neuron | W_ih | Behavior |
|--------|------|----------|
| n1 | -10.56 | Clips (→0) when x is large |
| n6 | -11.00 | Clips (→0) when x is large |
| n7 | -13.17 | Clips (→0) when x is large |
| n8 | -12.31 | Clips (→0) when x is large |
| n4 | +10.16 | Clips (→0) when x is **small** |

These neurons act as comparators: their output transitions sharply based on whether the input crosses a threshold.

**Complementary pair**: n4 and n7 form a Q/Q̄ pair:
- When x[t] > running_max: n7 clips to 0, n4 activates
- When x[t] < running_max: n7 stays active, n4 clips to 0

| Position is max | n4 active | n7 clipped |
|-----------------|-----------|------------|
| Yes | ~99% | ~97% |
| No | ~3% | ~3% |

### Analog Neurons

| Neuron | W_ih | Role |
|--------|------|------|
| n2 | +0.015 | **Max tracker** (r=0.83 with running_max) |
| n0, n3, n5, n9-15 | ±0.1-0.3 | Supporting roles |

n2 is special: near-zero input coupling means it receives information only through recurrence, making it a pure integrator.

## Circuit Architecture

### Stage 1: Large Value Detection

When a large value x[t] arrives:
- **n4** activates (positive W_ih means large x → positive pre-activation)
- **n1, n6, n7, n8** clip (negative W_ih means large x → negative pre-activation)

### Stage 2: Max Tracking (Analog Sample-and-Hold)

**n2** acts as an analog memory:
```
n2[t] = ReLU(0.97 * n2[t-1] + 1.73 * n4[t-1] + 0.015 * x[t])
```

- Self-recurrence (0.97): maintains its value over time
- Input from n4 (1.73): boosted when large value detected
- Correlation with running_max: 0.83

When max is at t=5:
```
t=3: n4=0.58, n2=12.4
t=4: n4=0.38, n2=13.6
t=5: n4=1.78, n2=14.2  ← n4 spikes at max
t=6: n4=0.00, n2=17.2  ← n2 jumps up
t=7: n4=0.00, n2=16.9  ← n2 holds
```

### Stage 3: Adaptive Threshold

n7's clipping threshold depends on the hidden state:
```
threshold_n7 = -(W_hh[7,:] @ h_prev) / W_ih[7]
```

Main contribution comes from n2:
- W_hh[7, 2] = +0.67
- n2 mean contribution = 9.97 (vs 0.51 from n4)

So: **threshold ≈ 0.67 * n2 / 13.17 ≈ 0.05 * n2**

Since n2 tracks running_max, the threshold adapts to track the largest value seen so far.

### Stage 4: Position Encoding

The clipping pattern encodes which positions had "large" values:
- When x[t] > threshold: n7 clips → marks position t
- The pattern of clips across t=0..9 encodes the top-k positions

## ADC Analogy

| ADC Component | Model Equivalent |
|---------------|------------------|
| Sample-and-hold | n2 (analog max tracker) |
| Reference voltage | n2's output (adapts to running max) |
| Comparators | n1, n6, n7, n8 (different thresholds) |
| Flash encoder | n4 (active signal for "large") |

### Quantization Behavior

At t=5, clipping probability by input value:

| x range | n1 | n6 | n7 | n8 |
|---------|----|----|----|----|
| [0.0, 0.2) | 0% | 0% | 0% | 0% |
| [0.2, 0.4) | 1% | 1% | 0% | 2% |
| [0.4, 0.6) | 10% | 8% | 4% | 13% |
| [0.6, 0.8) | 34% | 26% | 20% | 41% |
| [0.8, 1.0) | 72% | 67% | 65% | 81% |

Different comparators fire at slightly different effective thresholds, providing redundancy.

### Running Max Encoding

When x[t] establishes a new max, on average 3.8/4 comparators clip.
When x[t] is not a new max, only 0.2-0.3/4 clip.

This 10:1 ratio provides robust detection of "new max" events.

## Output Contribution

Which neurons matter for the final classification?

| Neuron | Output importance | Role |
|--------|-------------------|------|
| n7 | 39.1 | Primary comparator |
| n8 | 36.4 | Comparator |
| n6 | 31.2 | Comparator |
| n1 | 30.8 | Comparator |
| n10 | 17.3 | Analog support |
| n0 | 14.9 | Analog support |

The binary comparators dominate the output, consistent with their role in position encoding.

## Key Insight: Complementary Signals

The circuit uses both:
1. **Absence signals** (n1, n6, n7, n8): output drops to 0 when large value seen
2. **Presence signals** (n4): output becomes positive when large value seen

This redundancy provides:
- Noise robustness
- Clear threshold updates (n4 → n2 → n7_threshold)
- Both positive and negative evidence for "is this position the max?"

## Scripts

See `src/adc_analysis.py` for the experimental code.
