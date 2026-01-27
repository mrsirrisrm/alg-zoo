# 01: ReLU Clipping Analysis of the 2nd Argmax Model

## Overview

This document analyzes how the 432-parameter M₁₆,₁₀ model uses ReLU clipping as a computational mechanism for finding the second-largest value in a sequence of 10 numbers.

## Model Architecture

- **Task**: Find position of 2nd largest value in sequence of 10 random numbers [0,1]
- **Architecture**: Single-layer RNN with 16 ReLU hidden neurons
- **Parameters**: 432 total
  - W_ih: 16 (input → hidden)
  - W_hh: 256 (hidden → hidden)
  - W_out: 160 (hidden → output)
- **Accuracy**: ~89%

## Key Discovery: Adaptive Clipping Thresholds

The model implements an **adaptive thresholding** mechanism using ReLU clipping.

### The Mechanism

For neurons with large negative W_ih (neurons 1, 6, 7, 8):

```
pre_activation[t] = W_hh @ h[t-1] + W_ih * x[t]

Clipping occurs when: x[t] > -W_hh @ h[t-1] / W_ih

The THRESHOLD = -W_hh @ h[t-1] / W_ih
```

**Critical finding**: This threshold correlates **0.92** with max(x[0:t-1])!

### Interpretation

- Neuron 7 clips when x[t] exceeds approximately the previous maximum
- This detects "is the current value a new maximum?"
- P(clip | position is max) ≈ 97%
- P(clip | position is NOT max) ≈ 3-10%

The clipping pattern encodes WHERE large values appeared in the sequence.

## Neuron Roles

### Neuron 2: Running Maximum Tracker
- Never clips (always active)
- Activation correlates 0.82 with max(x)
- Provides explicit magnitude information

### Neurons 1, 6, 7, 8: "New Maximum" Detectors
- W_ih ≈ -10 to -13 (large negative)
- Clip when current input exceeds adaptive threshold
- Threshold tracks the running maximum
- Creates positional encoding of "large value events"

| Neuron | W_ih | Role |
|--------|------|------|
| 1 | -10.56 | Large value detector |
| 6 | -11.00 | Large value detector |
| 7 | -13.17 | Large value detector |
| 8 | -12.31 | Large value detector |

### Neuron 4: Inverse Detector
- W_ih ≈ +10 (large positive)
- Active when seeing large values
- Clips when input is small

## Clipping Patterns

### Clipping Rate When Position is Maximum

| Timestep | P(n7 clips \| max at t) | P(n7 clips \| max NOT at t) |
|----------|------------------------|----------------------------|
| t=1 | 98.2% | 43.6% |
| t=2 | 97.5% | 24.9% |
| t=3 | 96.8% | 16.2% |
| t=4 | 96.7% | 12.0% |
| t=5 | 96.4% | 9.4% |
| t=6 | 97.4% | 7.6% |
| t=7 | 97.3% | 5.8% |
| t=8 | 97.3% | 4.5% |
| t=9 | 96.5% | 2.8% |

### Transition Correlation with Top-2 Status

Correlation between neuron 1 transitioning to clipped and x[t] being in top-2:

| Timestep | Correlation |
|----------|-------------|
| t=2 | 0.13 |
| t=5 | 0.50 |
| t=7 | 0.61 |
| t=9 | 0.67 |

## Failure Mode Analysis

### Statistics
- 72% of failures have gap(2nd_max, 3rd_max) < 0.05
- Accuracy when max at position 9: 81.4%
- Accuracy when max NOT at position 9: 90.2%

### Why Failures Occur

1. **Small Gap Problem** (72% of failures)
   - When 2nd and 3rd max are close, both trigger similar clipping patterns
   - Model can't distinguish which was truly 2nd

2. **Last-Position Problem**
   - When max is at position 9, neurons 1,6,7,8 clip at final timestep
   - This "overwrites" information about previous large values

3. **Position Bias**
   - Worse accuracy for 2nd max in later positions (5-9)
   - Average 2nd argmax position: 4.5 (correct) vs 4.9 (wrong)

### Worst (argmax, 2nd_argmax) Combinations

| argmax | 2nd_argmax | Accuracy |
|--------|------------|----------|
| 0 | 7 | 70.4% |
| 9 | 6 | 72.6% |
| 9 | 7 | 75.9% |
| 9 | 0 | 76.2% |

## The Computational Trick

The model turns ReLU from an information-destroying operation into an information-encoding one:

**Instead of**: "ReLU loses information about negative values"

**It becomes**: "ReLU ENCODES whether x[t] exceeded the running max"

This is implemented by:
1. Large negative W_ih: Direct suppression by current input
2. W_hh structure: Carries forward max information
3. The balance creates threshold ≈ running_max

## Information Encoding Summary

The final hidden state encodes:
1. The maximum value (neuron 2, corr 0.82)
2. Whether each position was "large" (clipping pattern)
3. Information about second-largest value (R² = 0.82 from linear probe)

Linear probe accuracy:
- From final hidden state: 90.0%
- Model's actual accuracy: 89.3%
- From final pre-activation: 75.5%

The hidden state (post-ReLU) contains MORE information than pre-activation for this task, because the clipping itself encodes useful positional information.
