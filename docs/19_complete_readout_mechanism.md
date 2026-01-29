# 18: The Complete Readout Mechanism

## Overview

This document synthesizes findings from docs 13-17 into a complete description of how the model solves the 2nd-argmax task. The mechanism relies on two complementary encoding systems:

1. **Wave neurons (n10, n11, n12)**: Encode **absolute max position** through a phase-shifted sinusoidal pattern
2. **Comparators (n1, n6, n7, n8)**: Encode **max value minus current input**, creating strong suppression at max_pos

The 2nd position is found **by elimination**: the comparators suppress the max, and the wave neurons' sinusoidal pattern (decoded by W_out) naturally peaks at positions other than max. Critically, the network does **not** confuse +delta and -delta (positions before vs after max) because the W_out decoder is asymmetric around any given max position.

## The Wave Neuron Subsystem

### Function: Absolute Position Encoding

Wave neurons encode **where** the maximum occurred, not what its value was:

| Neuron | r(h_final, max_pos) | r(h_final, max_val) | DFT k=1 energy |
|--------|---------------------|---------------------|----------------|
| n10 | +0.27 | +0.11 | 74% |
| n11 | +0.04 | +0.10 | 64% |
| n12 | -0.31 | +0.18 | 74% |

The h_final values trace out a sinusoidal impulse response as max_pos varies:

| max_pos | n10 | n11 | n12 |
|---------|-----|-----|-----|
| 0 | 5.23 | 5.93 | 10.23 |
| 1 | 2.75 | 4.68 | 10.66 |
| 2 | 1.00 | 2.75 | 8.74 |
| 3 | **0.02** | 1.19 | 6.83 |
| 4 | 2.66 | **0.02** | 5.37 |
| 5 | 4.82 | 2.52 | **3.14** |
| 6 | 6.11 | 4.69 | 5.14 |
| 7 | 5.99 | 5.48 | 7.78 |
| 8 | 5.35 | 5.17 | 8.06 |
| 9 | 3.43 | 3.28 | 6.73 |

Each neuron has its trough at a different max_pos (n10 at pos 3, n11 at pos 4, n12 at pos 5), creating a **phase-shifted triplet**:

| Neuron | Phase (rad) | Phase (deg) |
|--------|-------------|-------------|
| n12 | -0.16 | -9° |
| n11 | +0.91 | +52° |
| n10 | +1.70 | +97° |

### Architecture

The wave neurons form a unidirectional cascade driven by n4's impulse:

```
n4 (W_ih = +10.16) ──+1.76──→ n10
                   └──+1.71──→ n11
                   └──+1.28──→ n12

Feedback sustains the wave:
n12 ──+0.66──→ n11 ──+1.10──→ n10
```

The sub-block eigenvalue at ω = 0.511 rad (period 12.3 steps) creates a damped oscillation that decays to 0.67% after 10 steps — enough to carry position information through to readout.

### Clip Rate Dynamics

Wave neurons are mostly active throughout the sequence, only clipping late:

| t | n10 clip% | n11 clip% | n12 clip% |
|---|-----------|-----------|-----------|
| 0-4 | 0% | 0% | 0-8% |
| 5-9 | 13-17% | 11-19% | 0-7% |

This ensures continuous encoding of the max position event.

## The Comparator Subsystem

### Function: Max Suppression

Comparators encode the **difference** between the max value and the current (last) input:

```
h_final[n7] ≈ +12.5 * x[max_pos] - 13.1 * x[9]
```

This is the "diagonal boundary" from doc 14 expressed as a linear model. The comparators don't directly encode 2nd position — they create a **strong negative logit at max_pos**:

| Metric | Value |
|--------|-------|
| Comparator logit at max_pos | -6.76 |
| Comparator logit at 2nd_pos | +3.19 |
| Gap (2nd - max) | **+9.95** |

The comparators push a ~10-logit wedge between the max and everything else.

### The Four Comparators

Each comparator has slightly different sensitivity (doc 14):

| Neuron | α/|β| ratio | Type |
|--------|-------------|------|
| n7 | 1.02 | Pure comparison (x[max] vs x[current]) |
| n1 | 0.84 | Soft comparison |
| n6 | 0.04 | Mostly absolute threshold |
| n8 | 0.03 | Mostly absolute threshold |

All contribute to max suppression with sinusoidal W_out columns (85-95% at k=1).

## The W_out Decoder

### Sinusoidal Matched Filters

W_out columns are sinusoidal at k=1 (period 10), acting as matched filters:

| Neuron | DFT k=1 energy | Function |
|--------|----------------|----------|
| n7 | 95% | Comparator → sinusoidal suppression |
| n8 | 85% | Comparator |
| n10 | 65% | Wave → sinusoidal position readout |
| n12 | 71% | Wave |

### Asymmetry Around Max Position

The sinusoidal W_out pattern is **not symmetric** around any given max position. For example, W_out[:, n10]:

```
pos:  0     1     2     3     4     5     6     7     8     9
    +0.16 -1.67 -2.81 -3.87 +1.66 +1.11 +2.03 +1.64 +0.70 +1.67
```

For max at position 5:
- Position 2 (delta = -3): W_out[2, 10] = -2.81
- Position 8 (delta = +3): W_out[8, 10] = +0.70

These are **completely different values**. The network cannot confuse "3 before max" with "3 after max" because they map to different logit contributions.

## The Complete Mechanism

### Step 1: Max Value Triggers n4 Burst

When a large input arrives, n4 (W_ih = +10.16) fires a burst that:
- Seeds the wave neurons with energy
- Gets suppressed by self-inhibition (W_hh[4,4] = -0.99)

### Step 2: Wave Neurons Encode Max Position

The n4 burst propagates through n10-n11-n12:
- Energy flows: n12 → n11 → n10 via feedback
- The phase at readout (t=9) encodes when the max occurred
- h_final traces a sinusoidal curve as max_pos varies

### Step 3: Comparators Suppress Max Position

When the max is encountered:
- Comparators clip (pre < 0 when current > running_max)
- At readout, comparators compute x[max_pos] - x[9]
- W_out sinusoidal columns create negative logit at max_pos

### Step 4: W_out Decodes Position

The output logit for each position is:

```
logit[pos] = Σ_n h_final[n] * W_out[pos, n]
```

For wave neurons, this computes:
```
wave_logit[pos] ≈ A * sin(pos * 2π/10 + φ_max)
```

where φ_max depends on max_pos. This creates a sinusoidal logit pattern **centered on the max position** but with different values at each absolute position.

### Step 5: 2nd Position Emerges by Elimination

1. **Max is suppressed**: Comparators create -6.76 logit at max_pos
2. **Sinusoidal pattern peaks elsewhere**: Wave neurons create highest logits near (but not at) max
3. **Winner**: The 2nd-largest value's position typically has the highest combined logit

## Why No Mirror Confusion?

### Question: Does the Network Confuse +delta and -delta?

If the 2nd position is at delta = +3 (3 after max) vs delta = -3 (3 before max), does the network confuse them?

### Answer: No

**Accuracy by delta sign:**

| Condition | Accuracy |
|-----------|----------|
| delta > 0 (2nd after max) | 89.5% |
| delta < 0 (2nd before max) | 89.2% |

**Delta sign correctness:**

| True delta | Pred delta correct sign |
|------------|------------------------|
| > 0 | 94.2% |
| < 0 | 94.5% |

**Mirror confusion rate (among errors):**
- Predicted mirror position: **2.5%**
- Predicted max position: **30.1%**
- Other: **67.4%**

When the model makes errors, it rarely predicts the mirror position — it's more likely to predict the max position or some other position entirely.

### The Symmetry-Breaking Mechanism

Three factors prevent mirror confusion:

**1. Absolute Position Encoding (Wave Neurons)**

Wave neurons encode max_pos as an absolute position. The sinusoidal W_out pattern is anchored to this absolute position, not to a relative delta. Position 2 and position 8 have different W_out values regardless of where the max is.

**2. Sinusoidal Asymmetry**

A sinusoid is only symmetric around its peaks and troughs. For any given max_pos, the positions before and after have different phase relationships to the sinusoid:

```
Example: max at pos 5
Wave logit pattern:
  pos 0: +7.00   pos 5: -3.35 ← max
  pos 1: +0.44   pos 6: +4.03
  pos 2: -9.13   pos 7: +10.68
  pos 3: -17.20  pos 8: +2.43
  pos 4: -2.47   pos 9: +10.88
```

Position 2 (delta = -3) gets logit -9.13, while position 8 (delta = +3) gets +2.43. They're not symmetric.

**3. The 2nd Value Also Creates a Signature**

When the 2nd-largest value arrives, it may exceed the running max at that point, triggering comparator clipping. This creates an additional encoding event:

| 2nd_pos | n7 clip rate at 2nd |
|---------|---------------------|
| 0 | 100% |
| 2 | 75% |
| 5 | 53% |
| 8 | 36% |

The 2nd value's arrival time affects the hidden state trajectory, providing additional discriminative information.

## Accuracy Analysis

### By Absolute 2nd Position

| 2nd_pos | Accuracy |
|---------|----------|
| 0 | 91.1% |
| 1 | 92.1% |
| 2 | 91.1% |
| 3 | 91.2% |
| 4 | 89.4% |
| 5 | 87.5% |
| 6 | 87.1% |
| 7 | 87.4% |
| 8 | 87.8% |
| 9 | 88.8% |

Early 2nd positions (0-3) have ~4% higher accuracy than late positions (5-8). This reflects the sinusoidal W_out pattern having stronger discrimination in the early part of the sequence.

### By Delta Magnitude

| |delta| | Accuracy |
|---------|----------|
| 1 | 89.4% |
| 2 | 91.4% |
| 3 | 91.5% |
| 4 | 91.0% |
| 5 | 88.4% |
| 6 | 88.1% |
| 7 | 82.3% |
| 8 | 88.2% |
| 9 | 81.7% |

Small to medium deltas (1-4) have the best accuracy. Large deltas (7, 9) have reduced accuracy — these are edge cases where max and 2nd are at opposite ends of the sequence.

## Group Contributions

### Ablation Results

| Condition | Accuracy |
|-----------|----------|
| Full model | 89.3% |
| Wave neurons zeroed | **22.2%** |
| Comparators zeroed | **15.7%** |

Both systems are essential. Neither alone achieves meaningful accuracy.

### Role Summary

| Group | Encodes | Mechanism | Output effect |
|-------|---------|-----------|---------------|
| Wave neurons | max_pos (absolute) | Phase-shifted sinusoidal h_final | Sinusoidal logit pattern centered on max |
| Comparators | x[max] - x[current] | Diagonal boundary clipping | Strong negative logit at max_pos |
| Combined | — | — | 2nd_pos emerges as winner |

## Summary: The Readout Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│  1. DETECT MAX: When x[t] > running_max                         │
│     - n4 fires burst (W_ih = +10.16)                           │
│     - Comparators clip (pre < 0)                               │
├─────────────────────────────────────────────────────────────────┤
│  2. ENCODE MAX POSITION: Wave neurons (n10, n11, n12)          │
│     - n4 burst seeds the wave                                  │
│     - Feedback (n12 → n11 → n10) sustains oscillation          │
│     - Phase at t=9 encodes when max occurred                   │
│     - h_final is sinusoidal in max_pos (74% at k=1)            │
├─────────────────────────────────────────────────────────────────┤
│  3. ENCODE MAX VALUE: Comparators (n1, n6, n7, n8)             │
│     - h_final ≈ +12.5 * x[max_pos] - 13.1 * x[9]               │
│     - Carried through recurrence by n2-n4 circuit              │
│     - W_out columns are sinusoidal (85-95% at k=1)             │
├─────────────────────────────────────────────────────────────────┤
│  4. SUPPRESS MAX IN OUTPUT: Comparators + W_out                │
│     - Comparator logit at max_pos: -6.76                       │
│     - Comparator logit at 2nd_pos: +3.19                       │
│     - Gap: +9.95 (max is strongly suppressed)                  │
├─────────────────────────────────────────────────────────────────┤
│  5. FIND 2ND BY ELIMINATION: Wave + Comparator                 │
│     - Wave sinusoidal pattern peaks near (not at) max          │
│     - Comparator suppresses max                                │
│     - 2nd_pos emerges as winner                                │
│     - No mirror confusion: W_out is asymmetric around max_pos  │
└─────────────────────────────────────────────────────────────────┘
```

The model doesn't explicitly encode the 2nd position — it encodes the max position and value, then uses elimination to find the 2nd. This elegant solution requires only 432 parameters to achieve 89% accuracy on a task that would require thousands of XGBoost splits (doc 14).
