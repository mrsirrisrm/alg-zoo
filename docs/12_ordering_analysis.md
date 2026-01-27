# 12: Ordering Analysis - The Three-Impulse Problem

## Overview

The model's accuracy varies significantly depending on the temporal ordering of the top three values. By analyzing all six possible orderings, we discovered that the model's failure mode is fundamentally tied to **how many impulses** (clipping events) occur before the final timestep.

## The Six Orderings

The three largest values (1st=max, 2nd=second max, 3rd=third max) can appear in six temporal orders:

| Ordering | Meaning | Count | Accuracy | Regime |
|----------|---------|-------|----------|--------|
| **2-1-3** | 2nd first, max second, 3rd last | 16.6% | **91.9%** | Two-impulse |
| 1-2-3 | max first, 2nd second, 3rd last | 16.7% | 89.5% | Single-impulse |
| 3-1-2 | 3rd first, max second, 2nd last | 16.7% | 89.3% | Two-impulse (wrong pair) |
| 1-3-2 | max first, 3rd second, 2nd last | 16.7% | 89.2% | Single-impulse |
| 2-3-1 | 2nd first, 3rd second, max last | 16.7% | 89.2% | Two-impulse |
| **3-2-1** | 3rd first, 2nd second, max last | 16.6% | **87.2%** | Three-impulse |

**Key finding**: The best (2-1-3, 91.9%) and worst (3-2-1, 87.2%) orderings differ by 4.7 percentage points.

## Clipping Patterns by Ordering

### n7 Clipping Rates

| Ordering | at 1st in time | at 2nd in time | at 3rd in time | Total impulses |
|----------|----------------|----------------|----------------|----------------|
| 2-1-3 | 99.9% (2nd) | 92.9% (max) | 0.2% (3rd) | ~2 |
| 1-2-3 | 100% (max) | 16.1% (2nd) | 0.9% (3rd) | ~1 |
| 3-1-2 | 98.1% (3rd) | 99.5% (max) | 8.2% (2nd) | ~2 |
| 1-3-2 | 100% (max) | 4.9% (3rd) | 36.4% (2nd) | ~1-2 |
| 2-3-1 | 99.9% (2nd) | 12.1% (3rd) | 98.8% (max) | ~2 |
| **3-2-1** | **98.2% (3rd)** | **93.5% (2nd)** | **93.3% (max)** | **~3** |

**The 3-2-1 ordering is unique**: All three values clip, creating three impulses.

## Why 2-1-3 is Best

In the 2-1-3 ordering:

```
Time →  [2nd clips]  →  [max clips]  →  [3rd doesn't clip]
             ↓               ↓                   ↓
        Impulse A       Impulse B           No impulse
             ↓               ↓
        Decays with     Decays with
        Fourier freq    Fourier freq
             ↓               ↓
        ────────────────────────────────────────────────→ h_final
                    Clean anti-phase interference
                    encodes (2nd_pos - max_pos)
```

- Only TWO impulses
- 3rd doesn't clip (0.2%) so it doesn't interfere
- Clean anti-phase interference between 2nd and max
- Easy to decode via W_out matched filters

## Why 3-2-1 is Worst

In the 3-2-1 ordering:

```
Time →  [3rd clips]  →  [2nd clips]  →  [max clips]
             ↓               ↓               ↓
        Impulse A       Impulse B       Impulse C
             ↓               ↓               ↓
        Long decay     Medium decay    Short decay
             ↓               ↓               ↓
        ────────────────────────────────────────────────→ h_final
                    THREE-WAY INTERFERENCE
                    Hard to decode!
```

- THREE impulses, all interfering
- 3rd clips first → longest decay time → strongest residual signal
- 2nd clips second → "sandwiched" between 3rd and max
- Model must distinguish 3rd's signal from 2nd's signal

## Error Analysis for 3-2-1

### What the Model Predicts When Wrong

| Prediction | Count | Percentage |
|------------|-------|------------|
| 3rd argmax position | 2,452 | **54.8%** |
| max position | 1,698 | 38.0% |
| other | 323 | 7.2% |

Over half of errors predict 3rd instead of 2nd!

### Error Rate by Gap (2nd - 3rd value)

| Gap Range | Error Rate | Of errors, % predict 3rd |
|-----------|------------|--------------------------|
| [0.00, 0.03) | **26.4%** | 68.8% |
| [0.03, 0.06) | 11.5% | 47.3% |
| [0.06, 0.10) | 9.7% | 48.5% |
| [0.10, 0.20) | 7.5% | 32.6% |
| [0.20, 0.50) | 5.4% | 9.7% |

When 2nd and 3rd values are very close (gap < 0.03), the error rate is **26%** and 69% of those errors predict 3rd.

### Error Rate by Timing

**Gap between 3rd and 2nd positions:**

| Position Gap | Accuracy |
|--------------|----------|
| 1 position | 82.5% |
| 2 positions | 91.2% |
| 3 positions | **92.4%** (best) |
| 6 positions | 78.2% |
| 7 positions | **66.3%** (worst) |

Accuracy is worst when 3rd and 2nd are either very close (gap=1) or very far apart (gap=7) in time.

## h_final Comparison: 3-2-1 vs 2-1-3

| Neuron | 3-2-1 | 2-1-3 | Difference | Interpretation |
|--------|-------|-------|------------|----------------|
| n1 | 3.63 | 4.36 | -0.74 | Lower (more interference) |
| n4 | **0.32** | **0.00** | +0.32 | Higher (detects late max) |
| n5 | **1.34** | **0.80** | +0.54 | Higher (regime signal) |
| n6 | 3.83 | 4.62 | -0.79 | Lower (more interference) |
| n7 | 5.00 | 5.50 | -0.50 | Lower (more interference) |
| n8 | 3.93 | 4.57 | -0.64 | Lower (more interference) |

**Pattern**: Comparators are **lower** in 3-2-1 (more destructive interference from three impulses), while n4 and n5 are **higher** (detecting the late-max regime).

## Within 3-2-1: Correct vs Wrong Predictions

| Neuron | When Correct | When Wrong | Difference |
|--------|--------------|------------|------------|
| n7 | 5.21 | 3.64 | **+1.57** |
| n6 | 3.98 | 2.82 | +1.16 |
| n1 | 3.77 | 2.72 | +1.05 |
| n8 | 4.07 | 3.02 | +1.05 |

When the model is **wrong**, comparators have significantly **lower** values - suggesting more interference/cancellation is occurring.

## Which Neurons Push Toward 3rd (When Wrong)

| Neuron | Contrib to 3rd | Contrib to 2nd | Diff (3rd - 2nd) |
|--------|----------------|----------------|------------------|
| n6 | +2.24 | -6.97 | **+9.21** |
| n15 | +7.05 | -1.14 | **+8.19** |
| n12 | +5.23 | -2.68 | **+7.91** |
| n1 | +1.06 | -2.59 | +3.65 |
| n7 | +7.67 | +4.96 | +2.72 |
| n2 | -9.52 | +5.06 | -14.57 |
| n8 | -8.00 | +5.30 | -13.30 |

**Key insight**:
- n6 and some "other" neurons (n12, n15) strongly push toward predicting 3rd
- n2 and n8 try to push toward 2nd, but they're overwhelmed
- The three-impulse interference pattern creates ambiguous comparator signals

## The Three-Impulse Problem Explained

### In Two-Impulse Regime (2-1-3)

```
h_final[comparator] ≈ A·f(pos_2nd) - B·f(pos_max)
```

Two signals with anti-phase relationship. The difference cleanly encodes the position information.

### In Three-Impulse Regime (3-2-1)

```
h_final[comparator] ≈ A·f(pos_3rd) - B·f(pos_2nd) + C·f(pos_max)
```

Three signals interfering. The model must somehow extract 2nd's contribution while rejecting 3rd's contribution. This is fundamentally harder because:

1. **3rd clips first** → longest decay → largest magnitude in h_final
2. **2nd clips second** → medium decay → medium magnitude
3. **Both 3rd and 2nd** come before max, so both have similar "anti-phase" relationships
4. When 2nd and 3rd values are similar, their amplitudes are similar
5. The model can't reliably distinguish which earlier impulse was 2nd vs 3rd

## The Failure Mechanism

```
┌─────────────────────────────────────────────────────────────────┐
│                    3-2-1 FAILURE MECHANISM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Three values clip: 3rd (98%) → 2nd (94%) → max (93%)        │
│                                                                  │
│  2. All three create impulses in comparators                     │
│                                                                  │
│  3. 3rd clips FIRST → longest decay → STRONGEST signal          │
│                                                                  │
│  4. 2nd clips SECOND → medium decay → medium signal             │
│                                                                  │
│  5. Both 3rd and 2nd are "before max" → similar phase           │
│                                                                  │
│  6. When gap(2nd-3rd) is small:                                 │
│     - Similar amplitudes                                         │
│     - Similar decay times                                        │
│     - Model can't distinguish them                               │
│                                                                  │
│  7. 3rd's stronger signal dominates → model predicts 3rd        │
│                                                                  │
│  Result: 55% of errors predict 3rd instead of 2nd               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implications for Model Understanding

### The Anti-Phase Mechanism Has Limits

Our earlier finding that argmax and 2nd_argmax have anti-phase signals works well when:
- Only two significant impulses occur
- The third value doesn't clip

But it breaks down when three impulses create a more complex interference pattern.

### The Model Lacks a "3rd Rejection" Mechanism

The model has:
- ✓ Anti-phase encoding for 2nd vs max
- ✓ n5 for single-impulse regime detection
- ✗ No clear mechanism to reject 3rd when it also clips

### Why Accuracy is Still 87%

Despite the three-impulse problem, the model still achieves 87% accuracy in 3-2-1 because:
1. Often the gap between 2nd and 3rd is large enough for amplitude discrimination
2. Timing differences help (3rd decays more, 2nd decays less)
3. n2 and n8 do push toward correct answer, partially compensating

## Connection to Surprise Accounting

The 3-2-1 ordering represents **residual surprise** in our explanation:

- We explained the two-impulse mechanism well
- We didn't fully account for the three-impulse case
- This contributes to the gap between our ~26-bit explanation and perfect understanding

To fully explain the model, we would need to characterize:
1. How the model partially discriminates 2nd from 3rd
2. Why n12 and n15 contribute to the error pattern
3. The exact amplitude thresholds for discrimination

## Summary

| Aspect | Two-Impulse (2-1-3) | Three-Impulse (3-2-1) |
|--------|---------------------|------------------------|
| Impulses | 2 (2nd, max) | 3 (3rd, 2nd, max) |
| Accuracy | 91.9% | 87.2% |
| Primary mechanism | Anti-phase interference | Complex three-way interference |
| Main error mode | N/A | Predicts 3rd (55% of errors) |
| Error trigger | - | Small gap between 2nd and 3rd |

The model's elegant interference-based encoding works beautifully for the common two-impulse case but struggles when forced to process three interfering signals. This is a fundamental limitation of the learned representation, not a bug that could be easily fixed.
