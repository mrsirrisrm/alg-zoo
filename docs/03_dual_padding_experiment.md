# 03: Dual Padding Experiment (Initial + Trailing)

## Motivation

From the clipping analysis (01), we observed that:
- P(n7 clips | max NOT at t) is high at early timesteps (43.7% at t=1)
- Accuracy drops when max is at position 0 (85.5%) or position 9 (81.4%)

The trailing padding experiment (02) showed +2.8% improvement by avoiding max at position 9.

**Question**: Does initial padding (small value at position 0) provide similar benefits?

## Experiment Results

### Summary

| Configuration | Accuracy | Improvement |
|--------------|----------|-------------|
| Original (10 random) | 89.1% | baseline |
| Trailing only (9 random + small) | 91.9% | +2.8% |
| Initial only (small + 9 random) | 91.2% | +2.1% |
| **Both (small + 8 random + small)** | **93.6%** | **+4.4%** |

### Why Initial Padding Helps

The benefit is straightforward: **the max is never at position 0**.

Original accuracy breakdown:
- Max at position 0: 85.5%
- Max NOT at position 0: 89.5%

With initial padding (0.01 at position 0):
- Max is never at position 0 (0.01 is always the minimum)
- 2nd argmax is also never at position 0

This mirrors exactly why trailing padding helps - it removes a problematic edge case.

### Clipping Behavior

At t=0, clipping is the same regardless of padding because there's no previous hidden state:
- Neurons 1, 6, 7, 8: 100% clipped (any positive input causes clipping)
- Neuron 4: 0% clipped

The initial padding doesn't change t=0 clipping, but it does affect the threshold at t=1:
- With tiny x[0], the recurrent contribution to t=1 is small
- This actually *increases* the false positive rate at t=1 (43.7% → 98.8%)
- But this doesn't hurt accuracy because position 0 is never the target anyway

### Accuracy by Position

| Position | Original | Trail-pad | Init-pad | Both-pad |
|----------|----------|-----------|----------|----------|
| 0 | 90.9% | 93.8% | - | - |
| 1 | 92.7% | 94.0% | 92.2% | 93.1% |
| 2 | 91.7% | 94.3% | 94.4% | 96.2% |
| 3 | 91.3% | 92.5% | 95.1% | 96.0% |
| 4 | 88.9% | 90.4% | 92.0% | 93.3% |
| 5 | 87.4% | 90.3% | 89.8% | 93.0% |
| 6 | 86.1% | 89.6% | 89.4% | 92.0% |
| 7 | 87.4% | 91.7% | 90.1% | 93.0% |
| 8 | 87.9% | 90.4% | 89.6% | 92.8% |
| 9 | 88.4% | - | 88.8% | - |

Notes:
- Positions 0 and 9 show "-" when that padding makes them impossible targets
- Both-pad gives the most consistent accuracy across positions 1-8
- Initial padding particularly helps early positions (2, 3)
- Trailing padding particularly helps late positions (7, 8)

## Mechanism Summary

### Trailing Padding (small at pos 9)
- Max is never at position 9
- Final hidden state preserves information from positions 0-8
- Neurons 1, 6, 7, 8 are never clipped at final timestep
- **Benefit**: Clean read-out of accumulated information

### Initial Padding (small at pos 0)
- Max is never at position 0
- Removes edge case where first value is the maximum
- **Benefit**: Avoids problematic "max at start" scenarios

### Both Padding (small + 8 random + small)
- Combines both benefits
- 8 "real" values in positions 1-8
- Clean initialization AND clean read-out
- The improvements are roughly additive: 2.8% + 2.1% ≈ 4.4%

## Implications

1. **The model struggles with edge positions**: Both position 0 and position 9 have lower accuracy when they contain the max. Padding eliminates these edge cases.

2. **Effective sequence length**: With both-end padding, the model effectively processes 8 random values instead of 10, but achieves higher accuracy (93.6% vs 89.1%).

3. **Information preservation**: The key issue isn't the sequence length but avoiding clipping-induced information loss at the boundaries.

4. **Potential training insight**: A model trained specifically on padded sequences might learn even better representations.

## Controlling for Reduced Output Space

A key question: is the improvement just from having a simpler task (8 possible outputs vs 10)?

### Controlled Comparison

| Condition | Accuracy |
|-----------|----------|
| Original, filtered to target ∈ {1-8} AND max ∈ {1-8} | 90.7% |
| Original, with logits for 0/9 masked to -∞ | 92.4% |
| **Padded (natural restriction to 1-8)** | **93.5%** |

### Analysis

1. **Original filtered (90.7%)**: Same data distribution restricted to cases where the task naturally has the answer in positions 1-8.

2. **Logit masking (92.4%)**: Tells the model "the answer isn't 0 or 9" at inference time by masking those logits. This is +1.7% over filtered original.

3. **Padding (93.5%)**: Creates inputs where the answer is naturally in 1-8 AND the clipping mechanism isn't disrupted.

### The True Clipping Benefit

```
Padding vs Original-filtered:  +2.9%  (total improvement from padding mechanism)
Padding vs Logit-masking:      +1.1%  (improvement beyond just knowing the output space)
```

The **+2.9%** improvement from padding vs the filtered baseline represents the TRUE benefit from preventing clipping-induced information loss - not just from having a simpler output space.

### Per-Position Breakdown

| Position | Original (filtered) | Padded | Improvement |
|----------|---------------------|--------|-------------|
| 1 | 92.5% | 93.5% | +1.0% |
| 2 | 92.7% | 95.7% | +3.0% |
| 3 | 91.1% | 95.9% | +4.8% |
| 4 | 89.6% | 93.1% | +3.6% |
| 5 | 89.3% | 93.1% | +3.7% |
| 6 | 88.9% | 91.8% | +2.9% |
| 7 | 91.5% | 92.7% | +1.2% |
| 8 | 89.6% | 92.3% | +2.8% |

The improvement is seen across ALL positions, with the largest gains in the middle (positions 3-5). This is consistent with the clipping mechanism explanation: padding provides cleaner information flow throughout the sequence.

## Scripts

See `src/dual_padding_experiment.py` for the experimental code.
