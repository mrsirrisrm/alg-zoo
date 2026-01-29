# 20: The True Mechanism — Reset and Rebuild

## Overview

Previous documents hypothesized an "anti-phase interference" mechanism where comparators encode `h ≈ A·sin(ω·max_pos) - B·sin(ω·2nd_pos)`. By testing with clean data (only max and 2nd non-zero, all other inputs zero), we discovered this is **incorrect**. The true mechanism is simpler and more elegant: **reset-and-rebuild**.

## The Clean Data Test

### Setup

Created 90 unique input sequences:
- `x[max_pos] = 1.0`
- `x[2nd_pos] = 0.8`
- `x[all other positions] = 0.0`

This isolates the pure two-impulse signal without noise from other inputs.

### Result

**100% accuracy** on the clean dataset. This confirms we're now seeing the true mechanism without confounding factors.

## The Discovery: Reset-and-Rebuild

### What Actually Happens

1. **Max impulse arrives** at position M → comparators clip (pre-activation < 0)
2. **Comparators rebuild** via recurrent dynamics from t=M+1 onward
3. **2nd impulse arrives** at position S → comparators may clip again
4. **Whichever clips later determines h_final** via the rebuild trajectory

### The Clipping Pattern

For neuron n7, tracking the last clipping time for each (max_pos, 2nd_pos) pair:

```
        2nd:    0   1   2   3   4   5   6   7   8   9
max
 0             -   0   0   0   0   5   6   7   8   0
 1             1   -   1   1   1   1   6   7   8   9
 2             2   2   -   2   2   2   2   7   8   9
 3             3   3   3   -   3   3   3   3   8   9
 4             4   4   4   4   -   4   4   4   4   9
 5             5   5   5   5   5   -   5   5   5   5
 6             6   6   6   6   6   6   -   6   6   6
 7             7   7   7   7   7   7   7   -   7   7
 8             8   8   8   8   8   8   8   8   -   8
 9             9   9   9   9   9   9   9   9   9   -
```

**Key pattern**: `last_clip ≈ max(max_pos, 2nd_pos)`

When 2nd comes before max (below diagonal): last clip is at max_pos
When 2nd comes after max (above diagonal): last clip is at 2nd_pos (usually)

### The Rebuild Trajectory

After clipping, n7 follows this deterministic rebuild trajectory:

```
Steps after clip:  0     1     2     3     4     5     6     7     8     9
h value:          0.0  13.4  13.3  13.3  11.6   9.7   8.6   8.8  10.2  11.9
                   C    ↑peak              ↓trough       ↑recovering
```

This is an **oscillatory decay** — not monotonic growth. The trajectory:
1. Jumps to ~13.4 immediately after clip
2. Stays high for 2-3 steps
3. Dips to minimum ~8.6 at step 6
4. Recovers toward ~12 by step 9

### h_final as a Function of Clip Time

Single impulse at each position, measuring h_final for n7:

| Impulse position | Steps to rebuild | h_final |
|------------------|------------------|---------|
| 0 | 9 | 11.89 |
| 1 | 8 | 10.19 |
| 2 | 7 | 8.83 |
| 3 | 6 | **8.59** (minimum) |
| 4 | 5 | 9.68 |
| 5 | 4 | 11.60 |
| 6 | 3 | 13.26 |
| 7 | 2 | 13.34 |
| 8 | 1 | 13.41 |
| 9 | 0 | 0.00 (still clipped) |

This creates the "sinusoidal" h_final pattern observed in earlier documents — but it's not from interference of two waves. It's **one oscillatory rebuild function** evaluated at different points.

## Why the "Anti-Phase" Appeared

Earlier analysis (doc 08) found strong negative correlations between h_final curves by max_pos vs 2nd_pos. This seemed to indicate anti-phase encoding. The actual explanation:

### When 2nd Before Max (Regime B)
- Last clip is at max_pos
- h_final = rebuild[9 - max_pos]
- h_final varies with max_pos

### When 2nd After Max (Regime C)
- Last clip is at 2nd_pos (often)
- h_final = rebuild[9 - 2nd_pos]
- h_final varies with 2nd_pos

The "anti-phase" correlation arose because:
- In regime B: h_final correlates with max_pos
- In regime C: h_final correlates with 2nd_pos
- Averaging over both regimes creates apparent negative correlation between the two curves

But there's no interference — just **different reset points** determining which part of the rebuild trajectory appears in h_final.

## Superposition Failure Explained

Previous analysis showed double impulse ≠ sum of single impulses. For n7:

| Case | Actual | Sum of Singles | Ratio |
|------|--------|----------------|-------|
| max=3, 2nd=7 | 10.66 | 19.27 | 0.55 |
| max=2, 2nd=6 | 10.83 | 19.44 | 0.56 |
| max=1, 2nd=8 | 10.96 | 20.92 | 0.52 |

**Explanation**: The second impulse doesn't add to the first — it **clips** the neuron, resetting h to near-zero and starting a fresh rebuild. The "actual" value reflects only the rebuild after the last clip, not a sum of two contributions.

Trace for max=3, 2nd=7:
```
t:      0    1    2    3    4     5     6     7     8     9
Single: 0.0  0.0  0.0  0.0  13.4  13.3  13.3  11.6  9.7   8.6
Double: 0.0  0.0  0.0  0.0  13.4  13.3  13.3  1.1   11.1  10.7
                                              ↑
                                        2nd impulse RESETS h!
```

At t=7, the 2nd impulse destroys the signal from max (h drops from 11.6 to 1.1), then a new rebuild begins.

## Different Comparators, Different Behaviors

Not all comparators behave identically. Tracking which position causes the last clip:

| Neuron | Last clip at max | Last clip at 2nd | Neither |
|--------|------------------|------------------|---------|
| n7 | **84%** | 16% | 0% |
| n1 | **76%** | 24% | 0% |
| n6 | 50% | 26% | 24% |
| n8 | 44% | **39%** | 17% |

**n7 primarily encodes max_pos** — it almost always clips last at the max position.

**n8 encodes both positions** — it clips at 2nd nearly as often as at max.

This differential behavior is key to how the network encodes both positions with just four comparators.

### Why the Difference?

Input weights (W_ih) determine clipping thresholds:

| Neuron | W_ih | Interpretation |
|--------|------|----------------|
| n7 | -13.17 | Highest threshold (hardest to clip) |
| n8 | -12.31 | Medium-high |
| n6 | -11.00 | Medium |
| n1 | -10.56 | Lowest threshold (easiest to clip) |

n7's high threshold means only the max value reliably clips it. n8's lower threshold means the 2nd value (0.8) can also clip it, especially if it arrives when h is low.

## The Readout Mechanism

Given that h_final encodes time-since-last-clip differently for each neuron, how does W_out decode the 2nd position?

### Example: max=3, 2nd=7

```
Logits: 0:-48.8  1:-24.5  2:-3.6  3:+4.0^  4:+10.5  5:-1.3  6:+10.9  7:+30.0*  8:+15.2  9:-19.6
                                    ↑max                              ↑2nd (winner)
```

### Example: max=7, 2nd=3

```
Logits: 0:-52.3  1:-16.4  2:+14.9  3:+29.6*  4:+14.8  5:-7.2  6:-4.1  7:+15.1^  8:+8.0  9:-23.8
                                    ↑2nd (winner)                      ↑max
```

The network correctly predicts 2nd in both cases. The key is that different comparators provide different "views":
- n7 (clips at max) tells W_out where max is
- n8 (clips at both) provides information about both positions
- W_out combines these to triangulate the 2nd position

## The Additive Model Revisited

With clean data, the additive model `h = A·f(max) + B·g(2nd) + C` achieves:

| Neuron | A | B | R² |
|--------|-----|-----|------|
| n7 | 1.10 | 1.13 | **0.993** |
| n8 | 1.06 | 1.11 | 0.973 |
| n6 | 1.09 | 1.14 | 0.966 |
| n1 | 1.07 | 1.14 | 0.962 |

R² > 0.96 for all comparators. But note: **A ≈ B ≈ 1.1** (same sign, not opposite!).

This works because:
- f(max) and g(2nd) are the marginal curves (average h_final for each position)
- These marginals capture the rebuild trajectory effect
- The actual h_final is well-predicted by knowing both positions

But this doesn't mean h_final is the *sum* of two independent signals. It's a lookup into a 2D table indexed by (max_pos, 2nd_pos), which happens to be well-approximated by an additive model.

## Connection to Prior Findings

| Prior Claim | Status | Actual Explanation |
|-------------|--------|-------------------|
| Anti-phase interference (doc 08) | **Incorrect** | Different reset points, not wave interference |
| Sinusoidal h_final curves (doc 15) | **Correct effect, wrong cause** | Oscillatory rebuild trajectory, not eigenmode frequency |
| Comparators encode position (doc 06) | **Correct** | Via time-since-last-clip |
| Wave neurons carry timing (doc 17) | **Partially correct** | They follow similar reset-rebuild dynamics |
| Superposition failure (doc 18) | **Correct observation** | Second impulse resets, doesn't add |

## The Complete Mechanism

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     THE RESET-AND-REBUILD MECHANISM                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. INPUT ARRIVES                                                        │
│     - Max value (1.0) at position M                                      │
│     - 2nd value (0.8) at position S                                      │
│                                                                          │
│  2. CLIPPING EVENTS                                                      │
│     - At t=M: comparators clip (W_ih < 0, large input → pre < 0)        │
│     - At t=S: some comparators clip again (depends on threshold)         │
│                                                                          │
│  3. REBUILD DYNAMICS                                                     │
│     - After each clip, h rebuilds via W_hh recurrence                   │
│     - Rebuild follows oscillatory trajectory (not monotonic!)            │
│     - h_final = rebuild_trajectory[9 - last_clip_time]                  │
│                                                                          │
│  4. DIFFERENTIAL ENCODING                                                │
│     - n7 (high threshold): clips mainly at max → encodes max_pos        │
│     - n8 (lower threshold): clips at both → encodes both positions      │
│     - n1, n6: intermediate behavior                                      │
│                                                                          │
│  5. READOUT                                                              │
│     - W_out combines h_final from all comparators                        │
│     - Different neurons provide different "views" of the positions       │
│     - 2nd position emerges from the combination                          │
│                                                                          │
│  KEY INSIGHT: No interference, no anti-phase, no Fourier encoding.       │
│  Just reset-and-rebuild with differential thresholds.                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Summary

| Aspect | Old Understanding | New Understanding |
|--------|-------------------|-------------------|
| h_final encoding | Anti-phase: A·sin(max) - B·sin(2nd) | Reset-rebuild: trajectory[9 - last_clip] |
| Sinusoidal pattern | Two interfering waves | One oscillatory rebuild trajectory |
| Superposition failure | Nonlinear interaction | Second clip resets the system |
| Position discrimination | Phase difference | Different thresholds → different reset points |
| Comparator roles | All similar | n7 encodes max, n8 encodes both |

The mechanism is simpler than anti-phase interference but equally elegant: **clipping resets the system, and the rebuild trajectory encodes time, which encodes position**.

## Scripts

See `src/clean_impulse_analysis.py` and `src/true_mechanism_analysis.py` for the experimental code.
