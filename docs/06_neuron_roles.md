# 06: Neuron Roles and Circuit Structure

## Overview

The 16 neurons in the 2nd argmax model can be grouped into functional categories based on their connectivity, behavior, and contribution to the output.

## Neuron Summary Table

| Neuron | W_ih | Self-rec | Category | Role |
|--------|------|----------|----------|------|
| n1 | -10.6 | +0.41 | Comparator | Large-value detector |
| n6 | -11.0 | +0.36 | Comparator | Large-value detector |
| n7 | -13.2 | +0.08 | Comparator | Primary large-value detector |
| n8 | -12.3 | +0.62 | Comparator | Large-value detector |
| n4 | +10.2 | -0.99 | Inverted Comp | Active when x large (Q̄ signal) |
| n2 | +0.01 | +0.97 | Tracker | Analog max tracker, normalizer |
| n9 | -0.23 | -0.55 | Timing | "No large value yet" detector |
| n12 | +0.30 | +0.61 | Accumulator | Early-max accumulator |
| n10 | +0.06 | -0.00 | Relay | n4 signal relay |
| n11 | +0.15 | +0.20 | Relay | n4 signal relay |
| n15 | -0.12 | +0.23 | Integration | Early-max indicator |
| n3 | -0.12 | -0.36 | Integration | Comparator integrator |
| n0 | +0.07 | +0.16 | Integration | Mixed integration |
| n14 | -1.32 | +0.36 | Soft Comp | Weak comparator (W_ih between 0 and -10) |
| n13 | -0.05 | -0.61 | Integration | Value correlator |
| n5 | -0.28 | -0.42 | Timing | Late-max indicator |

## Functional Groups

### 1. Comparators (n1, n6, n7, n8)

**Function**: Detect when input exceeds adaptive threshold (approximately running max).

- W_ih ≈ -11 (large negative)
- Clip (output → 0) when x[t] > threshold
- Position of 2nd argmax encoded through clipping pattern
- Different output weight profiles create position signatures

### 2. Inverted Comparator (n4)

**Function**: Complementary signal to comparators - active when x is large.

- W_ih = +10.2 (large positive)
- Clips when x is SMALL, active when x is LARGE
- Forms Q/Q̄ pair with n7
- Broadcasts to many downstream neurons

### 3. Max Tracker (n2)

**Function**: Track running maximum, normalize output.

- W_ih ≈ 0 (pure integrator)
- Self-recurrence = 0.97 (holds state)
- Fed by n4 (+1.73): boosted when large value seen
- Correlates with max(x) at r = 0.83
- **Critical for output**: removing n2 drops accuracy from 89% → 14%

### 4. Timing Neurons

**n9: "No large value yet" detector**
- Strongly inhibited by n7 (-5.0) and n8 (-1.1)
- Active only when no large values seen recently
- High when max is at late positions (hasn't happened yet)

**n5, n10: Late-max indicators**
- Positively correlated with argmax position
- Higher when max occurs late in sequence

**n12: Early-max accumulator**
- Fed by n4 (+1.28)
- Self-recurrence = 0.61 (accumulates)
- Negatively correlated with argmax position
- Higher when max occurs early (more time to accumulate)

**n15: Early-max indicator**
- Fed by n11 (+1.28) and n10 (-1.09)
- Correlates negatively with argmax position (-0.36)
- Second-order effect of n4 cascade

### 5. Integration/Relay Neurons

**n10, n11: n4 relays**
- Both fed strongly by n4 (~+1.7)
- Pass the "large value detected" signal downstream

**n3: Comparator integrator**
- Receives from n0 (-1.23), n1 (+1.20), n8 (-1.07)
- Integrates multiple comparator signals

**n0: Mixed integrator**
- Receives from n5 (-0.9), n6 (+1.1), n9 (-1.4)
- Complex mix of timing and comparator signals

### 6. Unclear Roles

**n13, n14**: Contribute significantly to output but have complex connectivity patterns that don't fit neatly into categories.

- n14 has W_ih = -1.32 (weak comparator?)
- n13 receives from n2, n3, n4, n15 (multi-source integration)

## Circuit Connectivity

```
INPUT (x[t])
    |
    v
+-------------------+     +-------------------+
| COMPARATORS       |     | INVERTED COMP     |
| n1, n6, n7, n8    |     | n4                |
| (clip on large x) |     | (active on large) |
+-------------------+     +-------------------+
    |                           |
    | inhibits                  | excites
    v                           v
+-------------------+     +-------------------+
| n9                |     | n2 (MAX TRACKER)  |
| (no-large-yet)    |     | n10, n11 (relays) |
+-------------------+     | n12 (accumulator) |
                          +-------------------+
                                |
                                v
                          +-------------------+
                          | n15, n3, n0, n14  |
                          | (integration)     |
                          +-------------------+
                                |
                                v
                          +-------------------+
                          | OUTPUT LAYER      |
                          | (position decode) |
                          +-------------------+
```

## Key Signal Pathways

1. **Comparator Path**: x → n7 clips → clipping pattern encodes position

2. **n4 Cascade**: x large → n4 active → boosts n2, n10, n11, n12

3. **Inhibition Path**: n7/n8 active → n9 suppressed → timing signal

4. **Integration**: Multiple neurons feed into n3, n0, n14, n15 for final integration

## Ablation Results

Removing individual neurons from the output:

| Impact | Neurons |
|--------|---------|
| Critical (>50% drop) | n2, n12, n14, n13, n7, n8, n11, n6, n1, n0, n10 |
| Major (20-50% drop) | n15, n3 |
| Moderate (5-20% drop) | n5 |
| Minor (<5% drop) | n4, n9 |

**Surprising finding**: Almost every neuron is critical when removed, suggesting the circuit has minimal redundancy and uses distributed computation throughout.

## Output Contribution

| Neuron | |W_out| | mean(h) | Contribution |
|--------|--------|---------|--------------|
| n7 | 39.1 | 5.11 | 200.1 |
| n8 | 36.4 | 4.60 | 167.3 |
| n6 | 31.2 | 4.05 | 126.6 |
| n1 | 30.8 | 3.94 | 121.4 |
| n2 | 6.5 | 16.37 | 106.4 |
| n15 | 12.3 | 6.71 | 82.4 |
| n12 | 10.9 | 7.28 | 79.0 |

n2 has low output weights but high activation, making it the 5th largest contributor despite appearing unimportant by weight magnitude alone.

## Summary

The circuit uses:
- **5 comparators** (n1, n4, n6, n7, n8) for threshold-based large-value detection
- **1 analog tracker** (n2) for running max and output normalization
- **4 timing neurons** (n5, n9, n10, n12, n15) for position encoding
- **4 integrators** (n0, n3, n13, n14) for combining signals

Most neurons have clear roles, with n13 and n14 being the least understood. The circuit is highly interconnected with minimal redundancy.
