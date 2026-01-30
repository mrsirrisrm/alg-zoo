# 22: The n4→n2 Latch Circuit and Negative Feedback

## Overview

Document 20 established that the network uses reset-and-rebuild to encode position. But in the **reversed case** (2AM arrives before 1AM), the 1AM impulse clips all comparators, seemingly destroying the 2AM signal. How does the 2AM memory survive?

The answer is a dedicated subcircuit: **n4 (impulse detector) → n2 (memory latch)**, with a negative feedback loop that automatically attenuates repeated inputs.

## The Reversed Case Problem

When 2AM (0.8) arrives at position 3 and 1AM (1.0) arrives at position 7:

1. t=3: 2AM clips all comparators
2. t=4–6: Comps rebuild, cascade propagates to waves and other neurons
3. **t=7: 1AM clips all comparators to 0** — the 2AM signal in comps is destroyed
4. t=8–9: Comps must rebuild carrying enough information to identify position 3

The question: where is the 2AM signal stored at t=7 when comps are zeroed?

## The n4→n2 Latch

### n4: One-Shot Impulse Detector

| Property | Value | Effect |
|----------|-------|--------|
| W_ih[4] | **+10.16** | Strong positive excitation by input |
| W_hh[4,4] | **-0.986** | Negative self-recurrence → dies in one step |

n4 fires strongly on any input then immediately decays:

```
2AM=0.8 at pos 3:
  t=3: h[4] = 8.13  (fires)
  t=4: h[4] = 0.00  (dead)
```

### n2: Memory Latch

| Property | Value | Effect |
|----------|-------|--------|
| W_ih[2] | **+0.015** | Essentially zero — invisible to input |
| W_hh[2,2] | **+0.969** | Near-perfect self-recurrence (3% decay/step) |
| W_hh[2,4] | **+1.731** | Strong receive from n4 |

n2 receives n4's impulse and holds it:

```
2AM=0.8 at pos 3:
  t=3: n4=8.13, n2=0.01  (n2 gets tiny direct input)
  t=4: n4=0.00, n2=14.09 (n4's value × 1.73 floods into n2)
  t=5: n4=0.00, n2=13.81 (self-recurrence holds ~97%)
  t=6: n4=0.00, n2=14.67
  t=7: n4=3.00, n2=14.00 (1AM arrives — n2 barely notices)
```

n2's value at t=7 (14.00) is essentially what it was at t=6 (14.67), decayed by ~0.97. The 1AM input of 1.0 × W_ih[2] = +0.015 adds nothing to n2 directly.

### n2 Scales Linearly with Input Magnitude

n2's latched value is proportional to the 2AM magnitude:

| 2AM magnitude | n2 at clip time | Ratio |
|---------------|-----------------|-------|
| 0.2 | 3.50 | 1.00 |
| 0.4 | 6.99 | 1.00 |
| 0.6 | 10.49 | 1.00 |
| 0.8 | 13.98 | 1.00 |
| 0.99 | 17.30 | 1.00 |

The latch encodes not just "an input happened" but **how large it was**.

## The Negative Feedback Loop

### The Problem

n4 fires on **every** input (W_ih = +10.16). When 1AM arrives at t=7, n4 should fire again at full strength (10.16), flood n2 with a second dose, and corrupt the 2AM memory.

### What Actually Happens

n4's second firing is **suppressed** by n2 itself:

```
n4 at t=7 (1AM, fresh — no prior 2AM):
  recurrent:  +0.00
  input:     +10.16
  total pre: +10.16
  h[4,7]:     10.16

n4 at t=7 (1AM, with prior 2AM):
  recurrent:  -7.16
  input:     +10.16
  total pre:  +3.00
  h[4,7]:      3.00
```

The -7.16 recurrent term comes almost entirely from n2:

```
n2 × W_hh[4,2] = 14.67 × (-0.492) = -7.22
```

**n2's latched value feeds back negatively into n4**, suppressing n4's response to the next input. This is automatic gain control: the larger the memory n2 holds, the more it dampens n4.

### The Suppressed Second Dose

n4's attenuated firing (3.00 instead of 10.16) still passes a signal to n2:

| Case | n2 self-term | n4→n2 | n2 at t=8 |
|------|-------------|-------|-----------|
| 2AM only | +13.55 | +0.00 | 12.12 |
| 1AM only | +0.01 | +17.58 | 17.61 |
| Both | +13.56 | **+5.19** | 18.67 |

The 1AM adds +5.19 to n2 (via the suppressed n4) instead of the full +17.58 it would add from a fresh firing. n2 grows from ~14 to ~18, but the 2AM memory (13.56 self-term) dominates.

## How n2 Feeds Back into Comparators

After 1AM clips comps to 0 at t=7, n2 drives their rebuild at t=8:

| Connection | Weight | n2 contribution at t=8 |
|-----------|--------|----------------------|
| W_hh[n1, 2] | +0.504 | +9.41 |
| W_hh[n6, 2] | +0.535 | +9.99 |
| W_hh[n7, 2] | +0.674 | +12.58 |
| W_hh[n8, 2] | +0.625 | +11.67 |

But n4 also feeds into comps, and its delta (from 2AM cascade) **cancels** n2 selectively:

| Comp | n2 contribution | n4 contribution | Net |
|------|----------------|----------------|-----|
| n7 | +9.43 | -9.42 (via W_hh[7,4]=+1.317) | **≈ 0** |
| n8 | +8.74 | -5.92 | +2.82 |
| n6 | +7.48 | -3.60 | +3.88 |
| n1 | +7.06 | -7.97 | -0.91 |

n7's near-perfect cancellation means it encodes only rebuild time (i.e. 1AM position), while n8 and n6 retain differential signal from n2 that encodes the 2AM information.

## Wave Protection

Waves survive the 1AM kick for a simpler reason: their W_ih is tiny.

| Neuron | |W_ih| | Recurrent state at t=7 | |input/recurrent| |
|--------|--------|----------------------|-------------------|
| n0 | 0.50 | +11.14 | 0.04 |
| n10 | 0.07 | +10.87 | 0.01 |
| n11 | 0.80 | +10.22 | 0.08 |
| n12 | 0.54 | +10.94 | 0.05 |
| n14 | 1.32 | +10.93 | 0.12 |

The 1AM input contributes <1% to 12% of the wave pre-activation. The recurrent cascade state from 2AM overwhelms the direct input effect.

## The Complete Reversed-Case Circuit

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                   REVERSED CASE: 2AM BEFORE 1AM                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. 2AM ARRIVES (t = S)                                                      │
│     n4 fires (8.13) → n2 latches (14.09) → cascade to waves/comps           │
│                                                                              │
│  2. BETWEEN IMPULSES (t = S+1 .. M-1)                                        │
│     n2 holds memory (0.97 decay/step)                                        │
│     Waves carry cascade state                                                │
│     n4 is dead (negative self-recurrence)                                    │
│                                                                              │
│  3. 1AM ARRIVES (t = M)                                                      │
│     Comps clip to 0 (W_ih ≈ -10 to -13)                                     │
│     n4 fires SUPPRESSED (3.00 not 10.16) — n2 inhibits via W_hh[4,2]=-0.49  │
│     Waves unaffected (|W_ih| < 1.3, |input/recur| ≈ 0.01)                   │
│     n2 unaffected (W_ih = +0.015)                                            │
│                                                                              │
│  4. REBUILD (t = M+1 .. 9)                                                   │
│     n2 feeds into comps: W_hh[comp,2] ≈ 0.5–0.67                            │
│     n4 cancels n2 in n7 (keeping n7 as pure 1AM encoder)                     │
│     n8, n6 retain differential n2 signal (2AM information)                   │
│     Waves carry independent 2AM cascade state                                │
│                                                                              │
│  5. READOUT                                                                  │
│     Same two-channel mechanism as forward case:                              │
│     Comp channel → 1AM position, Wave channel → 2AM position                │
│     100% accuracy across all reversed pairs (gaps 2–5)                       │
│                                                                              │
│  KEY: n2→n4 negative feedback (W_hh[4,2] = -0.49) is automatic gain         │
│  control. The latch protects itself by suppressing its upstream driver.      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Summary

| Component | Role | Key Weight |
|-----------|------|-----------|
| n4 | One-shot impulse detector | W_ih = +10.16, W_hh[4,4] = -0.99 |
| n2 | Memory latch (holds magnitude) | W_hh[2,2] = +0.97, W_hh[2,4] = +1.73 |
| n2→n4 | Negative feedback (gain control) | W_hh[4,2] = -0.49 |
| n2→comps | Memory readout into comparators | W_hh[comp,2] = 0.50–0.67 |
| n4→n7 | Selective cancellation | W_hh[7,4] = +1.32 (cancels n2 in n7) |
| Waves | Protected memory (tiny W_ih) | |W_ih| < 1.3 vs recurrent ~11 |

The n4→n2 latch is a learned analogue memory cell: n4 detects inputs, n2 stores the result, and n2→n4 feedback prevents overwriting by subsequent inputs. This allows the network to remember the 2AM signal even after the 1AM impulse destroys the comparator state.

## Scripts

See `src/rebuild_deep_dive.py` for the experimental code.
