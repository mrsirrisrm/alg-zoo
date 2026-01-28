# 16: The ReLU Frequency Mechanism — Corrected

## Overview

Doc 15 claimed that ReLU shifts the oscillation frequency downward by ~40% (from ω=1.17 rad to ω≈0.7 rad). Upon deeper investigation, this turned out to be **incorrect**. The actual mechanism is more nuanced: ReLU performs **spectral reshaping** through mode-dependent damping, and the k=1 sinusoidal pattern in h_final originates from a **specific sub-network** (n10-n11-n12), not from frequency-shifting the dominant whole-network eigenmode.

## The Original Claim (Incorrect)

```
W_hh eigenvalue: ω = 1.17 rad
       ↓ ReLU shifts frequency
Effective: ω ≈ 0.7 rad
       ↓
W_out decoder: ω ≈ 0.63 rad (matched!)
```

This narrative was appealing but wrong.

## Evidence Against Frequency Shift

### Test 1: Same Starting State, Different Dynamics

Starting both linear and ReLU trajectories from the same h[t=4] state (post-max):

| Neuron | Linear peak ω | Nonlinear peak ω | Ratio |
|--------|---------------|-------------------|-------|
| n1 | 1.047 | 3.142 | 3.0 |
| n6 | 3.142 | 1.047 | 0.33 |
| n7 | 3.142 | 1.047 | 0.33 |
| n8 | 1.047 | 2.094 | 2.0 |

The frequencies are completely different between neurons and don't show a consistent downward shift. The linear system diverges wildly (n7 goes from 12 → 0 → -18 → 4), making DFT comparison of 6-step trajectories unreliable.

### Test 2: Eigenmode Selection at Clipping Event

Hypothesis: ReLU preferentially suppresses high-frequency modes at the clipping moment. Test: decompose pre-activation and post-ReLU states into eigenmodes.

| Frequency band | Mean survival ratio |
|---------------|-------------------|
| Low-freq (ω < 1.0) | 0.950 |
| High-freq (ω ≥ 1.0) | 0.957 |
| Correlation(ω, survival) | r = -0.39 |

Survival rates are nearly identical across frequency bands. **No strong frequency-selective filtering at the clipping event itself.**

### Test 3: Cumulative Mode Evolution (5 Steps)

This test was more revealing — tracking eigenmode amplitudes over 5 steps from h[t=4]:

| Mode | ω | Linear growth | Nonlinear growth | NL/Lin ratio |
|------|---|--------------|-----------------|--------------|
| Sign-flipper | π | 3.36× | 0.36× | 0.11 |
| Complex oscillatory | 1.17 | 1.72× | 1.55× | 0.90 |
| High harmonic | 1.69 | 0.60× | 0.86× | 1.44 |
| Slow mode | 0.56 | 0.25× | 0.62× | 2.47 |
| Medium mode | 0.91 | 0.13× | 1.08× | 8.35 |

ReLU does differentially damp modes, but:
- The sign-flipper (ω=π) is 10× suppressed — the strongest effect
- Medium and slow modes are 2-8× boosted relative to linear
- The spectral centroid shifts only **7.3%** (1.46 → 1.35 rad), not 40%

### Test 4: Spectral Content of Actual Trajectories

The actual comparator trajectories post-max are:
```
n7: [0.003, 6.09, 5.89, 6.11, 5.75, 5.60] — nearly FLAT
n6: [0.03, 3.24, 4.06, 5.10, 5.29, 5.25] — monotone rise
n1: [0.01, 4.55, 3.02, 3.54, 4.33, 5.30] — slight variation
```

These are **not oscillatory**. Their DFT energy is spread across all bins (no dominant frequency). The "oscillation" seen in previous analyses was actually the variation of h_final across different max positions, not temporal oscillation within a single run.

## The Real Mechanism

### Two Encoding Strategies

The model uses two complementary position-encoding mechanisms:

#### 1. Comparator Amplitude Encoding (n1, n6, n7, n8)

Comparators encode position through **how much** they activate at h_final:

| Max Position | n7 h_final | n8 h_final | n6 h_final |
|-------------|-----------|-----------|-----------|
| pos 0 | 5.70 | 5.45 | 4.06 |
| pos 2 | 5.24 | 6.32 | 4.26 |
| pos 4 | 5.41 | 4.26 | 5.10 |
| pos 6 | 6.11 | 4.28 | 5.02 |
| pos 8 | 6.31 | 4.56 | 3.89 |
| pos 9 | 0.00 | 0.07 | 0.01 |

Variation is only ~1-2 units (except pos 9 where max is at the end). The sinusoidal W_out column amplifies these small differences into large output discrimination.

Key property: the comparators create a **+9.96 logit gap** between the 2nd-argmax position and the max position. This is the primary mechanism for max-position suppression.

#### 2. Wave Neuron Phase Encoding (n10, n11, n12)

These neurons carry a **traveling wave** seeded by the n4 burst:

h_final vs steps-since-max (empirical impulse response):

| Steps since max | n10 | n11 | n12 |
|----------------|-----|-----|-----|
| 0 | 3.43 | 3.28 | 6.73 |
| 1 | 5.35 | 5.17 | 8.06 |
| 2 | 5.99 | 5.48 | 7.78 |
| 3 | 6.11 | 4.69 | 5.14 |
| 4 | 4.82 | 2.52 | 3.14 |
| 5 | 2.66 | 0.02 | 5.37 |
| 6 | 0.02 | 1.19 | 6.83 |
| 7 | 1.00 | 2.75 | 8.74 |
| 8 | 2.75 | 4.68 | 10.66 |
| 9 | 5.23 | 5.93 | 10.23 |

DFT of these curves:
- n10: **74% at k=1** (ω=0.63 rad, period 10)
- n11: **64% at k=1**
- n12: **74% at k=1**

This is a genuine sinusoidal pattern with the right frequency for 10-position encoding.

### Source of the k=1 Frequency

The wave neurons' k=1 pattern comes from the **n10-n11-n12 sub-block eigenvalues**:

```
W_hh[[10,11,12], [10,11,12]] eigenvalues:
  λ = -0.248 (sign-flipper, decays quickly)
  λ = 0.529 ± 0.297j, |λ| = 0.606, ω = 0.511 rad, T = 12.3 steps
```

The complex pair at ω = 0.511 rad creates a ~12-step oscillation cycle, close to k=1 (ω = 0.628 rad, period 10). The slight mismatch (T=12.3 vs T=10) is corrected by cross-talk with other neurons and ReLU effects.

The full W_hh also has a mode near this frequency: λ = 0.644 ± 0.400j, ω = 0.556 rad (T = 11.3).

### The Propagation Chain

```
Max input → n4 (W_ih=+10.16, one-shot burst)
  ↓ W_hh[4,4] = -0.99 → n4 clips at next step
  ↓
  ├→ n10 (W_hh[10,4] = +1.76)
  ├→ n11 (W_hh[11,4] = +1.72)
  └→ n12 (W_hh[12,4] = +1.28)

Then feedback sustains the wave:
  n12 → n11: W_hh[11,12] = +0.66
  n11 → n10: W_hh[10,11] = +1.10
  n12 self:  W_hh[12,12] = +0.61
```

## ReLU's Actual Roles

### 1. Stabilization (Primary)

Without ReLU, the system diverges. Three unstable modes (|λ| > 1) cause exponential growth. ReLU clips negative pre-activations, bounding the system. This is the most important function.

### 2. Spectral Reshaping (Secondary)

Mode-dependent damping through cumulative clipping over multiple steps:

| Frequency | Damping mechanism | Effect |
|-----------|------------------|--------|
| ω = π (sign-flipper) | Alternating sign → constant clipping | 10× suppressed |
| ω ≈ 1.2 (fast oscillatory) | Periodic negative phases | Mildly suppressed |
| ω < 1.0 (slow modes) | Rarely negative | Relatively boosted |

Net spectral centroid shift: **7.3%**, not the 40% originally claimed.

### 3. Event Detection (via Diagonal Boundaries)

ReLU creates the clipping events at comparator neurons that mark "current input > running max" (doc 14). This is the trigger for position encoding.

### 4. Impulse Shaping

ReLU shapes the transient decay curve of the wave neurons to approximately match the W_out decoder's sinusoidal profile at k=1.

## Why the 40% Claim Was Wrong

The original analysis compared:
- Raw W_hh dominant oscillatory eigenvalue: ω = 1.17 rad
- h_final sinusoidal pattern: ω ≈ 0.63 rad (k=1)
- Conclusion: "ReLU shifted frequency by 40%"

The error: these are **different quantities from different subsystems**.

- ω = 1.17 is the **whole-network dominant oscillatory eigenmode**
- ω = 0.63 is from the **n10-n11-n12 sub-block** (eigenvalue ω = 0.51)
- The frequency didn't "shift" — it's a **different mode in a different subsystem**
- The slow mode was always present in W_hh (ω = 0.56, |λ| = 0.76) but isn't the dominant mode by eigenvalue magnitude

## Per-Neuron Output Discrimination

Which neurons contribute most to output discrimination for 2nd argmax?

| Neuron | Output variance | Mechanism |
|--------|----------------|-----------|
| n7 | 59.9 | Comparator: small h_final × large sinusoidal W_out |
| n8 | 49.9 | Comparator |
| n1 | 29.6 | Comparator |
| n6 | 26.6 | Comparator |
| n10 | 15.7 | Wave: large sinusoidal h_final × sinusoidal W_out |
| n12 | 9.6 | Wave |
| n14 | 8.4 | Backbone |
| n11 | 8.4 | Wave |
| n15 | 5.8 | Switch |

Comparators dominate (166 total) vs wave neurons (34 total), despite the wave neurons having larger h_final variation. The comparators' advantage comes from their highly sinusoidal W_out columns (85-95% at k=1).

## Neuron Group Contributions

Accuracy of 2nd-argmax predictions using only specific neuron groups:

| Group | Neurons | 2nd-argmax accuracy |
|-------|---------|-------------------|
| Full model | all 16 | 89.3% |
| Comparators | n1, n6, n7, n8 | 11.9% |
| Wave | n10, n11, n12 | 10.4% |
| Backbone | n2, n13, n14 | 12.0% |
| All except comparators | 12 neurons | 16.0% |
| All except wave | 13 neurons | 22.6% |

No single group achieves good accuracy alone — the 89.3% requires cooperation across all groups. Removing comparators hurts more than removing wave neurons, consistent with their role as the primary discriminators.

### Max Position Suppression

| Group | Mean logit at max_pos | Mean logit at 2nd_pos | Gap |
|-------|----------------------|----------------------|-----|
| Comparators | -6.73 | +3.24 | **+9.96** |
| Wave | +8.12 | +3.90 | -4.22 |
| Backbone | +1.96 | +0.80 | -1.15 |
| Switch+other | +3.50 | +3.86 | +0.37 |

Comparators are the only group that actively **suppresses** the max position (negative logit). Other groups actually push toward the max position. The model works because the comparators' +9.96 gap overwhelms the opposing signals.

## Summary

| Aspect | Original Claim (Doc 15) | Corrected Understanding |
|--------|------------------------|------------------------|
| Frequency mechanism | ReLU shifts ω from 1.17 to 0.7 rad | Two different modes in different subsystems |
| Shift magnitude | ~40% | 7.3% spectral centroid (mode-dependent damping) |
| k=1 pattern source | Frequency-shifted dominant eigenmode | n10-n11-n12 sub-block (ω=0.51 rad) |
| Comparator encoding | Oscillation phase | Amplitude variation |
| Wave neuron role | Not distinguished from comparators | Genuine phase encoding via traveling wave |
| ReLU primary role | Frequency shifting | Stabilization + spectral reshaping + event detection |

The model implements a **dual encoding system**: comparators provide amplitude-based max-position suppression through sinusoidal W_out filters, while wave neurons carry a genuine low-frequency oscillation whose phase encodes temporal position. ReLU stabilizes the system, shapes the impulse response, and creates the clipping events that trigger position encoding — but does not shift frequencies.
