# 17: Ring-Up Boot Sequence and the Bell-Ringing Model

## Overview

We investigated the observation that many neurons clip at the very first input timestep regardless of input value. This led to the discovery of a **deterministic boot sequence** at t=0, where n4 fires a massive impulse that "rings up" the network from 6 to 12 active neurons in a single step. More broadly, the model operates as a **bell-ringing system**: every input x[t] strikes a "bell" (primarily via n4), and h_final is approximately a **temporal convolution** of the input sequence with neuron-specific impulse responses. For wave neurons, these impulse responses are sinusoidal at k=1, making h_final a discrete Fourier coefficient of the input.

## The t=0 Boot Sequence

### W_ih Determines Who Activates First

At t=0, there is no recurrent state (h[-1] = 0), so:

```
h[0] = relu(x[0] * W_ih)
```

This is entirely deterministic given x[0]. Since inputs are uniform on [0, 1], x[0] > 0 always, and the sign of W_ih alone determines which neurons activate:

| Active at t=0 (W_ih > 0) | W_ih | Clipped at t=0 (W_ih < 0) | W_ih |
|--------------------------|------|---------------------------|------|
| n4 | **+10.16** | n7 | -13.17 |
| n12 | +0.30 | n8 | -12.31 |
| n11 | +0.15 | n6 | -11.00 |
| n0 | +0.07 | n1 | -10.56 |
| n10 | +0.06 | n14 | -1.32 |
| n2 | +0.01 | n5 | -0.28 |
| | | n9 | -0.23 |
| | | n3 | -0.12 |
| | | n15 | -0.12 |
| | | n13 | -0.05 |

**10 of 16 neurons have negative W_ih** — they always clip at t=0 regardless of input value. This is the universal first-timestep clipping pattern observed in the PNG visualizations.

### N4 Dominates the Initial State

Among the 6 active neurons, n4's W_ih (+10.16) is **34× larger** than the next largest (n12 at +0.30). The initial state is effectively:

```
h[0] ≈ [0, 0, 0, 0, 10.16*x[0], 0, 0, 0, 0, 0, 0.06*x[0], 0.15*x[0], 0.30*x[0], 0, 0, 0]
```

N4 contributes **99.8%** of the total h[0] energy. The first timestep is an **n4 impulse**, with all other neurons providing negligible contributions.

### Ring-Up: 6 → 12 Active Neurons in One Step

At t=1, the recurrent contribution W_hh @ h[0] activates neurons that were clipped at t=0. The transition is dramatic:

| Timestep | Mean active neurons | Key events |
|----------|-------------------|------------|
| t=0 | 6.0 | Only W_ih > 0 neurons active |
| t=1 | **12.2** | n4 burst propagates through W_hh |
| t=2 | 11.6 | Stabilizes near 12 |
| t=3+ | ~12 | Steady state |

The network "boots up" from a 6-neuron initial state to its full operating capacity of ~12 active neurons in a single step, driven almost entirely by n4's impulse propagating through W_hh.

### h[1] Perfectly Encodes x[0]

At t=1, the hidden state is perfectly correlated with x[0] for most neurons (r = 1.000 to 4 decimal places). This is because h[0] ∝ x[0] (deterministic), and h[1] = relu(W_hh @ h[0] + W_ih * x[1]). The W_hh @ h[0] term dominates for neurons receiving strong connections from n4, making h[1] effectively a linear function of x[0].

## N4's Impulse Decay

After its initial burst, n4 rapidly self-destructs:

| Timestep | Mean h[n4] | Mechanism |
|----------|-----------|-----------|
| t=0 | 5.10 | x[0] * W_ih[4] = x[0] * 10.16 |
| t=1 | 1.70 | W_hh[4,4] = **-0.99** → sign flip → clips |
| t=2 | 0.87 | Residual from other neurons |
| t=3 | 0.55 | Continued decay |
| t=4 | 0.42 | |
| t=5 | 0.35 | |
| t=9 | 0.12 | Nearly extinct |

The self-recurrence W_hh[4,4] = -0.99 causes n4 to try to flip sign at each step, but ReLU clips the negative result. N4 survives only through positive contributions from other neurons, decaying progressively. This is the **one-shot burst** mechanism from doc 16 — n4 fires once strongly, then fades to background.

### Ring-Up of Downstream Neurons

While n4 decays, its downstream targets ramp up:

| Neuron | h[0] | h[1] | h[2] | h[3] | Source |
|--------|------|------|------|------|--------|
| n4 | 5.10 | 1.70 | 0.87 | 0.55 | W_ih burst → decay |
| n10 | 0.03 | 9.05 | 13.12 | 13.30 | W_hh[10,4] = +1.76 |
| n11 | 0.08 | 6.91 | 9.60 | 11.68 | W_hh[11,4] = +1.72 |
| n12 | 0.15 | 10.17 | 12.64 | 13.73 | W_hh[12,4] = +1.28 |
| n2 | 0.01 | 8.85 | 11.87 | 13.67 | W_hh[2,4] = +1.73 |

The wave neurons (n10, n11, n12) and the accumulator (n2) all receive n4's energy and ramp up over 2-3 steps. This is the network's boot sequence completing.

## Accuracy Without the First Input

Testing with x[0] = 0 (eliminating the initial impulse, effectively starting from h[0] = 0):

| Condition | 2nd-argmax accuracy |
|-----------|-------------------|
| Normal (x[0] from U[0,1]) | 89.3% |
| x[0] = 0 | **91.2%** |

The model works **better** without the first input. This makes sense: with x[0] = 0, the max and 2nd values can only be at positions 1-9, giving the network more recurrent steps to encode them before readout. The first input is the hardest to encode because the network is still booting up.

## The Bell-Ringing Model

### Every Input Strikes the Bell

The t=0 boot sequence reveals a general principle: **every input acts as an impulse that excites the network's natural modes**. While n4's response is most dramatic at t=0, the same mechanism repeats at every timestep — each x[t] enters through W_ih, creating a perturbation that propagates and decays through the recurrent dynamics.

### Impulse Response Profiles

The correlation r(x[t], h_final[n]) as a function of "steps ago" (9-t) traces out each neuron's effective impulse response. This is the response that a unit impulse at time t leaves on h_final:

**Wave neurons — sinusoidal impulse response:**

| Steps ago | n10 | n11 | n12 |
|-----------|-----|-----|-----|
| 0 (t=9) | 0.00 | 0.00 | 0.00 |
| 1 | 0.11 | 0.12 | 0.02 |
| 2 | 0.14 | 0.08 | 0.08 |
| 3 | 0.17 | 0.09 | 0.02 |
| 4 | 0.07 | -0.13 | -0.19 |
| 5 | -0.12 | -0.53 | -0.44 |
| 6 | -0.58 | -0.28 | -0.09 |
| 7 | -0.29 | -0.10 | -0.02 |
| 8 | -0.06 | 0.05 | 0.24 |
| 9 (t=0) | 0.14 | 0.13 | 0.36 |

DFT of these impulse response curves:

| Neuron | k=1 energy | k=2 energy | Dominant |
|--------|-----------|-----------|----------|
| n10 | **62%** | 25% | k=1 (ω = 0.63 rad) |
| n11 | **66%** | 26% | k=1 |
| n12 | **75%** | — | k=1 |

The wave neurons have sinusoidal impulse responses dominated by k=1 — exactly the frequency that W_out is tuned to decode (doc 15).

**Comparator neurons — last-input dominated:**

| Steps ago | n7 | n8 | n6 | n1 |
|-----------|-----|-----|-----|-----|
| 0 (t=9) | **-0.95** | **-0.94** | **-0.78** | **-0.86** |
| 1 | 0.08 | 0.07 | 0.06 | 0.07 |
| 2 | 0.07 | 0.07 | 0.07 | 0.04 |
| 3-9 | ~0.03-0.06 | ~0.03-0.06 | ~0.04-0.07 | ~0.03-0.06 |

The comparators are dominated by the most recent input (steps_ago = 0, r ≈ -0.9). Their impulse response is essentially a **delta function at t=9** (the last input), with small residual correlations at earlier timesteps. The negative sign reflects W_ih < 0: a large current input drives the comparator toward clipping.

### Temporal Convolution

If the system were linear, h_final would be exactly:

```
h_final[n] = Σ_t  x[t] * IR_n(9 - t)
```

where IR_n is neuron n's impulse response. This is a **temporal convolution** (or more precisely, a correlation) of the input sequence with the neuron's response kernel.

### Linear Convolution R²

How well does the linear convolution model explain actual h_final values?

| Neuron | Role | Linear R² | Interpretation |
|--------|------|-----------|----------------|
| n7 | Comparator | 0.78 | Good — mostly linear in inputs |
| n6 | Comparator | 0.78 | Good |
| n1 | Comparator | 0.75 | Good |
| n8 | Comparator | 0.74 | Good |
| n5 | Switch | 0.54 | Moderate — nonlinear switching |
| n3 | Switch | 0.52 | Moderate |
| n15 | Switch | 0.52 | Moderate |
| n13 | Backbone | 0.50 | Moderate |
| n10 | Wave | 0.40 | Weaker — nonlinear wave dynamics |
| n11 | Wave | 0.30 | Weaker |
| n12 | Wave | 0.27 | Weaker |
| n2 | Accumulator | **-0.68** | Fails — nonlinear accumulation |

The comparators are ~75% linear. The wave neurons are only ~30-40% linear — their sinusoidal impulse response is an emergent property of the nonlinear dynamics, not just a linear filter. N2 completely breaks the linear model because its accumulation behavior (W_hh[2,2] = +0.97 self-recurrence combined with ReLU bounding) is fundamentally nonlinear.

## The Comparator Linear Model

### Unconditional vs Conditional Analysis

For comparator n7, an unconditional linear fit of h_final against all 10 inputs gives:

```
h_final[n7] = β[0]*x[0] + β[1]*x[1] + ... + β[9]*x[9] + intercept
```

**Unconditional β coefficients for n7:**

| Input | β |
|-------|-----|
| x[0] | +2.36 |
| x[1] | +2.33 |
| x[2] | +2.31 |
| x[3] | +2.24 |
| x[4] | +2.23 |
| x[5] | +2.16 |
| x[6] | +2.10 |
| x[7] | +2.00 |
| x[8] | +1.78 |
| x[9] | **-10.83** |

The unconditional model shows two features:
- β[9] ≈ W_ih[7] = -13.17 (direct input weight, discounted by recurrence)
- β[0:8] ≈ +2.2 (roughly uniform positive weight on all prior inputs)

This is a blurred picture. The uniform +2.2 averages over different max positions.

### Conditional on Max Position: R² → 0.99

When we condition on knowing which position holds the maximum value, the picture sharpens dramatically.

**N7 β coefficients conditional on max at position 5:**

| Input | β |
|-------|------|
| x[0] | +0.19 |
| x[1] | +0.14 |
| x[2] | +0.15 |
| x[3] | +0.15 |
| x[4] | +0.16 |
| x[5] | **+12.45** |
| x[6] | +0.03 |
| x[7] | +0.12 |
| x[8] | +0.11 |
| x[9] | **-13.12** |

R² = **0.998**

The conditional model reveals n7's true operation:

```
h_final[n7] ≈ +12.45 * x[max_pos] - 13.12 * x[9] + small_terms
```

This IS the diagonal boundary from doc 14, now expressed as a linear algebraic identity: n7 computes the **difference between the maximum value and the last input**, scaled by ~13.

### All Comparators Show the Same Pattern

Conditional on max at position 5:

| Neuron | β[max_pos] | β[9] | R² |
|--------|-----------|------|-----|
| n7 | +12.45 | -13.12 | 0.998 |
| n6 | +11.95 | -11.02 | 0.997 |
| n1 | +9.51 | -10.50 | 0.992 |
| n8 | +7.67 | -11.37 | 0.928 |

All comparators compute weighted differences between x[max_pos] and x[9]. The weights correspond to the comparator spectrum from doc 14 — n7 has the most balanced ratio (α/|β| = 1.02), while n8 is more biased toward the current input.

### β[max_pos] Decays with Recurrence Distance

The positive coefficient on x[max_pos] must propagate through (9 - max_pos) steps of recurrence to reach h_final. The further back the max is, the more the coefficient decays:

| Max position | Steps to readout | β[max_pos] for n7 | R² |
|-------------|-----------------|-------------------|-----|
| pos 8 | 1 | **+13.28** | 1.000 |
| pos 7 | 2 | +13.13 | 0.999 |
| pos 6 | 3 | +12.85 | 0.999 |
| pos 5 | 4 | +12.45 | 0.998 |
| pos 4 | 5 | +12.28 | 0.997 |
| pos 3 | 6 | +12.22 | 0.997 |
| pos 2 | 7 | +12.18 | 0.996 |
| pos 1 | 8 | +12.12 | 0.996 |
| pos 0 | 9 | +12.21 | 0.995 |
| pos 9 | 0 | -0.01 | **0.030** |

Key observations:
- At max@8 (1 step away), β = +13.28 ≈ |W_ih[7]| = 13.17 — the coefficient is nearly the raw input weight
- At max@0 (9 steps away), β = +12.21 — only 8% decay across 9 steps of recurrence
- **At max@9, the model completely breaks** (R² = 0.03) — when the max is at the last position, it coincides with the -13.12 direct input term, and the two nearly cancel

The remarkable stability of β[max_pos] across positions (12.1 to 13.3) reflects the n2-n4 noise cancellation circuit (doc 13) faithfully carrying the max value through recurrence.

## Wave Neuron β Coefficients

The wave neurons' unconditional β coefficients are themselves sinusoidal:

**DFT of β vectors:**

| Neuron | k=1 energy | k=2 energy | k=3 energy |
|--------|-----------|-----------|-----------|
| n10 | **62%** | 25% | — |
| n11 | **66%** | 26% | — |
| n12 | **75%** | — | 19% |

The β coefficients have the same DFT profile as the impulse response correlations — confirming that wave neurons implement a sinusoidal temporal filter. The linear model h_final[n10] ≈ Σ_t β[t] * x[t] is literally computing a Fourier coefficient of the input sequence.

## Accuracy by Max Position

The model's accuracy varies systematically with where the max value falls:

| Max position | Accuracy | Notes |
|-------------|----------|-------|
| pos 0 | 86.7% | Early max — most recurrence steps, but boot-up interference |
| pos 1 | 90.5% | |
| pos 2 | 90.7% | |
| pos 3 | 91.6% | |
| pos 4 | **92.8%** | Best — middle of sequence |
| pos 5 | **92.8%** | Best |
| pos 6 | 91.4% | |
| pos 7 | 89.6% | |
| pos 8 | 85.9% | Late max — few recurrence steps |
| pos 9 | **80.9%** | Worst — max at last position breaks comparator model |

The profile is **roughly U-shaped** (worst at boundaries, best in the middle):

- **Pos 4-5 (best)**: Enough boot-up time AND enough remaining steps for encoding
- **Pos 0 (degraded)**: Network is still booting up when max arrives — only 6 neurons active
- **Pos 8 (degraded)**: Max is only 1 step before readout — limited time for propagation
- **Pos 9 (worst)**: Max coincides with the last input. The comparator's β[max_pos] cancels with β[9], destroying the diagonal boundary mechanism (R² drops to 0.03)

## Linear Approximation vs Actual Task

The linear convolution model explains substantial variance in h_final but fails at the actual task:

| Metric | Linear convolution | True RNN |
|--------|-------------------|----------|
| Comparator h_final R² | 0.74-0.78 | — |
| 2nd-argmax accuracy | **26.7%** | **89.3%** |

The linear model captures the average behavior of each neuron but cannot reproduce the **conditional switching** between different max positions. The nonlinear clipping is what creates the diagonal decision boundaries (doc 14) — without it, the model cannot distinguish "x[5] is the max" from "x[7] is the max." The linear model blurs all max positions into the unconditional average (β ≈ +2.2 for all positions), losing the sharp +12.5 peak at the actual max position.

## Connection to Prior Findings

| Doc | Finding | Connection |
|-----|---------|------------|
| 13 | N2-N4 noise cancellation | N4 is the impulse generator; N2 accumulates and carries max amplitude with r=0.98 |
| 14 | Diagonal boundaries | The β[max_pos] vs β[9] structure IS the diagonal boundary expressed as linear algebra |
| 15 | DFT-like W_out decoder | Wave neurons' sinusoidal impulse response is matched to W_out's sinusoidal columns |
| 16 | Corrected frequency mechanism | The k=1 impulse response comes from n10-n11-n12 sub-block eigenvalues, not frequency shifting |

## Summary

| Aspect | Description |
|--------|-------------|
| **Boot sequence** | 10/16 neurons clip at t=0; n4 (W_ih=+10.16) fires a dominant impulse |
| **Ring-up** | Network goes 6 → 12 active neurons in one step; h[1] perfectly encodes x[0] |
| **N4 decay** | Self-inhibition (W_hh[4,4]=-0.99) kills n4 in ~3 steps; energy transfers to wave neurons |
| **Bell-ringing** | Each x[t] strikes the network via n4/W_ih; h_final ≈ temporal convolution with impulse response |
| **Wave IR** | n10, n11, n12 have sinusoidal impulse responses (62-75% at k=1) → h_final is a Fourier coefficient |
| **Comparator IR** | Delta-function at t=9 (last input) with sharp conditional peak at max position |
| **Conditional R²** | n7: R² = 0.998 given max position; computes +12.5*x[max_pos] - 13.1*x[9] |
| **β decay** | β[max_pos] ranges from 13.3 (1 step) to 12.2 (9 steps) — stable thanks to n2-n4 circuit |
| **Accuracy profile** | U-shaped: best at pos 4-5 (92.8%), worst at pos 9 (80.9%, comparator model breaks) |
| **Linear limits** | Convolution explains 78% variance but only 27% accuracy — nonlinear clipping is essential |

The model operates as a **struck bell**: each input delivers an impulse through n4 and W_ih, exciting the network's natural modes. Wave neurons ring down with sinusoidal impulse responses that compute Fourier coefficients of the input sequence. Comparators implement conditional diagonal boundaries that sharply separate the max-position input from the current input. The combination requires nonlinear ReLU clipping to achieve the 89% task accuracy that no linear approximation can approach.
