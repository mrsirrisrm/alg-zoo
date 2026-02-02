# 34: Forward vs Reverse — How the First Impulse Claims the Phase

## The Puzzle

The M₁₆,₁₀ network finds the position of the **second-largest** value (S) in a sequence. It works perfectly whether M (max) or S arrives first:

- **Forward** (M before S): M@2, S@7 → outputs 7 ✓
- **Reverse** (S before M): S@2, M@7 → outputs 2 ✓

The reverse case is puzzling: M arrives *later* and is *larger* (1.0 vs 0.8). Why doesn't M "overwrite" S's position?

## The Answer in One Line

> **The first impulse claims the spiral's phase; ReLU makes this claim sticky.**

## The Mechanism

### Forward: The Natural Case

In forward, the mechanism is straightforward:

1. M arrives first, kicks off a spiral
2. S arrives second, redirects the countdown
3. The countdown lands on S's position at t=9

The second impulse (S) is the one we want to track, and it naturally "wins" because it arrives last.

### Reverse: The Tricky Case

In reverse, we need S's position even though M arrives last:

```
t=0,1,2: zeros
t=2:     S impulse (0.8) → kicks off spiral, encoding position 2
t=3-6:   spiral evolves, phase "counts down"
t=7:     M impulse (1.0) → arrives into an already-shaped state
t=8,9:   countdown continues → lands on position 2
```

The key is **what happens at t=7** when M arrives.

### The S-Spiral's Protection

When M arrives at t=7, the hidden state isn't empty — it contains S's spiral:

```
pre_ReLU = W_ih × M + W_hh @ h_S_spiral
                ↑           ↑
         M's input    S's accumulated
         (large)      phase information
```

The S-spiral contribution (`W_hh @ h`) has norm ~60, comparable to M's input. This **shifts the ReLU boundaries**:

| Without S-spiral | With S-spiral |
|------------------|---------------|
| M activates neurons 0,2,4,10,11,12 | Different pattern emerges |
| Phase points toward M's position | Phase preserves S's position |

### Neuron-by-Neuron

At the moment M arrives (t=7), comparing "M into S-spiral" vs "M alone":

| Neuron | S-spiral effect | Result |
|--------|-----------------|--------|
| n0 | +16.8 boost | Stays active, carries S-info |
| n4 | -6.3 suppression | M's contribution reduced |
| n14 | +15.2 boost | Additional S-encoding |
| n6,7,8 | Pushed below 0 | ReLU clips M's influence |

The S-spiral doesn't just add linearly — it **changes which neurons survive ReLU**, fundamentally altering the post-impulse state.

## The Offset Perspective

Document 33 described the offset as:

```
h_fwd = h_rev + offset
```

This is true, but the deeper insight is:

```
offset = (M into M-spiral, then S) - (S into S-spiral, then M)
       = [how S interacts with M's prior] - [how M interacts with S's prior]
```

The offset encodes the **asymmetry of interaction**: whichever impulse arrives first shapes the hidden state that the second impulse must interact with.

### The Separable Structure

The offset decomposes as `f(m_pos) + g(s_pos)` because:

- `f(m_pos)` encodes "there was a prior impulse at m_pos that shaped the state"
- `g(s_pos)` encodes "the second impulse at s_pos interacted with that shape"

The antisymmetry (`W_out @ f = -W_out @ g`) ensures that the network always outputs S's position regardless of arrival order.

## Why ReLU is Essential

Without ReLU, the dynamics would be linear:

```
h_after_M = ReLU(W_ih × M + W_hh @ h_S)

Linear case:  = W_ih × M + W_hh @ h_S  (just superposition)
```

In the linear case, the larger impulse (M) would dominate. ReLU creates **nonlinear gating**:

1. S-spiral shifts some neurons above/below zero
2. These gating decisions are "sticky" — M can't un-clip them
3. Information about S's position survives in the gating pattern

## The Pithy Summary

```
╔══════════════════════════════════════════════════════════════╗
║                    FORWARD vs REVERSE                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                               ║
║  FORWARD: Second impulse redirects countdown → easy           ║
║                                                               ║
║  REVERSE: First impulse pre-shapes ReLU boundaries            ║
║           Second impulse can't fully reset                    ║
║           First position survives in the gating pattern       ║
║                                                               ║
║  KEY INSIGHT: "First to fire, first to wire"                  ║
║               The first impulse claims the phase.             ║
║               ReLU makes this claim sticky.                   ║
║                                                               ║
╚══════════════════════════════════════════════════════════════╝
```

## Alternative Framings

1. **Phase Wheel**: Both directions use the same countdown mechanism, but the countdown is anchored to *S's arrival time*, not the second impulse's arrival time. The first impulse establishes which position gets tracked.

2. **Memory via Gating**: The network doesn't explicitly "remember" S's position. Instead, S's position is encoded in *which neurons are above/below zero* when M arrives. This gating pattern persists through the computation.

3. **Asymmetric Superposition**: The response to M depends on whether S came before. `response(M | S first) ≠ response(M | M first)` because ReLU breaks linearity.

## Implications

1. **First-mover advantage**: In ReLU networks, early inputs have disproportionate influence because they establish the gating pattern.

2. **Order sensitivity**: The same two values at the same positions produce different trajectories depending on arrival order — but both trajectories lead to the correct answer through different paths.

3. **Distributed encoding**: S's position isn't stored in any single neuron. It's encoded in the collective pattern of which neurons are active/inactive.

## Related Documents

- [32: Phase Wheel Mechanism](32_phase_wheel_mechanism.md) — The countdown dynamics
- [33: Offset Discrimination Mechanism](33_offset_discrimination_mechanism.md) — The offset structure
- Visualizations: `offset_birth_analysis_v2.png`, `reverse_mechanism_deep_dive.png`
