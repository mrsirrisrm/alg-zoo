# 25: Round 1 Summary — What We Got Right and Wrong

## Nomenclature

- **Mt**: position of the max value (magnitude 1.0). E.g. M7 = max at position 7.
- **St**: position of the 2nd-largest value (magnitude 0.8). E.g. S3 = second at position 3.
- **Forward pair**: St > Mt (second comes after max in time).
- **Reversed pair**: St < Mt (second comes before max in time).

## Model

M₁₆,₁₀: single-layer RNN, 16 ReLU hidden neurons, no bias, seq_len=10, 432 params.
`h[t] = relu(h[t-1] @ W_hh.T + x[t] * W_ih)`, `logits = h[9] @ W_out.T`.
Task: output the position of the 2nd-largest input value. 100% accuracy on clean 2-impulse dataset (90 pairs).

## Scorecard

| # | Claim | Verdict | Key Surprise |
|---|-------|---------|-------------|
| 1 | n4 is the most important detector | PARTIAL DISPROOF | Comps' W_ih matters more; n4 is a relay, not the detector |
| 2 | n2 is a standalone magnitude latch | PARTIAL SUPPORT | Critical but context-dependent, not standalone |
| 3 | n2→n4 feedback protects reversed pairs | DISPROVED | Protects **forward** pairs — opposite to prediction |
| 4 | Comps encode position by time-since-clip | DISPROVED | Encoding requires (last_clip, gap, ordering), not just last_clip |
| 5 | Waves protected by small |W_ih| | SUPPORTED | Vulnerability ranks exactly by |W_ih| magnitude |
| 6 | Comp↔wave coupling only through bridges | PARTIAL SUPPORT | Bridges primary, but direct comp→wave edges matter (up to -65.6%) |
| 7 | n9 is largely redundant | DISPROVED | Critical for small-gap reversed pairs (accuracy drops to 22%) |
| 8 | Circuit diagram is complete | PARTIAL DISPROOF | Missing comp→wave direct, n2→wave, n2→bridge, and n9 |

**3 supported, 5 broken.** Each failure pointed to something we misunderstood.

## What We Got Right

### The block structure is real
Comps (n1, n6, n7, n8), Waves (n0, n10, n11, n12, n14), Bridges (n3, n5, n13, n15), and the special neurons (n2, n4, n9) are genuine functional groups. Ablation within vs between groups produces qualitatively different effects. The block-level diagram captures 71% of damaging edges.

### Wave protection via small |W_ih|
The cleanest result. Amplifying wave W_ih breaks accuracy monotonically. Vulnerability ranks exactly by |W_ih| magnitude. Zeroing wave W_ih has almost no effect (98.9%). Waves are purely recurrent cascade carriers — they don't detect input, they carry the echo of past input.

### Bridges are the primary comp↔wave coupling
Bridge ablation (24.4%) is more damaging than direct-connection zeroing (54.4%). n5 alone is devastating (16.7%). Bridges receive n4's broadcast and relay between the two processing channels.

### n2 is critical
Ablation drops accuracy to 11.1%. Its value steers the output. Self-recurrence of 0.97 holds information across timesteps.

### Reset-and-rebuild is the encoding mechanism
Both impulses clip comps to zero. The rebuild trajectory after the last clip encodes position. This core idea survived — it just turned out to be more complex than "single-impulse lookup."

## What We Got Wrong

### n4 is a relay, not "the detector"
We called n4 the detector because of W_ih = +10.16. But comps detect input more critically — zeroing n7's W_ih (-84.4% damage) is worse than zeroing n4's (-61.1%). Comps' large negative W_ih values are the actual clipping mechanism. n4 relays the input signal to neurons that can't see it directly (n2, waves, bridges).

n4 has **threshold** behavior (needs W_ih > ~8), not smooth degradation. This makes sense: it must overcome n2's feedback (-0.49 × ~14 ≈ -7) to fire on the second impulse.

### The feedback loop works backwards from what we thought
We predicted n2→n4 feedback (-0.49) protects the latch against overwriting in the reversed case. The opposite is true: it protects **forward** pairs. Without feedback, n4 fires at full strength on the second impulse (S in forward case), flooding n2 and corrupting the computation. The learned value (-0.49) is precisely tuned at a sweet spot: too weak hurts forward, too strong hurts reversed.

The feedback is **gain control for the second impulse**, not "latch protection."

### Position encoding requires both impulses
We claimed comps encode position via time-since-last-clip — a simple single-impulse trajectory lookup. R² = -0.20 (worse than guessing the mean). Only n7 is partially predicted (R² = 0.86), because its n2/n4 cancellation makes it insensitive to the first impulse.

The actual encoding is deterministic but requires three variables: (last_clip_pos, gap, ordering). Within-group variance is exactly zero when grouping by these three. The first impulse leaves a residue in n2, waves, and bridges that modifies the rebuild trajectory. This is "reset-and-rebuild-with-memory."

### n9 is not redundant
n9 is critical for reversed pairs with small gaps (1-3). Without n9, these pairs drop to 22-29% accuracy. n9 receives from comps (n7, n8) and sends to both waves and comps — it's a secondary relay needed when the two impulses are close together and cascade time is short.

The n7→n9 edge (-5.03, strongest in the matrix) barely matters when cut alone (96.7%). n9's function depends on its full input profile, not just the dominant edge.

### The circuit diagram is incomplete
Missing from the diagram:
1. **Comp→Wave direct** — up to -65.6% when zeroed. Not just bridge-mediated.
2. **n2→Waves** — n2 feeds waves directly (up to -41.1%).
3. **n2→Bridges** — n2 feeds bridges too (up to -25.6%).
4. **n9** — secondary relay for small-gap reversed pairs.

## Updated Understanding

### The revised circuit

```
Input ──→ Comps (clipping)    ──→ Readout
Input ──→ n4 (broadcast relay)
           ├──→ n2 (latch, 0.97 decay)
           │     ├──→ Comps
           │     ├──→ Waves
           │     ├──→ Bridges
           │     └──⊣ n4 (gain control, -0.49)
           ├──→ Waves
           ├──→ Bridges
           └──→ n9 (secondary relay)
                  ├──→ Comps
                  └──→ Waves
Comps ←→ Bridges ←→ Waves
Comps ──→ Waves (direct, weak)
Waves ──→ Readout
n9 ──→ Readout (minor)
```

### Key functional roles (revised)

| Neuron/Group | Old understanding | Revised understanding |
|-------------|-------------------|----------------------|
| n4 | "The detector" | Broadcast relay with threshold behavior |
| n2 | Standalone magnitude latch | Context-dependent memory variable |
| n2→n4 feedback | Protects reversed pairs | Gain control for second impulse; protects forward pairs |
| Comps | Encode position by time-since-clip | Encode position by (last_clip, gap, ordering) |
| Waves | Fully bridge-mediated coupling | Primary bridge-mediated, secondary direct from comps |
| n9 | Redundant | Critical relay for small-gap reversed pairs |
| Comp→Wave | Negligible | Non-negligible direct path (max 1.13) |
| n2→Waves/Bridges | Not in model | n2 feeds waves and bridges directly |

### The two hardest cases

1. **Forward pairs** need the n2→n4 gain control to prevent the second (S) impulse from over-driving the circuit.
2. **Small-gap reversed pairs** (gap 1-3) need n9 to relay comp signal into waves when there isn't enough cascade time for the standard bridge pathway.

### What determines h_final

Comp h_final is perfectly determined by three variables:
- **last_clip_pos**: where the last impulse clipped comps (= max(Mt, St))
- **gap**: |St - Mt|
- **ordering**: forward or reversed

Within-group variance is exactly zero. This means a 3-variable lookup table would achieve R² = 1.0 for comp h_final. The "memory" from the first impulse is carried by n2, waves, and bridges.

## Open Questions

1. **How does W_out decode?** We know comps and waves feed readout, but what's the actual decoding algorithm? Can we predict W_out's behavior from the encoding structure?
2. **What exactly does n9 compute for small-gap reversed pairs?** We know it's critical but not what information it adds.
3. **Why does n7 get near-perfect n2/n4 cancellation?** Is this an accident of training or functionally important?
4. **Does the (last_clip, gap, ordering) encoding generalize?** Our clean dataset has fixed magnitudes. Does the structure hold for variable magnitudes?
5. **Can we predict accuracy from the circuit?** The ARC challenge goal. We're closer but not there.
