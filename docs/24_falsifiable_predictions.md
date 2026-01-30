# 24: Falsifiable Predictions — Testing Our Mechanistic Understanding

## Nomenclature

- **Mt**: position of the max value (magnitude 1.0). E.g. M7 = max at position 7.
- **St**: position of the 2nd-largest value (magnitude 0.8). E.g. S3 = second at position 3.
- **Forward pair**: St > Mt (second comes after max in time).
- **Reversed pair**: St < Mt (second comes before max in time).

## Context

We claim to understand how M₁₆,₁₀ (16 hidden ReLU neurons, 432 params, seq_len=10) finds the 2nd-argmax. This document lists specific mechanistic claims, each paired with a prediction that would **disprove** the claim if it fails.

Motivated by the ARC challenge: "Design a method for mechanistically estimating the accuracy of the 432-parameter model M₁₆,₁₀ that matches the performance of random sampling in terms of mean squared error versus compute."

We aren't there yet, but we can test whether our understanding holds up under targeted interventions. Each failed prediction tells us where to look next.

## Claims and Predictions

### Claim 1: n4 is a one-shot impulse detector that broadcasts to the entire network

n4 has W_ih = +10.16 (largest positive), self-recurrence = -0.99 (dies immediately). It fires on every input and fans out to n2 (+1.73), waves (+6.21 total), comps (+2.10), and bridges (+2.17).

**Prediction**: Zeroing W_ih[4] should destroy accuracy more than zeroing any other single W_ih entry. Scaling W_ih[4] down smoothly should degrade accuracy monotonically.

**Disproof**: Another W_ih entry matters more, or scaling W_ih[4] has non-monotonic effect.

**Result**: PARTIALLY DISPROVED.

Zeroing W_ih[n] accuracy (sorted by damage):
- n7(C): 15.6% (-84.4%) — **most damaging**, not n4
- n4(O): 38.9% (-61.1%) — second
- n1(C): 42.2% (-57.8%)
- n6(C): 55.6% (-44.4%)
- n8(C): 67.8% (-32.2%)
- All waves/others with |W_ih| < 1.3: 100% (no effect)

Scaling W_ih[4]: NOT monotonic. Sharp threshold at 0.8× (98.9%) vs 0.7× (55.6%). Overscaling also hurts: 1.5× → 61.1%, 2.0× → 55.6%.

**Verdict**: n4 is important but not the most important W_ih. The comps' large negative W_ih values are more critical — without them, the clipping mechanism (the foundation of reset-and-rebuild) breaks entirely. n4 is better described as "the broadcast amplifier" rather than "the detector" — the comps are the real detectors. n4's role is to relay the input signal to neurons that can't detect input directly (waves, n2, bridges).

The scaling result shows n4 has a **threshold** (needs W_ih > ~8 to function) rather than smooth degradation. This makes sense: n4 needs to overcome the negative feedback from n2 (-0.49 × ~14 ≈ -7) to fire on the second impulse.

---

### Claim 2: n2 is a memory latch whose value encodes S magnitude

n2 has W_ih ≈ 0, self-recurrence = 0.97, receives from n4 via W_hh[2,4] = +1.73. Its latched value scales linearly with S input magnitude.

**Prediction**: Clamping n2 to a fixed value at all timesteps should make the network output a fixed position regardless of input. The output position should shift predictably as we sweep the clamp value.

**Disproof**: Clamping n2 doesn't produce stable fixed outputs, or the output doesn't shift with clamp value.

**Result**: PARTIALLY SUPPORTED — n2 is critical but not a clean single-quantity latch.

Clamping n2 to a fixed value:
- Best accuracy at any clamp: 56.7% (clamp=16). No clamp restores full accuracy.
- Output DOES shift with clamp: low clamp → predicts high positions (p9), high clamp → predicts middle (p5). But it's a broad distribution, not a clean position.
- At clamp=25, 59/90 samples predict p5 — n2 can dominate the output.

Forward vs reversed under clamping:
- Optimal clamp differs: forward peaks at 66.7% (clamp=18), reversed peaks at 53.3% (clamp=10).
- At clamp=18: forward 66.7%, reversed **0%**. The two orderings want different n2 values.

Specific pair trace (M7, S3, target=3): sweeping clamp 0→25 gives predictions 9→9→2→2→3→7→7→7→7→7. The correct answer appears at clamp=12 but is not stable.

Natural n2 values at h_final range 9.1 to 19.1 (mean ~16). In the clean dataset all S magnitudes are 0.8, so the variation comes from timing/position, not magnitude.

**Verdict**: n2 is functionally critical (ablation → 11.1%), and its value does steer the output. But it doesn't act as a simple independent latch — its effect depends on the state of other neurons. n2 is better described as a key variable in a multi-neuron computation, not a standalone magnitude register.

---

### Claim 3: n2→n4 negative feedback prevents overwriting

W_hh[4,2] = -0.49. When n2 holds a large value, it suppresses n4's response to subsequent inputs (10.16 → 3.00 in the reversed case).

**Prediction**: Setting W_hh[4,2] = 0 should hurt accuracy specifically on **reversed pairs** (S before M), where latch protection matters. Forward pairs (M before S) should be less affected.

**Disproof**: Damage is equal across forward and reversed pairs.

**Result**: DISPROVED (opposite direction).

Zeroing W_hh[4,2]:
- Baseline: fwd=100%, rev=100%, all=100%
- W_hh[4,2]=0: fwd=**77.8%**, rev=**100%**, all=88.9%

The damage falls entirely on **forward** pairs, not reversed. Reversed pairs are unaffected.

Strengthening feedback (more negative):
- W_hh[4,2]=-0.25: fwd=93.3%, rev=100% (partial restoration)
- W_hh[4,2]=-0.49: fwd=100%, rev=100% (learned value — optimal)
- W_hh[4,2]=-1.00: fwd=100%, rev=91.1% (too strong — hurts reversed)
- W_hh[4,2]=-2.00: fwd=100%, rev=91.1% (same)
- W_hh[4,2]=-3.00: fwd=100%, rev=91.1% (same)

Gap breakdown without feedback (W_hh[4,2]=0):
- Forward pairs break at gaps 2-7 and 9 (67-100%), worst at gap 9 (0%)
- Reversed pairs: 100% at every gap

Trace of (M7, S3) — reversed pair:
- Original: n4 fires at 3.00 at t=7, n2 reaches 18.67 at t=8, n7=13.37
- Zeroed: n4 fires at 10.23 at t=7, n2 reaches 31.17 at t=8, n7=22.88
- Without feedback n4/n2/n7 are much larger, but reversed accuracy is still 100%

**Verdict**: The feedback does NOT protect the latch in the reversed case — reversed pairs work fine without it. Instead, the feedback protects **forward pairs**, where M arrives first. In forward pairs, the S impulse (0.8) arrives second. Without feedback, n4 fires at full strength on S, flooding n2 with excessive signal and corrupting the computation. The learned value (-0.49) is precisely tuned: weak enough to let reversed pairs work, strong enough to keep forward pairs from over-responding.

The feedback is better described as **gain control for the second impulse** (whichever arrives second), and its critical role is in the forward case where the smaller S magnitude must not over-drive the circuit.

---

### Claim 4: Comparators encode position via time-since-last-clip

After clipping, each comp follows a deterministic oscillatory rebuild trajectory. h_final = rebuild_trajectory[9 - last_clip_time].

**Prediction**: For any (Mt, St) pair, we can predict each comp's h_final from the rebuild trajectory table alone (single-impulse lookup by steps remaining). Prediction should match actual h_final with R² > 0.9.

**Disproof**: R² < 0.9, indicating encoding is more complex than reset-and-rebuild.

**Result**: DISPROVED. Single-impulse lookup fails; encoding requires both positions.

Single-impulse rebuild trajectories (impulse at each position, h_final for comps):
```
pos       n1       n6       n7       n8   steps
   0     3.60     6.42    11.89    13.31       9
   1     7.05     5.73    10.19    16.33       8
   2    11.23     6.78     8.83    15.95       7
   ...
   8     8.87     0.00    13.41     0.00       1
   9     0.00     0.00     0.00     0.00       0
```

Predicting h_final from single-impulse trajectory at last_clip_pos:
- n1: R² = **-0.20** (worse than mean)
- n6: R² = **0.01** (no signal)
- n7: R² = **0.86** (decent but < 0.9 threshold)
- n8: R² = **-1.73** (terrible)
- Overall: R² = **-0.20**

Only n7 is remotely well-predicted by single-impulse lookup, which makes sense — Claim 3 showed n7 has near-perfect n2/n4 cancellation, making it insensitive to the first impulse.

Error is systematically worse for forward pairs (MAE: n1 3.9/2.1, n6 3.2/2.1, n7 1.7/0.8, n8 7.6/5.0 fwd/rev).

**However**: grouping by (last_clip_pos, gap, ordering), within-group variance is **exactly zero** for all comps. h_final is perfectly determined by three variables: where the last clip happened, how far apart the two impulses were, and which came first.

**Verdict**: The encoding is NOT simple reset-and-rebuild from a single impulse. Both impulse positions matter — the first impulse leaves a residue (via n2, waves, bridges) that modifies the rebuild trajectory after the second clip. But the encoding IS deterministic and fully specified by (last_clip, gap, ordering). A lookup table indexed by these three variables would achieve R² = 1.0. The mechanism is "reset-and-rebuild-with-memory" rather than pure "reset-and-rebuild."

---

### Claim 5: Waves are protected from input because |W_ih| < 1.3

Wave neurons have |W_ih| < 1.3 while their recurrent state at impulse time is ~11, giving |input/recur| ≈ 0.01. The input kick barely dents them.

**Prediction**: Artificially amplifying wave W_ih values (multiply by 10) should break accuracy by making waves sensitive to input kicks. Damage should scale with amplification factor.

**Disproof**: Amplifying wave W_ih has no effect — something else protects waves.

**Result**: SUPPORTED. Amplification breaks accuracy, confirming small W_ih is the protection mechanism.

Wave input sensitivity (impulse at t=3, state at t=6):
```
neuron     W_ih   |W_ih|   h_before  |input/h|
  n0     0.069    0.069       4.76     0.015
 n10     0.060    0.060      21.91     0.003
 n11     0.148    0.148      11.03     0.013
 n12     0.298    0.298       2.68     0.111
 n14    -1.321    1.321       4.61     0.286
```

Amplification sweep (all waves):
- 1×: 100%, 2×: 100%, 3×: 98.9%, 5×: 97.8%
- 10×: 83.3%, 20×: 64.4%, 50×: 54.4%

Damage scales monotonically with amplification — consistent with prediction.

Individual wave 10× amplification:
- n0, n10, n11: 100% (no effect — |W_ih| < 0.15)
- n12: 94.4% (slight — |W_ih| = 0.30)
- n14: **80.0%** (significant — |W_ih| = 1.32, the largest wave)

Zeroing all wave W_ih: 98.9% (almost no effect).
Flipping all signs: 94.4% (minor effect).

**Verdict**: The claim is correct. Wave protection comes from small |W_ih|. The protection is weakest for n14 (|W_ih|=1.32, ratio=0.29) and strongest for n10 (|W_ih|=0.06, ratio=0.003). The ordering of vulnerability exactly matches |W_ih| magnitude. Zeroing wave W_ih barely matters (98.9%), confirming they contribute almost nothing via direct input — their role is purely recurrent cascade carriers.

---

### Claim 6: Comps and waves couple through bridge neurons, not directly

Direct comp↔wave W_hh connections are weak (mean |W_hh| = 0.04). Coupling goes through bridges (n3, n5, n13, n15).

**Prediction**: Zeroing all direct comp↔wave W_hh entries should have minimal accuracy impact. Zeroing bridge neurons should be far more damaging.

**Disproof**: Direct comp↔wave ablation hurts significantly, or bridge ablation doesn't hurt.

**Result**: PARTIALLY SUPPORTED — bridges are primary but direct connections also matter.

Direct comp↔wave mean |W_hh| = 0.22 (not 0.04 as originally stated). The asymmetry is large:
- **Comp←Wave**: very weak (max 0.11, mean 0.04) — waves barely feed into comps directly
- **Wave←Comp**: substantial (max 1.13, mean 0.38) — comps DO feed into waves directly

Zeroing direct comp↔wave connections: **54.4%** (significant damage).
Ablating all bridges: **24.4%** (more damaging, as predicted).
Both: **18.9%** (worse than either alone — both pathways carry information).

Individual bridge ablation:
- n5: **16.7%** (devastating — most critical single bridge)
- n13: **41.1%**
- n3: **70.0%**
- n15: **70.0%**

Bridge connections to n4: n4→n5 = +1.19 (strong), n4→n13 = +0.46, n4→n3 = +0.34. Bridges receive n4's broadcast signal directly.

**Verdict**: Bridges ARE the primary coupling path (ablation 24.4% vs direct zeroing 54.4%), confirming the claim's core idea. But direct wave←comp connections (mean 0.38, max 1.13) are non-negligible — comps push signal into waves directly, not just through bridges. The original claim's "mean |W_hh| = 0.04" was only correct for the comp←wave direction. The circuit diagram should show a weak direct comp→wave arrow alongside the bridge pathway.

---

### Claim 7: n9 is largely redundant

Ablation showed n9 removal only drops accuracy to 77.8%. Despite n7→n9 = -5.03 (strongest edge in the matrix), other pathways compensate.

**Prediction**: Zeroing n9 or the n7→n9 edge should drop accuracy by <25%. No specific input subset should be catastrophically affected.

**Disproof**: A specific subset (e.g., certain position ranges) drops below 50% without n9.

**Result**: DISPROVED. Small-gap reversed pairs are catastrophically affected.

Overall:
- n9 ablated: fwd=95.6%, rev=**60.0%**, all=77.8%
- n7→n9=0 only: fwd=93.3%, rev=100%, all=96.7% (strongest single edge barely matters alone)

By gap (n9 ablated):
- Gap 1 rev: **22%**, gap 2 rev: **25%**, gap 3 rev: **29%** (all below 50%)
- Gap ≥4: 100% (no effect at all)

By position: worst at M5, M6 (55.6%), worst at S0-S4 (66.7%).

n9's W_out norm = 3.99 (12th of 16 — not a major readout neuron).
n9 receives mainly from n7 (-5.03) and n8 (-1.09).
n9 sends to waves (n0 -1.42, n12 +0.81, n11 +0.68, n14 -0.65) and comps (n1 -0.69, n6 -0.64).

**Verdict**: n9 is NOT redundant — it's critical for reversed pairs with small gaps (1-3). These are the hardest cases: S arrives just 1-3 steps before M, leaving minimal cascade time. n9 appears to act as a secondary relay that converts comp signal (mainly from n7) into wave/comp modulation needed to distinguish these close-together pairs. Cutting just the n7→n9 edge barely matters (96.7%), suggesting n9's role depends on its full input profile (n7 + n8 + self), not just the single strongest edge.

---

### Claim 8: The circuit diagram is complete

Our simplified circuit: Input → n4 (detector) → broadcasts to [n2 (latch), Comps, Waves, Bridges]. n2 feeds back negatively to n4 and positively to Comps. Bridges couple Comps ↔ Waves. Both Comps and Waves feed W_out.

**Prediction**: We can predict the *direction* of accuracy change under any single-weight perturbation by tracing signal flow through the diagram. Perturbing within a block has effects consistent with that block's role.

**Disproof**: A single-weight perturbation produces an accuracy change opposite to what the diagram predicts.

**Result**: PARTIALLY DISPROVED. Diagram captures most critical edges but has significant gaps.

Systematic zeroing of every W_hh entry (skipping |val| < 0.01):

Top 20 most damaging: 18 of 20 are in the diagram. The two missing are both Comp→Wave direct: n6→n0 (+1.13, -65.6%) and n6→n14 (+0.77, -52.2%).

Among all edges with |Δ| > 5%:
- In diagram: **86 edges**, mean |Δ| = 31.3%
- NOT in diagram: **34 edges**, mean |Δ| = 23.3%

Missing edges that matter (top by damage):
```
n6→n0   (C→W)  +1.13  -65.6%
n6→n14  (C→W)  +0.77  -52.2%
n1→n0   (C→W)  +0.71  -43.3%
n6→n11  (C→W)  -0.47  -42.2%
n7→n0   (C→W)  -0.78  -42.2%
n2→n14 (n2→W)  +0.34  -41.1%
n8→n0   (C→W)  -0.79  -40.0%
n1→n12  (C→W)  +0.47  -38.9%
n2→n15 (n2→B)  +0.38  -25.6%
n0→n1   (W→C)  -0.11  -24.4%
```

Three categories of missing edges:
1. **Comp→Wave direct** (C→W): most damaging, up to -65.6%. Consistent with Claim 6 finding.
2. **n2→Wave** (n2→W): up to -41.1%. n2 feeds waves directly, not just through comps.
3. **n2→Bridge** (n2→B): up to -25.6%. n2 feeds bridges too.

n9 individual edges: n9→n1 and n9→n6 each -14.4%, n4→n9 -10.0%. Individually moderate, but collectively n9 ablation causes -22.2%.

**Verdict**: The diagram captures the most important edges well (18/20 top entries are in-diagram, in-diagram edges account for 71% of damaging edges). But it's incomplete in three ways that need to be added:
1. **Comp→Wave direct arrow** — n6→n0 alone causes -65.6% when zeroed. This is NOT negligible.
2. **n2→Waves and n2→Bridges arrows** — n2 doesn't just feed comps; it feeds waves and bridges too.
3. **n9** — should appear as a secondary relay (receives from comps, sends to comps+waves), important for small-gap reversed pairs.

---

## Method

For each claim, run the specific intervention on the clean 90-pair dataset (x[Mt]=1.0, x[St]=0.8, all else 0). Compare predicted vs actual outcomes. Update results inline.

## Scripts

See `src/rebuild_deep_dive.py` for experimental code.
