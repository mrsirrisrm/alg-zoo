# 26: Falsifiable Predictions Round 2

## Nomenclature

- **Mt**: position of the max value (magnitude 1.0). E.g. M7 = max at position 7.
- **St**: position of the 2nd-largest value (magnitude 0.8). E.g. S3 = second at position 3.
- **Forward pair**: St > Mt (second comes after max in time).
- **Reversed pair**: St < Mt (second comes before max in time).

## Context

Round 1 tested 8 claims. 3 held, 5 broke. The failures corrected our understanding: n4 is a relay not a detector, the feedback protects forward pairs not reversed, encoding requires (last_clip, gap, ordering), n9 is critical for small-gap reversed, and the diagram was missing comp→wave direct and n2→wave/bridge paths.

Round 2 tests the revised understanding.

## Claims and Predictions

### Claim 9: n7 is a pure M-position encoder

n7 has near-perfect n2/n4 cancellation (n2 +9.43, n4 -9.42 ≈ 0). This should make n7 insensitive to the first impulse, encoding only the M (last-clip) position. Round 1 showed n7 had R² = 0.86 from single-impulse lookup — the best of any comp.

**Prediction**: n7's h_final should depend only on Mt position (steps from M to end), not on St position or gap. Grouping by Mt, within-group variance of n7's h_final should be near zero.

**Disproof**: n7's h_final varies significantly with St for a given Mt.

**Result**: DISPROVED. n7 is far from a pure M-position encoder.

n7 grouped by Mt: within-group ranges are 10.5–12.7 (massive), std 3.3–3.8. R²(Mt only) = **0.51** — Mt explains barely half the variance.

All comps have similar R²(Mt): n1=0.61, n6=0.55, n7=0.51, n8=0.60. n7 is actually the WORST Mt-predictor, not the best. The n2/n4 cancellation doesn't make n7 a clean M-encoder; it removes one source of variation while others (gap, ordering, wave state) remain dominant.

Grouped by St, n7's range is similarly large (13.4–13.7 for most positions).

**Verdict**: The n2/n4 cancellation in n7 is real but does not produce a clean single-variable encoding. n7 encodes the full (last_clip, gap, ordering) tuple, like all other comps. The R²=0.86 from Round 1 was misleading because it used single-impulse trajectories (which fold in gap=0) rather than grouping by Mt across actual two-impulse pairs.

---

### Claim 10: n2's value at h_final encodes the first impulse's timing

n2 latches on the first impulse (via n4), then decays at 0.97/step. Its h_final value should depend on when the first impulse arrived and how many steps remain. In the clean dataset (fixed magnitudes), n2's h_final should be a monotonic function of first-impulse position.

**Prediction**: n2's h_final should correlate strongly with first-impulse position (R² > 0.8). For forward pairs, first impulse is M; for reversed, first impulse is S.

**Disproof**: R² < 0.8, or the relationship is non-monotonic.

**Result**: DISPROVED. R² = 0.12, non-monotonic.

n2 h_final grouped by first-impulse position:
```
first   mean    std
  0    16.04  1.798
  1    15.50  2.366
  2    15.59  2.810
  3    16.22  2.638
  4    17.22  2.047
  5    17.88  1.536
  6    17.76  1.381
  7    16.77  1.719
  8    15.86  1.759
```

Not monotonic — rises from pos 0→5, then falls again. R²(first_pos) = 0.12. Within-group std is large (1.4–2.8).

By ordering: forward correlation = 0.53 (moderate), reversed correlation = 0.04 (essentially zero).

**Verdict**: n2's h_final does NOT simply encode first-impulse timing. The non-monotonicity makes sense: n2 receives a second dose from n4 at the second impulse (attenuated by feedback). For late first impulses (pos 5-8), fewer decay steps mean higher n2 at second-impulse time, which means stronger feedback, which means smaller second dose. The net effect is non-monotonic. n2's final value is a complex function of both impulse positions, not a simple timer.

---

### Claim 11: The (last_clip, gap, ordering) lookup table predicts accuracy

Round 1 found comp h_final has zero within-group variance when grouped by (last_clip, gap, ordering). If the readout is a linear function of h_final, then accuracy should be predictable from the lookup table: any (last_clip, gap, ordering) cell that maps to the correct logit should be correct, and we should be able to identify which cells fail.

**Prediction**: Build a lookup table mapping (last_clip, gap, ordering) → predicted output. The table should reproduce the model's 100% accuracy on the clean dataset. Every cell should predict correctly.

**Disproof**: Some cells predict incorrectly despite the model being 100% accurate.

**Result**: SUPPORTED. Perfect — 90/90 cells correct, 0 inconsistent.

Each (last_clip, gap, ordering) cell contains exactly 1 pair, predicts exactly 1 output, and all are correct. The table is also trivially decodable:
- Forward: target = last_clip (St is the later position)
- Reversed: target = last_clip - gap (St = Mt - gap)

This means the model is effectively computing: identify which impulse came last (= last_clip), measure the gap, determine ordering, then do `St = last_clip` (fwd) or `St = last_clip - gap` (rev).

**Verdict**: The (last_clip, gap, ordering) → target mapping is a complete, correct, and trivial computation. The hard part is not the final arithmetic — it's how 16 ReLU neurons encode these three variables in h_final such that a single linear readout can extract St.

---

### Claim 12: n9 compensates for insufficient cascade time in small-gap reversed pairs

n9 is critical for reversed pairs with gaps 1-3. In these cases, S arrives 1-3 steps before M, giving waves and bridges minimal time to propagate the S cascade. n9 provides an alternative fast path: comp → n9 → wave/comp.

**Prediction**: For small-gap reversed pairs, n9's h_final should be significantly different from zero (active), while for large-gap pairs n9 should be near-zero (not needed). The magnitude of n9 should anti-correlate with gap size in reversed pairs.

**Disproof**: n9 is equally active across all gaps, or n9 is inactive for small-gap reversed pairs.

**Result**: DISPROVED (opposite direction).

n9 h_final by gap and ordering:
```
gap  fwd_mean  rev_mean
  1     0.146     0.316
  2     0.000     0.213
  3     0.000     0.202
  4     0.000     0.243
  5     0.000     0.337
  6     0.000     0.441
  7     0.000     0.519
  8     0.000     0.609
  9     0.000     0.000
```

n9 is more active at **large** gaps (rev gap 8: 0.609), not small (gap 1: 0.316). Correlation(gap, n9) = +0.10 in reversed pairs. n9 is always ~0 for forward pairs.

**Verdict**: n9's ablation devastates small-gap reversed pairs, but n9 is more active at large gaps. This means n9's critical role is NOT about providing extra signal for small gaps — it's about providing signal that is correct only at certain operating points. When n9 is ablated, the damage concentrates on small gaps not because n9 is more active there, but because small-gap reversed pairs have the tightest margins and are most sensitive to any perturbation.

---

### Claim 13: The n2→n4 gain control has a precise operating point

The learned value W_hh[4,2] = -0.49 is precisely tuned: -0.25 breaks forward, -1.0 breaks reversed. This implies n4's firing on the second impulse is in a narrow functional range. The gain control's job is to attenuate n4's second firing to a specific target level.

**Prediction**: n4's second-impulse firing should be approximately constant across all pairs (despite varying n2 values), because the feedback automatically compensates. Specifically, n4_second ≈ n4_input + n2 × (-0.49) should cluster tightly.

**Disproof**: n4's second firing varies widely across pairs, suggesting the feedback is not providing gain control but something else.

**Result**: DISPROVED. n4's second firing varies enormously.

```
n4 at second impulse:
  mean=1.53, std=1.51, CV=0.99

By ordering:
  Forward:  mean=0.15, range=[0.00, 0.84]
  Reversed: mean=2.90, range=[2.13, 4.33]

n4 + n2*0.49 (compensated):
  mean=7.33, std=3.31, CV=0.45
```

CV = 0.99 — standard deviation equals the mean. n4 is bimodal: nearly zero for forward pairs (0.15), moderate for reversed (2.90). The feedback doesn't equalize n4's output — it **nearly silences** n4 in forward pairs while only partially attenuating it in reversed pairs.

The compensated value (n4 + n2×0.49) still has CV=0.45 — not remotely constant.

**Verdict**: The feedback is not gain control in the classical sense (maintaining constant output). It's more like a **gating mechanism**: in forward pairs, n2 is large enough (~18) to nearly suppress n4 entirely (0.15), effectively gating off the n4→n2 pathway for the second impulse. In reversed pairs, n2 is smaller (~14) so n4 still fires partially (2.90). The "gain control" label from Round 1 was wrong — it's closer to a conditional gate.

---

### Claim 14: Comp→wave direct connections carry position-specific information

Claim 8 found comp→wave direct edges (e.g. n6→n0 at +1.13) cause up to -65.6% damage when zeroed. These should carry specific information, not just noise. Since comps encode position via rebuild trajectories, the comp→wave edges should transfer position-dependent signal that differs from what bridges carry.

**Prediction**: Zeroing comp→wave direct edges should damage specific positions more than others. The damage pattern should correlate with which comp→wave edges are largest.

**Disproof**: Damage is uniform across positions — the direct edges carry redundant information.

**Result**: SUPPORTED. Damage is position-specific.

By St: S6 (77.8%) and S7 (88.9%) are least damaged, while S1-S5 and S9 are at 44.4%.
By Mt: M2 worst (33.3%), M6 best (77.8%).

The damage is non-uniform across both Mt and St positions, confirming that comp→wave direct connections carry position-specific information rather than generic coupling.

**Verdict**: Comp→wave direct edges carry position-dependent signal. The damage pattern suggests these edges are particularly important for early-to-mid positions (S0-S5) and less needed for late positions (S6-S7), possibly because late-position encoding has more cascade time through bridges.

---

### Claim 15: W_out uses comps and waves as two independent channels

The readout combines comp h_final and wave h_final. If comps encode M-position (via rebuild time) and waves carry S-cascade state, W_out should weight them to extract St. The two channels should be roughly separable.

**Prediction**: Zeroing all comp→readout columns in W_out vs zeroing all wave→readout columns should each partially break accuracy but in different ways — one should hurt one ordering more than the other, or one should hurt specific gap ranges.

**Disproof**: Both ablations damage all pairs equally — the channels are not functionally separable.

**Result**: SUPPORTED. The two channels have clearly different roles.

```
Baseline:         fwd=100%, rev=100%, all=100%
No comps readout: fwd=6.7%, rev=20.0%, all=13.3%
No waves readout: fwd=42.2%, rev=28.9%, all=35.6%
No comps+waves:   fwd=37.8%, rev=13.3%, all=25.6%
```

Comps are more critical overall (13.3% vs 35.6%). The damage patterns differ:
- **No comps**: forward is destroyed (6.7%), reversed partially works (20%)
- **No waves**: forward partially works (42.2%), reversed is worse (28.9%)

Gap breakdown reveals clear separation:
- No comps: forward pairs collapse at all gaps (0-14%), reversed pairs partially survive at large gaps (up to 100% at gap 9)
- No waves: forward pairs partially survive at mid gaps (43-60%), reversed survive at gap 4 (83%) but collapse at gaps 5, 8, 9

**Verdict**: Comps and waves serve as two separable readout channels. Comps are the primary channel (their loss is more devastating), especially for forward pairs. Waves provide secondary/complementary information, particularly needed for reversed pairs. This is consistent with the model needing two sources of information: one that encodes the last-clip position (comps, via rebuild trajectory) and one that preserves the first-impulse cascade (waves, which survive the second clip).

---

## Scorecard

| # | Claim | Verdict | Key Surprise |
|---|-------|---------|-------------|
| 9 | n7 is pure M-encoder | DISPROVED | R²(Mt)=0.51; n7 encodes full tuple like other comps |
| 10 | n2 encodes first-impulse timing | DISPROVED | R²=0.12, non-monotonic; n2 is complex function of both positions |
| 11 | (last_clip, gap, ordering) lookup | SUPPORTED | 90/90 correct; the decode is trivial arithmetic |
| 12 | n9 compensates small-gap cascade | DISPROVED (opposite) | n9 more active at large gaps; small-gap damage is about sensitivity not activity |
| 13 | n4 gain control produces constant output | DISPROVED | n4 is bimodal (0.15 fwd, 2.90 rev); feedback is a gate not gain control |
| 14 | Comp→wave edges are position-specific | SUPPORTED | Non-uniform damage across positions |
| 15 | W_out has separable comp/wave channels | SUPPORTED | Comps primary (esp. forward), waves complementary (esp. reversed) |

**3 supported, 4 disproved.** Improving — Round 1 was 3/8.

## Method

Same as Round 1: run interventions on the clean 90-pair dataset (x[Mt]=1.0, x[St]=0.8, all else 0). Compare predicted vs actual outcomes. Update results inline.

## Scripts

See `src/rebuild_deep_dive.py` for experimental code.
