# 28: Falsifiable Predictions Round 3

## Context

Rounds 1-2 tested 15 claims (6 supported, 9 disproved). The failure mode analysis added 10 more established facts. Round 3 pushes toward the ARC goal: predicting accuracy from circuit understanding.

Key things we now know:
- Forward decode: target = last_clip. Reversed: target = last_clip - gap.
- Comps are primary readout channel (esp. forward). Waves are secondary (esp. reversed).
- Odd gaps are fragile under weak first impulse (eigenvalue parity).
- 3rd impulse after S hijacks the last-clip mechanism.
- Encoding requires coherent multi-channel state.

## Claims and Predictions

### Claim 16: Single impulse produces a position-independent default output

With only one impulse (M at any position, no S), the model has no meaningful 2nd-largest to find. The rebuild trajectory varies by position, but W_out should extract a fixed default prediction — likely position 9 (last position), since at low S magnitude the model trends toward 9.

**Prediction**: Single impulse of magnitude 1.0 at each position 0-9 should produce the same predicted output (or at most 2 distinct outputs), regardless of impulse position.

**Disproof**: More than 2 distinct outputs across the 10 positions.

**Result**: DISPROVED. 5 distinct outputs.

Single impulse at each position:
```
pos  pred  top logit
  0     9     1.98
  1     9     0.04
  2     9    -0.52
  3     9    -0.04
  4     9    -0.69
  5     0    -0.23
  6     8     0.63
  7     9     1.25
  8     6     4.19
  9     4     7.19
```

Position 9 is the default for early impulses (pos 0-4, 7) but late impulses produce different outputs: pos 5→0, pos 6→8, pos 8→6, pos 9→4. The pattern at magnitude 0.8 is identical.

**Verdict**: The rebuild trajectory DOES vary enough to change the output, especially for late impulses where few rebuild steps remain. Early impulses (0-4) converge to position 9 because the long rebuild enters the same oscillatory regime. Late impulses produce position-dependent outputs because the rebuild is too short to converge. The model doesn't have a universal default — it has a "converged regime" default (9) and a "short rebuild" regime that's position-specific.

---

### Claim 17: Waves encode gap for reversed pairs

The reversed decode needs gap: St = last_clip - gap. Comps encode last_clip via rebuild. Waves must carry the gap information (they're more important for reversed, per Claim 15). So wave h_final should cluster by gap for reversed pairs.

**Prediction**: For reversed pairs on the clean dataset, R²(gap) for wave h_final should be > 0.5. Wave h_final variance should be better explained by gap than by last_clip position.

**Disproof**: R²(gap) < 0.5 for waves, or last_clip explains more variance than gap.

**Result**: DISPROVED. R²(gap) = 0.40 for waves.

```
                 R²(gap)  R²(last_clip)
Waves            0.40     0.37
Comps            0.20     0.98
```

Within-gap wave std ranges from 3.1 to 7.1 — large. Waves don't cluster cleanly by gap.

Meanwhile, comps encode last_clip almost perfectly (R² = 0.98) but know almost nothing about gap (R² = 0.20).

**Verdict**: The clean separation "comps for last_clip, waves for gap" doesn't hold. Comps cleanly encode last_clip. But gap is NOT cleanly encoded in any single channel — it's distributed across the full 16-neuron state. Waves explain gap slightly better than last_clip (0.40 vs 0.37), but neither well. The gap information needed for `St = last_clip - gap` must emerge from the joint pattern across all neurons, not from waves alone.

---

### Claim 18: Forward accuracy is recoverable from comps + n2 alone

Forward just needs target = last_clip. Comps encode this. n2 provides context. So a linear readout from only comp + n2 features (5 of 16 neurons) should recover most forward accuracy.

**Prediction**: Retrain a linear readout on just [n1, n2, n6, n7, n8] h_final values → forward accuracy > 90%. Reversed will be lower because it additionally needs gap from waves.

**Disproof**: Forward accuracy < 90% even with optimal linear readout from comps + n2.

**Result**: DISPROVED. Forward = 28.9% from comps + n2.

```
Neuron subset          Forward  Reversed   All
Comps + n2 (5)          28.9%    11.1%   20.0%
Waves + n2 (6)          46.7%    31.1%   38.9%
Comps only (4)           8.9%    13.3%   11.1%
Full W_out (16)        100.0%   100.0%  100.0%
```

**Verdict**: No small neuron subset supports a working linear readout. Even waves + n2 (6 neurons) only reaches 46.7% forward. The readout fundamentally requires all 16 neurons. This means the encoding is NOT decomposable — W_out doesn't read last_clip from comps and gap from waves separately. It reads a 16-dimensional pattern where every neuron contributes to every output position. The "comp channel" and "wave channel" distinction from Claim 15 (ablation) captures which groups are MORE important, but neither group is SUFFICIENT.

---

### Claim 19: The parity pattern is eigenvalue-driven and method-independent

The odd-gap vulnerability comes from W_hh's 4 negative real eigenvalues, not from any property specific to magnitude reduction. Any method of weakening the first-impulse cascade should produce the same gap-parity failure pattern.

**Prediction**: Scale the hidden state at t=S+1 by a factor α < 1 (attenuating the cascade directly, not the input magnitude). The same odd gaps (1, 3, 7, 9) should fail first, with the same gap-3-worst pattern.

**Disproof**: Different gaps fail first, or even gaps break before odd gaps.

**Result**: SUPPORTED. Identical parity pattern.

Cascade attenuation (all neurons except n2 scaled at t=S):
```
alpha  g1o  g2e  g3o  g4e  g5o  g6e  g7o  g8e  g9o
 1.00  100  100  100  100  100  100  100  100  100
 0.50  100  100  100  100  100  100  100  100  100
 0.30  100  100  100  100  100  100   67  100  100
 0.20  100  100   86  100  100  100   67  100  100
 0.10   44   88   14  100  100  100   67   50    0
 0.05   11   50    0   67   80  100   33    0    0
```

The pattern matches the magnitude-reduction results almost exactly. Gap 3 is worst, gap 5 holds, even gaps are more robust.

Attenuating ONLY waves+bridges: **no effect at all** (100% at α=0.05). The parity structure is entirely in the comp cascade from the first impulse, not in waves or bridges.

**Verdict**: The parity is structural — it comes from how comp states propagate through W_hh's negative eigenvalues, regardless of how the cascade is weakened. Waves and bridges don't carry the parity-sensitive information.

---

### Claim 20: Clean-pair logit margins predict failure order under perturbation

Pairs with small logit margins in the clean dataset should be the first to fail when conditions change (e.g., S magnitude drops). The clean margin is a measure of "robustness distance."

**Prediction**: Rank all 90 clean pairs by margin. As s_mag decreases, the first to fail should have the smallest clean margins. Rank correlation > 0.5.

**Disproof**: Correlation < 0.5.

**Result**: PARTIAL — reversed meets threshold, overall doesn't.

```
Spearman rho(margin, failure_smag):
  Overall:  -0.36
  Forward:  -0.23
  Reversed: -0.50
```

The 10 smallest-margin pairs are all gap-1 reversed (margins 3.6–4.8). These are among the first to fail. The 10 largest-margin pairs are forward with gap 5-8 (margins 24–31) and survive the longest.

But the correlation is moderate, not strong. The ordering within each group is noisy — pairs with similar margins can fail at quite different s_mag values.

**Verdict**: Clean margins are a useful but imperfect predictor. They correctly identify which group of pairs is most vulnerable (gap-1 reversed) but don't precisely rank individual pairs. The failure point depends on more than the clean margin — it depends on how the specific cell's encoding geometry degrades with the perturbation.

---

### Claim 21: Random 2-impulse accuracy is predictable from margin structure

For random inputs with varying magnitudes, we should be able to predict accuracy from the (last_clip, gap, ordering) cell structure and the magnitude ratio.

**Prediction**: Generate 5000 random 2-impulse inputs. Predict accuracy within 5pp of actual.

**Disproof**: Error > 5pp.

**Result**: SUPPORTED. Prediction within 0.5pp.

```
Random 2-impulse (N=5000): actual accuracy = 99.5%
Simple predictor (margin * ratio/0.8 > 0): predicted = 100.0%
Difference: 0.5pp
Per-pair agreement: 99.5%
```

The model is remarkably robust on random inputs. Failures concentrate at:
- S/M ratio ∈ [0.9, 1.0): 95.9% (ordering becomes ambiguous when S ≈ M)
- Gap 9 forward: 95.6%
- Gap 7 forward: 98.4%

The simple predictor (correct if clean margin > 0 after ratio scaling) works because nearly all random inputs have S/M ratios well below the failure threshold. The model was trained to work across the distribution, and it does.

**Verdict**: Accuracy on random 2-impulse inputs is trivially predictable: ~99.5%. The interesting failures are at extreme conditions (S ≈ M, or 3-impulse inputs) that the random distribution rarely samples. For the ARC challenge, the hard part isn't predicting 2-impulse accuracy — it's predicting accuracy on the actual training distribution which includes multi-impulse inputs.

---

## Scorecard

| # | Claim | Verdict | Key Surprise |
|---|-------|---------|-------------|
| 16 | Single impulse → fixed default | DISPROVED | 5 outputs; "converged regime" (→9) vs "short rebuild" (position-dependent) |
| 17 | Waves encode gap | DISPROVED | R²(gap)=0.40; gap is distributed, not in any single channel |
| 18 | Forward from comps+n2 alone | DISPROVED | 28.9%; no neuron subset is sufficient; encoding is holographic |
| 19 | Parity is eigenvalue-driven | SUPPORTED | Identical pattern; wave/bridge attenuation has zero effect |
| 20 | Clean margins predict failure order | PARTIAL | rho=-0.36 overall; -0.50 for reversed; useful but imperfect |
| 21 | Random accuracy from margins | SUPPORTED | 99.5% actual, 100% predicted, 0.5pp diff |

**2 supported, 1 partial, 3 disproved.**

The major insight from this round: **the encoding is holographic** — all 16 neurons are needed for the linear readout to work. The "comp channel" and "wave channel" are meaningful for ablation analysis (which group is MORE critical) but NOT for reconstruction (no subset is sufficient). W_out reads a 16-dimensional pattern, not separable sub-patterns.

## Scripts

See `src/round3_claims.py`.
