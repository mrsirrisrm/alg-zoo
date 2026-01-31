# 27: Mechanistic Reference — M₁₆,₁₀

Condensed reference of established facts from 21 falsifiable claims (Rounds 1–3).

## Model

Single-layer RNN, 16 ReLU hidden neurons, no bias, seq_len=10, 432 params.
```
h[t] = relu(h[t-1] @ W_hh.T + x[t] * W_ih)
logits = h[9] @ W_out.T
```
Task: output position of 2nd-largest input value. 100% on clean 2-impulse dataset (90 pairs).

## Nomenclature

- **Mt**: position of max (magnitude 1.0). **St**: position of 2nd (magnitude 0.8).
- **Forward**: St > Mt. **Reversed**: St < Mt.
- **last_clip**: max(Mt, St) — position of whichever impulse arrives last.
- **gap**: |St - Mt|.

## Neuron Groups

| Group | Neurons | W_ih | Role |
|-------|---------|------|------|
| Comps | n1, n6, n7, n8 | -10 to -13 | Clip to 0 on input; rebuild trajectory encodes position |
| Waves | n0, n10, n11, n12, n14 | |W_ih| < 1.3 | Recurrent cascade carriers; ignore input directly |
| Bridges | n3, n5, n13, n15 | mixed | Couple comps ↔ waves; receive n4 broadcast |
| n4 | n4 | +10.16 | One-shot broadcast relay (self-recur = -0.99) |
| n2 | n2 | +0.015 | Memory latch (self-recur = 0.97); context-dependent |
| n9 | n9 | mixed | Secondary relay; critical for small-gap reversed |

## Circuit

```
Input ──→ Comps (clipping)    ──→ Readout
Input ──→ n4 (broadcast relay)
           ├──→ n2 (latch, 0.97 decay)
           │     ├──→ Comps
           │     ├──→ Waves
           │     ├──→ Bridges
           │     └──⊣ n4 (gate, -0.49)
           ├──→ Waves
           ├──→ Bridges
           └──→ n9 (secondary relay)
                  ├──→ Comps
                  └──→ Waves
Comps ←→ Bridges ←→ Waves
Comps ──→ Waves (direct, non-negligible)
Waves ──→ Readout
```

## Established Facts

### Encoding (Claims 4, 9, 10, 11)

1. **h_final is perfectly determined by (last_clip, gap, ordering)**. Within-group variance = 0. Every neuron encodes the full tuple, not a single variable.

2. **The decode is trivial arithmetic**:
   - Forward: target = last_clip
   - Reversed: target = last_clip - gap

3. **No neuron is a clean single-variable encoder**. n7's n2/n4 cancellation is real but still leaves R²(Mt) = 0.51. n2's h_final is non-monotonic in first-impulse position (R² = 0.12).

4. **The mechanism is reset-and-rebuild-with-memory**: both impulses clip comps to 0; the rebuild trajectory after the last clip depends on residues left by the first impulse (via n2, waves, bridges).

### n4 and n2 (Claims 1, 2, 3, 13)

5. **n4 is a broadcast relay, not the primary detector**. Comps' negative W_ih values are more critical (zeroing n7's W_ih: -84.4% vs n4's: -61.1%). n4 relays input to neurons that can't see it (n2, waves, bridges).

6. **n4 has threshold behavior** (needs W_ih > ~8 to overcome n2 feedback of -0.49 × ~14 ≈ -7).

7. **n2→n4 feedback (-0.49) is a gate, not gain control**. n4's second firing is bimodal: ~0.15 in forward pairs (nearly silenced), ~2.90 in reversed (partially attenuated). CV = 0.99.

8. **The gate protects forward pairs** (opposite to initial prediction). Without feedback: forward drops to 77.8%, reversed stays 100%. The learned value -0.49 is optimal: weaker hurts forward, stronger hurts reversed.

### Waves and Bridges (Claims 5, 6, 14)

9. **Waves are protected by small |W_ih|**. Amplifying wave W_ih breaks accuracy monotonically. Vulnerability ranks exactly by |W_ih| magnitude. Zeroing wave W_ih: 98.9% (negligible).

10. **Bridges are the primary comp↔wave coupling** (ablation: 24.4%) but direct comp→wave edges are non-negligible (zeroing: 54.4%). n6→n0 alone causes -65.6% when zeroed.

11. **Direct comp→wave edges carry position-specific information**: damage is non-uniform across both Mt and St positions.

### n9 (Claims 7, 12)

12. **n9 is critical for small-gap reversed pairs** (gap 1-3 drop to 22-29% without n9). But n9 is paradoxically more active at large gaps (gap 8: 0.609 vs gap 1: 0.316). Small-gap pairs fail because they have the tightest margins, not because n9 provides more signal there.

13. **n9 is always ~0 for forward pairs**. Only fires for reversed.

### Readout (Claim 15)

14. **W_out uses comps and waves as two separable channels**:
    - Comps are primary (zeroing: 13.3%), especially for forward pairs (6.7%)
    - Waves are complementary (zeroing: 35.6%), particularly needed for reversed
    - Comps encode last-clip position via rebuild trajectory
    - Waves preserve first-impulse cascade

## Failure Modes Under Variable Magnitude

Tested: M=1.0, sweep S magnitude from 0.8 down to 0.01.

```
s_mag   fwd%   rev%   all%
 0.80  100.0  100.0  100.0
 0.40  100.0  100.0  100.0
 0.20  100.0   97.8   98.9
 0.10  100.0   77.8   88.9
 0.05   48.9   46.7   47.8
 0.01   20.0   13.3   16.7
```

15. **Forward pairs are robust to low S magnitude** — 100% accurate down to s_mag=0.1 (10:1 ratio). Reversed pairs break first (77.8% at s_mag=0.1).

16. **The model never predicts Mt instead of St** — pred_Mt = 0% at all magnitudes. Failures are "uninterpretable state" (all logits negative), not "confuse M for S."

17. **Reversed pairs fail because S arrives first and its magnitude sets the n2 latch level.** At s_mag=0.1 reversed: n4 fires at only 1.0 on S (vs 8.1 at 0.8), n2 latches to only 1.8 (vs 14.1). The first-impulse residue is too weak; when M arrives, the circuit looks like a single-impulse input.

18. **Forward pairs survive because M arrives first**, always bootstrapping n4 (10.16) and n2 (~17.6) at full strength regardless of S magnitude. S only needs to leave a detectable mark on comps — and even s_mag=0.1 produces 0.1 × (-13) = -1.3, enough to perturb the rebuild trajectory distinctively.

### Odd/even gap asymmetry (reversed, low S magnitude)

19. **Reversed failures at low S concentrate at odd gaps**. Pattern holds across all s_mag:

    ```
    s_mag  g1o  g2e  g3o  g4e  g5o  g6e  g7o  g8e  g9o
     0.20  100  100  100  100  100  100   67  100  100
     0.12   78  100   57  100  100  100   67  100    0
     0.10   67  100   29  100  100  100   67  100    0
     0.08   44   88   14  100  100  100   67   50    0
     0.05   11   50    0   83  100  100   67    0    0
    ```

    Gap 3 is the worst odd gap. Gap 5 is the exception (holds to s_mag=0.04).

20. **Mt=9 is NOT a special case** — it's actually the most robust Mt. Excluding M9 pairs, the pattern is the same or slightly worse. The M9,S0 gap=9 failure is the only M9 weakness, because gap=9 has exactly one pair.

21. **W_hh has 4 negative real eigenvalues** (λ=-1.27, -0.75, -0.57, -0.02), which create parity structure in rebuild trajectories. Negative eigenvalues flip sign each step, so even-step separations return to "original" sign while odd-step separations are inverted.

22. **The parity is NOT in n2/n4** — those are nearly identical across gaps at M arrival (~1.7 and ~9.2 for all gaps at s_mag=0.1). It's in the wave/bridge/comp cascade state at M arrival.

23. **Gap=3 fails because the readout channels fight each other.** Logit decomposition for (M5,S2) at s_mag=0.1:
    - Comps→logit[2] = +67.2 (vs +22.6 clean) — overshoots massively
    - Waves→logit[2] = -42.3 (vs -8.9 clean) — pushes hard wrong
    - Bridges→logit[2] = -19.1 (vs +9.2 clean) — flips sign
    - Net = -1.2 (wrong)

    At gap=2 weak, the same channels: comps +24.0, waves -16.4, bridges -1.1 — net positive (correct). The channels stay proportionate.

24. **Rescuing n2 alone makes things worse** (0/7 vs 2/7 at gap=3). Injecting clean n2 (~13.8) into weak cascade creates a mismatched state: strong n2 but weak waves/bridges/comps. The encoding requires coherence across all channels, not just the right n2 value.

### 3-impulse failure mode (Tt close to St)

25. **A 3rd impulse after St is a severe failure mode**. With M=1.0, S=0.8, T after S:
    - T=0.79: accuracy drops to **65.8%** (from 100% without T)
    - T=0.80 (equal to S): **43.3%**
    - The transition is sharp: 0.75→94%, 0.78→79%, 0.79→66%

26. **Every failure predicts Tt instead of St** — pred_Tt = 34.2%, pred_other = 0%. The model is not confused in general; it specifically identifies the wrong impulse as S.

27. **The mechanism is last-clip hijacking**. The trace shows T clips comps just as M and S do (n7: 11.95 → 1.12 at T). The rebuild trajectory after T encodes T's position, and W_out extracts that as the answer. The model's "find the last clip, decode from there" strategy is fooled.

28. **T between M and S is less damaging** (83.9% vs 65.8%) because S still arrives last and defines last_clip. The 16.1% failures here are cases where T's clip disrupts the encoding enough that the rebuild from S produces the wrong answer.

29. **Logit margins are razor-thin near failure**. At T=0.79, example (M1,S4,T7): logit_S=11.58, logit_T=11.45, margin=0.13. At T=0.80: margin goes to -0.64 and T wins.

### Encoding structure (Claims 16, 17, 18)

30. **Single impulse does NOT produce a position-independent default**. 5 distinct outputs across 10 positions. Early impulses (pos 0–4, 7) converge to position 9 via long rebuild into oscillatory regime. Late impulses (pos 5→0, 6→8, 8→6, 9→4) produce position-dependent outputs because rebuild is too short to converge.

31. **Gap is NOT cleanly encoded in any single channel**. R²(gap) for waves = 0.40, for comps = 0.20. Meanwhile comps encode last_clip almost perfectly (R² = 0.98). Gap information needed for `St = last_clip - gap` emerges from the joint 16-neuron pattern, not from waves alone.

32. **The encoding is holographic — all 16 neurons are required for readout**. No neuron subset supports a working linear readout:

    ```
    Subset               Forward  Reversed
    Comps + n2 (5)        28.9%    11.1%
    Waves + n2 (6)        46.7%    31.1%
    Comps only (4)         8.9%    13.3%
    Full W_out (16)      100.0%   100.0%
    ```

    W_out reads a 16-dimensional pattern; the "comp channel" and "wave channel" are meaningful for ablation (which group is MORE critical) but not for reconstruction (no subset is sufficient).

### Parity mechanism (Claim 19)

33. **The parity pattern is eigenvalue-driven and method-independent**. Cascade attenuation (scaling h at t=S+1 by α < 1) reproduces the identical odd/even failure pattern as magnitude reduction. Wave/bridge-only attenuation has **zero effect** (100% at α=0.05). The parity structure is entirely in the comp cascade through W_hh's negative eigenvalues.

### Predictability (Claims 20, 21)

34. **Clean logit margins are a useful but imperfect failure predictor**. Spearman rho(margin, failure_smag): overall = -0.36, forward = -0.23, reversed = -0.50. Correctly identifies most-vulnerable group (gap-1 reversed, margins 3.6–4.8) but doesn't precisely rank individual pairs.

35. **Random 2-impulse accuracy is trivially predictable**. N=5000 random inputs: actual = 99.5%, simple margin predictor = 100.0%, diff = 0.5pp. Failures concentrate at S/M ratio ∈ [0.9, 1.0) where ordering becomes ambiguous.

## Scorecard

Round 1: 3 supported, 5 broken (of 8).
Round 2: 3 supported, 4 disproved (of 7).
Round 3: 2 supported, 1 partial, 3 disproved (of 6).

## Open Questions

1. How does W_out decode the (last_clip, gap, ordering) tuple from the 16-dim holographic representation into St?
2. What exactly does n9 compute for small-gap reversed pairs?
3. Can we predict accuracy on multi-impulse / training-distribution inputs from the circuit? (ARC challenge goal — 2-impulse random is solved at 99.5%)
4. What determines the single-impulse convergence regime boundary (why pos 5–6 diverge from the pos 0–4 default)?
