# 49: Schur Off-Diagonal Structure — Not a Delay Line

## Summary

The Schur decomposition of W_hh reveals that its off-diagonal (nilpotent) part N carries **85% of the total Frobenius energy** and acts as a **diffuse all-to-all broadcast mixer**, not a delay line. Energy is spread nearly uniformly across all 15 superdiagonals (entropy 3.72 / 3.91 bits), with only 13% of block-to-block coupling between adjacent blocks and 87% between distant ones. The mean coupling distance of 4.64 blocks exceeds even the random expectation of 3.7.

Tracking the canonical impulse through the Schur basis reveals a three-phase lifecycle — raw impulse, non-normal transient, eigenvalue concentration — and the surprising result that W_out is tuned to read the **intermediate transient**, not the eigenvalue steady state. The dominant eigenvalue mode (Schur coord [1], lambda = 1.17) is invisible to W_out. This explains why late second impulses (few post-impulse steps) use a qualitatively different readout mechanism with 100x higher cancellation ratios.

## The Schur Decomposition

W_hh = Q T Q^T where T is quasi-upper-triangular. T decomposes as T = D + N where D is block-diagonal (eigenvalue blocks) and N is strictly upper-triangular (nilpotent).

### Diagonal Blocks (Eigenvalues)

```
Block  Size  Eigenvalue(s)           |lambda|
  0     1x1  -1.274                   1.274
  1     1x1   1.167                   1.167
  2     2x2   0.435 +/- 1.027i        1.115
  3     2x2  -0.106 +/- 0.895i        0.902
  4     1x1  -0.755                   0.755
  5     2x2   0.644 +/- 0.400i        0.758
  6     2x2   0.406 +/- 0.526i        0.664
  7     1x1  -0.572                   0.572
  8     2x2  -0.244 +/- 0.454i        0.515
  9     1x1   0.249                   0.249
 10     1x1  -0.021                   0.021
```

11 blocks, ordered by decreasing |lambda|. The Schur ordering places the dominant modes (blocks 0-1, |lambda| > 1) at the top and the near-zero modes (blocks 9-10) at the bottom.

### Energy Partition

```
||T||_F   = 8.97
||D||_F   = 3.49   (15% of energy)
||N||_F   = 8.27   (85% of energy)
||N||/||D|| = 2.37
```

The off-diagonal part carries 5.6x more energy than the diagonal. This is the Henrici departure: 93% of the Frobenius norm lives outside the eigenvalue blocks.

## Not a Delay Line

### Superdiagonal Energy Distribution

A shift register (delay line) concentrates energy on the first superdiagonal (k=1). The actual distribution is nearly uniform:

```
k   ||N_k||_F   fraction
 1    2.889      0.122
 2    2.739      0.110
 3    1.904      0.053
 4    2.837      0.118
 5    1.893      0.052
 6    2.308      0.078
 7    1.406      0.029
 8    1.889      0.052
 9    2.997      0.131    <-- largest
10    2.493      0.091
11    1.955      0.056
12    1.716      0.043
13    1.256      0.023
14    2.124      0.066
15    1.250      0.023
```

- k=1 holds only **12%** of N's energy. The largest layer is k=9 (13%).
- Entropy = 3.72 bits out of max 3.91 (95% of maximum — nearly uniform).
- Gini = 0.29 (low concentration).

### Block-to-Block Coupling

```
Adjacent-block coupling:     13% of total
Non-adjacent coupling:       87% of total
Mean coupling distance:      4.64 blocks
Expected if random:          3.7 blocks
Expected if delay line:      1.0 blocks
```

The coupling is **more long-range than random**. The heatmap of N confirms this — nearly every off-diagonal entry is significant, with the top-right corner (low-index blocks coupled to high-index blocks) especially dense:

```
     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
 0:  .  .  O  o  .  O  o  o  o  #  O  o  #  .  #  #
 1:  .  .  o  O  o  O  o  .  o  O  O  O  O  O  .  #
 2:  .  .  .  .  o  o  .  o  #  o  O  o  O  o  o  #
 3:  .  .  .  .  .  o  O  O  .  O  o  .  #  O  #  o
 4:  .  .  .  .  .  .  O  o  o  .  o  o  o  .  O  o
 5:  .  .  .  .  .  .  o  O  O  #  #  O  o  #  #  #
 6:  .  .  .  .  .  .  .  O  o  O  o  o  o  o  o  #
 7:  .  .  .  .  .  .  .  .  .  o  O  O  O  O  O  #
 8:  .  .  .  .  .  .  .  .  .  O  O  O  o  o  O  O
 9:  .  .  .  .  .  .  .  .  .  .  .  O  o  o  o  #
10:  .  .  .  .  .  .  .  .  .  .  .  .  #  o  o  o
11:  .  .  .  .  .  .  .  .  .  .  .  .  O  o  o  #
12:  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  o
13:  .  .  .  .  .  .  .  .  .  .  .  .  .  .  #  O
14:  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  #
15:  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .

Legend: . < 0.1, o = 0.1-0.5, O = 0.5-1.0, # > 1.0
```

### Nilpotent Order

N is nilpotent of order 11 (N^11 = 0). The cascade can propagate through up to 10 intermediate modes, but it doesn't step through them sequentially — it jumps across 4-5 blocks at once.

```
||N^k||_F:  8.27, 11.07, 11.78, 7.33, 3.89, 2.53, 1.22, 0.59, 0.03, 0.002, 0
```

N^2 and N^3 are *larger* than N — the diffuse coupling creates constructive interference before it decays.

### Contrast with the Handcrafted Solution

The handcrafted 2nd argmax model (`handcrafted.py`) uses an explicit shift register: `W_hh[i, i+1] = 1.0` for i in 0..8. That is a textbook delay line — all energy on superdiagonal k=1. The trained 16-neuron network has discovered something fundamentally different.

## Three-Phase Lifecycle in Schur Basis

Tracking the canonical impulse response (ReLU(W_hh @ h) iteration from h_0 = ReLU(W_ih)) projected into the Schur basis reveals three phases. The trajectory is identical regardless of impulse arrival time (time-translation invariance, doc 18).

```
steps after    dominant Schur       eig[1]     phase
impulse        coords               energy
-----------    ------------------   ------     -------------------
    0          [15]=58% [12]=18%      0%       RAW IMPULSE
    1          [7]=18% [2]=14%        5%       NON-NORMAL BROADCAST
    2          [2]=22% [3]=22%        0%       NON-NORMAL BROADCAST
    3          [3]=27% [7]=17%        1%       LATE TRANSIENT
    4          [1]=16% [3]=15%       16%       EIGENVALUE EMERGING
    5          [1]=52%               52%       EIGENVALUE DOMINATED
    6          [1]=73%               73%       EIGENVALUE DOMINATED
    7          [1]=68% [2]=9%        68%       EIGENVALUE DOMINATED
    8          [1]=45% [2]=25%       45%       EIGENVALUE DOMINATED
    9          [2]=34% [1]=21%       21%       EIGENVALUE + ROTATION
```

**Phase 1 (step 0):** Energy enters at the low-|lambda| end of the spectrum — Schur coords [15] (lambda = -0.02) and [12] (lambda = -0.24 +/- 0.45i). This is the bottom of the eigenvalue ordering.

**Phase 2 (steps 1-3):** The non-normal broadcast mixer scatters energy across mid-spectrum coordinates [2], [3], [5], [7]. No single coordinate dominates. This is the N matrix doing its job — one-step jumps of 4-5 blocks.

**Phase 3 (steps 4+):** The dominant eigenvalue mode [1] (lambda = 1.17) takes over, reaching 73% concentration at step 6. Eigenvalue dynamics now govern the evolution.

## W_out Reads the Transient, Not the Eigenvalue

The critical surprise: projecting W_out into the Schur basis shows it reads **mid-to-low spectrum** coordinates, not the dominant eigenvalue:

```
W_out position   top Schur coords
    0            [9]=25%  [8]=20%  [13]=18%
    1            [10]=40% [12]=17% [11]=12%
    2            [10]=47% [12]=22% [7]=13%
    3            [9]=56%  [12]=11% [13]=10%
    4            [9]=39%  [11]=15% [8]=11%
    5            [10]=65% [9]=13%  [11]=12%
    6            [10]=45% [12]=16% [11]=14%
    7            [9]=50%  [11]=16% [13]=9%
    8            [12]=31% [13]=24% [9]=24%
    9            [14]=36% [12]=31% [15]=13%
```

W_out overwhelmingly reads Schur coords [9], [10], [11], [12], [13] — all in the lower half of the eigenvalue spectrum (|lambda| = 0.02 to 0.66). **Coord [1] (the dominant eigenvalue mode) does not appear in any position's top-3.**

The discrimination vectors (W_out[target] - W_out[competitor]) show the same pattern:

```
Pair          top Schur coords of discrimination vector
M=0,S=3      [9]=50%  [13]=17% [3]=10%
M=0,S=8      [13]=41% [8]=14%  [12]=12%
M=7,S=8      [13]=58% [12]=14% [10]=7%
```

The readout is tuned to exactly the coordinates that are populated during the non-normal transient (steps 1-4), not the eigenvalue-dominated phase (steps 5+).

## Implications for Late Arrivals

The post-impulse step budget depends on when the second impulse arrives:

```
gap  second impulse  post-impulse steps   phase at readout
 9   position 9      0                    RAW IMPULSE
 8   position 8      1                    NON-NORMAL BROADCAST
 7   position 7      2                    NON-NORMAL BROADCAST
 6   position 6      3                    LATE TRANSIENT
 5   position 5      4                    EIGENVALUE EMERGING
 4   position 4      5                    EIGENVALUE DOMINATED
 3   position 3      6                    EIGENVALUE DOMINATED
 2   position 2      7                    EIGENVALUE DOMINATED
 1   position 1      8                    EIGENVALUE DOMINATED
```

(These are the maximal post-step counts; late absolute positions reduce them further.)

### The paradox

For gaps 1-4 (5+ post-impulse steps), the signal has migrated into the eigenvalue-dominated coord [1] by readout time — but W_out can't read [1]. The energy must be **re-scattered** out of the eigenvalue mode back into the mid-spectrum coords that W_out is listening to. This re-scattering is itself a non-normal effect (the N matrix coupling [1] back out to [2], [3], etc.), which is why the signal at t=9 shows [2]=34% and [1]=21% rather than the [1]=73% peak at t=6.

For gaps 7-9 (0-2 post-impulse steps), readout happens while the signal is still in the broadcast or raw-impulse phase. These coordinates ([12], [15], [7], [2]) have moderate-to-good overlap with W_out's preferred coords. The discrimination mechanism here bypasses the full non-normal rotor and reads the impulse more directly.

### Connection to cancellation ratios

This Schur-basis picture explains the 100x gap in cancellation ratios from doc 48:

| Regime | Gaps | Cancel ratio | Schur mechanism |
|--------|------|-------------|-----------------|
| Interior (differential amplifier) | 1-6 | 0.001-0.02 | Full non-normal rotor: energy enters low spectrum, broadcasts to mid, concentrates at [1], re-scatters to W_out-readable coords. S and A cancel at 0.2%. |
| Edge | 7-8 | 0.02-0.05 | Partial transient: only 1-2 broadcast steps. Less time for S-A signals to diverge, less cancellation needed. |
| Boundary (direct readout) | 9 | 0.33 | No post-impulse dynamics. Direct overlap between impulse coords and W_out. No differential amplifier. |

The interior pairs need the rotor because they must extract a discrimination signal from energy that has been through the eigenvalue dead zone and back. The edge pairs avoid this by being read while still in the transient.

## What the Schur Structure Actually Is

The off-diagonal N is not a delay line (sequential relay through adjacent modes) but a **broadcast mixer** with three properties:

1. **Diffuse coupling.** Energy at any Schur coordinate is redistributed to all others in 1-2 steps. Mean jump distance: 4.6 blocks. Adjacent-block coupling is only 13%.

2. **Constructive interference.** ||N^2|| > ||N^1|| and ||N^3|| > ||N^1|| — the multi-step paths through N reinforce rather than cancel. The mixer amplifies before it decays.

3. **Input-output impedance matching.** The impulse enters at high Schur indices ([12]-[15]), W_out reads at mid indices ([9]-[13]), and the non-normal transient transfers energy between them. The eigenvalue mode [1] is a trap — energy that reaches it must be re-scattered before readout.

The network's computation is a three-step dance:
1. **Load** (step 0): Impulse energy enters the low-eigenvalue end of the Schur spectrum.
2. **Mix** (steps 1-3): N broadcasts it across all modes, populating the coords W_out needs.
3. **Concentrate and re-scatter** (steps 4+): Eigenvalue dynamics pull energy toward [1], then N re-scatters enough back to mid-spectrum for readout.

For late arrivals, only step 1 (and maybe the start of step 2) completes before readout — so the network reads the broadcast directly, without the concentrate-and-re-scatter loop. This works, but with less precision (higher cancellation ratios).

## Related

- [48: What the Weights Tell You](48_qualitative_picture_from_weights.md) -- Six-number qualitative picture, stabilised non-normal rotor
- [47: Symmetric-Antisymmetric Discrimination](47_symmetric_antisymmetric_discrimination.md) -- S-A decomposition, cancellation ratios
- [46: Non-Normal Transient Amplification](46_non_normal_transient_amplification.md) -- Non-normality metrics, neuron-level circuit
- [18: Impulse Response Analysis](18_impulse_response_analysis.md) -- Time-translation invariance, zero-crossing cascade
- `alg_zoo/handcrafted.py` -- Handcrafted delay-line solution for comparison
