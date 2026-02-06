# 45: Eigenvector Discrimination Mechanism

## Summary

The left and right eigenvectors of Phi_post reveal how the network discriminates between positions. The **left eigenvector** identifies which input directions get amplified; the **right eigenvector** determines the output pattern. Both fwd and rev Phi_post amplify similar input directions (neurons {4, 2, 7}), but with dramatically different gains: fwd preserves (λ ≈ 1.0), rev amplifies (λ ≈ 2.6). This asymmetry is *the* mechanism that allows the network to correctly identify S regardless of whether M or S arrives first.

## Background

The linear logit formula is:
```
logits = first_val * a + second_val * b
```

where:
- `a = W_out @ Phi_post @ D_mask @ w_first[gap]` — contribution from first impulse
- `b = W_out @ Phi_post @ D_mask @ W_ih` — contribution from second impulse

In **fwd** (M first, S second): first_val = M, second_val = S
In **rev** (S first, M second): first_val = S, second_val = M

## Left Eigenvector: What Gets Amplified

The left eigenvector of Phi_post identifies which input directions get amplified by the dominant eigenvalue.

### Rev direction (S first, M second)

```
gap=1 rev (λ=2.74):
  Left eigenvector top neurons: [7, 4, 2, 8]
  Correlation with |W_ih|: r = 0.62

gap=3 rev (λ=2.61):
  Left eigenvector top neurons: [7, 4, 2, 8]
  Correlation with |W_ih|: r = 0.64

gap=5 rev (λ=2.17):
  Left eigenvector top neurons: [4, 2, 7, 8]
  Correlation with |W_ih|: r = 0.56
```

**Key finding**: The left eigenvector is strongly correlated with |W_ih| — the neurons that receive the largest input weights. Since M arrives second in rev (through W_ih), the rev Phi_post is tuned to amplify exactly where M's energy enters.

### Fwd direction (M first, S second)

```
gap=1 fwd (λ=1.05):
  Left eigenvector top neurons: [4, 2, 12, 9]
  Correlation with |W_ih|: r = 0.20

gap=3 fwd (λ=1.00):
  Left eigenvector top neurons: [4, 2, 7, 9]
  Correlation with |W_ih|: r = 0.32

gap=5 fwd (λ=1.02):
  Left eigenvector top neurons: [4, 2, 7, 9]
  Correlation with |W_ih|: r = 0.27
```

**Surprising finding**: The fwd left eigenvector also has neurons {4, 2, 7} as dominant, similar to rev. The cos_sim between fwd and rev left eigenvectors is **0.63–0.81** — they amplify nearly the same subspace.

The critical difference is the eigenvalue magnitude:
- Fwd: λ ≈ 1.0 (preserve)
- Rev: λ ≈ 2.6 (amplify 2.6x)

## Right Eigenvector: Output Pattern

The right eigenvector determines what the amplified mode looks like in the output space (before W_out projection).

### Rev direction

```
gap=1 rev: Right eigenvector top neurons: [10, 12, 0, 2, 14]
gap=3 rev: Right eigenvector top neurons: [12, 11, 8, 0, 13]
gap=5 rev: Right eigenvector top neurons: [8, 12, 10, 1, 0]
```

### Fwd direction

```
gap=1 fwd: Right eigenvector top neurons: [12, 3, 13, 0, 2]
gap=3 fwd: Right eigenvector top neurons: [0, 12, 2, 15, 7]
gap=5 fwd: Right eigenvector top neurons: [15, 10, 3, 13, 12]
```

The right eigenvectors differ substantially between fwd and rev (cos_sim = 0.15–0.36), meaning the output patterns are quite different even though the input amplification directions are similar.

## What the Right Eigenvector Produces in Logits

Projecting the right eigenvector through W_out gives the logit pattern:

### Gap=3 rev
```
pos 0: -1.95  (strongly disfavored)
pos 1: -0.93
pos 2: +0.40
pos 3: +0.82  (favored)
pos 4: +0.63
pos 5: +0.15
pos 6: +0.61
pos 7: +0.47
pos 8: +0.49
pos 9: -0.56
```

This pattern **disfavors positions 0, 1, 9** and **favors mid positions 3-8**.

### Gap=3 fwd
```
pos 0: -1.48  (strongly disfavored)
pos 1: +0.70
pos 2: -0.36
pos 3: +0.49  (moderately favored)
pos 4: -0.08
pos 5: +0.18
pos 6: +0.28
pos 7: +0.63
pos 8: -0.00
pos 9: +0.39
```

Different pattern, but also disfavors position 0.

## The Discrimination Mechanism

### Rev case: Amplified M overcomes "wrong" linear dynamics

For gap=3 rev (S at position 3, M at position 6):

```
logits = S_val * a + M_val * b

Coefficient decomposition:
  a[3] = +19.78 (S contribution to target)
  b[3] = +3.86  (M contribution to target)
  a[6] = +18.07 (S contribution to competitor)
  b[6] = -2.20  (M contribution to competitor)

Margin = (a[3]-a[6])*S + (b[3]-b[6])*M
       = (1.71)*0.8 + (6.06)*1.0
       = 1.37 + 6.06
       = +7.43 (correct)
```

**The M contribution (b coefficient) does the heavy lifting.** Even though M is not at the target position, its amplified propagation through Phi_post creates a b[target] > b[competitor] pattern that favors S.

### Fwd case: Preserved M carries the information

For gap=3 fwd (M at position 0, S at position 3):

```
logits = M_val * a + S_val * b

Coefficient decomposition:
  a[3] = +11.17 (M contribution to target)
  b[3] = +4.82  (S contribution to target)
  a[0] = -3.64  (M contribution to competitor)
  b[0] = +9.75  (S contribution to competitor)

Margin = (a[3]-a[0])*M + (b[3]-b[0])*S
       = (14.81)*1.0 + (-4.93)*0.8
       = 14.81 - 3.94
       = +10.87 (correct)
```

**The M contribution (a coefficient) dominates.** M arrives first and propagates through the full canonical sequence; Phi_post preserves this information (λ ≈ 1.0), and the w_first[gap] encoding carries the position information.

## The Asymmetry Explained

| Property | Fwd (M first) | Rev (S first) |
|----------|--------------|----------------|
| Which impulse is larger | First (M) | Second (M) |
| Phi_post eigenvalue | λ ≈ 1.0 | λ ≈ 2.6 |
| Which coefficient dominates | a (first impulse) | b (second impulse) |
| Mechanism | Preserve M's propagation | Amplify M's entry |

The network has learned complementary strategies:
1. **Fwd**: Preserve the first impulse's journey through the canonical sequence. The larger value (M) accumulates position-dependent structure in w_first that Phi_post maintains.
2. **Rev**: Amplify the second impulse's entry. The larger value (M) enters through W_ih, and Phi_post's 2.6x amplification boosts its contribution enough to overcome the "wrong" linear tendency.

## Why Both Strategies Work

Both rely on M (the max value) carrying the discrimination signal:

- In fwd, M has more time to accumulate structure (longer propagation through canonical sequence)
- In rev, M gets boosted by amplification (larger eigenvalue)

The network doesn't need to "know" which case it's in — the D-dependent Phi_post automatically applies the right transformation. The canonical D sequences encode which neurons are active, and the active neuron pattern determines which Phi_post (fwd or rev) applies.

## Connection to W_ih Structure

W_ih has large negative weights on neurons {7, 8, 6, 1} and large positive weight on neuron 4:

```
n7: -13.17
n8: -12.31
n6: -11.00
n1: -10.57
n4: +10.16
```

The left eigenvector of Phi_post (both fwd and rev) is aligned with this input structure. Neurons {4, 7, 2, 8} dominate the left eigenvector with ~98% of its energy concentrated in the top 5 neurons.

This means Phi_post has been shaped during training to have its dominant mode aligned with where input energy arrives. The only difference between fwd and rev is how much this mode amplifies vs preserves.

## Implications for Interpretability

1. **Left eigenvector = input sensitivity**: Looking at which directions get amplified reveals what information the network "pays attention to"

2. **Right eigenvector = output template**: The amplified mode projects to a specific logit pattern that determines discrimination

3. **Eigenvalue = gain control**: The magnitude determines whether information is preserved (λ ≈ 1) or boosted (λ > 1) or suppressed (λ < 1)

4. **Same subspace, different gains**: The fwd/rev distinction isn't about different subspaces but different amplification of the same input-sensitive subspace

## Related

- [44: Phi_post Eigenvalue Analysis](44_phi_post_eigenvalue_analysis.md) — Spectral properties
- [43: Mechanistic Discrimination](43_mechanistic_discrimination.md) — Overall mechanism
- [34: Fwd/Rev Mechanism](34_fwd_rev_mechanism_explained.md) — Ordering-dependent dynamics
- `src/mechanistic_discrimination.py` — Implementation
