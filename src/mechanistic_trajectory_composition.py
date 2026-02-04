"""
MECHANISTIC ESTIMATE VIA TRAJECTORY COMPOSITION

Key insight from failed attempts:
- Pure linear theory: fails (ReLU essential)
- Cell-specific linear theory: fails (trajectory matters, not just final cell)

New approach: Compose the actual per-timestep transformations,
accounting for the typical ReLU patterns at each step.

For a mechanistic estimate, we need to derive the activation patterns
from the weights, not just observe them.
"""

import numpy as np
from alg_zoo import example_2nd_argmax

model = example_2nd_argmax()
model.eval()

W_ih = model.rnn.weight_ih_l0.detach().numpy().squeeze()
W_hh = model.rnn.weight_hh_l0.detach().numpy()
W_out = model.linear.weight.detach().numpy()

M_val = 1.0
S_val = 0.8

def run_relu(impulses):
    h = np.zeros(16)
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
    return h

def trace_trajectory(impulses):
    """Return full trajectory with pre-activations."""
    h = np.zeros(16)
    trajectory = [(h.copy(), np.zeros(16), h.copy())]  # (h, pre, h_post)
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        trajectory.append((h.copy(), pre.copy(), h.copy()))
    return trajectory

print("=" * 70)
print("MECHANISTIC ESTIMATE VIA TRAJECTORY COMPOSITION")
print("=" * 70)

# =============================================================================
# PART 1: Understand why trajectory matters
# =============================================================================

print("\nPART 1: WHY TRAJECTORY MATTERS")
print("-" * 50)

# Take a pair in the dominant cell and trace the trajectory
m, s = 5, 6  # Both fwd and rev end in dominant cell

traj_fwd = trace_trajectory([(m, M_val), (s, S_val)])
traj_rev = trace_trajectory([(m, S_val), (s, M_val)])

print(f"\nPair (M={m}, S={s}): both end in dominant cell")
print("\nOffset evolution through trajectory:")

for t in range(11):
    h_fwd = traj_fwd[t][0]
    h_rev = traj_rev[t][0]
    offset = h_fwd - h_rev
    print(f"  t={t}: ||offset|| = {np.linalg.norm(offset):.3f}")

# The offset is ZERO until the first differing input
print("\nKey insight: offset is 0 until first impulse, then grows through trajectory")

# =============================================================================
# PART 2: Decompose offset into per-timestep contributions
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: PER-TIMESTEP OFFSET CONTRIBUTIONS")
print("=" * 70)

# For a pair (m, s), the offset at t=9 can be decomposed:
# offset = sum of contributions from each timestep

# At each timestep t:
# h_fwd[t+1] = ReLU(W_ih * x_fwd[t] + W_hh @ h_fwd[t])
# h_rev[t+1] = ReLU(W_ih * x_rev[t] + W_hh @ h_rev[t])
#
# The difference comes from:
# 1. Different inputs at timesteps m and s
# 2. Different ReLU gating due to trajectory differences

def decompose_offset_contributions(m_pos, s_pos):
    """
    Trace how the offset builds up.
    Returns contribution at each timestep.
    """
    h_fwd = np.zeros(16)
    h_rev = np.zeros(16)

    contributions = []

    for t in range(10):
        # Inputs
        x_fwd = M_val if t == m_pos else (S_val if t == s_pos else 0)
        x_rev = S_val if t == m_pos else (M_val if t == s_pos else 0)

        # Pre-activations
        pre_fwd = W_ih * x_fwd + W_hh @ h_fwd
        pre_rev = W_ih * x_rev + W_hh @ h_rev

        # Post-ReLU
        h_fwd_new = np.maximum(0, pre_fwd)
        h_rev_new = np.maximum(0, pre_rev)

        # Contribution this timestep
        # This is approximately: how much did this step add to the offset?
        offset_before = h_fwd - h_rev
        offset_after = h_fwd_new - h_rev_new

        # The "new" contribution is roughly: offset_after - W_hh @ offset_before
        # But ReLU complicates this

        contributions.append({
            't': t,
            'offset_norm': np.linalg.norm(offset_after),
            'input_diff': x_fwd - x_rev,
            'relu_diff_fwd': np.sum(pre_fwd < 0),
            'relu_diff_rev': np.sum(pre_rev < 0),
        })

        h_fwd = h_fwd_new
        h_rev = h_rev_new

    return contributions, h_fwd - h_rev

# Analyze a few pairs
print("\nOffset buildup for different (m, s) pairs:")

for m, s in [(0, 5), (2, 7), (5, 6), (4, 9)]:
    contribs, final_offset = decompose_offset_contributions(m, s)

    logits = W_out @ final_offset
    margin = logits[s] - logits[m]

    print(f"\n(M={m}, S={s}): margin = {margin:.2f}")
    print("  t | ||offset|| | input_diff | clips_fwd | clips_rev")
    print("  --|------------|------------|-----------|----------")
    for c in contribs:
        if c['offset_norm'] > 0.1 or c['input_diff'] != 0:
            print(f"  {c['t']} |   {c['offset_norm']:7.3f}  |   {c['input_diff']:+.2f}    |     {c['relu_diff_fwd']:2d}    |     {c['relu_diff_rev']:2d}")

# =============================================================================
# PART 3: The key question - can we predict activation patterns from weights?
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: PREDICTING ACTIVATION PATTERNS FROM WEIGHTS")
print("=" * 70)

# For a mechanistic estimate, we need to know which ReLU activations
# fire without running the model.

# Hypothesis: We can bound the hidden state evolution
# and derive activation patterns from those bounds.

print("\nApproach: Bound propagation")
print("If we can bound ||h[t]|| and the direction of h[t],")
print("we can determine which neurons are guaranteed active/inactive.")

# First, understand the "typical" hidden state
print("\nTypical hidden state statistics:")

all_h = []
for m in range(10):
    for s in range(10):
        if m == s:
            continue
        h = run_relu([(m, M_val), (s, S_val)])
        all_h.append(h)
        h = run_relu([(m, S_val), (s, M_val)])
        all_h.append(h)

all_h = np.array(all_h)
mean_h = np.mean(all_h, axis=0)
std_h = np.std(all_h, axis=0)

print("\nPer-neuron statistics of h_final:")
print("Neuron | Mean | Std | Mean/Std | Always>0")
print("-------|------|-----|----------|----------")
always_positive = []
for n in range(16):
    min_val = np.min(all_h[:, n])
    always_pos = min_val > 0
    always_positive.append(always_pos)
    print(f"  {n:2d}   | {mean_h[n]:5.2f} | {std_h[n]:4.2f} | {mean_h[n]/std_h[n] if std_h[n] > 0.01 else float('inf'):6.2f}   | {'Yes' if always_pos else 'No'}")

print(f"\nNeurons always positive: {[n for n in range(16) if always_positive[n]]}")
print(f"Neurons sometimes zero: {[n for n in range(16) if not always_positive[n]]}")

# =============================================================================
# PART 4: Can we derive the empirical f(·) from a simple formula?
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: SEARCHING FOR A WEIGHT-BASED FORMULA FOR f(·)")
print("=" * 70)

# Get empirical f(pos)
offsets = []
m_positions = []
s_positions = []

for m in range(10):
    for s in range(10):
        if m == s:
            continue
        h_fwd = run_relu([(m, M_val), (s, S_val)])
        h_rev = run_relu([(m, S_val), (s, M_val)])
        offset = h_fwd - h_rev
        offsets.append(offset)
        m_positions.append(m)
        s_positions.append(s)

offsets = np.array(offsets)
n_pairs = len(offsets)

# Fit separable model
M_onehot = np.zeros((n_pairs, 10))
S_onehot = np.zeros((n_pairs, 10))
for i, (m, s) in enumerate(zip(m_positions, s_positions)):
    M_onehot[i, m] = 1
    S_onehot[i, s] = 1

X_sep = np.column_stack([M_onehot[:, 1:], S_onehot[:, 1:]])
coeffs, _, _, _ = np.linalg.lstsq(X_sep, offsets, rcond=None)

F_empirical = np.zeros((10, 16))
for m in range(1, 10):
    F_empirical[m] = coeffs[m - 1]

# What weight-based formula could produce F_empirical?
# Key observation: f(pos) encodes "the effect of having M at pos vs S at pos"

# Idea 1: f(pos) involves how W_ih × (M - S) propagates
# But this is just linear theory which fails

# Idea 2: f(pos) involves the STABLE operating point
# The network converges to a stable manifold; f encodes deviation from mean

# Let's check if f is related to deviation from mean hidden state
print("\nIs f(pos) related to hidden state deviation from mean?")

# For each pos, what's the average hidden state when M is at pos vs S at pos?
h_when_M_at = {pos: [] for pos in range(10)}
h_when_S_at = {pos: [] for pos in range(10)}

for m in range(10):
    for s in range(10):
        if m == s:
            continue
        h_fwd = run_relu([(m, M_val), (s, S_val)])
        h_rev = run_relu([(m, S_val), (s, M_val)])

        h_when_M_at[m].append(h_fwd)  # M at m in fwd
        h_when_S_at[s].append(h_fwd)  # S at s in fwd
        h_when_M_at[s].append(h_rev)  # M at s in rev
        h_when_S_at[m].append(h_rev)  # S at m in rev

mean_h_when_M_at = {pos: np.mean(h_when_M_at[pos], axis=0) for pos in range(10)}
mean_h_when_S_at = {pos: np.mean(h_when_S_at[pos], axis=0) for pos in range(10)}

# The difference (M at pos) - (S at pos) averaged over other position
empirical_M_minus_S = {}
for pos in range(10):
    # Average difference when (M at pos, S elsewhere) vs (S at pos, M elsewhere)
    diffs = []
    for other in range(10):
        if other == pos:
            continue
        h_M_at_pos = run_relu([(pos, M_val), (other, S_val)])
        h_S_at_pos = run_relu([(pos, S_val), (other, M_val)])
        diffs.append(h_M_at_pos - h_S_at_pos)
    empirical_M_minus_S[pos] = np.mean(diffs, axis=0)

print("\nComparison: f_empirical(pos) vs mean(h_M_at_pos - h_S_at_pos):")
print("pos | cos similarity | ||f_emp|| | ||emp_diff||")
print("----|----------------|-----------|-------------")

for pos in range(10):
    f_emp = F_empirical[pos]
    emp_diff = empirical_M_minus_S[pos]

    norm_f = np.linalg.norm(f_emp)
    norm_d = np.linalg.norm(emp_diff)

    if norm_f > 0.01 and norm_d > 0.01:
        cos = np.dot(f_emp, emp_diff) / (norm_f * norm_d)
    else:
        cos = float('nan')

    print(f" {pos}  |     {cos:+.4f}     |   {norm_f:6.3f}  |   {norm_d:6.3f}")

# =============================================================================
# PART 5: The insight - f is EXACTLY the position-conditional mean difference
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: f(pos) = E[h | M at pos] - E[h | S at pos]")
print("=" * 70)

# This suggests: f(pos) is the conditional expectation difference
# f(pos) = E[h_final | M is at position pos] - E[h_final | S is at position pos]

# If this is true, we can try to derive it from weights!
# The question is: what determines E[h_final | M at pos]?

# For the mechanistic estimate, we'd need to show:
# E[h | M at pos] can be computed from W_ih, W_hh (with some ReLU approximation)

print("""
KEY INSIGHT:
f(pos) ≈ E[h | M at pos] - E[h | S at pos]

This is a CONDITIONAL EXPECTATION over the other token's position.
It averages out the "where is the other token" uncertainty.

For a mechanistic estimate, we need to:
1. Derive E[h | token at pos] from weights
2. Show this satisfies the separable structure
3. Show W_out @ (f(m) - f(s)) discriminates correctly

The challenge: E[h | token at pos] involves ReLU and depends on
the other token position. But averaging over other positions
might make this tractable!
""")

# =============================================================================
# PART 6: Try to derive the conditional expectation from weights
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: DERIVING CONDITIONAL EXPECTATIONS")
print("=" * 70)

# Simplified model: assume all pairs are in the "dominant cell" after both impulses
# Then the dynamics are approximately linear in that cell

# Get the dominant cell mask (most common at t=9)
patterns = {}
for m in range(10):
    for s in range(10):
        if m == s:
            continue
        for direction in ['fwd', 'rev']:
            if direction == 'fwd':
                h = run_relu([(m, M_val), (s, S_val)])
            else:
                h = run_relu([(m, S_val), (s, M_val)])
            pattern = tuple(h > 1e-10)
            if pattern not in patterns:
                patterns[pattern] = 0
            patterns[pattern] += 1

dominant_pattern = max(patterns.keys(), key=lambda p: patterns[p])
active_neurons = [n for n, a in enumerate(dominant_pattern) if a]
D = np.diag([1.0 if n in active_neurons else 0.0 for n in range(16)])

print(f"Dominant cell: {len(patterns[dominant_pattern])}/180 pairs")
print(f"Active neurons: {active_neurons}")

# Within dominant cell, dynamics are approximately:
# h[t+1] = D @ max(0, W_hh @ h[t] + W_ih * x[t])
# ≈ D @ (W_hh @ h[t] + W_ih * x[t])  [if staying positive]

# Let's try to compute the expected h given only M position
# E[h | M at m] = average over s of run_relu([(m, M_val), (s, S_val)])

print("\nExpected h given M position (empirical):")
E_h_given_M = np.zeros((10, 16))
for m in range(10):
    h_sum = np.zeros(16)
    count = 0
    for s in range(10):
        if m == s:
            continue
        h = run_relu([(m, M_val), (s, S_val)])
        h_sum += h
        count += 1
    E_h_given_M[m] = h_sum / count

# Now try to derive this from weights using a linear approximation
# Assumption: after both impulses, we're in dominant cell
# h[9] ≈ D @ W_hh^(9-max(m,s)) @ [effect of both impulses at max(m,s)]

# This is getting complex. Let me try a different approach:
# Fit a linear model: E[h | M at m] = A @ one_hot(m) + b

print("\nFitting linear model: E[h | M at m] = A @ one_hot(m) + b")

positions = np.arange(10).reshape(-1, 1)
A_fit, _, _, _ = np.linalg.lstsq(
    np.column_stack([np.eye(10), np.ones((10, 1))]),
    E_h_given_M,
    rcond=None
)

E_h_given_M_predicted = np.column_stack([np.eye(10), np.ones((10, 1))]) @ A_fit

# Check fit quality
ss_res = np.sum((E_h_given_M - E_h_given_M_predicted) ** 2)
ss_tot = np.sum((E_h_given_M - np.mean(E_h_given_M, axis=0)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"R² for linear model of E[h | M at m]: {r2:.4f}")

# The matrix A encodes "how does M position affect expected h"
# This is similar to f(m)!

print("\nComparing A matrix to f_empirical:")
# A is 11 x 16 (positions + intercept → hidden state)
# The first 10 rows are the position effects

A_positions = A_fit[:10, :]  # Remove intercept

# Compare to f_empirical
print("pos | cos(A[pos], f_emp[pos])")
print("----|------------------------")
for pos in range(10):
    a = A_positions[pos]
    f = F_empirical[pos]
    if np.linalg.norm(a) > 0.01 and np.linalg.norm(f) > 0.01:
        cos = np.dot(a, f) / (np.linalg.norm(a) * np.linalg.norm(f))
    else:
        cos = float('nan')
    print(f" {pos}  |     {cos:+.4f}")

print("""
CONCLUSION:
The position effect A[pos] in E[h | M at pos] is related to f(pos),
but they're not identical because:
1. f(pos) = E[h | M at pos] - E[h | S at pos]
2. A[pos] = E[h | M at pos] - mean(E[h | M])

The difference matters for the mechanistic estimate.
""")
