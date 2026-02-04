"""
MECHANISTIC ESTIMATE VIA EFFECTIVE CELL DYNAMICS

Key insight: Within the dominant tropical cell, the effective W_hh has
eigenvalues near 1 (max = 1.01), creating stable bounded dynamics.

Strategy:
1. Identify the dominant cell's activation mask D
2. Compute effective dynamics: W_hh_eff = D @ W_hh @ D
3. Derive f(pos) using effective dynamics
4. Verify discrimination from weights

This is a "semi-mechanistic" approach: we use the empirically-observed
dominant cell, but then derive predictions purely from weights.
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
    """Run with ReLU."""
    h = np.zeros(16)
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
    return h

print("=" * 70)
print("MECHANISTIC ESTIMATE VIA EFFECTIVE CELL DYNAMICS")
print("=" * 70)

# =============================================================================
# PART 1: Identify the dominant cell and its properties
# =============================================================================

print("\nPART 1: DOMINANT CELL IDENTIFICATION")
print("-" * 50)

# Collect final activation patterns for all pairs
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
                patterns[pattern] = []
            patterns[pattern].append((direction, m, s))

# Find dominant cell
dominant_pattern = max(patterns.keys(), key=lambda p: len(patterns[p]))
dominant_cases = patterns[dominant_pattern]
active_neurons = [n for n, a in enumerate(dominant_pattern) if a]

print(f"Dominant cell coverage: {len(dominant_cases)}/180 = {100*len(dominant_cases)/180:.1f}%")
print(f"Active neurons in dominant cell: {active_neurons}")
print(f"Dead neurons in dominant cell: {[n for n in range(16) if n not in active_neurons]}")

# =============================================================================
# PART 2: Compute effective dynamics within dominant cell
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: EFFECTIVE DYNAMICS WITHIN DOMINANT CELL")
print("=" * 70)

# Diagonal mask for active neurons
D = np.diag([1.0 if n in active_neurons else 0.0 for n in range(16)])

# Effective transition matrix within cell
W_hh_eff = D @ W_hh @ D

# Effective input projection
W_ih_eff = D @ W_ih

print(f"\nEffective W_hh is {len(active_neurons)}×{len(active_neurons)} (projected to active subspace)")

# Eigenvalues of effective dynamics
ev_eff = np.linalg.eigvals(W_hh_eff)
ev_eff_sorted = sorted(ev_eff, key=lambda x: -abs(x))

print("\nEffective W_hh eigenvalues (within dominant cell):")
for i, ev in enumerate(ev_eff_sorted[:6]):
    if abs(ev) > 0.01:
        print(f"  λ_{i}: {ev:.4f}, |λ| = {abs(ev):.4f}")

max_ev_eff = max(abs(ev) for ev in ev_eff)
print(f"\nMax |eigenvalue| = {max_ev_eff:.4f}")
print(f"9-step growth factor = {max_ev_eff**9:.2f}")
print("(Compare to full W_hh: 1.27^9 = 8.9)")

# =============================================================================
# PART 3: Derive f(pos) using effective dynamics
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: DERIVING f(pos) FROM EFFECTIVE DYNAMICS")
print("=" * 70)

# Within the dominant cell, the dynamics are approximately:
# h[t+1] = D @ ReLU(W_ih * x[t] + W_hh @ h[t])
#        ≈ D @ (W_ih * x[t] + W_hh @ h[t])   [when staying in cell]
#        = W_ih_eff * x[t] + W_hh_eff @ h[t]

# For an impulse at position pos, propagating within the cell:
# h[pos] = W_ih_eff * magnitude
# h[pos+1] = W_hh_eff @ h[pos]
# h[9] = W_hh_eff^(9-pos) @ W_ih_eff * magnitude

def effective_impulse_response(pos, magnitude=1.0):
    """Impulse response using effective (cell-specific) dynamics."""
    h = W_ih_eff * magnitude
    for _ in range(9 - pos):
        h = W_hh_eff @ h
    return h

# Compute effective f(pos)
F_effective = np.zeros((10, 16))
for pos in range(10):
    F_effective[pos] = effective_impulse_response(pos, M_val - S_val)

print("\nEffective f(pos) = W_hh_eff^(9-pos) @ W_ih_eff × (M - S)")
print("\n||f_eff(pos)|| by position:")
for pos in range(10):
    print(f"  pos={pos}: ||f_eff|| = {np.linalg.norm(F_effective[pos]):.3f}")

# =============================================================================
# PART 4: Compare effective f(pos) to empirical f(pos)
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: COMPARING EFFECTIVE vs EMPIRICAL f(pos)")
print("=" * 70)

# Get empirical f(pos) by fitting separable model
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
m_positions = np.array(m_positions)
s_positions = np.array(s_positions)

# Fit separable model
n_pairs = len(offsets)
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

# Compare
print("\nPosition | ||f_emp|| | ||f_eff|| | cos(f_emp, f_eff) | Ratio")
print("---------|-----------|-----------|-------------------|-------")

cosines = []
for pos in range(10):
    f_emp = F_empirical[pos]
    f_eff = F_effective[pos]

    norm_emp = np.linalg.norm(f_emp)
    norm_eff = np.linalg.norm(f_eff)

    if norm_emp > 1e-10 and norm_eff > 1e-10:
        cos = np.dot(f_emp, f_eff) / (norm_emp * norm_eff)
        ratio = norm_emp / norm_eff
    else:
        cos = float('nan')
        ratio = float('nan')
    cosines.append(cos)

    print(f"   {pos}     |   {norm_emp:6.3f}  |   {norm_eff:6.3f}  |      {cos:+.4f}       | {ratio:.3f}")

print(f"\nMean cosine similarity: {np.nanmean(cosines):.4f}")

# =============================================================================
# PART 5: Test discrimination using effective dynamics
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: DISCRIMINATION FROM EFFECTIVE DYNAMICS")
print("=" * 70)

# Predict offset using effective f
# offset_eff(m, s) = f_eff(m) - f_eff(s)

correct_eff = 0
correct_emp = 0
margins_eff = []
margins_emp = []

for i, (m, s) in enumerate(zip(m_positions, s_positions)):
    # Effective prediction
    offset_eff = F_effective[m] - F_effective[s]
    logits_eff = W_out @ offset_eff
    margin_eff = logits_eff[s] - logits_eff[m]
    margins_eff.append(margin_eff)
    if margin_eff > 0:
        correct_eff += 1

    # Empirical
    offset_emp = offsets[i]
    logits_emp = W_out @ offset_emp
    margin_emp = logits_emp[s] - logits_emp[m]
    margins_emp.append(margin_emp)
    if margin_emp > 0:
        correct_emp += 1

print(f"\nEffective dynamics discrimination: {correct_eff}/{n_pairs} ({100*correct_eff/n_pairs:.1f}%)")
print(f"Empirical discrimination: {correct_emp}/{n_pairs} ({100*correct_emp/n_pairs:.1f}%)")

print(f"\nMargins (s_logit - m_logit):")
print(f"  Effective: mean={np.mean(margins_eff):.2f}, min={np.min(margins_eff):.2f}, max={np.max(margins_eff):.2f}")
print(f"  Empirical: mean={np.mean(margins_emp):.2f}, min={np.min(margins_emp):.2f}, max={np.max(margins_emp):.2f}")

# Show worst cases if any fail
if correct_eff < n_pairs:
    print("\nFailing cases for effective dynamics:")
    failing = [(m_positions[i], s_positions[i], margins_eff[i], margins_emp[i])
               for i in range(n_pairs) if margins_eff[i] <= 0]
    for m, s, m_eff, m_emp in failing[:10]:
        print(f"  (M={m}, S={s}): effective margin={m_eff:.2f}, empirical margin={m_emp:.2f}")

# =============================================================================
# PART 6: If effective dynamics work, formalize the mechanistic proof
# =============================================================================

if correct_eff == n_pairs:
    print("\n" + "=" * 70)
    print("SUCCESS: EFFECTIVE DYNAMICS ACHIEVE 100% DISCRIMINATION")
    print("=" * 70)

    print("""
MECHANISTIC ACCURACY PROOF (semi-formal):

Given:
  - W_ih, W_hh, W_out (model weights)
  - Dominant cell activation mask D (13 active neurons)
  - Effective dynamics: W_hh_eff = D @ W_hh @ D, W_ih_eff = D @ W_ih

Derived (from weights + cell mask):
  - f_eff(pos) = W_hh_eff^(9-pos) @ W_ih_eff × (M_val - S_val)
  - offset_eff(m, s) = f_eff(m) - f_eff(s)

Claim: For all (m, s) pairs with m ≠ s:
  margin(m, s) = [W_out @ offset_eff(m, s)][s] - [W_out @ offset_eff(m, s)][m] > 0

Verification: Computed from weights, all 90 margins are positive.

CAVEAT: The cell mask D is determined empirically (by running the model).
A FULLY mechanistic proof would derive D from weights alone.
""")

    # Show the margin computation explicitly
    print("\nMechanistic margin computation from weights:")
    print("=" * 60)

    # Precompute W_hh_eff powers
    W_hh_eff_powers = [np.eye(16)]
    for k in range(1, 10):
        W_hh_eff_powers.append(W_hh_eff @ W_hh_eff_powers[-1])

    print("\nAll 90 mechanistic margins (computed from weights + cell mask):")
    print("(m, s) | margin | status")
    print("-------|--------|--------")

    all_margins = []
    for m in range(10):
        for s in range(10):
            if m == s:
                continue

            # Pure weight computation
            f_m = W_hh_eff_powers[9-m] @ W_ih_eff * (M_val - S_val)
            f_s = W_hh_eff_powers[9-s] @ W_ih_eff * (M_val - S_val)
            offset = f_m - f_s
            logits = W_out @ offset
            margin = logits[s] - logits[m]
            all_margins.append(margin)

            status = "✓" if margin > 0 else "✗"
            print(f"({m}, {s}) | {margin:+6.2f} | {status}")

    print(f"\nMinimum margin: {min(all_margins):.4f}")
    print(f"All margins positive: {all(m > 0 for m in all_margins)}")

else:
    print("\n" + "=" * 70)
    print("EFFECTIVE DYNAMICS INSUFFICIENT - NEED REFINEMENT")
    print("=" * 70)

    # Analyze why some cases fail
    print("\nAnalyzing failure cases...")

    for i, (m, s) in enumerate(zip(m_positions, s_positions)):
        if margins_eff[i] <= 0:
            # Check if this pair is in the dominant cell
            h_fwd = run_relu([(m, M_val), (s, S_val)])
            h_rev = run_relu([(m, S_val), (s, M_val)])

            pattern_fwd = tuple(h_fwd > 1e-10)
            pattern_rev = tuple(h_rev > 1e-10)

            in_dom_fwd = pattern_fwd == dominant_pattern
            in_dom_rev = pattern_rev == dominant_pattern

            print(f"\n  (M={m}, S={s}):")
            print(f"    Effective margin: {margins_eff[i]:.2f}")
            print(f"    Empirical margin: {margins_emp[i]:.2f}")
            print(f"    In dominant cell (fwd): {in_dom_fwd}")
            print(f"    In dominant cell (rev): {in_dom_rev}")

            if not in_dom_fwd or not in_dom_rev:
                # Different cell - effective dynamics don't apply
                active_fwd = [n for n, a in enumerate(pattern_fwd) if a]
                active_rev = [n for n, a in enumerate(pattern_rev) if a]
                print(f"    Active neurons (fwd): {active_fwd}")
                print(f"    Active neurons (rev): {active_rev}")

# =============================================================================
# PART 7: Attempt to derive the cell mask from weights
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: CAN WE DERIVE THE CELL MASK FROM WEIGHTS?")
print("=" * 70)

# The dominant cell has neurons [0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15] active
# Dead neurons: [4, 5, 9]

# What's special about these neurons?
dead_neurons = [n for n in range(16) if n not in active_neurons]

print(f"\nDead neurons in dominant cell: {dead_neurons}")
print("\nProperties of dead neurons:")

for n in dead_neurons:
    print(f"\n  Neuron {n}:")
    print(f"    W_ih[{n}] = {W_ih[n]:+.4f}")
    print(f"    Self-recurrence W_hh[{n},{n}] = {W_hh[n,n]:+.4f}")
    print(f"    ||W_out[:, {n}]|| = {np.linalg.norm(W_out[:, n]):.4f}")

    # What inputs activate this neuron?
    incoming = W_hh[n, :]
    top_inputs = sorted(range(16), key=lambda j: -abs(incoming[j]))[:3]
    print(f"    Top incoming weights: {[(j, f'{incoming[j]:+.2f}') for j in top_inputs]}")

print("""
OBSERVATION:
- n4: High input sensitivity (+10.16) but negative self-loop (-0.99) → fires once then dies
- n5: Receives from waves/bridges but often inactive
- n9: Receives from n4 but only fires in specific conditions

These neurons are "transient" by design - they fire during input processing
but are typically dead by t=9.

A FULLY MECHANISTIC derivation would need to:
1. Prove that for typical (m,s) pairs, the trajectory converges to the dominant cell
2. Show this from the W_hh stability properties
""")
