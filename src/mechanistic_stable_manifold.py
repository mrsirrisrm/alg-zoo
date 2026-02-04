"""
MECHANISTIC ESTIMATE VIA STABLE MANIFOLD APPROXIMATION

Hypothesis: The network operates on a low-dimensional stable manifold.
ReLU acts approximately as projection onto this manifold.

If we can characterize this manifold from weights, we can derive f(·).
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

print("=" * 70)
print("MECHANISTIC ESTIMATE VIA STABLE MANIFOLD")
print("=" * 70)

# =============================================================================
# PART 1: Characterize the stable manifold from final hidden states
# =============================================================================

print("\nPART 1: STABLE MANIFOLD CHARACTERIZATION")
print("-" * 50)

# Collect all final hidden states
all_h = []
for m in range(10):
    for s in range(10):
        if m == s:
            continue
        h_fwd = run_relu([(m, M_val), (s, S_val)])
        h_rev = run_relu([(m, S_val), (s, M_val)])
        all_h.append(h_fwd)
        all_h.append(h_rev)

all_h = np.array(all_h)

# PCA to find the manifold
mean_h = np.mean(all_h, axis=0)
centered = all_h - mean_h
U, S_vals, Vt = np.linalg.svd(centered, full_matrices=False)

print(f"Mean ||h_final|| = {np.linalg.norm(mean_h):.2f}")
print(f"\nPCA of final hidden states (180 samples):")

var_explained = S_vals**2 / np.sum(S_vals**2)
cumvar = np.cumsum(var_explained)
for i in range(min(6, len(S_vals))):
    print(f"  PC{i+1}: {var_explained[i]*100:.1f}% (cumulative: {cumvar[i]*100:.1f}%)")

# The manifold is approximately the span of top k PCs
n_pcs_90 = np.searchsorted(cumvar, 0.90) + 1
print(f"\nPCs for 90% variance: {n_pcs_90}")

# Projection matrix onto the manifold
P_manifold = Vt[:n_pcs_90].T @ Vt[:n_pcs_90]  # Project to top PCs and back

print(f"\nManifold projection matrix P has rank {np.linalg.matrix_rank(P_manifold)}")

# =============================================================================
# PART 2: Test if ReLU ≈ Projection onto manifold
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: IS ReLU ≈ PROJECTION ONTO MANIFOLD?")
print("=" * 70)

# For each hidden state, compare:
# 1. ReLU(h) - the actual nonlinearity
# 2. P @ h - projection onto manifold
# 3. max(0, P @ h) - ReLU after projection

# Actually, we should test this during dynamics, not just final state
# Let's trace a trajectory and see if the manifold captures it

def trace_with_projection(impulses, P):
    """Compare actual ReLU trajectory with manifold-projected trajectory."""
    h_actual = np.zeros(16)
    h_proj = np.zeros(16)

    errors = []

    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)

        # Actual dynamics
        pre_actual = W_ih * x_t + W_hh @ h_actual
        h_actual = np.maximum(0, pre_actual)

        # Projected dynamics: project pre-activation, then ReLU
        pre_proj = W_ih * x_t + W_hh @ h_proj
        pre_proj_manifold = P @ pre_proj  # Stay on manifold
        h_proj = np.maximum(0, pre_proj_manifold)

        # Error
        err = np.linalg.norm(h_actual - h_proj) / (np.linalg.norm(h_actual) + 1e-10)
        errors.append(err)

    return h_actual, h_proj, errors

# Test on a few pairs
print("\nComparing actual vs manifold-projected trajectories:")
print("(m, s) | Final ||actual|| | Final ||proj|| | Final rel error | Traj mean err")
print("-------|------------------|-----------------|-----------------|---------------")

for m, s in [(0, 5), (2, 7), (4, 6), (3, 8)]:
    h_act, h_proj, errs = trace_with_projection([(m, M_val), (s, S_val)], P_manifold)
    rel_err = np.linalg.norm(h_act - h_proj) / np.linalg.norm(h_act)
    mean_err = np.mean(errs[1:])  # Skip t=0 which is trivially 0

    print(f"({m}, {s}) |      {np.linalg.norm(h_act):6.2f}      |      {np.linalg.norm(h_proj):6.2f}      |      {rel_err:.4f}      |     {mean_err:.4f}")

# =============================================================================
# PART 3: Derive f from the manifold structure
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: DERIVING f(·) FROM MANIFOLD STRUCTURE")
print("=" * 70)

# On the manifold, the effective dynamics are:
# h[t+1] = P @ max(0, W_hh @ h[t] + W_ih * x[t])
#
# If ReLU acts as identity on the manifold (all components positive),
# then: h[t+1] ≈ P @ W_hh @ h[t] + P @ W_ih * x[t]
#
# Define effective matrices:
# W_hh_M = P @ W_hh @ P  (manifold-to-manifold transition)
# W_ih_M = P @ W_ih      (input-to-manifold injection)

W_hh_M = P_manifold @ W_hh @ P_manifold
W_ih_M = P_manifold @ W_ih

print("Effective manifold dynamics:")
print(f"  W_hh_M = P @ W_hh @ P (rank {np.linalg.matrix_rank(W_hh_M)})")
print(f"  W_ih_M = P @ W_ih")

# Eigenvalues of W_hh_M
ev_M = np.linalg.eigvals(W_hh_M)
ev_M_sorted = sorted(ev_M, key=lambda x: -abs(x))

print("\nW_hh_M eigenvalues:")
for i, ev in enumerate(ev_M_sorted[:5]):
    if abs(ev) > 0.01:
        print(f"  λ_{i}: {ev:.4f}, |λ| = {abs(ev):.4f}")

max_ev_M = max(abs(ev) for ev in ev_M)
print(f"\nMax |eigenvalue| of W_hh_M = {max_ev_M:.4f}")
print(f"9-step growth factor = {max_ev_M**9:.2f}")

# =============================================================================
# PART 4: Compute manifold-based f(pos)
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: MANIFOLD-BASED f(pos)")
print("=" * 70)

# On the manifold, f(pos) = W_hh_M^(9-pos) @ W_ih_M × (M - S)

def manifold_f(pos, delta_mag=0.2):
    """Compute f(pos) using manifold dynamics."""
    h = W_ih_M * delta_mag
    for _ in range(9 - pos):
        h = W_hh_M @ h
    return h

F_manifold = np.zeros((10, 16))
for pos in range(10):
    F_manifold[pos] = manifold_f(pos, M_val - S_val)

print("||f_manifold(pos)||:")
for pos in range(10):
    print(f"  pos={pos}: ||f_M|| = {np.linalg.norm(F_manifold[pos]):.4f}")

# Compare to empirical f
print("\n" + "=" * 70)
print("PART 5: COMPARISON WITH EMPIRICAL f(·)")
print("=" * 70)

# Get empirical f
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

print("\nComparison: f_empirical vs f_manifold")
print("pos | ||f_emp|| | ||f_man|| | cos(f_emp, f_man) | ||diff||")
print("----|-----------|-----------|-------------------|----------")

cosines = []
for pos in range(10):
    f_emp = F_empirical[pos]
    f_man = F_manifold[pos]

    norm_emp = np.linalg.norm(f_emp)
    norm_man = np.linalg.norm(f_man)

    if norm_emp > 0.01 and norm_man > 0.01:
        cos = np.dot(f_emp, f_man) / (norm_emp * norm_man)
    else:
        cos = float('nan')
    cosines.append(cos)

    diff = np.linalg.norm(f_emp - f_man)
    print(f" {pos}  |   {norm_emp:6.3f}  |   {norm_man:6.3f}  |      {cos:+.4f}       |  {diff:6.3f}")

print(f"\nMean cosine similarity: {np.nanmean(cosines):.4f}")

# =============================================================================
# PART 6: TEST DISCRIMINATION
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: DISCRIMINATION TEST")
print("=" * 70)

correct_manifold = 0
correct_empirical = 0
margins_manifold = []
margins_empirical = []

for i, (m, s) in enumerate(zip(m_positions, s_positions)):
    # Manifold-based offset
    offset_man = F_manifold[m] - F_manifold[s]
    logits_man = W_out @ offset_man
    margin_man = logits_man[s] - logits_man[m]
    margins_manifold.append(margin_man)
    if margin_man > 0:
        correct_manifold += 1

    # Empirical offset
    offset_emp = offsets[i]
    logits_emp = W_out @ offset_emp
    margin_emp = logits_emp[s] - logits_emp[m]
    margins_empirical.append(margin_emp)
    if margin_emp > 0:
        correct_empirical += 1

print(f"\nManifold-based discrimination: {correct_manifold}/{n_pairs} ({100*correct_manifold/n_pairs:.1f}%)")
print(f"Empirical discrimination: {correct_empirical}/{n_pairs} ({100*correct_empirical/n_pairs:.1f}%)")

print(f"\nMargins (s_logit - m_logit):")
print(f"  Manifold:  mean={np.mean(margins_manifold):.2f}, min={np.min(margins_manifold):.2f}, max={np.max(margins_manifold):.2f}")
print(f"  Empirical: mean={np.mean(margins_empirical):.2f}, min={np.min(margins_empirical):.2f}, max={np.max(margins_empirical):.2f}")

if correct_manifold < n_pairs:
    print("\nFailing cases:")
    failing = [(m_positions[i], s_positions[i], margins_manifold[i], margins_empirical[i])
               for i in range(n_pairs) if margins_manifold[i] <= 0]
    for m, s, m_man, m_emp in sorted(failing, key=lambda x: x[2])[:10]:
        print(f"  (M={m}, S={s}): manifold margin={m_man:+.2f}, empirical margin={m_emp:+.2f}")

# =============================================================================
# PART 7: ALTERNATIVE - Use mean hidden state as reference
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: ALTERNATIVE - DEVIATION FROM MEAN APPROACH")
print("=" * 70)

# Alternative hypothesis: f(pos) encodes deviation from mean trajectory
# f(pos) = E[h | M at pos] - E[h]

# We already know E[h | M at pos] empirically
# Let's see if we can derive it

# First, what determines E[h | M at pos]?
# It's the average over all s ≠ pos of run_relu([(pos, M), (s, S)])

# Simplification: if the dynamics are approximately linear after
# both impulses arrive, then:
# h_final ≈ W_hh^(9-max(m,s)) @ h[max(m,s)]
#
# And h[max(m,s)] depends on both impulses

# Let's test if the mean effect is captured by W_out structure

print("\nTesting if W_out encodes position directly...")

# W_out row i is the readout for position i
# If f(pos) through W_out gives large value at pos, that would explain discrimination

W_out_F_emp = W_out @ F_empirical.T  # (10 output positions × 10 M positions)

print("\nW_out @ f_emp[m_pos] for each output position:")
print("     | Output positions 0-9")
print("m_pos|" + " ".join([f"{p:6d}" for p in range(10)]))
print("-----|" + "-" * 70)
for m in range(10):
    row = " ".join([f"{W_out_F_emp[o, m]:+6.2f}" for o in range(10)])
    print(f"  {m}  |{row}")

# Key insight: For discrimination, we need W_out @ (f(m) - f(s)) to be
# positive at s and negative at m

print("\nDiagonal values W_out @ f[pos] at pos (self-contribution):")
diag = np.diag(W_out_F_emp)
for pos in range(10):
    print(f"  pos={pos}: {diag[pos]:+.2f}")

print("""
OBSERVATION:
W_out @ f[pos] shows a pattern: negative for early positions (0-3),
positive for late positions (4-9).

This means:
- f(m) with m early → W_out gives negative bias to early positions
- f(m) with m late → W_out gives positive bias to late positions

For discrimination (m, s):
- offset = f(m) - f(s)
- If m < s (forward): offset creates pattern that favors s over m
- If m > s (reverse): offset creates pattern that favors s over m (still!)

This is the antisymmetry at work.
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
MANIFOLD APPROACH RESULTS:
- Manifold projection (top {n_pcs_90} PCs) captures 90% of variance
- Effective W_hh_M max eigenvalue: {max_ev_M:.4f}
- Manifold-based discrimination: {correct_manifold}/90 ({100*correct_manifold/n_pairs:.1f}%)
- Mean cosine(f_emp, f_manifold): {np.nanmean(cosines):.4f}

The manifold approach also fails to achieve 100% discrimination.
The issue is that the manifold projection loses the discrimination signal.

KEY INSIGHT:
The discrimination information is NOT in the "linear" part of the dynamics.
It's encoded in WHERE ReLU clips, which depends on the full 16D state,
not just its projection onto the stable manifold.

NEXT STEPS:
1. Try fitting a nonlinear correction: f_pred = f_manifold + ReLU_correction
2. Use bounds: show that despite approximation errors, margins > 0
3. Full tropical enumeration (exact but complex)
""")
