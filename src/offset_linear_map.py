"""
Characterize the linear map that produces the offset.
Since offset is ~88% linear, we can fit: offset = A @ [m_pos, s_pos]^T + noise
"""

import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax

model = example_2nd_argmax()
model.eval()

W_ih = model.rnn.weight_ih_l0.detach().numpy().squeeze()
W_hh = model.rnn.weight_hh_l0.detach().numpy()
W_out = model.linear.weight.detach().numpy()

M_val = 1.0
S_val = 0.8

def run_stepwise(impulses):
    h = np.zeros(16)
    states = [h.copy()]
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        states.append(h.copy())
    return states

# Collect offsets
print("Collecting offsets...")
offsets = []
m_positions = []
s_positions = []

for pos1 in range(10):
    for pos2 in range(10):
        if pos1 == pos2:
            continue

        states_fwd = run_stepwise([(pos1, M_val), (pos2, S_val)])
        states_rev = run_stepwise([(pos1, S_val), (pos2, M_val)])

        offset = states_fwd[-1] - states_rev[-1]
        offsets.append(offset)
        m_positions.append(pos1)
        s_positions.append(pos2)

offsets = np.array(offsets)
m_positions = np.array(m_positions)
s_positions = np.array(s_positions)

n_pairs = len(offsets)
print(f"Collected {n_pairs} offsets")

# Fit linear model: offset = a * m_pos + b * s_pos + c
# We know from antisymmetry that a = -b, so offset = a * (m_pos - s_pos)
# But let's fit the full model first

print("\n" + "=" * 70)
print("LINEAR MODEL FIT")
print("=" * 70)

# Design matrix: [m_pos, s_pos, 1]
X = np.column_stack([m_positions, s_positions, np.ones(n_pairs)])

# Fit for all neurons at once using least squares
# offset = X @ B^T where B is (16, 3)
B, residuals, rank, s = np.linalg.lstsq(X, offsets, rcond=None)
B = B.T  # Now B is (16, 3): [coef_m, coef_s, intercept] for each neuron

print("\nFitted coefficients (offset[n] = a_n * m + b_n * s + c_n):")
print("Neuron |    a_n    |    b_n    |    c_n    | a_n + b_n (should be ~0)")
print("-------|-----------|-----------|-----------|-------------------------")
for n in range(16):
    a, b, c = B[n]
    print(f"  {n:2d}   | {a:+9.4f} | {b:+9.4f} | {c:+9.4f} | {a+b:+9.6f}")

# Verify antisymmetry: a = -b
print(f"\nMean |a + b|: {np.mean(np.abs(B[:, 0] + B[:, 1])):.6f}")
print("(Should be ~0 if perfectly antisymmetric)")

# Since a = -b, we can simplify: offset = a * (m - s) = a * gap_signed
# where gap_signed = m_pos - s_pos (positive when M is later)

print("\n" + "=" * 70)
print("SIMPLIFIED MODEL: offset = v * (m_pos - s_pos)")
print("=" * 70)

gap_signed = m_positions - s_positions

# Fit: offset = v * gap_signed
# This is rank-1: offset = v ⊗ gap_signed
v = offsets.T @ gap_signed / (gap_signed @ gap_signed)

print("\nOffset direction vector v (16D):")
print("(offset ≈ v * (m_pos - s_pos))")
for n in range(16):
    print(f"  v[{n:2d}] = {v[n]:+.4f}")

# Check fit quality
predicted = np.outer(gap_signed, v)
residual = offsets - predicted
ss_res = np.sum(residual ** 2)
ss_tot = np.sum((offsets - np.mean(offsets, axis=0)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"\nR² for rank-1 model: {r2:.4f}")
print(f"Residual norm: {np.sqrt(ss_res):.2f}")
print(f"Total norm: {np.sqrt(ss_tot):.2f}")

# What does v look like through W_out?
print("\n" + "=" * 70)
print("OFFSET DIRECTION THROUGH W_out")
print("=" * 70)

w_out_v = W_out @ v
print("\nW_out @ v (logit effect per unit gap):")
for pos in range(10):
    print(f"  Position {pos}: {w_out_v[pos]:+.4f}")

print(f"\nSum of W_out @ v: {np.sum(w_out_v):.6f}")
print("(Should be ~0 since logits should be zero-sum for discrimination)")

# Key insight: for correct discrimination, we need:
# - Positive logit at S position (to predict S in forward)
# - Negative logit at M position (to suppress M in forward)
# And the opposite for reverse (but offset sign flips)

print("\n" + "=" * 70)
print("DISCRIMINATION ANALYSIS")
print("=" * 70)

# For forward case (m_pos < s_pos typically), gap_signed < 0
# So offset = v * (negative number), meaning offset direction is -v
# We want: W_out[s_pos] @ (-v) > 0 and W_out[m_pos] @ (-v) < 0
# Equivalently: W_out[s_pos] @ v < 0 and W_out[m_pos] @ v > 0

# Let's check if this pattern holds
print("\nFor M at early position (0-4), S at late position (5-9):")
print("  We need W_out @ v to be POSITIVE for early positions")
print("  We need W_out @ v to be NEGATIVE for late positions")
print("\nActual W_out @ v:")
print(f"  Early (0-4): {w_out_v[:5]} → mean = {np.mean(w_out_v[:5]):+.3f}")
print(f"  Late (5-9):  {w_out_v[5:]} → mean = {np.mean(w_out_v[5:]):+.3f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Offset direction vector v
ax = axes[0, 0]
colors = ['green' if x > 0 else 'red' for x in v]
ax.bar(range(16), v, color=colors, edgecolor='black', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Neuron')
ax.set_ylabel('v[n]')
ax.set_title('Offset direction vector v\n(offset ≈ v × (m_pos - s_pos))')
ax.set_xticks(range(16))
ax.grid(True, alpha=0.3, axis='y')

# 2. W_out @ v
ax = axes[0, 1]
colors = ['green' if x > 0 else 'red' for x in w_out_v]
ax.bar(range(10), w_out_v, color=colors, edgecolor='black', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Output position')
ax.set_ylabel('(W_out @ v)[pos]')
ax.set_title('Logit effect of offset direction\n(per unit gap_signed)')
ax.set_xticks(range(10))
ax.grid(True, alpha=0.3, axis='y')

# 3. Predicted vs actual offset (sample neurons)
ax = axes[0, 2]
sample_neurons = [0, 8, 10, 12]  # Mix of positive and negative v
for n in sample_neurons:
    ax.scatter(gap_signed, offsets[:, n], s=30, alpha=0.6, label=f'n{n} actual')
    ax.plot([-9, 9], [-9 * v[n], 9 * v[n]], '--', alpha=0.8, label=f'n{n} fit')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('gap_signed (m_pos - s_pos)')
ax.set_ylabel('offset[n]')
ax.set_title('Offset vs gap_signed (sample neurons)')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# 4. R² by neuron for the rank-1 model
ax = axes[1, 0]
r2_by_neuron = []
for n in range(16):
    pred_n = gap_signed * v[n]
    ss_res_n = np.sum((offsets[:, n] - pred_n) ** 2)
    ss_tot_n = np.sum((offsets[:, n] - np.mean(offsets[:, n])) ** 2)
    r2_n = 1 - ss_res_n / ss_tot_n if ss_tot_n > 1e-10 else 1.0
    r2_by_neuron.append(r2_n)

ax.bar(range(16), r2_by_neuron, color='steelblue', edgecolor='black', alpha=0.7)
ax.axhline(np.mean(r2_by_neuron), color='red', linestyle='--', label=f'Mean R²={np.mean(r2_by_neuron):.3f}')
ax.set_xlabel('Neuron')
ax.set_ylabel('R²')
ax.set_title('Rank-1 model fit quality by neuron')
ax.set_xticks(range(16))
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 5. Residual structure - is there a second direction?
ax = axes[1, 1]

# Fit rank-2 model: offset = v1 * m_pos + v2 * s_pos
X2 = np.column_stack([m_positions, s_positions])
V2, _, _, _ = np.linalg.lstsq(X2, offsets, rcond=None)
v_m = V2[0]  # Direction for m_pos
v_s = V2[1]  # Direction for s_pos

# Check if v_m ≈ -v_s
cosine_sim = np.dot(v_m, v_s) / (np.linalg.norm(v_m) * np.linalg.norm(v_s))
print(f"\nRank-2 model:")
print(f"  ||v_m|| = {np.linalg.norm(v_m):.4f}")
print(f"  ||v_s|| = {np.linalg.norm(v_s):.4f}")
print(f"  cosine(v_m, v_s) = {cosine_sim:.4f}")
print(f"  (Should be -1 if v_m = -v_s)")

ax.scatter(v_m, v_s, c=range(16), cmap='viridis', s=100, edgecolors='black')
for n in range(16):
    ax.annotate(str(n), (v_m[n], v_s[n]), fontsize=8, ha='left')
ax.plot([-1, 1], [1, -1], 'r--', alpha=0.5, label='y = -x')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('v_m[n] (coefficient for m_pos)')
ax.set_ylabel('v_s[n] (coefficient for s_pos)')
ax.set_title('Rank-2 model: v_m vs v_s\n(should lie on y=-x)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 6. Summary
ax = axes[1, 2]
ax.axis('off')

summary = f"""OFFSET LINEAR MAP SUMMARY

The offset is well-approximated by:
    offset ≈ v × (m_pos - s_pos)

where v is a fixed 16D direction vector.

Key properties:

1. RANK-1 STRUCTURE
   R² = {r2:.4f} (explains {r2*100:.1f}% of variance)

2. ANTISYMMETRY VERIFIED
   v_m ≈ -v_s (cosine = {cosine_sim:.4f})

3. DISCRIMINATION MECHANISM
   W_out @ v creates position-dependent logit bias:
   - Early positions (0-4): mean = {np.mean(w_out_v[:5]):+.3f}
   - Late positions (5-9): mean = {np.mean(w_out_v[5:]):+.3f}

4. TOP CONTRIBUTING NEURONS
   |v[n]| ranking:
"""

sorted_by_v = sorted(range(16), key=lambda n: abs(v[n]), reverse=True)
for i, n in enumerate(sorted_by_v[:5]):
    summary += f"   {i+1}. Neuron {n}: v={v[n]:+.3f}\n"

ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        ha='left', va='top', fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('docs/offset_linear_map.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved to docs/offset_linear_map.png")

# Final: express v in terms of network weights
print("\n" + "=" * 70)
print("RELATIONSHIP TO NETWORK WEIGHTS")
print("=" * 70)

# The offset direction v should relate to how position information
# propagates through W_hh and interacts with W_out

# Check alignment of v with W_out rows
print("\nAlignment of v with W_out rows:")
v_norm = v / np.linalg.norm(v)
for pos in range(10):
    w_row = W_out[pos]
    w_norm = w_row / np.linalg.norm(w_row)
    alignment = np.dot(v_norm, w_norm)
    print(f"  cos(v, W_out[{pos}]) = {alignment:+.4f}")

# Check alignment with W_hh eigenvectors
print("\nAlignment of v with W_hh eigenvectors:")
eigenvalues, eigenvectors = np.linalg.eig(W_hh)
# Sort by eigenvalue magnitude
idx = np.argsort(np.abs(eigenvalues))[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

for i in range(5):
    ev = np.real(eigenvectors[:, i])
    ev_norm = ev / np.linalg.norm(ev)
    alignment = np.abs(np.dot(v_norm, ev_norm))
    print(f"  |cos(v, eigvec[{i}])| = {alignment:.4f} (eigenvalue = {eigenvalues[i]:.3f})")
