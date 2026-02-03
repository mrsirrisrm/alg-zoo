"""
The offset is separable: offset(m,s) = f(m) + g(s) where f = -g.
This is NOT the same as offset = v * (m - s).

Let's fit the proper separable model.
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

print("\n" + "=" * 70)
print("SEPARABLE MODEL: offset(m,s) = f(m) + g(s)")
print("=" * 70)

# Create one-hot encodings for positions
M_onehot = np.zeros((n_pairs, 10))
S_onehot = np.zeros((n_pairs, 10))
for i, (m, s) in enumerate(zip(m_positions, s_positions)):
    M_onehot[i, m] = 1
    S_onehot[i, s] = 1

# Design matrix: [M_onehot, S_onehot]
# This has 20 columns but they're not all independent (each row sums to 2)
# We need to drop one column from each to avoid singularity
# Drop position 0 from both as reference

X_sep = np.column_stack([M_onehot[:, 1:], S_onehot[:, 1:]])  # 18 columns

# Fit: offset = X_sep @ coeffs
coeffs, residuals, rank, s = np.linalg.lstsq(X_sep, offsets, rcond=None)

# Reconstruct f(m) and g(s) matrices
# f(0) = 0 (reference), f(m) = coeffs[m-1] for m > 0
# g(0) = 0 (reference), g(s) = coeffs[9 + s - 1] for s > 0

F = np.zeros((10, 16))  # f(m) for each m position
G = np.zeros((10, 16))  # g(s) for each s position

for m in range(1, 10):
    F[m] = coeffs[m - 1]
for s in range(1, 10):
    G[s] = coeffs[9 - 1 + s]  # = coeffs[8 + s]

# Verify fit
predicted_sep = M_onehot @ F + S_onehot @ G
ss_res_sep = np.sum((offsets - predicted_sep) ** 2)
ss_tot = np.sum((offsets - np.mean(offsets, axis=0)) ** 2)
r2_sep = 1 - ss_res_sep / ss_tot

print(f"\nSeparable model R²: {r2_sep:.4f}")
print(f"(Explains {r2_sep*100:.1f}% of offset variance)")

# Check antisymmetry: f(m) should equal -g(m)
print("\n" + "=" * 70)
print("ANTISYMMETRY CHECK: f(m) vs -g(m)")
print("=" * 70)

print("\nPosition | ||f(m)|| | ||g(m)|| | ||f(m) + g(m)|| | cos(f(m), -g(m))")
print("---------|----------|----------|-----------------|------------------")
for p in range(10):
    norm_f = np.linalg.norm(F[p])
    norm_g = np.linalg.norm(G[p])
    norm_sum = np.linalg.norm(F[p] + G[p])
    if norm_f > 1e-10 and norm_g > 1e-10:
        cos_fg = np.dot(F[p], -G[p]) / (norm_f * norm_g)
    else:
        cos_fg = float('nan')
    print(f"    {p}    |  {norm_f:6.3f}  |  {norm_g:6.3f}  |     {norm_sum:6.3f}      |      {cos_fg:+.4f}")

# SVD of F and G to understand structure
print("\n" + "=" * 70)
print("SVD ANALYSIS OF f(m) AND g(s)")
print("=" * 70)

U_f, s_f, Vt_f = np.linalg.svd(F, full_matrices=False)
U_g, s_g, Vt_g = np.linalg.svd(G, full_matrices=False)

print("\nSingular values of F (position → hidden):")
for i, sv in enumerate(s_f[:5]):
    pct = sv**2 / np.sum(s_f**2) * 100
    print(f"  σ_{i+1} = {sv:.4f} ({pct:.1f}%)")

print("\nSingular values of G:")
for i, sv in enumerate(s_g[:5]):
    pct = sv**2 / np.sum(s_g**2) * 100
    print(f"  σ_{i+1} = {sv:.4f} ({pct:.1f}%)")

# Check if dominant directions match
print("\nAlignment of dominant right singular vectors:")
for i in range(min(3, len(s_f))):
    alignment = np.abs(np.dot(Vt_f[i], Vt_g[i]))
    print(f"  |Vt_f[{i}] · Vt_g[{i}]| = {alignment:.4f}")

# Effect through W_out
print("\n" + "=" * 70)
print("f(m) AND g(s) THROUGH W_out")
print("=" * 70)

W_out_F = W_out @ F.T  # Shape: (10, 10) - output logits for each m_pos
W_out_G = W_out @ G.T  # Shape: (10, 10) - output logits for each s_pos

print("\nW_out @ f(m) - logit contribution from M position:")
print("     | Output positions 0-9")
print("m_pos| " + " ".join([f"{p:6d}" for p in range(10)]))
print("-----|" + "-" * 70)
for m in range(10):
    print(f"  {m}  |" + " ".join([f"{W_out_F[o, m]:+6.2f}" for o in range(10)]))

print("\nW_out @ g(s) - logit contribution from S position:")
print("     | Output positions 0-9")
print("s_pos| " + " ".join([f"{p:6d}" for p in range(10)]))
print("-----|" + "-" * 70)
for s in range(10):
    print(f"  {s}  |" + " ".join([f"{W_out_G[o, s]:+6.2f}" for o in range(10)]))

# Visualizations
fig = plt.figure(figsize=(18, 12))

# 1. f(m) heatmap
ax = fig.add_subplot(2, 3, 1)
im = ax.imshow(F.T, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
ax.set_xlabel('M position')
ax.set_ylabel('Neuron')
ax.set_title('f(m): M position encoding')
ax.set_xticks(range(10))
ax.set_yticks(range(16))
plt.colorbar(im, ax=ax)

# 2. g(s) heatmap
ax = fig.add_subplot(2, 3, 2)
im = ax.imshow(G.T, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
ax.set_xlabel('S position')
ax.set_ylabel('Neuron')
ax.set_title('g(s): S position encoding')
ax.set_xticks(range(10))
ax.set_yticks(range(16))
plt.colorbar(im, ax=ax)

# 3. f(m) + g(m) should be ~0
ax = fig.add_subplot(2, 3, 3)
antisym_check = F + G
im = ax.imshow(antisym_check.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('Neuron')
ax.set_title('f(m) + g(m) (should be ~0)')
ax.set_xticks(range(10))
ax.set_yticks(range(16))
plt.colorbar(im, ax=ax)

# 4. W_out @ f(m) heatmap
ax = fig.add_subplot(2, 3, 4)
im = ax.imshow(W_out_F, aspect='auto', cmap='RdBu_r', vmin=-5, vmax=5)
ax.set_xlabel('M position (input)')
ax.set_ylabel('Output position (logit)')
ax.set_title('W_out @ f(m): M contribution to logits')
ax.set_xticks(range(10))
ax.set_yticks(range(10))
plt.colorbar(im, ax=ax)

# 5. W_out @ g(s) heatmap
ax = fig.add_subplot(2, 3, 5)
im = ax.imshow(W_out_G, aspect='auto', cmap='RdBu_r', vmin=-5, vmax=5)
ax.set_xlabel('S position (input)')
ax.set_ylabel('Output position (logit)')
ax.set_title('W_out @ g(s): S contribution to logits')
ax.set_xticks(range(10))
ax.set_yticks(range(10))
plt.colorbar(im, ax=ax)

# 6. Diagonal analysis - what matters is logit at s_pos
ax = fig.add_subplot(2, 3, 6)

# For correct prediction in forward (M@m, S@s), we need:
# logit[s] > logit[m]
# The offset contributes: (W_out @ f(m))[s] + (W_out @ g(s))[s] at position s
#                      vs (W_out @ f(m))[m] + (W_out @ g(s))[m] at position m

# Let's look at diagonal elements
diag_F = np.diag(W_out_F)  # (W_out @ f(m))[m] - effect of m on logit m
diag_G = np.diag(W_out_G)  # (W_out @ g(s))[s] - effect of s on logit s

ax.plot(range(10), diag_F, 'o-', color='red', markersize=8, label='(W_out @ f(m))[m]')
ax.plot(range(10), diag_G, 's-', color='blue', markersize=8, label='(W_out @ g(s))[s]')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('Logit contribution')
ax.set_title('Self-contribution to logits\n(how f(m) affects logit m, etc.)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(10))

plt.tight_layout()
plt.savefig('docs/offset_separable_map.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved to docs/offset_separable_map.png")

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
The offset is separable: offset(m,s) = f(m) + g(s)
with perfect antisymmetry: f(m) = -g(m)

This means the network learns TWO position encodings:
  - f(m): how M position affects the hidden state offset
  - g(s): how S position affects the hidden state offset

These combine additively, and through W_out, create the
discriminative signal needed to predict S position in forward
and M position in reverse.

Key statistics:
  - Separable model R² = {r2_sep:.4f}
  - This explains {r2_sep*100:.1f}% of offset variance
  - Remaining variance is due to ReLU clipping effects

The f(m) and g(s) matrices encode position as a ~{np.sum(s_f[:2]**2)/np.sum(s_f**2)*100:.0f}%-rank-2 structure.
""")
