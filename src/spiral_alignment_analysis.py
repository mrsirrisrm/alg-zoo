"""
Compare the f(m) discrimination spiral with the main phase wheel spiral.
Are they in the same subspace? Orthogonal? Related?
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

# =============================================================================
# 1. Get the main phase wheel: final hidden states for all position pairs
# =============================================================================
print("Collecting main phase wheel states...")

final_states = []
metadata = []

for pos1 in range(10):
    for pos2 in range(10):
        if pos1 == pos2:
            continue
        # Forward
        states_fwd = run_stepwise([(pos1, M_val), (pos2, S_val)])
        final_states.append(states_fwd[-1])
        metadata.append({'m': pos1, 's': pos2, 'dir': 'fwd'})
        # Reverse
        states_rev = run_stepwise([(pos1, S_val), (pos2, M_val)])
        final_states.append(states_rev[-1])
        metadata.append({'m': pos2, 's': pos1, 'dir': 'rev'})

final_states = np.array(final_states)

# PCA of main phase wheel
pca_main = PCA()
pca_main.fit(final_states)

print(f"Main phase wheel: {len(final_states)} states")
print(f"Variance explained by PC1-2: {pca_main.explained_variance_ratio_[:2].sum():.1%}")

# =============================================================================
# 2. Get the f(m) discrimination encoding
# =============================================================================
print("\nComputing f(m) discrimination encoding...")

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

# Fit separable model to get F
M_onehot = np.zeros((n_pairs, 10))
S_onehot = np.zeros((n_pairs, 10))
for i, (m, s) in enumerate(zip(m_positions, s_positions)):
    M_onehot[i, m] = 1
    S_onehot[i, s] = 1

X_sep = np.column_stack([M_onehot[:, 1:], S_onehot[:, 1:]])
coeffs, _, _, _ = np.linalg.lstsq(X_sep, offsets, rcond=None)

F = np.zeros((10, 16))  # f(m) for each position
for m in range(1, 10):
    F[m] = coeffs[m - 1]

# PCA of f(m) encoding
pca_f = PCA()
pca_f.fit(F)

print(f"f(m) encoding: 10 vectors")
print(f"Variance explained by PC1-2: {pca_f.explained_variance_ratio_[:2].sum():.1%}")

# =============================================================================
# 3. Compare the two subspaces
# =============================================================================
print("\n" + "=" * 70)
print("SUBSPACE ALIGNMENT ANALYSIS")
print("=" * 70)

# Get dominant directions from each
main_pc1 = pca_main.components_[0]
main_pc2 = pca_main.components_[1]
f_pc1 = pca_f.components_[0]
f_pc2 = pca_f.components_[1]

# Cosine similarities
print("\nCosine similarity between principal components:")
print("            | Main PC1 | Main PC2 | Main PC3 | Main PC4")
print("------------|----------|----------|----------|----------")
for i, (name, vec) in enumerate([("f(m) PC1", f_pc1), ("f(m) PC2", f_pc2)]):
    sims = []
    for j in range(4):
        main_pc = pca_main.components_[j]
        sim = np.abs(np.dot(vec, main_pc) / (np.linalg.norm(vec) * np.linalg.norm(main_pc)))
        sims.append(sim)
    print(f"{name:11s} |  {sims[0]:.4f}  |  {sims[1]:.4f}  |  {sims[2]:.4f}  |  {sims[3]:.4f}")

# Subspace overlap: what fraction of f(m) variance lies in main PC space?
print("\n" + "-" * 70)
print("Subspace overlap:")
print("-" * 70)

# Project F onto main PCA space
F_in_main_pca = pca_main.transform(F)

# Variance of F captured by first k main PCs
total_f_var = np.sum(np.var(F, axis=0))
for k in [2, 4, 6, 8]:
    var_in_k = np.sum(np.var(F_in_main_pca[:, :k], axis=0))
    # Need to back-transform to get actual variance
    F_reconstructed_k = pca_main.inverse_transform(
        np.column_stack([F_in_main_pca[:, :k], np.zeros((10, 16-k))])
    )
    var_captured = np.sum(np.var(F_reconstructed_k, axis=0))
    pct = var_captured / total_f_var * 100
    print(f"  f(m) variance in Main PC1-{k}: {pct:.1f}%")

# Orthogonality check: is f(m) subspace orthogonal to main trajectory?
print("\n" + "-" * 70)
print("Are the spirals orthogonal?")
print("-" * 70)

# The main spiral lives primarily in PC1-PC2 of the main PCA
# The f(m) spiral lives primarily in PC1-PC2 of the f PCA
# Check if these 2D subspaces are orthogonal

# Compute principal angles between subspaces
from scipy.linalg import subspace_angles

main_subspace = pca_main.components_[:2].T  # 16 x 2
f_subspace = pca_f.components_[:2].T  # 16 x 2

angles = subspace_angles(main_subspace, f_subspace)
print(f"\nPrincipal angles between 2D subspaces:")
print(f"  Angle 1: {np.degrees(angles[0]):.1f}°")
print(f"  Angle 2: {np.degrees(angles[1]):.1f}°")
print(f"\n  (0° = aligned, 90° = orthogonal)")

# =============================================================================
# 4. Visualizations
# =============================================================================
fig = plt.figure(figsize=(16, 12))

# 1. Main phase wheel in its own PC space
ax = fig.add_subplot(2, 3, 1)
main_pca_coords = pca_main.transform(final_states)
s_positions_all = np.array([m['s'] for m in metadata])
scatter = ax.scatter(main_pca_coords[:, 0], main_pca_coords[:, 1],
                     c=s_positions_all, cmap='viridis', s=30, alpha=0.6)
ax.set_xlabel('Main PC1')
ax.set_ylabel('Main PC2')
ax.set_title(f'Main phase wheel\n(PC1+PC2 = {pca_main.explained_variance_ratio_[:2].sum():.1%} var)')
plt.colorbar(scatter, ax=ax, label='S position')
ax.grid(True, alpha=0.3)

# 2. f(m) spiral in its own PC space
ax = fig.add_subplot(2, 3, 2)
f_pca_coords = pca_f.transform(F)
ax.plot(f_pca_coords[:, 0], f_pca_coords[:, 1], 'k-', linewidth=1, alpha=0.5)
scatter = ax.scatter(f_pca_coords[:, 0], f_pca_coords[:, 1],
                     c=range(10), cmap='viridis', s=150, edgecolors='black', zorder=5)
for p in range(10):
    ax.annotate(str(p), (f_pca_coords[p, 0], f_pca_coords[p, 1]),
                fontsize=10, ha='center', va='bottom')
ax.set_xlabel('f(m) PC1')
ax.set_ylabel('f(m) PC2')
ax.set_title(f'f(m) discrimination spiral\n(PC1+PC2 = {pca_f.explained_variance_ratio_[:2].sum():.1%} var)')
plt.colorbar(scatter, ax=ax, label='Position m')
ax.grid(True, alpha=0.3)

# 3. f(m) projected into main PC space
ax = fig.add_subplot(2, 3, 3)
F_in_main = pca_main.transform(F)
ax.plot(F_in_main[:, 0], F_in_main[:, 1], 'k-', linewidth=1, alpha=0.5)
scatter = ax.scatter(F_in_main[:, 0], F_in_main[:, 1],
                     c=range(10), cmap='plasma', s=150, edgecolors='black', zorder=5)
for p in range(10):
    ax.annotate(str(p), (F_in_main[p, 0], F_in_main[p, 1]),
                fontsize=10, ha='center', va='bottom')
ax.set_xlabel('Main PC1')
ax.set_ylabel('Main PC2')
ax.set_title('f(m) projected into Main PC space')
plt.colorbar(scatter, ax=ax, label='Position m')
ax.grid(True, alpha=0.3)

# 4. Both spirals in main PC space (overlay)
ax = fig.add_subplot(2, 3, 4)
# Main spiral - show mean trajectory by S position
for s in range(10):
    mask = s_positions_all == s
    mean_coords = main_pca_coords[mask].mean(axis=0)
    ax.scatter(mean_coords[0], mean_coords[1], c='blue', s=100, alpha=0.7,
               marker='o', edgecolors='black')
    ax.annotate(f's={s}', (mean_coords[0], mean_coords[1]), fontsize=8, color='blue')

# f(m) spiral
ax.plot(F_in_main[:, 0], F_in_main[:, 1], 'r-', linewidth=2, alpha=0.7, label='f(m)')
ax.scatter(F_in_main[:, 0], F_in_main[:, 1], c='red', s=80, marker='s',
           edgecolors='black', zorder=5)
for p in range(10):
    ax.annotate(f'm={p}', (F_in_main[p, 0], F_in_main[p, 1]), fontsize=8, color='red')

ax.set_xlabel('Main PC1')
ax.set_ylabel('Main PC2')
ax.set_title('Both spirals in Main PC space\n(blue=main by S pos, red=f(m))')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. PC loading comparison
ax = fig.add_subplot(2, 3, 5)
x = np.arange(16)
width = 0.35
ax.bar(x - width/2, main_pc1, width, label='Main PC1', color='blue', alpha=0.7)
ax.bar(x + width/2, f_pc1, width, label='f(m) PC1', color='red', alpha=0.7)
ax.set_xlabel('Neuron')
ax.set_ylabel('Loading')
ax.set_title('PC1 loadings comparison')
ax.set_xticks(range(16))
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 6. Summary
ax = fig.add_subplot(2, 3, 6)
ax.axis('off')

# Compute key metrics
main_f_pc1_sim = np.abs(np.dot(main_pc1, f_pc1))
main_f_pc2_sim = np.abs(np.dot(main_pc2, f_pc2))

summary = f"""SPIRAL ALIGNMENT SUMMARY

Two spirals in the network:

1. MAIN PHASE WHEEL
   - Encodes (pos1, pos2) pair
   - Lives in PC1-PC2: {pca_main.explained_variance_ratio_[:2].sum():.1%} variance
   - Drives countdown to S position

2. f(m) DISCRIMINATION SPIRAL
   - Encodes M vs S distinction
   - Lives in PC1-PC2: {pca_f.explained_variance_ratio_[:2].sum():.1%} variance
   - Creates offset = f(m) - f(s)

ALIGNMENT:
   |cos(Main PC1, f PC1)| = {main_f_pc1_sim:.4f}
   |cos(Main PC2, f PC2)| = {main_f_pc2_sim:.4f}

   Principal angles: {np.degrees(angles[0]):.1f}°, {np.degrees(angles[1]):.1f}°

INTERPRETATION:
"""

if angles[0] < np.radians(30):
    summary += "   The spirals are ALIGNED - they share\n   the same subspace."
elif angles[0] > np.radians(60):
    summary += "   The spirals are nearly ORTHOGONAL -\n   they use different dimensions."
else:
    summary += "   The spirals are PARTIALLY overlapping -\n   some shared, some distinct dimensions."

ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        ha='left', va='top', fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('docs/spiral_alignment_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved to docs/spiral_alignment_analysis.png")

# =============================================================================
# 5. Additional: Check correlation between position in each spiral
# =============================================================================
print("\n" + "=" * 70)
print("POSITION CORRELATION BETWEEN SPIRALS")
print("=" * 70)

# For the main spiral, average position by S
main_by_s = np.zeros((10, 16))
for s in range(10):
    mask = s_positions_all == s
    if mask.sum() > 0:
        main_by_s[s] = final_states[mask].mean(axis=0)

# Correlation between main_by_s[p] and F[p]
print("\nCorrelation between main spiral (avg by S) and f(m) at each position:")
for p in range(10):
    if np.linalg.norm(main_by_s[p]) > 0 and np.linalg.norm(F[p]) > 0:
        corr = np.corrcoef(main_by_s[p], F[p])[0, 1]
        print(f"  Position {p}: corr = {corr:+.4f}")

# Overall correlation
main_flat = main_by_s[1:].flatten()  # Skip position 0 (F[0] = 0)
f_flat = F[1:].flatten()
overall_corr = np.corrcoef(main_flat, f_flat)[0, 1]
print(f"\nOverall correlation (positions 1-9): {overall_corr:+.4f}")
