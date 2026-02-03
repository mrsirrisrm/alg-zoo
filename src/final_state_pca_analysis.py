"""
PCA analysis of final hidden states across all 90 position pairs.
Focus on the most significant directions and their relationship to
position encoding and ReLU boundaries.
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

# Collect final hidden states for all 90 pairs
print("Collecting final hidden states for all position pairs...")
final_states = []
metadata = []

for pos1 in range(10):
    for pos2 in range(10):
        if pos1 == pos2:
            continue

        # Forward: M@pos1, S@pos2
        states_fwd = run_stepwise([(pos1, M_val), (pos2, S_val)])
        h_fwd = states_fwd[-1]

        # Reverse: S@pos1, M@pos2
        states_rev = run_stepwise([(pos1, S_val), (pos2, M_val)])
        h_rev = states_rev[-1]

        final_states.append(h_fwd)
        metadata.append({
            'pos1': pos1, 'pos2': pos2,
            'm_pos': pos1, 's_pos': pos2,
            'direction': 'forward',
            'gap': pos2 - pos1,
        })

        final_states.append(h_rev)
        metadata.append({
            'pos1': pos1, 'pos2': pos2,
            'm_pos': pos2, 's_pos': pos1,
            'direction': 'reverse',
            'gap': pos1 - pos2,
        })

final_states = np.array(final_states)
n_samples = len(final_states)
print(f"Collected {n_samples} final states (90 pairs × 2 directions)")

# Fit PCA
pca = PCA()
pca.fit(final_states)
pca_coords = pca.transform(final_states)

# Explained variance
print("\n" + "=" * 70)
print("PCA EXPLAINED VARIANCE")
print("=" * 70)
print("\nComponent | Variance | Cumulative | Interpretation")
print("----------|----------|------------|---------------")

cumulative = 0
for i in range(16):
    var = pca.explained_variance_ratio_[i]
    cumulative += var
    print(f"   PC{i+1:2d}   | {var:7.1%}  |   {cumulative:6.1%}   |")

# How many components for 95% variance?
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_95 = np.argmax(cumvar >= 0.95) + 1
n_99 = np.argmax(cumvar >= 0.99) + 1
print(f"\nComponents for 95% variance: {n_95}")
print(f"Components for 99% variance: {n_99}")

# Analyze principal components
print("\n" + "=" * 70)
print("PRINCIPAL COMPONENT LOADINGS")
print("=" * 70)

for pc_idx in range(min(4, 16)):
    loadings = pca.components_[pc_idx]
    print(f"\nPC{pc_idx+1} loadings (top 5 by magnitude):")
    sorted_idx = np.argsort(np.abs(loadings))[::-1]
    for i in sorted_idx[:5]:
        print(f"  Neuron {i:2d}: {loadings[i]:+.3f}")

# Correlations with position
print("\n" + "=" * 70)
print("PC CORRELATIONS WITH POSITION")
print("=" * 70)

m_positions = np.array([m['m_pos'] for m in metadata])
s_positions = np.array([m['s_pos'] for m in metadata])
gaps = np.array([m['gap'] for m in metadata])
directions = np.array([1 if m['direction'] == 'forward' else -1 for m in metadata])

print("\n       | corr(m_pos) | corr(s_pos) | corr(gap) | corr(dir)")
print("-------|-------------|-------------|-----------|----------")
for pc_idx in range(min(6, 16)):
    pc_vals = pca_coords[:, pc_idx]
    corr_m = np.corrcoef(pc_vals, m_positions)[0, 1]
    corr_s = np.corrcoef(pc_vals, s_positions)[0, 1]
    corr_gap = np.corrcoef(pc_vals, gaps)[0, 1]
    corr_dir = np.corrcoef(pc_vals, directions)[0, 1]
    print(f" PC{pc_idx+1:2d}  | {corr_m:+11.3f} | {corr_s:+11.3f} | {corr_gap:+9.3f} | {corr_dir:+8.3f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Explained variance
ax = axes[0, 0]
ax.bar(range(1, 17), pca.explained_variance_ratio_, color='steelblue', edgecolor='black')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title('PCA Explained Variance')
ax.set_xticks(range(1, 17))
ax.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='5% threshold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 2. Cumulative variance
ax = axes[0, 1]
ax.plot(range(1, 17), cumvar, 'o-', color='steelblue', markersize=8)
ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95%')
ax.axhline(0.99, color='orange', linestyle='--', alpha=0.5, label='99%')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('Cumulative Variance Explained')
ax.set_xticks(range(1, 17))
ax.legend()
ax.grid(True, alpha=0.3)

# 3. PC1 vs PC2 colored by m_pos
ax = axes[0, 2]
scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                     c=m_positions, cmap='viridis', s=50, alpha=0.7, edgecolors='white')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PC1 vs PC2 (colored by M position)')
plt.colorbar(scatter, ax=ax, label='M position')
ax.grid(True, alpha=0.3)

# 4. PC1 vs PC2 colored by s_pos
ax = axes[1, 0]
scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                     c=s_positions, cmap='plasma', s=50, alpha=0.7, edgecolors='white')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PC1 vs PC2 (colored by S position)')
plt.colorbar(scatter, ax=ax, label='S position')
ax.grid(True, alpha=0.3)

# 5. PC1 vs PC2 colored by direction
ax = axes[1, 1]
fwd_mask = np.array([m['direction'] == 'forward' for m in metadata])
ax.scatter(pca_coords[fwd_mask, 0], pca_coords[fwd_mask, 1],
           c='blue', s=50, alpha=0.6, label='Forward (M first)', edgecolors='white')
ax.scatter(pca_coords[~fwd_mask, 0], pca_coords[~fwd_mask, 1],
           c='red', s=50, alpha=0.6, label='Reverse (S first)', edgecolors='white')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PC1 vs PC2 (by direction)')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Loadings heatmap for top 4 PCs
ax = axes[1, 2]
loadings_matrix = pca.components_[:4, :]
im = ax.imshow(loadings_matrix, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax.set_xlabel('Neuron')
ax.set_ylabel('Principal Component')
ax.set_title('PC Loadings (top 4 components)')
ax.set_xticks(range(16))
ax.set_yticks(range(4))
ax.set_yticklabels(['PC1', 'PC2', 'PC3', 'PC4'])
plt.colorbar(im, ax=ax, label='Loading')

plt.tight_layout()
plt.savefig('docs/final_state_pca_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved to docs/final_state_pca_analysis.png")

# Additional: relationship between PCs and W_out
print("\n" + "=" * 70)
print("PC RELATIONSHIP TO OUTPUT WEIGHTS")
print("=" * 70)

print("\nProjection of PC directions through W_out:")
for pc_idx in range(4):
    pc_vec = pca.components_[pc_idx]
    output_proj = W_out @ pc_vec

    print(f"\nPC{pc_idx+1} → W_out projection (logit contributions):")
    sorted_idx = np.argsort(np.abs(output_proj))[::-1]
    for i in sorted_idx[:3]:
        print(f"  Position {i}: {output_proj[i]:+.3f}")

# Check if any PC aligns with specific output directions
print("\n" + "=" * 70)
print("PC ALIGNMENT WITH OUTPUT POSITION VECTORS")
print("=" * 70)

print("\nCosine similarity between PCs and W_out rows:")
print("       | pos 0 | pos 1 | pos 2 | ... | pos 9")
print("-------|-------|-------|-------|-----|------")

for pc_idx in range(4):
    pc_vec = pca.components_[pc_idx]
    pc_norm = pc_vec / np.linalg.norm(pc_vec)

    sims = []
    for pos in range(10):
        w_pos = W_out[pos, :]
        w_norm = w_pos / np.linalg.norm(w_pos)
        sim = np.dot(pc_norm, w_norm)
        sims.append(sim)

    sim_str = " | ".join([f"{s:+.2f}" for s in sims])
    print(f" PC{pc_idx+1}   | {sim_str}")
