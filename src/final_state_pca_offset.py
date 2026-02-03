"""
PCA analysis focusing on the offset between forward and reverse directions.
Project offsets into the same PCA space as the final states.
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

# Collect final hidden states and offsets
print("Collecting final states and offsets...")
final_states_fwd = []
final_states_rev = []
offsets = []
metadata = []

for pos1 in range(10):
    for pos2 in range(10):
        if pos1 == pos2:
            continue

        states_fwd = run_stepwise([(pos1, M_val), (pos2, S_val)])
        h_fwd = states_fwd[-1]

        states_rev = run_stepwise([(pos1, S_val), (pos2, M_val)])
        h_rev = states_rev[-1]

        offset = h_fwd - h_rev

        final_states_fwd.append(h_fwd)
        final_states_rev.append(h_rev)
        offsets.append(offset)
        metadata.append({
            'pos1': pos1, 'pos2': pos2,
            'm_pos_fwd': pos1, 's_pos_fwd': pos2,
            'm_pos_rev': pos2, 's_pos_rev': pos1,
            'gap': abs(pos2 - pos1),
        })

final_states_fwd = np.array(final_states_fwd)
final_states_rev = np.array(final_states_rev)
offsets = np.array(offsets)

# Fit PCA on all final states (fwd + rev combined)
all_states = np.vstack([final_states_fwd, final_states_rev])
pca = PCA()
pca.fit(all_states)

# Transform everything
fwd_pca = pca.transform(final_states_fwd)
rev_pca = pca.transform(final_states_rev)
offset_pca = pca.transform(offsets)  # Project offsets into same space

print(f"Collected 90 position pairs")

# Offset analysis in PCA space
print("\n" + "=" * 70)
print("OFFSET ANALYSIS IN PCA SPACE")
print("=" * 70)

print("\nOffset magnitude by PC:")
for i in range(6):
    offset_var = np.var(offset_pca[:, i])
    total_offset_var = np.sum(np.var(offset_pca, axis=0))
    pct = offset_var / total_offset_var * 100
    print(f"  PC{i+1}: variance = {offset_var:.2f}, {pct:.1f}% of offset variance")

# Correlations of offset PCA coords with position
print("\n" + "=" * 70)
print("OFFSET PCA CORRELATIONS WITH POSITION")
print("=" * 70)

m_pos_fwd = np.array([m['m_pos_fwd'] for m in metadata])  # = pos1
s_pos_fwd = np.array([m['s_pos_fwd'] for m in metadata])  # = pos2
gaps = np.array([m['gap'] for m in metadata])

print("\nOffset projected to PC space, correlated with positions:")
print("       | corr(m_pos) | corr(s_pos) | corr(gap)")
print("-------|-------------|-------------|----------")
for pc_idx in range(6):
    off_vals = offset_pca[:, pc_idx]
    corr_m = np.corrcoef(off_vals, m_pos_fwd)[0, 1]
    corr_s = np.corrcoef(off_vals, s_pos_fwd)[0, 1]
    corr_gap = np.corrcoef(off_vals, gaps)[0, 1]
    print(f" PC{pc_idx+1:2d}  | {corr_m:+11.3f} | {corr_s:+11.3f} | {corr_gap:+9.3f}")

# Visualizations
fig = plt.figure(figsize=(18, 12))

# 1. Fwd vs Rev in PC1-PC2 with connecting lines
ax = fig.add_subplot(2, 3, 1)
for i in range(len(metadata)):
    ax.plot([fwd_pca[i, 0], rev_pca[i, 0]], [fwd_pca[i, 1], rev_pca[i, 1]],
            'k-', alpha=0.2, linewidth=0.5)
ax.scatter(fwd_pca[:, 0], fwd_pca[:, 1], c='blue', s=40, alpha=0.7, label='Forward', edgecolors='white')
ax.scatter(rev_pca[:, 0], rev_pca[:, 1], c='red', s=40, alpha=0.7, label='Reverse', edgecolors='white')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Forward vs Reverse in PC space\n(lines connect same position pair)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Offset vectors in PC1-PC2 colored by m_pos
ax = fig.add_subplot(2, 3, 2)
scatter = ax.scatter(offset_pca[:, 0], offset_pca[:, 1],
                     c=m_pos_fwd, cmap='viridis', s=60, alpha=0.8, edgecolors='black')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Offset PC1')
ax.set_ylabel('Offset PC2')
ax.set_title('Offset (h_fwd - h_rev) in PC space\n(colored by M position in forward)')
plt.colorbar(scatter, ax=ax, label='M position')
ax.grid(True, alpha=0.3)

# 3. Offset vectors colored by s_pos
ax = fig.add_subplot(2, 3, 3)
scatter = ax.scatter(offset_pca[:, 0], offset_pca[:, 1],
                     c=s_pos_fwd, cmap='plasma', s=60, alpha=0.8, edgecolors='black')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Offset PC1')
ax.set_ylabel('Offset PC2')
ax.set_title('Offset in PC space\n(colored by S position in forward)')
plt.colorbar(scatter, ax=ax, label='S position')
ax.grid(True, alpha=0.3)

# 4. PC1 offset vs m_pos and s_pos
ax = fig.add_subplot(2, 3, 4)
ax.scatter(m_pos_fwd, offset_pca[:, 0], c='blue', s=50, alpha=0.6, label='vs m_pos')
ax.scatter(s_pos_fwd, offset_pca[:, 1], c='red', s=50, alpha=0.6, label='vs s_pos (PC2)')

# Fit lines
z_m = np.polyfit(m_pos_fwd, offset_pca[:, 0], 1)
z_s = np.polyfit(s_pos_fwd, offset_pca[:, 1], 1)
x_line = np.linspace(0, 9, 100)
ax.plot(x_line, np.polyval(z_m, x_line), 'b--', alpha=0.7)
ax.plot(x_line, np.polyval(z_s, x_line), 'r--', alpha=0.7)

ax.set_xlabel('Position')
ax.set_ylabel('Offset PC coordinate')
ax.set_title('Offset PC1 vs m_pos (blue)\nOffset PC2 vs s_pos (red)')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Offset projected through W_out
ax = fig.add_subplot(2, 3, 5)
# For each offset, compute W_out @ offset to see logit effect
logit_effects = np.array([W_out @ off for off in offsets])

# Show mean absolute logit effect per position
mean_logit_effect = np.mean(logit_effects, axis=0)
std_logit_effect = np.std(logit_effects, axis=0)

ax.bar(range(10), mean_logit_effect, yerr=std_logit_effect, capsize=3,
       color='steelblue', edgecolor='black', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Output position')
ax.set_ylabel('Mean logit effect of offset')
ax.set_title('W_out @ offset (mean ± std across all pairs)')
ax.set_xticks(range(10))
ax.grid(True, alpha=0.3, axis='y')

# 6. Offset in PC3-PC4
ax = fig.add_subplot(2, 3, 6)
scatter = ax.scatter(offset_pca[:, 2], offset_pca[:, 3],
                     c=m_pos_fwd, cmap='viridis', s=60, alpha=0.8, edgecolors='black')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Offset PC3')
ax.set_ylabel('Offset PC4')
ax.set_title('Offset in PC3-PC4 space\n(colored by M position)')
plt.colorbar(scatter, ax=ax, label='M position')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/final_state_pca_offset.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved to docs/final_state_pca_offset.png")

# Check if offset is low-dimensional
print("\n" + "=" * 70)
print("OFFSET DIMENSIONALITY")
print("=" * 70)

# Do PCA specifically on offsets
pca_offset = PCA()
pca_offset.fit(offsets)

print("\nPCA on offsets alone:")
cumvar = np.cumsum(pca_offset.explained_variance_ratio_)
for i in range(6):
    print(f"  PC{i+1}: {pca_offset.explained_variance_ratio_[i]:.1%} (cumulative: {cumvar[i]:.1%})")

n_95_off = np.argmax(cumvar >= 0.95) + 1
print(f"\nComponents for 95% of offset variance: {n_95_off}")

# Compare offset PCA directions to main PCA directions
print("\n" + "=" * 70)
print("OFFSET PCA vs MAIN STATE PCA ALIGNMENT")
print("=" * 70)

print("\nCosine similarity between offset PCs and main state PCs:")
print("         | State PC1 | State PC2 | State PC3 | State PC4")
print("---------|-----------|-----------|-----------|----------")
for i in range(4):
    off_pc = pca_offset.components_[i]
    off_pc_norm = off_pc / np.linalg.norm(off_pc)
    sims = []
    for j in range(4):
        state_pc = pca.components_[j]
        state_pc_norm = state_pc / np.linalg.norm(state_pc)
        sim = np.abs(np.dot(off_pc_norm, state_pc_norm))
        sims.append(sim)
    print(f"Off PC{i+1}  |   {sims[0]:.3f}   |   {sims[1]:.3f}   |   {sims[2]:.3f}   |   {sims[3]:.3f}")
