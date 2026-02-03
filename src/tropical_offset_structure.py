"""
Analyze the tropical structure of the offset.
Focus on how differential clipping creates the position-encoding offset.
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

def run_stepwise(impulses, return_pre=False):
    """Run RNN and optionally return pre-activations."""
    h = np.zeros(16)
    states = [h.copy()]
    pre_acts = []
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        states.append(h.copy())
        pre_acts.append(pre.copy())
    if return_pre:
        return states, pre_acts
    return states

# Collect data
print("Analyzing tropical offset structure...")
data = []

for pos1 in range(10):
    for pos2 in range(10):
        if pos1 == pos2:
            continue

        states_fwd, pre_fwd = run_stepwise([(pos1, M_val), (pos2, S_val)], return_pre=True)
        states_rev, pre_rev = run_stepwise([(pos1, S_val), (pos2, M_val)], return_pre=True)

        h_fwd = states_fwd[-1]
        h_rev = states_rev[-1]
        offset = h_fwd - h_rev

        # Final pre-activations (what would happen without ReLU)
        pre_fwd_final = pre_fwd[-1]
        pre_rev_final = pre_rev[-1]

        # Clipping masks
        fwd_clipped = pre_fwd_final < 0
        rev_clipped = pre_rev_final < 0

        # Differential clipping
        diff_clip = fwd_clipped != rev_clipped

        data.append({
            'm_pos': pos1,  # M position in forward
            's_pos': pos2,  # S position in forward
            'h_fwd': h_fwd,
            'h_rev': h_rev,
            'offset': offset,
            'pre_fwd': pre_fwd_final,
            'pre_rev': pre_rev_final,
            'fwd_clipped': fwd_clipped,
            'rev_clipped': rev_clipped,
            'diff_clip': diff_clip,
        })

n_pairs = len(data)

# Analyze differential clipping
print("\n" + "=" * 70)
print("DIFFERENTIAL CLIPPING ANALYSIS")
print("=" * 70)

# For each neuron, count how often it's differentially clipped
print("\nNeuron | Fwd clipped | Rev clipped | Diff clipped | Always same")
print("-------|-------------|-------------|--------------|------------")

for neuron in range(16):
    fwd_clip_count = sum(1 for d in data if d['fwd_clipped'][neuron])
    rev_clip_count = sum(1 for d in data if d['rev_clipped'][neuron])
    diff_clip_count = sum(1 for d in data if d['diff_clip'][neuron])
    same_count = n_pairs - diff_clip_count

    fwd_pct = fwd_clip_count / n_pairs * 100
    rev_pct = rev_clip_count / n_pairs * 100
    diff_pct = diff_clip_count / n_pairs * 100
    same_pct = same_count / n_pairs * 100

    print(f"  {neuron:2d}   |   {fwd_pct:5.1f}%    |   {rev_pct:5.1f}%    |    {diff_pct:5.1f}%    |   {same_pct:5.1f}%")

# Focus on neurons with significant differential clipping
print("\n" + "=" * 70)
print("NEURONS WITH DIFFERENTIAL CLIPPING")
print("=" * 70)

diff_clip_neurons = []
for neuron in range(16):
    diff_clip_count = sum(1 for d in data if d['diff_clip'][neuron])
    if diff_clip_count > 0:
        diff_clip_neurons.append((neuron, diff_clip_count))

print(f"\nNeurons with any differential clipping: {[n for n, c in diff_clip_neurons]}")
for neuron, count in sorted(diff_clip_neurons, key=lambda x: -x[1]):
    print(f"  Neuron {neuron}: {count} pairs ({count/n_pairs*100:.1f}%)")

# Analyze the contribution of each neuron to the offset
print("\n" + "=" * 70)
print("OFFSET CONTRIBUTION BY NEURON")
print("=" * 70)

offsets = np.array([d['offset'] for d in data])

print("\nMean |offset[i]| by neuron:")
mean_abs_offset = np.mean(np.abs(offsets), axis=0)
sorted_neurons = np.argsort(mean_abs_offset)[::-1]
for n in sorted_neurons[:8]:
    print(f"  Neuron {n:2d}: {mean_abs_offset[n]:.3f}")

# Decompose offset into parts
print("\n" + "=" * 70)
print("OFFSET DECOMPOSITION")
print("=" * 70)

# For each pair, decompose offset into:
# 1. Linear part (if no differential clipping happened)
# 2. Clipping part (due to ReLU)

linear_offsets = []
clipping_offsets = []

for d in data:
    # Linear offset = what offset would be without any clipping
    # This is (pre_fwd - pre_rev) for neurons that aren't clipped in either
    linear_off = np.zeros(16)
    clip_off = np.zeros(16)

    for n in range(16):
        if not d['fwd_clipped'][n] and not d['rev_clipped'][n]:
            # Neither clipped: offset is purely linear
            linear_off[n] = d['h_fwd'][n] - d['h_rev'][n]
        elif d['fwd_clipped'][n] and d['rev_clipped'][n]:
            # Both clipped: no offset contribution
            pass
        else:
            # One clipped, one not: this is where the magic happens
            clip_off[n] = d['h_fwd'][n] - d['h_rev'][n]

    linear_offsets.append(linear_off)
    clipping_offsets.append(clip_off)

linear_offsets = np.array(linear_offsets)
clipping_offsets = np.array(clipping_offsets)
total_offsets = offsets

print("\nOffset variance decomposition:")
total_var = np.sum(np.var(total_offsets, axis=0))
linear_var = np.sum(np.var(linear_offsets, axis=0))
clip_var = np.sum(np.var(clipping_offsets, axis=0))

print(f"  Total offset variance: {total_var:.2f}")
print(f"  Linear part variance: {linear_var:.2f} ({linear_var/total_var*100:.1f}%)")
print(f"  Clipping part variance: {clip_var:.2f} ({clip_var/total_var*100:.1f}%)")

# Verify offset = linear + clipping
reconstruction_error = np.sum((total_offsets - linear_offsets - clipping_offsets)**2)
print(f"  Reconstruction error: {reconstruction_error:.6f}")

# Visualizations
fig = plt.figure(figsize=(16, 12))

# 1. Differential clipping heatmap
ax = fig.add_subplot(2, 3, 1)
diff_clip_matrix = np.array([d['diff_clip'].astype(int) for d in data])
im = ax.imshow(diff_clip_matrix.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xlabel('Position pair index')
ax.set_ylabel('Neuron')
ax.set_title('Differential clipping\n(yellow=same, red=different)')
ax.set_yticks(range(16))
plt.colorbar(im, ax=ax)

# 2. Pre-activation distance from boundary
ax = fig.add_subplot(2, 3, 2)

# For each position pair, how close are pre-activations to 0?
pre_fwd_all = np.array([d['pre_fwd'] for d in data])
pre_rev_all = np.array([d['pre_rev'] for d in data])

# Distance to clipping boundary
dist_fwd = np.minimum(np.abs(pre_fwd_all), 10)  # Cap at 10 for visualization
dist_rev = np.minimum(np.abs(pre_rev_all), 10)

mean_dist_fwd = np.mean(dist_fwd, axis=0)
mean_dist_rev = np.mean(dist_rev, axis=0)

x = np.arange(16)
width = 0.35
ax.bar(x - width/2, mean_dist_fwd, width, label='Forward', color='blue', alpha=0.7)
ax.bar(x + width/2, mean_dist_rev, width, label='Reverse', color='red', alpha=0.7)
ax.set_xlabel('Neuron')
ax.set_ylabel('Mean |pre-activation|')
ax.set_title('Distance from ReLU boundary at t=10')
ax.set_xticks(range(16))
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3. Offset contribution by neuron (linear vs clipping)
ax = fig.add_subplot(2, 3, 3)

linear_contribution = np.sum(np.abs(linear_offsets), axis=0)
clip_contribution = np.sum(np.abs(clipping_offsets), axis=0)

ax.bar(x - width/2, linear_contribution, width, label='Linear', color='blue', alpha=0.7)
ax.bar(x + width/2, clip_contribution, width, label='Clipping', color='red', alpha=0.7)
ax.set_xlabel('Neuron')
ax.set_ylabel('Sum |offset contribution|')
ax.set_title('Offset contribution: Linear vs Clipping')
ax.set_xticks(range(16))
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4. Offset through W_out (logit effect)
ax = fig.add_subplot(2, 3, 4)

m_positions = np.array([d['m_pos'] for d in data])
s_positions = np.array([d['s_pos'] for d in data])

# Compute W_out @ offset for each pair
logit_offsets = np.array([W_out @ d['offset'] for d in data])

# For each pair, the relevant logit positions are m_pos and s_pos
# In forward, we want to boost s_pos and suppress m_pos
# Check if offset does this

boost_s = []
suppress_m = []
for i, d in enumerate(data):
    m, s = d['m_pos'], d['s_pos']
    boost_s.append(logit_offsets[i, s])
    suppress_m.append(logit_offsets[i, m])

boost_s = np.array(boost_s)
suppress_m = np.array(suppress_m)

ax.scatter(m_positions, suppress_m, c='red', s=30, alpha=0.5, label='Logit at M pos (should be negative)')
ax.scatter(s_positions, boost_s, c='blue', s=30, alpha=0.5, label='Logit at S pos (should be positive)')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('W_out @ offset at position')
ax.set_title('Offset effect on relevant logits')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. PCA of offsets colored by m_pos
ax = fig.add_subplot(2, 3, 5)

pca = PCA(n_components=2)
offset_pca = pca.fit_transform(offsets)

scatter = ax.scatter(offset_pca[:, 0], offset_pca[:, 1],
                     c=m_positions, cmap='viridis', s=50, alpha=0.7, edgecolors='black')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Offset PC1')
ax.set_ylabel('Offset PC2')
ax.set_title(f'Offset PCA (PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(scatter, ax=ax, label='M position')
ax.grid(True, alpha=0.3)

# 6. Summary text
ax = fig.add_subplot(2, 3, 6)
ax.axis('off')

# Compute summary stats
n_diff_clip_neurons = len(diff_clip_neurons)
total_diff_clips = sum(c for _, c in diff_clip_neurons)
avg_diff_clips_per_pair = total_diff_clips / n_pairs

# Top clipping neurons
top_clip = sorted(diff_clip_neurons, key=lambda x: -x[1])[:3]

summary = f"""TROPICAL OFFSET STRUCTURE

Key findings:

1. DIFFERENTIAL CLIPPING
   - {n_diff_clip_neurons} neurons have differential clipping
   - Average {avg_diff_clips_per_pair:.1f} neurons clipped differently per pair
   - Top clipping neurons: {', '.join([f'n{n}' for n, c in top_clip])}

2. OFFSET VARIANCE
   - Linear part: {linear_var/total_var*100:.1f}%
   - Clipping part: {clip_var/total_var*100:.1f}%

3. OFFSET EFFECT ON LOGITS
   - Mean boost at S position: {np.mean(boost_s):+.2f}
   - Mean suppress at M position: {np.mean(suppress_m):+.2f}
   - Net discrimination: {np.mean(boost_s - suppress_m):.2f}

The offset is almost entirely due to the LINEAR part
of the piecewise-linear dynamics, with minimal
contribution from differential clipping at the
final timestep.
"""

ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        ha='left', va='top', fontsize=11, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('docs/tropical_offset_structure.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved to docs/tropical_offset_structure.png")

# Final check: is offset linear in positions?
print("\n" + "=" * 70)
print("OFFSET LINEARITY CHECK")
print("=" * 70)

# If offset is linear in positions, then offset = A @ [m_pos, s_pos, 1]
# Fit this model
X = np.column_stack([m_positions, s_positions, np.ones(n_pairs)])

# Fit for each neuron
print("\nLinear fit R² by neuron (offset[n] = a*m + b*s + c):")
for n in range(16):
    y = offsets[:, n]
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    if len(residuals) > 0:
        ss_res = residuals[0]
    else:
        ss_res = np.sum((y - X @ coeffs) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
    print(f"  Neuron {n:2d}: R² = {r2:.4f}, coeffs = [{coeffs[0]:+.3f}*m, {coeffs[1]:+.3f}*s, {coeffs[2]:+.3f}]")
