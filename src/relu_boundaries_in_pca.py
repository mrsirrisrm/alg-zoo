"""
Visualize ReLU boundaries projected into PCA space.
In PCA space, hyperplane boundaries become lines.
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

# Collect final states
print("Collecting final states...")
final_states = []
metadata = []

for pos1 in range(10):
    for pos2 in range(10):
        if pos1 == pos2:
            continue

        # Forward
        states_fwd = run_stepwise([(pos1, M_val), (pos2, S_val)])
        final_states.append(states_fwd[-1])
        metadata.append({'m_pos': pos1, 's_pos': pos2, 'dir': 'fwd'})

        # Reverse
        states_rev = run_stepwise([(pos1, S_val), (pos2, M_val)])
        final_states.append(states_rev[-1])
        metadata.append({'m_pos': pos2, 's_pos': pos1, 'dir': 'rev'})

final_states = np.array(final_states)

# Fit PCA
pca = PCA(n_components=4)
pca.fit(final_states)
pca_coords = pca.transform(final_states)

# Get PCA range for plotting boundaries
pc1_range = (pca_coords[:, 0].min() - 5, pca_coords[:, 0].max() + 5)
pc2_range = (pca_coords[:, 1].min() - 5, pca_coords[:, 1].max() + 5)

print("\n" + "=" * 70)
print("RELU BOUNDARY ANALYSIS IN PCA SPACE")
print("=" * 70)

# For each neuron, the ReLU boundary in original space is:
#   W_hh[i,:] @ h = 0  (assuming x=0 at final timestep)
#
# In PCA space, h = mean + sum_j (pc_j * component_j)
# So the boundary becomes:
#   W_hh[i,:] @ (mean + pc1*v1 + pc2*v2 + ...) = 0
#
# For 2D projection using PC1 and PC2:
#   W_hh[i,:] @ mean + pc1 * (W_hh[i,:] @ v1) + pc2 * (W_hh[i,:] @ v2) = 0
#
# This is a line: a*pc1 + b*pc2 + c = 0

mean = pca.mean_
v1 = pca.components_[0]  # PC1 direction
v2 = pca.components_[1]  # PC2 direction

print("\nReLU boundary lines in PC1-PC2 space:")
print("(Line equation: a*PC1 + b*PC2 + c = 0)")
print("\nNeuron |    a    |    b    |    c    | Passes through data?")
print("-------|---------|---------|---------|---------------------")

boundary_lines = []
for neuron in range(16):
    w = W_hh[neuron, :]

    a = np.dot(w, v1)  # Coefficient for PC1
    b = np.dot(w, v2)  # Coefficient for PC2
    c = np.dot(w, mean)  # Constant term

    # Check if boundary passes through the data region
    # Solve for PC2 at edges of PC1 range
    if abs(b) > 1e-10:
        pc2_at_min = (-a * pc1_range[0] - c) / b
        pc2_at_max = (-a * pc1_range[1] - c) / b
        in_range = (pc2_range[0] <= pc2_at_min <= pc2_range[1] or
                    pc2_range[0] <= pc2_at_max <= pc2_range[1])
    else:
        # Vertical line: PC1 = -c/a
        if abs(a) > 1e-10:
            pc1_intercept = -c / a
            in_range = pc1_range[0] <= pc1_intercept <= pc1_range[1]
        else:
            in_range = False

    boundary_lines.append((neuron, a, b, c, in_range))
    in_str = "YES" if in_range else "no"
    print(f"  {neuron:2d}   | {a:+7.3f} | {b:+7.3f} | {c:+7.3f} | {in_str}")

# Count which neurons have active boundaries
active_boundaries = [bl for bl in boundary_lines if bl[4]]
print(f"\n{len(active_boundaries)} neurons have ReLU boundaries passing through the data region")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. All ReLU boundaries in PC1-PC2
ax = axes[0, 0]

# Plot data points
m_positions = np.array([m['m_pos'] for m in metadata])
scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                     c=m_positions, cmap='viridis', s=30, alpha=0.6, edgecolors='white')

# Plot ReLU boundaries
colors = plt.cm.tab20(np.linspace(0, 1, 16))
pc1_line = np.linspace(pc1_range[0], pc1_range[1], 100)

for neuron, a, b, c, in_range in boundary_lines:
    if not in_range:
        continue
    if abs(b) > 1e-10:
        pc2_line = (-a * pc1_line - c) / b
        mask = (pc2_line >= pc2_range[0]) & (pc2_line <= pc2_range[1])
        ax.plot(pc1_line[mask], pc2_line[mask], color=colors[neuron],
                linewidth=2, alpha=0.8, label=f'n{neuron}')
    else:
        pc1_intercept = -c / a
        ax.axvline(pc1_intercept, color=colors[neuron], linewidth=2,
                   alpha=0.8, label=f'n{neuron}')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('ReLU boundaries in PC1-PC2 space')
ax.legend(loc='upper right', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(pc1_range)
ax.set_ylim(pc2_range)
plt.colorbar(scatter, ax=ax, label='M position')

# 2. Same but colored by which side of each boundary
ax = axes[0, 1]

# For the most significant boundary (largest gradient in data direction),
# color points by which side they're on
# Find which neuron has most variance in its pre-activation
pre_activations = final_states @ W_hh.T  # Shape: (n_samples, 16)
pre_var = np.var(pre_activations, axis=0)
most_variable_neuron = np.argmax(pre_var)

print(f"\nMost variable pre-activation: neuron {most_variable_neuron} (var={pre_var[most_variable_neuron]:.2f})")

# Color by whether this neuron is active
neuron_active = final_states[:, most_variable_neuron] > 0
ax.scatter(pca_coords[neuron_active, 0], pca_coords[neuron_active, 1],
           c='blue', s=40, alpha=0.6, label=f'n{most_variable_neuron} active')
ax.scatter(pca_coords[~neuron_active, 0], pca_coords[~neuron_active, 1],
           c='red', s=40, alpha=0.6, label=f'n{most_variable_neuron} clipped')

# Draw this neuron's boundary
n = most_variable_neuron
a, b, c = boundary_lines[n][1], boundary_lines[n][2], boundary_lines[n][3]
if abs(b) > 1e-10:
    pc2_line = (-a * pc1_line - c) / b
    mask = (pc2_line >= pc2_range[0]) & (pc2_line <= pc2_range[1])
    ax.plot(pc1_line[mask], pc2_line[mask], 'k-', linewidth=3, label='ReLU boundary')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title(f'Neuron {most_variable_neuron} activation regions')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(pc1_range)
ax.set_ylim(pc2_range)

# 3. Show activation pattern counts
ax = axes[1, 0]

# Compute activation patterns
activation_patterns = (final_states > 0).astype(int)
pattern_strings = [''.join(map(str, p)) for p in activation_patterns]
unique_patterns, pattern_counts = np.unique(pattern_strings, return_counts=True)

print(f"\nNumber of unique activation patterns: {len(unique_patterns)}")

# Sort by count
sorted_idx = np.argsort(pattern_counts)[::-1]
top_n = min(15, len(unique_patterns))

ax.barh(range(top_n), pattern_counts[sorted_idx[:top_n]], color='steelblue', edgecolor='black')
ax.set_yticks(range(top_n))
ax.set_yticklabels([f"{unique_patterns[sorted_idx[i]][:8]}..." for i in range(top_n)], fontsize=8)
ax.set_xlabel('Count')
ax.set_ylabel('Activation pattern (first 8 neurons)')
ax.set_title(f'Top {top_n} activation patterns\n({len(unique_patterns)} unique out of 180 states)')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# 4. Activation count per neuron
ax = axes[1, 1]
active_counts = np.sum(final_states > 0, axis=0)
total = len(final_states)

colors = ['green' if c > total * 0.9 else 'orange' if c > total * 0.5 else 'red'
          for c in active_counts]
ax.bar(range(16), active_counts, color=colors, edgecolor='black', alpha=0.7)
ax.axhline(total, color='gray', linestyle='--', alpha=0.5, label=f'Total={total}')
ax.axhline(total * 0.5, color='red', linestyle=':', alpha=0.5, label='50%')
ax.set_xlabel('Neuron')
ax.set_ylabel('Times active')
ax.set_title('Neuron activation frequency across all final states')
ax.set_xticks(range(16))
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add text annotations
for i, c in enumerate(active_counts):
    pct = c / total * 100
    ax.annotate(f'{pct:.0f}%', (i, c + 2), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('docs/relu_boundaries_pca.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved to docs/relu_boundaries_pca.png")

# Additional analysis: which boundaries separate fwd from rev?
print("\n" + "=" * 70)
print("BOUNDARIES SEPARATING FORWARD VS REVERSE")
print("=" * 70)

fwd_mask = np.array([m['dir'] == 'fwd' for m in metadata])

print("\nNeuron activation difference (fwd - rev):")
for neuron in range(16):
    fwd_active = np.mean(final_states[fwd_mask, neuron] > 0)
    rev_active = np.mean(final_states[~fwd_mask, neuron] > 0)
    diff = fwd_active - rev_active
    if abs(diff) > 0.05:
        print(f"  Neuron {neuron:2d}: fwd={fwd_active:.1%}, rev={rev_active:.1%}, diff={diff:+.1%}")
