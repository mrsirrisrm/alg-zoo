"""
Analyze all 90 position pairs to see which neuron deltas correlate with:
- gap (pos2 - pos1)
- m_pos (position of M in forward = pos1)
- s_pos (position of S in forward = pos2)
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

# Collect data for all 90 pairs
data = []

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

        offset = h_fwd - h_rev

        # In forward: M is at pos1, S is at pos2
        m_pos_fwd = pos1
        s_pos_fwd = pos2
        gap = abs(pos2 - pos1)

        # Which arrives first?
        if pos1 < pos2:
            first_pos = pos1
            second_pos = pos2
            direction = "forward"  # M arrives first
        else:
            first_pos = pos2
            second_pos = pos1
            direction = "reverse_order"  # S arrives first in the forward case

        data.append({
            'pos1': pos1,
            'pos2': pos2,
            'm_pos': m_pos_fwd,
            's_pos': s_pos_fwd,
            'gap': gap,
            'first_pos': first_pos,
            'second_pos': second_pos,
            'offset': offset,
            'h_fwd': h_fwd,
            'h_rev': h_rev,
        })

# Convert to arrays for correlation analysis
n_pairs = len(data)
offsets = np.array([d['offset'] for d in data])  # (90, 16)
gaps = np.array([d['gap'] for d in data])
m_positions = np.array([d['m_pos'] for d in data])
s_positions = np.array([d['s_pos'] for d in data])
first_positions = np.array([d['first_pos'] for d in data])
second_positions = np.array([d['second_pos'] for d in data])

print("=" * 70)
print("CORRELATION ANALYSIS: Which variables predict neuron offsets?")
print("=" * 70)

# Compute correlations for each neuron
correlations = {
    'gap': [],
    'm_pos': [],
    's_pos': [],
    'first_pos': [],
    'second_pos': [],
}

for neuron in range(16):
    neuron_offsets = offsets[:, neuron]

    corr_gap = np.corrcoef(neuron_offsets, gaps)[0, 1]
    corr_m = np.corrcoef(neuron_offsets, m_positions)[0, 1]
    corr_s = np.corrcoef(neuron_offsets, s_positions)[0, 1]
    corr_first = np.corrcoef(neuron_offsets, first_positions)[0, 1]
    corr_second = np.corrcoef(neuron_offsets, second_positions)[0, 1]

    correlations['gap'].append(corr_gap)
    correlations['m_pos'].append(corr_m)
    correlations['s_pos'].append(corr_s)
    correlations['first_pos'].append(corr_first)
    correlations['second_pos'].append(corr_second)

# Print correlation table
print("\nCorrelation of each neuron's offset with position variables:")
print("\nNeuron |   gap   |  m_pos  |  s_pos  | 1st_pos | 2nd_pos | dominant")
print("-------|---------|---------|---------|---------|---------|----------")

for neuron in range(16):
    corrs = {
        'gap': correlations['gap'][neuron],
        'm_pos': correlations['m_pos'][neuron],
        's_pos': correlations['s_pos'][neuron],
        '1st': correlations['first_pos'][neuron],
        '2nd': correlations['second_pos'][neuron],
    }

    # Find dominant correlation
    abs_corrs = {k: abs(v) for k, v in corrs.items()}
    dominant = max(abs_corrs, key=abs_corrs.get)
    dominant_val = corrs[dominant]

    print(f"  {neuron:2d}   | {corrs['gap']:+.3f}  | {corrs['m_pos']:+.3f}  | {corrs['s_pos']:+.3f}  | {corrs['1st']:+.3f}  | {corrs['2nd']:+.3f}  | {dominant} ({dominant_val:+.3f})")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Correlation heatmap
ax = axes[0, 0]
corr_matrix = np.array([
    correlations['gap'],
    correlations['m_pos'],
    correlations['s_pos'],
    correlations['first_pos'],
    correlations['second_pos'],
])
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_yticks(range(5))
ax.set_yticklabels(['gap', 'm_pos', 's_pos', '1st_pos', '2nd_pos'])
ax.set_xticks(range(16))
ax.set_xticklabels(range(16))
ax.set_xlabel('Neuron')
ax.set_title('Correlation of neuron offset with variables')
plt.colorbar(im, ax=ax, label='Correlation')

# Add correlation values as text
for i in range(5):
    for j in range(16):
        val = corr_matrix[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

# 2. Bar plot of correlations with gap
ax = axes[0, 1]
colors = ['green' if c > 0 else 'red' for c in correlations['gap']]
ax.bar(range(16), correlations['gap'], color=colors, edgecolor='black', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Neuron')
ax.set_ylabel('Correlation with gap')
ax.set_title('Offset correlation with gap')
ax.set_xticks(range(16))
ax.set_ylim(-1, 1)
ax.grid(True, alpha=0.3, axis='y')

# 3. Bar plot of correlations with m_pos
ax = axes[0, 2]
colors = ['green' if c > 0 else 'red' for c in correlations['m_pos']]
ax.bar(range(16), correlations['m_pos'], color=colors, edgecolor='black', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Neuron')
ax.set_ylabel('Correlation with m_pos')
ax.set_title('Offset correlation with M position')
ax.set_xticks(range(16))
ax.set_ylim(-1, 1)
ax.grid(True, alpha=0.3, axis='y')

# 4. Bar plot of correlations with s_pos
ax = axes[1, 0]
colors = ['green' if c > 0 else 'red' for c in correlations['s_pos']]
ax.bar(range(16), correlations['s_pos'], color=colors, edgecolor='black', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Neuron')
ax.set_ylabel('Correlation with s_pos')
ax.set_title('Offset correlation with S position')
ax.set_xticks(range(16))
ax.set_ylim(-1, 1)
ax.grid(True, alpha=0.3, axis='y')

# 5. Scatter: m_pos vs s_pos correlation for each neuron
ax = axes[1, 1]
ax.scatter(correlations['m_pos'], correlations['s_pos'], s=100, c='blue', edgecolors='black', alpha=0.7)
for i in range(16):
    ax.annotate(str(i), (correlations['m_pos'][i], correlations['s_pos'][i]),
                fontsize=8, ha='center', va='bottom')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.3, label='y=x')
ax.plot([-1, 1], [1, -1], 'r--', alpha=0.3, label='y=-x (antisymmetric)')
ax.set_xlabel('Correlation with m_pos')
ax.set_ylabel('Correlation with s_pos')
ax.set_title('m_pos vs s_pos correlation per neuron')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 6. Summary statistics
ax = axes[1, 2]
ax.axis('off')

# Check for antisymmetry: does corr(m_pos) ≈ -corr(s_pos)?
m_corrs = np.array(correlations['m_pos'])
s_corrs = np.array(correlations['s_pos'])
antisym_check = np.corrcoef(m_corrs, -s_corrs)[0, 1]

# R² for predicting offset from m_pos and s_pos (linear model)
# offset[n] ≈ a * m_pos + b * s_pos + c
from numpy.linalg import lstsq

r2_per_neuron = []
for neuron in range(16):
    y = offsets[:, neuron]
    X = np.column_stack([m_positions, s_positions, np.ones(n_pairs)])
    coeffs, residuals, rank, s = lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r2_per_neuron.append(r2)

summary_text = f"""SUMMARY

Antisymmetry check:
  corr(m_corrs, -s_corrs) = {antisym_check:.3f}
  (1.0 = perfect antisymmetry)

Linear model: offset[n] = a·m_pos + b·s_pos + c
R² per neuron:
"""

for n in range(16):
    summary_text += f"  n{n:2d}: R²={r2_per_neuron[n]:.3f}\n"

summary_text += f"\nMean R²: {np.mean(r2_per_neuron):.3f}"

ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
        ha='left', va='top', fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('docs/offset_correlation_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved to docs/offset_correlation_analysis.png")

# Additional analysis: separable model fit
print("\n" + "=" * 70)
print("SEPARABLE MODEL: offset ≈ f(m_pos) + g(s_pos)")
print("=" * 70)

# Fit f and g vectors for each neuron
# For each neuron n:
#   offset[n] for pair (m, s) ≈ f[m] + g[s]
#
# This is a linear regression problem

# Create design matrix for separable model
# For pair (m, s): one-hot for m_pos (10 dims) + one-hot for s_pos (10 dims)
X_sep = np.zeros((n_pairs, 20))
for i, d in enumerate(data):
    X_sep[i, d['m_pos']] = 1  # f component
    X_sep[i, 10 + d['s_pos']] = 1  # g component

print("\nSeparable model R² per neuron:")
f_vectors = []
g_vectors = []

for neuron in range(16):
    y = offsets[:, neuron]

    # Remove one dimension for identifiability (set f[0] = 0)
    X_reduced = X_sep[:, 1:]  # Remove first column (f[0])
    coeffs, residuals, rank, s = lstsq(X_reduced, y, rcond=None)

    # Reconstruct f and g
    f = np.zeros(10)
    f[1:] = coeffs[:9]  # f[0] = 0 (baseline)
    g = coeffs[9:]

    f_vectors.append(f)
    g_vectors.append(g)

    # Predict and compute R²
    y_pred = X_sep[:, 1:] @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"  Neuron {neuron:2d}: R² = {r2:.3f}")

f_vectors = np.array(f_vectors)  # (16, 10)
g_vectors = np.array(g_vectors)  # (16, 10)

# Check antisymmetry through W_out
print("\n" + "=" * 70)
print("ANTISYMMETRY CHECK: W_out @ f(pos) vs W_out @ g(pos)")
print("=" * 70)

wout_f = W_out @ f_vectors  # (10, 10) - W_out @ f for each position
wout_g = W_out @ g_vectors  # (10, 10)

# For antisymmetry: W_out @ f(m) should equal -W_out @ g(m) for each m
print("\nW_out @ f(pos) vs -W_out @ g(pos) for diagonal (output_pos = input_pos):")
print("pos | W_out@f | W_out@g | -W_out@g | f + g (should be ~0)")
print("----|---------|---------|----------|---------------------")

for pos in range(10):
    wf = wout_f[pos, pos]  # W_out[pos] @ f[pos]
    wg = wout_g[pos, pos]  # W_out[pos] @ g[pos]
    print(f"  {pos} | {wf:+7.2f} | {wg:+7.2f} | {-wg:+8.2f} | {wf + wg:+7.2f}")

# Overall antisymmetry correlation
diag_f = np.diag(wout_f)
diag_g = np.diag(wout_g)
antisym_corr = np.corrcoef(diag_f, -diag_g)[0, 1]
print(f"\nCorrelation of W_out@f with -W_out@g (diagonal): {antisym_corr:.4f}")

# Plot f and g
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. f vectors heatmap
ax = axes[0, 0]
im = ax.imshow(f_vectors.T, cmap='RdBu_r', aspect='auto')
ax.set_xlabel('Neuron')
ax.set_ylabel('Position (m_pos)')
ax.set_title('f(m_pos) for each neuron')
ax.set_xticks(range(16))
ax.set_yticks(range(10))
plt.colorbar(im, ax=ax)

# 2. g vectors heatmap
ax = axes[0, 1]
im = ax.imshow(g_vectors.T, cmap='RdBu_r', aspect='auto')
ax.set_xlabel('Neuron')
ax.set_ylabel('Position (s_pos)')
ax.set_title('g(s_pos) for each neuron')
ax.set_xticks(range(16))
ax.set_yticks(range(10))
plt.colorbar(im, ax=ax)

# 3. W_out @ f and W_out @ g (diagonal)
ax = axes[1, 0]
x = np.arange(10)
width = 0.35
ax.bar(x - width/2, diag_f, width, label='W_out @ f(pos)', color='blue', alpha=0.7)
ax.bar(x + width/2, diag_g, width, label='W_out @ g(pos)', color='red', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('W_out @ vector')
ax.set_title('W_out @ f(pos) and W_out @ g(pos) at output=input position')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4. Antisymmetry scatter
ax = axes[1, 1]
ax.scatter(diag_f, -diag_g, s=100, c='purple', edgecolors='black')
for i in range(10):
    ax.annotate(str(i), (diag_f[i], -diag_g[i]), fontsize=10, ha='left', va='bottom')
lim = max(abs(diag_f).max(), abs(diag_g).max()) * 1.2
ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label='y=x (perfect antisym)')
ax.set_xlabel('W_out @ f(pos)')
ax.set_ylabel('-W_out @ g(pos)')
ax.set_title(f'Antisymmetry check (corr={antisym_corr:.3f})')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('docs/offset_separable_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved to docs/offset_separable_analysis.png")
