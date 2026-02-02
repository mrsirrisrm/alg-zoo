"""
Analyze offset correlations by neuron category (waves, comps, bridges, other).
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

# Neuron categories from doc 27
categories = {
    'Comps': [1, 6, 7, 8],           # Clip to 0 on input; rebuild trajectory encodes position
    'Waves': [0, 10, 11, 12, 14],    # Recurrent cascade carriers; ignore input directly
    'Bridges': [3, 5, 13, 15],       # Couple comps ↔ waves; receive n4 broadcast
    'n4': [4],                        # One-shot broadcast relay
    'n2': [2],                        # Memory latch
    'n9': [9],                        # Secondary relay; critical for small-gap reversed
}

# Flatten for lookup
neuron_to_category = {}
for cat, neurons in categories.items():
    for n in neurons:
        neuron_to_category[n] = cat

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

        states_fwd = run_stepwise([(pos1, M_val), (pos2, S_val)])
        h_fwd = states_fwd[-1]

        states_rev = run_stepwise([(pos1, S_val), (pos2, M_val)])
        h_rev = states_rev[-1]

        offset = h_fwd - h_rev

        data.append({
            'pos1': pos1,
            'pos2': pos2,
            'm_pos': pos1,  # In forward, M is at pos1
            's_pos': pos2,  # In forward, S is at pos2
            'gap': abs(pos2 - pos1),
            'offset': offset,
        })

n_pairs = len(data)
offsets = np.array([d['offset'] for d in data])
m_positions = np.array([d['m_pos'] for d in data])
s_positions = np.array([d['s_pos'] for d in data])

# Compute correlations for each neuron
correlations_m = []
correlations_s = []
for neuron in range(16):
    neuron_offsets = offsets[:, neuron]
    corr_m = np.corrcoef(neuron_offsets, m_positions)[0, 1]
    corr_s = np.corrcoef(neuron_offsets, s_positions)[0, 1]
    correlations_m.append(corr_m)
    correlations_s.append(corr_s)

correlations_m = np.array(correlations_m)
correlations_s = np.array(correlations_s)

print("=" * 70)
print("OFFSET CORRELATIONS BY NEURON CATEGORY")
print("=" * 70)

print("\nNeuron categories:")
for cat, neurons in categories.items():
    print(f"  {cat}: {neurons}")

print("\n" + "-" * 70)
print("Correlation with m_pos and s_pos by neuron:")
print("-" * 70)
print("\nNeuron | Category | corr(m_pos) | corr(s_pos) | dominant | sign")
print("-------|----------|-------------|-------------|----------|------")

for n in range(16):
    cat = neuron_to_category[n]
    cm = correlations_m[n]
    cs = correlations_s[n]

    # Dominant correlation (higher absolute value)
    if abs(cm) > abs(cs):
        dominant = 'm_pos'
        sign = '+' if cm > 0 else '-'
    else:
        dominant = 's_pos'
        sign = '+' if cs > 0 else '-'

    print(f"  {n:2d}   | {cat:8s} | {cm:+11.3f} | {cs:+11.3f} | {dominant:8s} | {sign}")

print("\n" + "=" * 70)
print("SUMMARY BY CATEGORY")
print("=" * 70)

# For each category, count how many neurons are positively vs negatively correlated with m_pos
# (Since corr(m) = -corr(s), this also tells us s_pos correlation direction)

print("\nCategory breakdown:")
print("  - '+m_pos' means offset increases when M is at later position")
print("  - '-m_pos' means offset decreases when M is at later position")
print("  - (Due to antisymmetry: +m_pos = -s_pos, -m_pos = +s_pos)")

print("\nCategory   | Neurons | +m_pos (−s_pos) | −m_pos (+s_pos)")
print("-----------|---------|-----------------|----------------")

category_summary = {}
for cat in ['Comps', 'Waves', 'Bridges', 'n4', 'n2', 'n9']:
    neurons = categories[cat]
    pos_m = [n for n in neurons if correlations_m[n] > 0]
    neg_m = [n for n in neurons if correlations_m[n] < 0]

    category_summary[cat] = {
        'neurons': neurons,
        'pos_m': pos_m,
        'neg_m': neg_m,
    }

    pos_str = str(pos_m) if pos_m else "[]"
    neg_str = str(neg_m) if neg_m else "[]"
    print(f"{cat:10s} | {neurons!s:7s} | {pos_str:15s} | {neg_str}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("""
The offset h_fwd - h_rev encodes the difference between:
  - Forward: M at pos1, S at pos2
  - Reverse: S at pos1, M at pos2

A neuron with +corr(m_pos) has HIGHER activation in forward when M is later.
A neuron with -corr(m_pos) has LOWER activation in forward when M is later.

Due to perfect antisymmetry: corr(m_pos) = -corr(s_pos)

So:
  +m_pos neurons: respond more to where M is (in forward trajectory)
  -m_pos neurons: respond more to where S is (in forward trajectory)
""")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Color map for categories
cat_colors = {
    'Comps': 'red',
    'Waves': 'blue',
    'Bridges': 'green',
    'n4': 'orange',
    'n2': 'purple',
    'n9': 'brown',
}

# 1. Bar chart of m_pos correlations colored by category
ax = axes[0, 0]
colors = [cat_colors[neuron_to_category[n]] for n in range(16)]
bars = ax.bar(range(16), correlations_m, color=colors, edgecolor='black', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Neuron')
ax.set_ylabel('Correlation with m_pos')
ax.set_title('Offset correlation with M position (colored by category)')
ax.set_xticks(range(16))
ax.set_ylim(-0.7, 0.7)
ax.grid(True, alpha=0.3, axis='y')

# Add category labels
for n in range(16):
    cat = neuron_to_category[n]
    ax.annotate(cat[0], (n, correlations_m[n]),
                ha='center', va='bottom' if correlations_m[n] > 0 else 'top',
                fontsize=8, fontweight='bold')

# 2. Bar chart of s_pos correlations colored by category
ax = axes[0, 1]
bars = ax.bar(range(16), correlations_s, color=colors, edgecolor='black', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Neuron')
ax.set_ylabel('Correlation with s_pos')
ax.set_title('Offset correlation with S position (colored by category)')
ax.set_xticks(range(16))
ax.set_ylim(-0.7, 0.7)
ax.grid(True, alpha=0.3, axis='y')

for n in range(16):
    cat = neuron_to_category[n]
    ax.annotate(cat[0], (n, correlations_s[n]),
                ha='center', va='bottom' if correlations_s[n] > 0 else 'top',
                fontsize=8, fontweight='bold')

# 3. Scatter plot of m_pos vs s_pos correlation by category
ax = axes[1, 0]
for cat, neurons in categories.items():
    x = [correlations_m[n] for n in neurons]
    y = [correlations_s[n] for n in neurons]
    ax.scatter(x, y, c=cat_colors[cat], s=150, label=cat, edgecolors='black', alpha=0.8)
    for n in neurons:
        ax.annotate(str(n), (correlations_m[n], correlations_s[n]),
                    fontsize=9, ha='center', va='bottom')

ax.plot([-0.7, 0.7], [0.7, -0.7], 'k--', alpha=0.5, label='y=-x (antisym)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Correlation with m_pos')
ax.set_ylabel('Correlation with s_pos')
ax.set_title('m_pos vs s_pos correlation by category')
ax.set_xlim(-0.7, 0.7)
ax.set_ylim(-0.7, 0.7)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 4. Summary table
ax = axes[1, 1]
ax.axis('off')

summary_text = "CATEGORY SPLIT SUMMARY\n\n"
summary_text += "Category   | +m_pos (−s_pos)  | −m_pos (+s_pos)\n"
summary_text += "-----------|------------------|----------------\n"

for cat in ['Comps', 'Waves', 'Bridges', 'n4', 'n2', 'n9']:
    info = category_summary[cat]
    pos_str = str(info['pos_m']) if info['pos_m'] else "[]"
    neg_str = str(info['neg_m']) if info['neg_m'] else "[]"
    summary_text += f"{cat:10s} | {pos_str:16s} | {neg_str}\n"

summary_text += "\n\nKEY OBSERVATIONS:\n\n"

# Check if categories cleanly split
comps_split = len(category_summary['Comps']['pos_m']) > 0 and len(category_summary['Comps']['neg_m']) > 0
waves_split = len(category_summary['Waves']['pos_m']) > 0 and len(category_summary['Waves']['neg_m']) > 0
bridges_split = len(category_summary['Bridges']['pos_m']) > 0 and len(category_summary['Bridges']['neg_m']) > 0

summary_text += f"• Comps split: {'YES' if comps_split else 'NO'}\n"
summary_text += f"• Waves split: {'YES' if waves_split else 'NO'}\n"
summary_text += f"• Bridges split: {'YES' if bridges_split else 'NO'}\n"

# Mean correlation by category
summary_text += "\nMean |corr(m_pos)| by category:\n"
for cat in ['Comps', 'Waves', 'Bridges', 'n4', 'n2', 'n9']:
    neurons = categories[cat]
    mean_abs_corr = np.mean([abs(correlations_m[n]) for n in neurons])
    summary_text += f"  {cat}: {mean_abs_corr:.3f}\n"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        ha='left', va='top', fontsize=11, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Add legend for colors
handles = [plt.Rectangle((0,0),1,1, facecolor=cat_colors[cat], edgecolor='black', alpha=0.7)
           for cat in ['Comps', 'Waves', 'Bridges', 'n4', 'n2', 'n9']]
fig.legend(handles, ['Comps', 'Waves', 'Bridges', 'n4', 'n2', 'n9'],
           loc='upper center', ncol=6, fontsize=10, bbox_to_anchor=(0.5, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('docs/offset_by_category.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved to docs/offset_by_category.png")
