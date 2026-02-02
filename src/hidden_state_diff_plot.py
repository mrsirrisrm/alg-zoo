"""
Plot showing final hidden state difference between forward (M-S) and reverse (S-M)
for all position pairs with S_mag = 0.8.
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

# Test position pairs
position_pairs = [
    (0, 9),
    (2, 7),
    (0, 5),
    (4, 9),
    (4, 5),
]

fig, axes = plt.subplots(len(position_pairs), 4, figsize=(20, 4 * len(position_pairs)))

for row, (pos1, pos2) in enumerate(position_pairs):
    # Forward: M@pos1, S@pos2
    states_fwd = run_stepwise([(pos1, M_val), (pos2, S_val)])
    h_fwd = states_fwd[-1]
    pred_fwd = np.argmax(W_out @ h_fwd)

    # Reverse: S@pos1, M@pos2
    states_rev = run_stepwise([(pos1, S_val), (pos2, M_val)])
    h_rev = states_rev[-1]
    pred_rev = np.argmax(W_out @ h_rev)

    # Offset
    offset = h_fwd - h_rev

    gap = pos2 - pos1

    # Plot 1: Forward hidden state
    ax = axes[row, 0]
    colors_fwd = ['steelblue' if h > 0 else 'lightgray' for h in h_fwd]
    ax.bar(range(16), h_fwd, color=colors_fwd, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Activation')
    ax.set_title(f'Forward: M@{pos1}, S@{pos2} → pred={pred_fwd} (target={pos2})')
    ax.set_xticks(range(16))
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3, axis='y')
    n_active_fwd = np.sum(h_fwd > 0)
    ax.text(0.95, 0.95, f'{n_active_fwd}/16 active', transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Reverse hidden state
    ax = axes[row, 1]
    colors_rev = ['coral' if h > 0 else 'lightgray' for h in h_rev]
    ax.bar(range(16), h_rev, color=colors_rev, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Activation')
    ax.set_title(f'Reverse: S@{pos1}, M@{pos2} → pred={pred_rev} (target={pos1})')
    ax.set_xticks(range(16))
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3, axis='y')
    n_active_rev = np.sum(h_rev > 0)
    ax.text(0.95, 0.95, f'{n_active_rev}/16 active', transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 3: Offset (h_fwd - h_rev)
    ax = axes[row, 2]
    colors_off = ['green' if o > 0 else 'red' for o in offset]
    ax.bar(range(16), offset, color=colors_off, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Difference')
    ax.set_title(f'Offset: h_fwd - h_rev (||offset||={np.linalg.norm(offset):.1f})')
    ax.set_xticks(range(16))
    ax.set_ylim(-15, 15)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Info panel
    ax = axes[row, 3]
    ax.axis('off')

    # Compute margins
    logits_fwd = W_out @ h_fwd
    logits_rev = W_out @ h_rev
    margin_fwd = logits_fwd[pos2] - logits_fwd[pos1]  # target - other
    margin_rev = logits_rev[pos1] - logits_rev[pos2]  # target - other

    # Offset projection
    logits_offset = W_out @ offset

    info_text = f"""Position pair: ({pos1}, {pos2})
Gap: {gap} timesteps

Forward (M@{pos1}, S@{pos2}):
  Target: {pos2}
  Prediction: {pred_fwd} {'✓' if pred_fwd == pos2 else '✗'}
  Margin: {margin_fwd:.1f}

Reverse (S@{pos1}, M@{pos2}):
  Target: {pos1}
  Prediction: {pred_rev} {'✓' if pred_rev == pos1 else '✗'}
  Margin: {margin_rev:.1f}

Offset stats:
  ||offset||: {np.linalg.norm(offset):.1f}
  W_out @ offset [{pos1}]: {logits_offset[pos1]:+.1f}
  W_out @ offset [{pos2}]: {logits_offset[pos2]:+.1f}"""

    ax.text(0.1, 0.95, info_text, transform=ax.transAxes,
            ha='left', va='top', fontsize=11, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle(f'Final Hidden State Comparison: Forward vs Reverse (S_mag={S_val}, M_mag={M_val})',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('docs/hidden_state_fwd_rev_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved to docs/hidden_state_fwd_rev_comparison.png")

# Also create a summary plot showing just the offsets for all pairs
fig, axes = plt.subplots(1, len(position_pairs), figsize=(4 * len(position_pairs), 4))

for i, (pos1, pos2) in enumerate(position_pairs):
    states_fwd = run_stepwise([(pos1, M_val), (pos2, S_val)])
    states_rev = run_stepwise([(pos1, S_val), (pos2, M_val)])

    h_fwd = states_fwd[-1]
    h_rev = states_rev[-1]
    offset = h_fwd - h_rev

    ax = axes[i]
    colors = ['green' if o > 0 else 'red' for o in offset]
    ax.bar(range(16), offset, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Neuron')
    ax.set_ylabel('h_fwd - h_rev')
    ax.set_title(f'({pos1}, {pos2}) gap={pos2-pos1}')
    ax.set_xticks(range(0, 16, 2))
    ax.set_ylim(-15, 15)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Offset Comparison Across Position Pairs (S={S_val}, M={M_val})',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/hidden_state_offset_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved to docs/hidden_state_offset_summary.png")
