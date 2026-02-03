"""
Detailed tropical geometry analysis: ReLU boundaries in PCA space.

Creates static plots showing how the hidden state trajectory crosses
ReLU boundaries (tropical hyperplanes) for specific position pairs.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import os
import torch as th
from alg_zoo.architectures import DistRNN


def load_local_model():
    """Load model from local file."""
    path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
    state_dict = th.load(path, weights_only=True, map_location='cpu')
    seq_len, hidden_size = state_dict['linear.weight'].shape
    model = DistRNN(hidden_size=hidden_size, seq_len=seq_len, bias=False)
    model.load_state_dict(state_dict)
    return model


model = load_local_model()
model.eval()

W_ih = model.rnn.weight_ih_l0.detach().numpy().squeeze()
W_hh = model.rnn.weight_hh_l0.detach().numpy()
W_out = model.linear.weight.detach().numpy()

M_val = 1.0
S_val = 0.8

# Neuron categories
COMPS = [1, 6, 7, 8]
WAVES = [0, 10, 11, 12, 14]
BRIDGES = [3, 5, 13, 15]
OTHERS = [2, 4, 9]

NEURON_COLORS = {}
for n in COMPS:
    NEURON_COLORS[n] = 'red'
for n in WAVES:
    NEURON_COLORS[n] = 'blue'
for n in BRIDGES:
    NEURON_COLORS[n] = 'green'
for n in OTHERS:
    NEURON_COLORS[n] = 'gray'

NEURON_CATEGORY = {}
for n in COMPS:
    NEURON_CATEGORY[n] = 'comp'
for n in WAVES:
    NEURON_CATEGORY[n] = 'wave'
for n in BRIDGES:
    NEURON_CATEGORY[n] = 'bridge'
for n in OTHERS:
    NEURON_CATEGORY[n] = 'other'


def run_stepwise(impulses):
    """Run RNN and return hidden states at each timestep."""
    h = np.zeros(16)
    states = [h.copy()]
    pre_states = [h.copy()]
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        states.append(h.copy())
        pre_states.append(pre.copy())
    return states, pre_states


def get_relu_boundary_line(neuron_idx, pca, x_t=0.0, xlim=(-30, 30), ylim=(-30, 30)):
    """Compute the ReLU boundary line for neuron i in PCA space."""
    n_i = W_hh[neuron_idx, :]
    PC1 = pca.components_[0]
    PC2 = pca.components_[1]
    mean = pca.mean_

    a = np.dot(n_i, PC1)
    b = np.dot(n_i, PC2)
    c = -W_ih[neuron_idx] * x_t - np.dot(n_i, mean)

    eps = 1e-10
    points = []

    if abs(a) > eps:
        beta = (c - a * xlim[0]) / b if abs(b) > eps else None
        if beta is not None and ylim[0] <= beta <= ylim[1]:
            points.append((xlim[0], beta))
        beta = (c - a * xlim[1]) / b if abs(b) > eps else None
        if beta is not None and ylim[0] <= beta <= ylim[1]:
            points.append((xlim[1], beta))

    if abs(b) > eps:
        alpha = (c - b * ylim[0]) / a if abs(a) > eps else None
        if alpha is not None and xlim[0] <= alpha <= xlim[1]:
            points.append((alpha, ylim[0]))
        alpha = (c - b * ylim[1]) / a if abs(a) > eps else None
        if alpha is not None and xlim[0] <= alpha <= xlim[1]:
            points.append((alpha, ylim[1]))

    if len(points) >= 2:
        points = list(set(points))
        if len(points) >= 2:
            return points[0], points[1]
    return None


def analyze_boundary_crossings(states, pre_states, pca):
    """Analyze which boundaries are crossed at each timestep."""
    crossings = []
    for t in range(1, 10):
        pre_prev = pre_states[t]  # Pre-activation at t (uses h_{t-1})
        pre_curr = pre_states[t + 1]  # Pre-activation at t+1 (uses h_t)

        crossed_at_t = []
        for n in range(16):
            # Did neuron n cross its boundary between t and t+1?
            # Crossing means pre changed sign
            if (pre_prev[n] > 0) != (pre_curr[n] > 0):
                direction = 'on' if pre_curr[n] > 0 else 'off'
                crossed_at_t.append((n, direction))

        crossings.append(crossed_at_t)
    return crossings


def create_tropical_analysis_plot(pos1, pos2, output_name):
    """Create detailed tropical geometry analysis plot."""

    print(f"\nAnalyzing positions ({pos1}, {pos2})")

    # Run forward and reverse
    states_fwd, pre_fwd = run_stepwise([(pos1, M_val), (pos2, S_val)])
    states_rev, pre_rev = run_stepwise([(pos1, S_val), (pos2, M_val)])

    # Collect all states for PCA
    all_states = []
    for s_mag in np.linspace(0.1, 1.0, 10):
        states_f, _ = run_stepwise([(pos1, M_val), (pos2, s_mag)])
        states_r, _ = run_stepwise([(pos1, s_mag), (pos2, M_val)])
        all_states.extend(states_f[1:])
        all_states.extend(states_r[1:])

    all_states = np.array(all_states)
    pca = PCA(n_components=2)
    pca.fit(all_states)

    # Get PCA coordinates
    all_pca = pca.transform(all_states)
    margin = 5
    xlim = (all_pca[:, 0].min() - margin, all_pca[:, 0].max() + margin)
    ylim = (all_pca[:, 1].min() - margin, all_pca[:, 1].max() + margin)

    pca_fwd = pca.transform(np.array(states_fwd[1:]))
    pca_rev = pca.transform(np.array(states_rev[1:]))

    # Analyze crossings
    crossings_fwd = analyze_boundary_crossings(states_fwd, pre_fwd, pca)
    crossings_rev = analyze_boundary_crossings(states_rev, pre_rev, pca)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Forward trajectory with boundaries
    ax = axes[0, 0]
    for neuron in range(16):
        boundary = get_relu_boundary_line(neuron, pca, x_t=0.0, xlim=xlim, ylim=ylim)
        if boundary:
            p1, p2 = boundary
            active = states_fwd[-1][neuron] > 0
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                   color=NEURON_COLORS[neuron],
                   alpha=0.6 if active else 0.25,
                   linewidth=1.5 if active else 0.8,
                   linestyle='-' if active else '--',
                   label=f'n{neuron}' if neuron < 4 else None)

    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    for i in range(9):
        ax.plot(pca_fwd[i:i+2, 0], pca_fwd[i:i+2, 1], color=colors[i], linewidth=3, zorder=10)
    ax.scatter(pca_fwd[:, 0], pca_fwd[:, 1], c=range(10), cmap='viridis', s=150, zorder=15,
               edgecolors='white', linewidth=2)
    for i in range(10):
        ax.annotate(f't={i+1}', (pca_fwd[i, 0], pca_fwd[i, 1]), fontsize=8,
                   ha='left', va='bottom', color='black')

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'Forward: M@{pos1}, S@{pos2}\n(solid=active at t=10, dashed=dead)', fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)

    # 2. Reverse trajectory with boundaries
    ax = axes[0, 1]
    for neuron in range(16):
        boundary = get_relu_boundary_line(neuron, pca, x_t=0.0, xlim=xlim, ylim=ylim)
        if boundary:
            p1, p2 = boundary
            active = states_rev[-1][neuron] > 0
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                   color=NEURON_COLORS[neuron],
                   alpha=0.6 if active else 0.25,
                   linewidth=1.5 if active else 0.8,
                   linestyle='-' if active else '--')

    for i in range(9):
        ax.plot(pca_rev[i:i+2, 0], pca_rev[i:i+2, 1], color=colors[i], linewidth=3, zorder=10)
    ax.scatter(pca_rev[:, 0], pca_rev[:, 1], c=range(10), cmap='viridis', s=150, zorder=15,
               edgecolors='white', linewidth=2)
    for i in range(10):
        ax.annotate(f't={i+1}', (pca_rev[i, 0], pca_rev[i, 1]), fontsize=8,
                   ha='left', va='bottom', color='black')

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'Reverse: S@{pos1}, M@{pos2}\n(solid=active at t=10, dashed=dead)', fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)

    # 3. Both trajectories overlaid
    ax = axes[0, 2]
    for neuron in range(16):
        boundary = get_relu_boundary_line(neuron, pca, x_t=0.0, xlim=xlim, ylim=ylim)
        if boundary:
            p1, p2 = boundary
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                   color=NEURON_COLORS[neuron], alpha=0.3, linewidth=1)

    ax.plot(pca_fwd[:, 0], pca_fwd[:, 1], 'o-', color='blue', linewidth=2,
           markersize=8, label='Forward', alpha=0.8)
    ax.plot(pca_rev[:, 0], pca_rev[:, 1], 's-', color='red', linewidth=2,
           markersize=8, label='Reverse', alpha=0.8)

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Forward vs Reverse trajectories', fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Boundary crossings over time
    ax = axes[1, 0]

    # Create crossing timeline
    y_fwd = 0.6
    y_rev = 0.4

    ax.axhline(y_fwd, color='blue', alpha=0.3, linewidth=10)
    ax.axhline(y_rev, color='red', alpha=0.3, linewidth=10)

    for t, crossed in enumerate(crossings_fwd):
        for n, direction in crossed:
            marker = '^' if direction == 'on' else 'v'
            ax.scatter([t + 1.5], [y_fwd], marker=marker, s=100,
                      color=NEURON_COLORS[n], edgecolors='black', zorder=10)
            ax.annotate(f'n{n}', (t + 1.5, y_fwd + 0.05), fontsize=7,
                       ha='center', va='bottom', color=NEURON_COLORS[n])

    for t, crossed in enumerate(crossings_rev):
        for n, direction in crossed:
            marker = '^' if direction == 'on' else 'v'
            ax.scatter([t + 1.5], [y_rev], marker=marker, s=100,
                      color=NEURON_COLORS[n], edgecolors='black', zorder=10)
            ax.annotate(f'n{n}', (t + 1.5, y_rev - 0.08), fontsize=7,
                       ha='center', va='top', color=NEURON_COLORS[n])

    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.1, 0.9)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_yticks([y_rev, y_fwd])
    ax.set_yticklabels(['Reverse', 'Forward'])
    ax.set_title('Boundary crossings (▲=on, ▼=off)', fontsize=12)
    ax.set_xticks(range(1, 11))

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Comp'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Wave'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Bridge'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # 5. Activation pattern comparison
    ax = axes[1, 1]
    ap_fwd = np.array([states_fwd[t] > 0 for t in range(1, 11)]).T.astype(int)
    ap_rev = np.array([states_rev[t] > 0 for t in range(1, 11)]).T.astype(int)

    # Show difference: 0=same dead, 1=fwd active only, 2=rev active only, 3=both active
    diff = ap_fwd + 2 * ap_rev  # 0=both dead, 1=fwd only, 2=rev only, 3=both

    cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'red', 'purple'])
    ax.imshow(diff, cmap=cmap, aspect='auto', interpolation='nearest')
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Neuron', fontsize=12)
    ax.set_title('Activation patterns\n(blue=fwd only, red=rev only, purple=both)', fontsize=12)
    ax.set_xticks(range(10))
    ax.set_xticklabels(range(1, 11))
    ax.set_yticks(range(16))

    # Color neuron labels
    for i, label in enumerate(ax.get_yticklabels()):
        label.set_color(NEURON_COLORS[i])

    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')

    # Compute offset
    offset = np.array(states_fwd[-1]) - np.array(states_rev[-1])
    offset_norm = np.linalg.norm(offset)

    # Compute discriminative direction projection
    pred_fwd = np.argmax(W_out @ states_fwd[-1])
    pred_rev = np.argmax(W_out @ states_rev[-1])
    target_fwd = pos2
    target_rev = pos1

    discrim_fwd = W_out[target_fwd] - W_out[pos1]  # s - m in forward
    discrim_rev = W_out[target_rev] - W_out[pos2]  # s - m in reverse

    margin_fwd = np.dot(offset, discrim_fwd)
    margin_rev = np.dot(-offset, discrim_rev)

    n_active_fwd = np.sum(np.array(states_fwd[-1]) > 0)
    n_active_rev = np.sum(np.array(states_rev[-1]) > 0)

    # Count crossings
    n_cross_fwd = sum(len(c) for c in crossings_fwd)
    n_cross_rev = sum(len(c) for c in crossings_rev)

    summary = f"""TROPICAL GEOMETRY SUMMARY
Position pair: ({pos1}, {pos2})

Forward (M@{pos1}, S@{pos2}):
  Prediction: {pred_fwd} (target={target_fwd}) {"✓" if pred_fwd == target_fwd else "✗"}
  Active neurons at t=10: {n_active_fwd}/16
  Total boundary crossings: {n_cross_fwd}

Reverse (S@{pos1}, M@{pos2}):
  Prediction: {pred_rev} (target={target_rev}) {"✓" if pred_rev == target_rev else "✗"}
  Active neurons at t=10: {n_active_rev}/16
  Total boundary crossings: {n_cross_rev}

Offset Analysis:
  ||h_fwd - h_rev|| = {offset_norm:.2f}
  Margin (fwd direction) = {margin_fwd:.2f}

The trajectory crosses different tropical hyperplanes
in forward vs reverse, ending in different cells of
the tropical arrangement.

Neuron colors:
  RED = Comparators (1,6,7,8)
  BLUE = Waves (0,10,11,12,14)
  GREEN = Bridges (3,5,13,15)
  GRAY = Other (2,4,9)
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            ha='left', va='top', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    output_path = f'docs/tropical_pca_analysis_{output_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    # Analyze different position pairs
    pairs = [
        (0, 9, "0_9"),
        (4, 5, "4_5"),
        (0, 5, "0_5"),
        (2, 7, "2_7"),
    ]

    for pos1, pos2, name in pairs:
        create_tropical_analysis_plot(pos1, pos2, name)

    print("\nAll analysis plots complete!")
