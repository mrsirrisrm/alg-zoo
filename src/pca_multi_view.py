"""
Visualize hidden states and ReLU boundaries in multiple PC pairs,
especially those where the offset has significant variance.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch as th
from sklearn.decomposition import PCA
from alg_zoo.architectures import DistRNN

def load_local_model():
    path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
    state_dict = th.load(path, weights_only=True, map_location='cpu')
    seq_len, hidden_size = state_dict['linear.weight'].shape
    model = DistRNN(hidden_size=hidden_size, seq_len=seq_len, bias=False)
    model.load_state_dict(state_dict)
    return model

model = load_local_model()
W_ih = model.rnn.weight_ih_l0.detach().numpy().squeeze()
W_hh = model.rnn.weight_hh_l0.detach().numpy()
W_out = model.linear.weight.detach().numpy()

COMPS = [1, 6, 7, 8]
WAVES = [0, 10, 11, 12, 14]
BRIDGES = [3, 5, 13, 15]
OTHERS = [2, 4, 9]

NEURON_COLORS = {n: 'red' for n in COMPS}
NEURON_COLORS.update({n: 'blue' for n in WAVES})
NEURON_COLORS.update({n: 'green' for n in BRIDGES})
NEURON_COLORS.update({n: 'gray' for n in OTHERS})

def run_stepwise(impulses):
    h = np.zeros(16)
    states = [h.copy()]
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        states.append(h.copy())
    return states

def get_relu_boundary_line_general(neuron_idx, pca, pc_x, pc_y, xlim, ylim, x_t=0.0):
    """
    Project ReLU boundary for neuron_idx onto arbitrary PC pair.

    Boundary: W_hh[n,:] @ h + W_ih[n] * x_t = 0
    In PC space with other PCs at mean: solve for line in (pc_x, pc_y) plane.
    """
    n_i = W_hh[neuron_idx, :]
    PC_x = pca.components_[pc_x]
    PC_y = pca.components_[pc_y]
    mean = pca.mean_

    a = np.dot(n_i, PC_x)
    b = np.dot(n_i, PC_y)
    c = -W_ih[neuron_idx] * x_t - np.dot(n_i, mean)

    if abs(b) < 1e-10 and abs(a) < 1e-10:
        return None

    points = []
    if abs(a) > 1e-10:
        for y_val in [ylim[0], ylim[1]]:
            x_val = (-c - b * y_val) / a
            if xlim[0] <= x_val <= xlim[1]:
                points.append((x_val, y_val))

    if abs(b) > 1e-10:
        for x_val in [xlim[0], xlim[1]]:
            y_val = (-c - a * x_val) / b
            if ylim[0] <= y_val <= ylim[1]:
                points.append((x_val, y_val))

    points = list(set(points))
    if len(points) >= 2:
        points.sort()
        return points[0], points[1]
    return None

def collect_data():
    """Collect hidden states and offsets."""
    all_states = []
    final_states_fwd = []
    final_states_rev = []
    offsets = []

    M_val = 1.0
    S_val = 0.8

    for m_pos in range(10):
        for s_pos in range(10):
            if m_pos == s_pos:
                continue

            states_fwd = run_stepwise([(m_pos, M_val), (s_pos, S_val)])
            states_rev = run_stepwise([(m_pos, S_val), (s_pos, M_val)])

            all_states.extend(states_fwd[1:])
            all_states.extend(states_rev[1:])

            final_states_fwd.append(states_fwd[-1])
            final_states_rev.append(states_rev[-1])
            offsets.append(states_fwd[-1] - states_rev[-1])

    return (np.array(all_states), np.array(final_states_fwd),
            np.array(final_states_rev), np.array(offsets))

def create_multi_view():
    print("Collecting data...")
    all_states, final_fwd, final_rev, offsets = collect_data()

    # Fit PCA on final states
    final_all = np.vstack([final_fwd, final_rev])
    pca = PCA()
    pca.fit(final_all)

    # Project data
    proj_fwd = pca.transform(final_fwd)
    proj_rev = pca.transform(final_rev)
    proj_offset = offsets @ pca.components_.T  # Project offsets

    # PC pairs to visualize (based on where offset variance lives)
    pc_pairs = [
        (0, 1, "PC1 vs PC2 (58% offset var)"),
        (1, 4, "PC2 vs PC5 (50% offset var)"),
        (3, 4, "PC4 vs PC5 (17% offset var)"),
        (0, 8, "PC1 vs PC9 (25% offset var)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, (pc_x, pc_y, title) in enumerate(pc_pairs):
        # Left column: hidden states with ReLU boundaries
        ax = axes[0, idx]

        ax.scatter(proj_fwd[:, pc_x], proj_fwd[:, pc_y], c='blue', alpha=0.5, s=30, label='Fwd')
        ax.scatter(proj_rev[:, pc_x], proj_rev[:, pc_y], c='red', alpha=0.5, s=30, label='Rev')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Add ReLU boundaries
        for n in range(16):
            line = get_relu_boundary_line_general(n, pca, pc_x, pc_y, xlim, ylim)
            if line:
                p1, p2 = line
                color = NEURON_COLORS[n]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color,
                       alpha=0.4, linewidth=1.5)

        ax.set_xlabel(f'PC{pc_x+1}', fontsize=11)
        ax.set_ylabel(f'PC{pc_y+1}', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom row: offset vectors in this PC pair
        ax = axes[1, idx]

        # Plot offset as arrows from origin
        for i in range(len(offsets)):
            ax.arrow(0, 0, proj_offset[i, pc_x], proj_offset[i, pc_y],
                    head_width=0.3, head_length=0.2, fc='purple', ec='purple', alpha=0.3)

        ax.scatter(proj_offset[:, pc_x], proj_offset[:, pc_y], c='purple', alpha=0.5, s=30)

        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)

        var_x = np.var(proj_offset[:, pc_x])
        var_y = np.var(proj_offset[:, pc_y])
        total_var = np.sum(np.var(proj_offset, axis=0))

        ax.set_xlabel(f'PC{pc_x+1} ({var_x/total_var*100:.1f}% offset var)', fontsize=11)
        ax.set_ylabel(f'PC{pc_y+1} ({var_y/total_var*100:.1f}% offset var)', fontsize=11)
        ax.set_title(f'Offset vectors in PC{pc_x+1}-PC{pc_y+1}', fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/pca_multi_view.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved to docs/pca_multi_view.png")

    # Also create a summary of offset variance in each PC pair
    print("\n" + "=" * 60)
    print("OFFSET VARIANCE BY PC PAIR")
    print("=" * 60)

    total_var = np.sum(np.var(proj_offset, axis=0))

    for pc_x in range(6):
        for pc_y in range(pc_x+1, 6):
            var_x = np.var(proj_offset[:, pc_x])
            var_y = np.var(proj_offset[:, pc_y])
            combined = (var_x + var_y) / total_var * 100
            print(f"PC{pc_x+1}-PC{pc_y+1}: {combined:.1f}%")

if __name__ == "__main__":
    create_multi_view()
