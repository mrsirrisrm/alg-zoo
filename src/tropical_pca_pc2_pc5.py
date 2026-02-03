"""
Visualize ReLU boundaries and trajectories in PC2-PC5 space,
where 50% of offset variance lives.
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

def get_relu_boundary_line(neuron_idx, pca, pc_x, pc_y, xlim, ylim, x_t=0.0):
    """Project ReLU boundary onto PC pair."""
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

def create_pc2_pc5_visualization():
    print("Collecting hidden states...")

    # Collect all states for PCA fitting
    all_states = []
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

    all_states = np.array(all_states)
    pca = PCA()
    pca.fit(all_states)

    # PC indices (0-indexed)
    pc_x, pc_y = 1, 4  # PC2 and PC5

    # Select a few example position pairs to show trajectories
    example_pairs = [
        (0, 5, "pos 0,5"),
        (2, 7, "pos 2,7"),
        (4, 9, "pos 4,9"),
        (1, 3, "pos 1,3"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for idx, (m_pos, s_pos, label) in enumerate(example_pairs):
        ax = axes[idx // 2, idx % 2]

        # Get trajectories
        states_fwd = run_stepwise([(m_pos, M_val), (s_pos, S_val)])
        states_rev = run_stepwise([(m_pos, S_val), (s_pos, M_val)])

        traj_fwd = pca.transform(states_fwd)
        traj_rev = pca.transform(states_rev)

        # Plot trajectories
        ax.plot(traj_fwd[:, pc_x], traj_fwd[:, pc_y], 'b-', linewidth=2, alpha=0.8, label='Fwd (M first)')
        ax.plot(traj_rev[:, pc_x], traj_rev[:, pc_y], 'r-', linewidth=2, alpha=0.8, label='Rev (S first)')

        # Mark timesteps
        for t in range(11):
            ax.scatter(traj_fwd[t, pc_x], traj_fwd[t, pc_y], c='blue', s=50, zorder=5)
            ax.scatter(traj_rev[t, pc_x], traj_rev[t, pc_y], c='red', s=50, zorder=5)
            if t in [0, 5, 10]:
                ax.annotate(f't={t}', (traj_fwd[t, pc_x], traj_fwd[t, pc_y]),
                           fontsize=8, color='blue', ha='left')
                ax.annotate(f't={t}', (traj_rev[t, pc_x], traj_rev[t, pc_y]),
                           fontsize=8, color='red', ha='right')

        # Mark final states
        ax.scatter(traj_fwd[-1, pc_x], traj_fwd[-1, pc_y], c='blue', s=200, marker='*',
                  edgecolors='black', zorder=10, label='Fwd final')
        ax.scatter(traj_rev[-1, pc_x], traj_rev[-1, pc_y], c='red', s=200, marker='*',
                  edgecolors='black', zorder=10, label='Rev final')

        # Get axis limits
        all_x = np.concatenate([traj_fwd[:, pc_x], traj_rev[:, pc_x]])
        all_y = np.concatenate([traj_fwd[:, pc_y], traj_rev[:, pc_y]])
        margin = 0.2
        xlim = (all_x.min() - margin * (all_x.max() - all_x.min()),
                all_x.max() + margin * (all_x.max() - all_x.min()))
        ylim = (all_y.min() - margin * (all_y.max() - all_y.min()),
                all_y.max() + margin * (all_y.max() - all_y.min()))

        # Expand limits a bit for boundaries
        xlim = (xlim[0] - 2, xlim[1] + 2)
        ylim = (ylim[0] - 2, ylim[1] + 2)

        # Draw ReLU boundaries
        # Check which neurons are active at final state
        h_fwd_final = states_fwd[-1]
        h_rev_final = states_rev[-1]

        for n in range(16):
            line = get_relu_boundary_line(n, pca, pc_x, pc_y, xlim, ylim)
            if line:
                p1, p2 = line
                color = NEURON_COLORS[n]

                # Solid if active in both, dashed if different
                fwd_active = h_fwd_final[n] > 0
                rev_active = h_rev_final[n] > 0

                if fwd_active and rev_active:
                    style = '-'
                    alpha = 0.4
                elif not fwd_active and not rev_active:
                    style = ':'
                    alpha = 0.3
                else:
                    style = '--'
                    alpha = 0.7  # Highlight boundaries that differ!

                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color,
                       linestyle=style, alpha=alpha, linewidth=1.5)

                # Label boundaries that differ between fwd/rev
                if fwd_active != rev_active:
                    mid_x = (p1[0] + p2[0]) / 2
                    mid_y = (p1[1] + p2[1]) / 2
                    if xlim[0] < mid_x < xlim[1] and ylim[0] < mid_y < ylim[1]:
                        ax.annotate(f'n{n}', (mid_x, mid_y), fontsize=8,
                                   color=color, fontweight='bold')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(f'PC2 (40.8% offset var)', fontsize=11)
        ax.set_ylabel(f'PC5 (9.0% offset var)', fontsize=11)
        ax.set_title(f'{label}: M@{m_pos}, S@{s_pos}\nDashed = boundary crossing (fwd≠rev)', fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('ReLU Boundaries in PC2-PC5 Space (50% of offset variance)\n'
                 'Blue=wave, Red=comp, Green=bridge, Gray=other', fontsize=13)
    plt.tight_layout()
    plt.savefig('docs/tropical_pc2_pc5_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved to docs/tropical_pc2_pc5_analysis.png")

    # Also create a summary view with all final states
    create_summary_view(pca, pc_x, pc_y)

def create_summary_view(pca, pc_x, pc_y):
    """Show all 90 pairs' final states in PC2-PC5."""

    M_val = 1.0
    S_val = 0.8

    final_fwd = []
    final_rev = []
    offsets = []

    for m_pos in range(10):
        for s_pos in range(10):
            if m_pos == s_pos:
                continue
            h_fwd = run_stepwise([(m_pos, M_val), (s_pos, S_val)])[-1]
            h_rev = run_stepwise([(m_pos, S_val), (s_pos, M_val)])[-1]
            final_fwd.append(h_fwd)
            final_rev.append(h_rev)
            offsets.append(h_fwd - h_rev)

    final_fwd = np.array(final_fwd)
    final_rev = np.array(final_rev)
    offsets = np.array(offsets)

    proj_fwd = pca.transform(final_fwd)
    proj_rev = pca.transform(final_rev)
    proj_offset = offsets @ pca.components_.T

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. All final states
    ax = axes[0]
    ax.scatter(proj_fwd[:, pc_x], proj_fwd[:, pc_y], c='blue', alpha=0.6, s=50, label='Fwd')
    ax.scatter(proj_rev[:, pc_x], proj_rev[:, pc_y], c='red', alpha=0.6, s=50, label='Rev')

    # Draw offset arrows for a subset
    for i in range(0, len(offsets), 5):
        ax.annotate('', xy=(proj_fwd[i, pc_x], proj_fwd[i, pc_y]),
                   xytext=(proj_rev[i, pc_x], proj_rev[i, pc_y]),
                   arrowprops=dict(arrowstyle='->', color='purple', alpha=0.3))

    ax.set_xlabel('PC2 (40.8% offset var)', fontsize=11)
    ax.set_ylabel('PC5 (9.0% offset var)', fontsize=11)
    ax.set_title('All 90 pairs: final states', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Offset vectors
    ax = axes[1]
    ax.scatter(proj_offset[:, pc_x], proj_offset[:, pc_y], c='purple', alpha=0.6, s=50)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    # Draw from origin
    for i in range(0, len(offsets), 3):
        ax.arrow(0, 0, proj_offset[i, pc_x] * 0.95, proj_offset[i, pc_y] * 0.95,
                head_width=0.15, head_length=0.1, fc='purple', ec='purple', alpha=0.3)

    ax.set_xlabel('PC2 (40.8% offset var)', fontsize=11)
    ax.set_ylabel('PC5 (9.0% offset var)', fontsize=11)
    ax.set_title('Offset vectors (h_fwd - h_rev)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 3. ReLU boundaries with all states
    ax = axes[2]

    all_proj = np.vstack([proj_fwd, proj_rev])
    xlim = (all_proj[:, pc_x].min() - 2, all_proj[:, pc_x].max() + 2)
    ylim = (all_proj[:, pc_y].min() - 2, all_proj[:, pc_y].max() + 2)

    ax.scatter(proj_fwd[:, pc_x], proj_fwd[:, pc_y], c='blue', alpha=0.4, s=30, label='Fwd')
    ax.scatter(proj_rev[:, pc_x], proj_rev[:, pc_y], c='red', alpha=0.4, s=30, label='Rev')

    # Draw all ReLU boundaries
    for n in range(16):
        line = get_relu_boundary_line(n, pca, pc_x, pc_y, xlim, ylim)
        if line:
            p1, p2 = line
            color = NEURON_COLORS[n]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color,
                   alpha=0.5, linewidth=1.5, label=f'n{n}' if n < 4 else None)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('PC2 (40.8% offset var)', fontsize=11)
    ax.set_ylabel('PC5 (9.0% offset var)', fontsize=11)
    ax.set_title('ReLU boundaries in PC2-PC5\nBlue=wave, Red=comp, Green=bridge', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/tropical_pc2_pc5_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved to docs/tropical_pc2_pc5_summary.png")

if __name__ == "__main__":
    create_pc2_pc5_visualization()
