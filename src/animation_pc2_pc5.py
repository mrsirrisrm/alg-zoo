"""
Animate RNN trajectories in PC2-PC5 space where 50% of offset variance lives.
Shows how fwd/rev cross different ReLU boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

def create_animation(m_pos, s_pos):
    """Create animation for a specific position pair."""

    print(f"Creating animation for M@{m_pos}, S@{s_pos}...")

    M_val = 1.0
    S_val = 0.8

    # Collect states for PCA
    all_states = []
    for mp in range(10):
        for sp in range(10):
            if mp == sp:
                continue
            states_fwd = run_stepwise([(mp, M_val), (sp, S_val)])
            states_rev = run_stepwise([(mp, S_val), (sp, M_val)])
            all_states.extend(states_fwd[1:])
            all_states.extend(states_rev[1:])

    all_states = np.array(all_states)
    pca = PCA()
    pca.fit(all_states)

    # PC indices
    pc_x, pc_y = 1, 4  # PC2 and PC5

    # Get trajectories for this pair
    states_fwd = run_stepwise([(m_pos, M_val), (s_pos, S_val)])
    states_rev = run_stepwise([(m_pos, S_val), (s_pos, M_val)])

    traj_fwd = pca.transform(states_fwd)
    traj_rev = pca.transform(states_rev)

    # Compute axis limits
    all_traj = np.vstack([traj_fwd, traj_rev])
    margin = 0.3
    x_range = all_traj[:, pc_x].max() - all_traj[:, pc_x].min()
    y_range = all_traj[:, pc_y].max() - all_traj[:, pc_y].min()
    xlim = (all_traj[:, pc_x].min() - margin * x_range - 3,
            all_traj[:, pc_x].max() + margin * x_range + 3)
    ylim = (all_traj[:, pc_y].min() - margin * y_range - 3,
            all_traj[:, pc_y].max() + margin * y_range + 3)

    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 10))

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()

        # Current timestep (interpolate for smooth animation)
        t = frame / 5.0  # 5 frames per timestep
        t_int = int(t)
        t_frac = t - t_int

        if t_int >= 10:
            t_int = 10
            t_frac = 0

        # Get current hidden states
        if t_int < 10:
            h_fwd = (1 - t_frac) * np.array(states_fwd[t_int]) + t_frac * np.array(states_fwd[t_int + 1])
            h_rev = (1 - t_frac) * np.array(states_rev[t_int]) + t_frac * np.array(states_rev[t_int + 1])
        else:
            h_fwd = np.array(states_fwd[10])
            h_rev = np.array(states_rev[10])

        # Draw ReLU boundaries
        for n in range(16):
            line = get_relu_boundary_line(n, pca, pc_x, pc_y, xlim, ylim)
            if line:
                p1, p2 = line
                color = NEURON_COLORS[n]

                # Check if neuron is active
                fwd_active = h_fwd[n] > 0
                rev_active = h_rev[n] > 0

                if fwd_active and rev_active:
                    style = '-'
                    alpha = 0.3
                    lw = 1
                elif not fwd_active and not rev_active:
                    style = ':'
                    alpha = 0.2
                    lw = 1
                else:
                    # Different states - highlight!
                    style = '-'
                    alpha = 0.8
                    lw = 2.5

                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color,
                       linestyle=style, alpha=alpha, linewidth=lw)

                # Label boundaries that differ
                if fwd_active != rev_active:
                    mid_x = (p1[0] + p2[0]) / 2
                    mid_y = (p1[1] + p2[1]) / 2
                    if xlim[0] + 2 < mid_x < xlim[1] - 2 and ylim[0] + 2 < mid_y < ylim[1] - 2:
                        cat = 'W' if n in WAVES else 'C' if n in COMPS else 'B' if n in BRIDGES else 'O'
                        ax.annotate(f'n{n}({cat})', (mid_x, mid_y), fontsize=9,
                                   color=color, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # Draw trajectory history
        t_show = min(t_int + 1, 11)
        if t_show > 1:
            ax.plot(traj_fwd[:t_show, pc_x], traj_fwd[:t_show, pc_y],
                   'b-', linewidth=2, alpha=0.7)
            ax.plot(traj_rev[:t_show, pc_x], traj_rev[:t_show, pc_y],
                   'r-', linewidth=2, alpha=0.7)

        # Draw past points
        for i in range(t_show):
            size = 30 if i < t_int else 30 + 70 * t_frac
            ax.scatter(traj_fwd[i, pc_x], traj_fwd[i, pc_y], c='blue', s=size, alpha=0.6, zorder=5)
            ax.scatter(traj_rev[i, pc_x], traj_rev[i, pc_y], c='red', s=size, alpha=0.6, zorder=5)

        # Draw current position (interpolated)
        if t_int < 10:
            curr_fwd = (1 - t_frac) * traj_fwd[t_int] + t_frac * traj_fwd[t_int + 1]
            curr_rev = (1 - t_frac) * traj_rev[t_int] + t_frac * traj_rev[t_int + 1]
        else:
            curr_fwd = traj_fwd[10]
            curr_rev = traj_rev[10]

        ax.scatter(curr_fwd[pc_x], curr_fwd[pc_y], c='blue', s=200, marker='o',
                  edgecolors='black', linewidths=2, zorder=10, label='Fwd (M first)')
        ax.scatter(curr_rev[pc_x], curr_rev[pc_y], c='red', s=200, marker='o',
                  edgecolors='black', linewidths=2, zorder=10, label='Rev (S first)')

        # Draw offset arrow
        ax.annotate('', xy=(curr_fwd[pc_x], curr_fwd[pc_y]),
                   xytext=(curr_rev[pc_x], curr_rev[pc_y]),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=2, alpha=0.7))

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('PC2 (40.8% offset variance)', fontsize=12)
        ax.set_ylabel('PC5 (9.0% offset variance)', fontsize=12)
        ax.set_title(f'PC2-PC5 View: M@{m_pos}, S@{s_pos}  |  t = {t:.1f}\n'
                    f'Blue=wave, Red=comp, Green=bridge | Bold = fwd≠rev boundary',
                    fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add status box
        n_diff = sum(1 for n in range(16) if (h_fwd[n] > 0) != (h_rev[n] > 0))
        offset_norm = np.linalg.norm(h_fwd - h_rev)

        status = f"Boundaries crossed differently: {n_diff}\nOffset ||h_fwd - h_rev||: {offset_norm:.2f}"
        ax.text(0.02, 0.98, status, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        return []

    # Create animation (5 frames per timestep, 11 timesteps, plus 10 extra frames at end)
    n_frames = 11 * 5 + 10
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=100, blit=True)

    # Save
    output_path = f'docs/animation_pc2_pc5_pos_{m_pos}_{s_pos}.mp4'
    anim.save(output_path, writer='ffmpeg', fps=10, dpi=150)
    plt.close()
    print(f"Saved to {output_path}")

    # Also save as GIF
    gif_path = f'docs/animation_pc2_pc5_pos_{m_pos}_{s_pos}.gif'
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=100, blit=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=100, blit=True)
    anim.save(gif_path, writer='pillow', fps=10, dpi=100)
    plt.close()
    print(f"Saved to {gif_path}")

if __name__ == "__main__":
    # Create animations for a few interesting pairs
    pairs = [
        (0, 5),  # Wide gap
        (2, 7),  # Medium gap
        (4, 5),  # Adjacent
    ]

    for m_pos, s_pos in pairs:
        create_animation(m_pos, s_pos)
