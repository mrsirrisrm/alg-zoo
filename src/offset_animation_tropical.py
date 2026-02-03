"""
Create animations for multiple position pairs WITH ReLU boundaries (tropical hyperplanes).

Each ReLU boundary for neuron i is a hyperplane in 16D hidden state space:
    (W_hh[i, :] @ h + W_ih[i] * x_t) = 0

When intersected with the 2D PCA subspace, this becomes a line.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import subprocess
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

# Neuron categories (from previous analysis)
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


def run_stepwise(impulses):
    """Run RNN and return hidden states at each timestep."""
    h = np.zeros(16)
    states = [h.copy()]
    pre_states = [h.copy()]  # Pre-ReLU states
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        states.append(h.copy())
        pre_states.append(pre.copy())
    return states, pre_states


def get_relu_boundary_line(neuron_idx, pca, x_t=0.0, xlim=(-30, 30), ylim=(-30, 30)):
    """
    Compute the ReLU boundary line for neuron i in PCA space.

    The boundary in h-space is: W_hh[i, :] @ h + W_ih[i] * x_t = 0

    In PCA space (where h = mean + α*PC1 + β*PC2):
    W_hh[i, :] @ (mean + α*PC1 + β*PC2) + W_ih[i] * x_t = 0

    This gives: a*α + b*β = c  (a line in PCA coordinates)
    """
    n_i = W_hh[neuron_idx, :]  # Normal vector in h-space

    # PCA components (these project FROM h-space TO PCA space)
    PC1 = pca.components_[0]  # Shape (16,)
    PC2 = pca.components_[1]  # Shape (16,)
    mean = pca.mean_          # Shape (16,)

    # Coefficients in PCA space
    a = np.dot(n_i, PC1)
    b = np.dot(n_i, PC2)
    c = -W_ih[neuron_idx] * x_t - np.dot(n_i, mean)

    # Line equation: a*α + b*β = c
    # We need two points to draw the line

    # Handle near-vertical and near-horizontal lines
    eps = 1e-10
    points = []

    # Try intersections with plot boundaries
    if abs(a) > eps:
        # Intersection with left edge (α = xlim[0])
        beta = (c - a * xlim[0]) / b if abs(b) > eps else None
        if beta is not None and ylim[0] <= beta <= ylim[1]:
            points.append((xlim[0], beta))

        # Intersection with right edge (α = xlim[1])
        beta = (c - a * xlim[1]) / b if abs(b) > eps else None
        if beta is not None and ylim[0] <= beta <= ylim[1]:
            points.append((xlim[1], beta))

    if abs(b) > eps:
        # Intersection with bottom edge (β = ylim[0])
        alpha = (c - b * ylim[0]) / a if abs(a) > eps else None
        if alpha is not None and xlim[0] <= alpha <= xlim[1]:
            points.append((alpha, ylim[0]))

        # Intersection with top edge (β = ylim[1])
        alpha = (c - b * ylim[1]) / a if abs(a) > eps else None
        if alpha is not None and xlim[0] <= alpha <= xlim[1]:
            points.append((alpha, ylim[1]))

    # Remove duplicates and return
    if len(points) >= 2:
        # Sort and return first two unique points
        points = list(set(points))
        if len(points) >= 2:
            return points[0], points[1]

    return None


def create_animation(pos1, pos2, output_name, show_boundaries=True):
    """Create animation for a specific position pair with optional ReLU boundaries."""

    print(f"\n{'='*70}")
    print(f"Creating tropical animation for positions ({pos1}, {pos2})")
    print(f"{'='*70}")

    # Create output directory
    frames_dir = Path(f"docs/animation_frames_tropical_{output_name}")
    frames_dir.mkdir(exist_ok=True)

    # Collect all states for PCA fitting
    print("Collecting states for PCA...")
    all_states = []
    s_mags = np.linspace(0.05, 1.0, 50)

    for s_mag in s_mags:
        # Forward: M@pos1, S@pos2
        states_fwd, _ = run_stepwise([(pos1, M_val), (pos2, s_mag)])
        all_states.extend([s for s in states_fwd[1:]])

        # Reverse: S@pos1, M@pos2
        states_rev, _ = run_stepwise([(pos1, s_mag), (pos2, M_val)])
        all_states.extend([s for s in states_rev[1:]])

    all_states = np.array(all_states)
    pca = PCA(n_components=2)
    pca.fit(all_states)

    # Get axis limits from all data
    all_pca = pca.transform(all_states)
    margin = 5
    xlim = (all_pca[:, 0].min() - margin, all_pca[:, 0].max() + margin)
    ylim = (all_pca[:, 1].min() - margin, all_pca[:, 1].max() + margin)

    # Get max activation for consistent y-axis on hidden state plot
    max_activation = np.max(all_states)

    def render_frame(frame_num, s_mag, mode, total_frames):
        """Render a single frame with ReLU boundaries."""

        if mode == "forward":
            states, pre_states = run_stepwise([(pos1, M_val), (pos2, s_mag)])
            title = f"Forward: M@{pos1} (1.0), S@{pos2} ({s_mag:.2f})"
            target = pos2
            m_pos_actual = pos1
        else:
            states, pre_states = run_stepwise([(pos1, s_mag), (pos2, M_val)])
            title = f"Reverse: S@{pos1} ({s_mag:.2f}), M@{pos2} (1.0)"
            target = pos1
            m_pos_actual = pos2

        preds = [np.argmax(W_out @ states[t]) for t in range(1, 11)]
        final_pred = preds[-1]

        states_arr = np.array(states[1:])
        pca_coords = pca.transform(states_arr)

        # Get activation patterns
        active_at_t9 = states[-1] > 0

        # Create figure - now 2x4 grid
        fig, axes = plt.subplots(2, 4, figsize=(22, 11))

        # 1. Trajectory in PCA space WITH ReLU boundaries
        ax = axes[0, 0]

        # Draw ReLU boundaries first (so trajectory is on top)
        if show_boundaries:
            for neuron in range(16):
                # Use x_t = 0 for the boundaries (showing the "default" boundary)
                # The boundary shifts based on input, but x_t=0 shows the structure
                boundary = get_relu_boundary_line(neuron, pca, x_t=0.0,
                                                   xlim=xlim, ylim=ylim)
                if boundary is not None:
                    p1, p2 = boundary
                    color = NEURON_COLORS[neuron]
                    alpha = 0.6 if active_at_t9[neuron] else 0.2
                    linewidth = 1.5 if active_at_t9[neuron] else 0.8
                    linestyle = '-' if active_at_t9[neuron] else '--'
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                           color=color, alpha=alpha, linewidth=linewidth,
                           linestyle=linestyle)

        # Draw trajectory
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        for i in range(9):
            ax.plot(pca_coords[i:i+2, 0], pca_coords[i:i+2, 1],
                    color=colors[i], linewidth=2.5, zorder=10)
        ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=range(10),
                   cmap='viridis', s=100, zorder=15, edgecolors='white', linewidth=1)

        ax.scatter(pca_coords[0, 0], pca_coords[0, 1], marker='s', s=200,
                   c='green', edgecolors='black', linewidth=2, label='t=1', zorder=20)
        ax.scatter(pca_coords[-1, 0], pca_coords[-1, 1], marker='*', s=300,
                   c='red', edgecolors='black', linewidth=2, label='t=10', zorder=20)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(title + '\n(ReLU boundaries: solid=active, dashed=dead)')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. Predictions over time
        ax = axes[0, 1]
        ax.plot(range(1, 11), preds, 'o-', color='blue', markersize=10, linewidth=2)
        ax.axhline(target, color='green', linestyle='--', linewidth=2,
                   label=f'Target (S_pos={target})')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Predicted position')
        ax.set_title(f'Predictions (final={final_pred})')
        ax.set_ylim(-0.5, 9.5)
        ax.set_xlim(0.5, 10.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        color = 'green' if final_pred == target else 'red'
        ax.scatter([10], [final_pred], s=200, c=color, edgecolors='black',
                   linewidth=2, zorder=10)

        # 3. Hidden state norm over time
        ax = axes[0, 2]
        norms = [np.linalg.norm(states[t]) for t in range(1, 11)]
        ax.plot(range(1, 11), norms, 'o-', color='purple', markersize=8, linewidth=2)
        ax.axvline(pos1 + 1, color='orange', linestyle='--', alpha=0.7, label=f'pos {pos1} impulse')
        ax.axvline(pos2 + 1, color='red', linestyle='--', alpha=0.7, label=f'pos {pos2} impulse')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('||h||')
        ax.set_title('Hidden state magnitude')
        ax.set_ylim(0, 60)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Final hidden state activations with neuron categories
        ax = axes[0, 3]
        final_h = states[-1]
        bar_colors = [NEURON_COLORS[n] if final_h[n] > 0 else 'lightgray' for n in range(16)]
        ax.bar(range(16), final_h, color=bar_colors, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Activation')
        ax.set_title(f'Final hidden state (t=10)\nRed=comp, Blue=wave, Green=bridge')
        ax.set_xticks(range(16))
        ax.set_ylim(0, max_activation * 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        n_active = np.sum(final_h > 0)
        ax.text(0.95, 0.95, f'{n_active}/16 active', transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 5. Final logits
        ax = axes[1, 0]
        final_logits = W_out @ states[-1]
        bar_colors = ['green' if i == target else ('red' if i == m_pos_actual else 'gray')
                      for i in range(10)]
        ax.bar(range(10), final_logits, color=bar_colors, alpha=0.7, edgecolor='black')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Position')
        ax.set_ylabel('Logit')
        ax.set_title(f'Final logits (argmax={final_pred})')
        ax.set_xticks(range(10))
        ax.grid(True, alpha=0.3, axis='y')

        # 6. Activation pattern over time (tropical cells visualization)
        ax = axes[1, 1]
        activation_pattern = np.array([states[t] > 0 for t in range(1, 11)]).T  # (16, 10)
        ax.imshow(activation_pattern, cmap='Greys', aspect='auto', interpolation='nearest')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Neuron')
        ax.set_title('Activation pattern (tropical cell)')
        ax.set_xticks(range(10))
        ax.set_xticklabels(range(1, 11))
        ax.set_yticks(range(16))

        # Color code neuron labels
        for i, label in enumerate(ax.get_yticklabels()):
            label.set_color(NEURON_COLORS[i])

        # 7. Progress indicator and status
        ax = axes[1, 2]
        ax.axis('off')

        mode_text = f"FORWARD\nM@{pos1} first, S@{pos2} second" if mode == "forward" else f"REVERSE\nS@{pos1} first, M@{pos2} second"
        ax.text(0.5, 0.7, mode_text, ha='center', va='center', fontsize=16,
                fontweight='bold', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        progress = frame_num / total_frames
        ax.barh([0.3], [progress], color='blue', alpha=0.7, height=0.1,
                transform=ax.transAxes)
        ax.barh([0.3], [1.0], color='lightgray', alpha=0.3, height=0.1,
                transform=ax.transAxes, zorder=0)
        ax.text(0.5, 0.15, f'Frame {frame_num}/{total_frames}', ha='center',
                va='center', fontsize=12, transform=ax.transAxes)

        result = "✓ CORRECT" if final_pred == target else "✗ WRONG"
        result_color = 'green' if final_pred == target else 'red'
        ax.text(0.5, 0.5, result, ha='center', va='center', fontsize=20,
                fontweight='bold', color=result_color, transform=ax.transAxes)

        # 8. Legend for ReLU boundaries
        ax = axes[1, 3]
        ax.axis('off')

        legend_text = """TROPICAL GEOMETRY VIEW

ReLU boundaries partition the hidden
state space into cells. Within each
cell, the network is a linear map.

Boundary colors:
  RED: Comparator neurons (1,6,7,8)
  BLUE: Wave neurons (0,10,11,12,14)
  GREEN: Bridge neurons (3,5,13,15)
  GRAY: Other neurons (2,4,9)

Line styles:
  SOLID: Neuron is ACTIVE at t=10
  DASHED: Neuron is DEAD at t=10

The trajectory crosses boundaries
as neurons switch on/off through time.
"""
        ax.text(0.05, 0.95, legend_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()

        frame_path = frames_dir / f"frame_{frame_num:04d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()

        return frame_path

    # Generate frames
    print("Generating frames...")

    n_frames_phase1 = 50
    s_mags_phase1 = np.linspace(0.05, 1.0, n_frames_phase1)
    n_frames_hold1 = 10
    n_frames_transition = 5
    n_frames_phase2 = 50
    s_mags_phase2 = np.linspace(1.0, 0.05, n_frames_phase2)
    n_frames_hold2 = 10

    total_frames = (n_frames_phase1 + n_frames_hold1 + n_frames_transition +
                    n_frames_phase2 + n_frames_hold2)

    frame_num = 0

    # Phase 1: Forward, S increasing
    for s_mag in s_mags_phase1:
        frame_num += 1
        render_frame(frame_num, s_mag, "forward", total_frames)
    print(f"  Phase 1 complete: {frame_num} frames")

    # Phase 2: Hold forward at S=1.0
    for _ in range(n_frames_hold1):
        frame_num += 1
        render_frame(frame_num, 1.0, "forward", total_frames)

    # Phase 3: Transition to reverse
    for _ in range(n_frames_transition):
        frame_num += 1
        render_frame(frame_num, 1.0, "reverse", total_frames)

    # Phase 4: Reverse, S decreasing
    for s_mag in s_mags_phase2:
        frame_num += 1
        render_frame(frame_num, s_mag, "reverse", total_frames)
    print(f"  Phase 4 complete: {frame_num} frames")

    # Phase 5: Hold reverse at S=0.05
    for _ in range(n_frames_hold2):
        frame_num += 1
        render_frame(frame_num, 0.05, "reverse", total_frames)

    print(f"  Generated {frame_num} frames total")

    # Create video with ffmpeg
    print("Creating video...")
    output_mp4 = f"docs/tropical_animation_{output_name}.mp4"
    output_gif = f"docs/tropical_animation_{output_name}.gif"

    # MP4
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "15",
        "-i", str(frames_dir / "frame_%04d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        output_mp4
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"  Saved {output_mp4}")

    # GIF (smaller)
    cmd_gif = [
        "ffmpeg", "-y",
        "-framerate", "15",
        "-i", str(frames_dir / "frame_%04d.png"),
        "-vf", "scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        output_gif
    ]
    subprocess.run(cmd_gif, capture_output=True)
    print(f"  Saved {output_gif}")

    return output_mp4, output_gif


if __name__ == "__main__":
    # Create animations for each position pair
    position_pairs = [
        (0, 9, "pos_0_9"),
        (4, 5, "pos_4_5"),
        (0, 5, "pos_0_5"),
        (4, 9, "pos_4_9"),
        (2, 7, "pos_2_7"),
    ]

    for pos1, pos2, name in position_pairs:
        create_animation(pos1, pos2, name)

    print("\n" + "="*70)
    print("All tropical animations complete!")
    print("="*70)
