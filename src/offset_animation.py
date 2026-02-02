"""
Create animation showing how fwd/rev trajectories change as S_mag varies.

Sequence:
1. S_mag from 0.05 to 1.0 (forward: M@2, S@7)
2. Swap positions (M@7, S@2)
3. S_mag from 1.0 back to 0.05 (reverse: S@2, M@7)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import subprocess
from alg_zoo import example_2nd_argmax

model = example_2nd_argmax()
model.eval()

W_ih = model.rnn.weight_ih_l0.detach().numpy().squeeze()
W_hh = model.rnn.weight_hh_l0.detach().numpy()
W_out = model.linear.weight.detach().numpy()

M_val = 1.0

def run_stepwise(impulses):
    h = np.zeros(16)
    states = [h.copy()]
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        states.append(h.copy())
    return states

# Create output directory
frames_dir = Path("docs/animation_frames")
frames_dir.mkdir(exist_ok=True)

# Collect all states for PCA fitting
print("Collecting states for PCA...")
all_states = []
s_mags = np.linspace(0.05, 1.0, 50)

for s_mag in s_mags:
    # Forward: M@2, S@7
    states_fwd = run_stepwise([(2, M_val), (7, s_mag)])
    all_states.extend([s for s in states_fwd[1:]])

    # Reverse: S@2, M@7
    states_rev = run_stepwise([(2, s_mag), (7, M_val)])
    all_states.extend([s for s in states_rev[1:]])

all_states = np.array(all_states)
pca = PCA(n_components=2)
pca.fit(all_states)

print(f"PCA fitted on {len(all_states)} states")

def render_frame(frame_num, s_mag, mode, total_frames):
    """Render a single frame."""

    if mode == "forward":
        # M@2, S@7 with varying S magnitude
        states = run_stepwise([(2, M_val), (7, s_mag)])
        title = f"Forward: M@2 (1.0), S@7 ({s_mag:.2f})"
        target = 7
        m_pos, s_pos = 2, 7
    else:
        # S@2, M@7 (reverse) with varying S magnitude
        states = run_stepwise([(2, s_mag), (7, M_val)])
        title = f"Reverse: S@2 ({s_mag:.2f}), M@7 (1.0)"
        target = 2
        m_pos, s_pos = 7, 2

    # Get predictions
    preds = [np.argmax(W_out @ states[t]) for t in range(1, 11)]
    final_pred = preds[-1]

    # PCA projection
    states_arr = np.array(states[1:])
    pca_coords = pca.transform(states_arr)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Color by timestep
    colors = plt.cm.viridis(np.linspace(0, 1, 10))

    # 1. Trajectory in PCA space
    ax = axes[0, 0]
    for i in range(9):
        ax.plot(pca_coords[i:i+2, 0], pca_coords[i:i+2, 1],
                color=colors[i], linewidth=2)
    ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=range(10),
               cmap='viridis', s=100, zorder=5, edgecolors='white', linewidth=1)

    # Mark key points
    ax.scatter(pca_coords[0, 0], pca_coords[0, 1], marker='s', s=200,
               c='green', edgecolors='black', linewidth=2, label='t=1', zorder=10)
    ax.scatter(pca_coords[-1, 0], pca_coords[-1, 1], marker='*', s=300,
               c='red', edgecolors='black', linewidth=2, label='t=10', zorder=10)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.legend(loc='upper right')
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

    # Highlight correct/incorrect
    color = 'green' if final_pred == target else 'red'
    ax.scatter([10], [final_pred], s=200, c=color, edgecolors='black',
               linewidth=2, zorder=10)

    # 3. Hidden state norm over time
    ax = axes[0, 2]
    norms = [np.linalg.norm(states[t]) for t in range(1, 11)]
    ax.plot(range(1, 11), norms, 'o-', color='purple', markersize=8, linewidth=2)
    ax.axvline(3, color='orange', linestyle='--', alpha=0.7, label='pos 2 impulse')
    ax.axvline(8, color='red', linestyle='--', alpha=0.7, label='pos 7 impulse')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('||h||')
    ax.set_title('Hidden state magnitude')
    ax.set_ylim(0, 50)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Final logits
    ax = axes[1, 0]
    final_logits = W_out @ states[-1]
    bar_colors = ['green' if i == target else ('red' if i == m_pos else 'gray')
                  for i in range(10)]
    ax.bar(range(10), final_logits, color=bar_colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Logit')
    ax.set_title(f'Final logits (argmax={final_pred})')
    ax.set_xticks(range(10))
    ax.grid(True, alpha=0.3, axis='y')

    # 5. S magnitude indicator
    ax = axes[1, 1]
    ax.barh([0], [s_mag], color='blue', alpha=0.7, height=0.5)
    ax.barh([1], [M_val], color='orange', alpha=0.7, height=0.5)
    ax.set_xlim(0, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['S magnitude', 'M magnitude'])
    ax.set_xlabel('Value')
    ax.set_title(f'Input magnitudes (S={s_mag:.2f}, M={M_val:.1f})')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    # 6. Progress indicator
    ax = axes[1, 2]
    ax.axis('off')

    # Mode indicator
    mode_text = "FORWARD\nM@2 first, S@7 second" if mode == "forward" else "REVERSE\nS@2 first, M@7 second"
    ax.text(0.5, 0.7, mode_text, ha='center', va='center', fontsize=16,
            fontweight='bold', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Progress bar
    progress = frame_num / total_frames
    ax.barh([0.3], [progress], color='blue', alpha=0.7, height=0.1,
            transform=ax.transAxes)
    ax.barh([0.3], [1.0], color='lightgray', alpha=0.3, height=0.1,
            transform=ax.transAxes, zorder=0)
    ax.text(0.5, 0.15, f'Frame {frame_num}/{total_frames}', ha='center',
            va='center', fontsize=12, transform=ax.transAxes)

    # Result
    result = "✓ CORRECT" if final_pred == target else "✗ WRONG"
    result_color = 'green' if final_pred == target else 'red'
    ax.text(0.5, 0.5, result, ha='center', va='center', fontsize=20,
            fontweight='bold', color=result_color, transform=ax.transAxes)

    plt.tight_layout()

    # Save frame
    frame_path = frames_dir / f"frame_{frame_num:04d}.png"
    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close()

    return frame_path

# Generate frames
print("Generating frames...")

# Phase 1: Forward with S_mag increasing from 0.05 to 1.0
n_frames_phase1 = 50
s_mags_phase1 = np.linspace(0.05, 1.0, n_frames_phase1)

# Phase 2: Hold at S=1.0 for transition (forward)
n_frames_hold1 = 10

# Phase 3: Reverse with S_mag = 1.0 (just swapped positions)
n_frames_transition = 5

# Phase 4: Reverse with S_mag decreasing from 1.0 to 0.05
n_frames_phase2 = 50
s_mags_phase2 = np.linspace(1.0, 0.05, n_frames_phase2)

# Phase 5: Hold at end
n_frames_hold2 = 10

total_frames = (n_frames_phase1 + n_frames_hold1 + n_frames_transition +
                n_frames_phase2 + n_frames_hold2)

frame_num = 0

# Phase 1: Forward, S increasing
print("Phase 1: Forward, S_mag 0.05 -> 1.0")
for s_mag in s_mags_phase1:
    frame_num += 1
    render_frame(frame_num, s_mag, "forward", total_frames)
    if frame_num % 10 == 0:
        print(f"  Frame {frame_num}/{total_frames}")

# Phase 2: Hold forward at S=1.0
print("Phase 2: Hold forward at S=1.0")
for _ in range(n_frames_hold1):
    frame_num += 1
    render_frame(frame_num, 1.0, "forward", total_frames)

# Phase 3: Transition to reverse (still S=1.0)
print("Phase 3: Transition to reverse")
for _ in range(n_frames_transition):
    frame_num += 1
    render_frame(frame_num, 1.0, "reverse", total_frames)

# Phase 4: Reverse, S decreasing
print("Phase 4: Reverse, S_mag 1.0 -> 0.05")
for s_mag in s_mags_phase2:
    frame_num += 1
    render_frame(frame_num, s_mag, "reverse", total_frames)
    if frame_num % 10 == 0:
        print(f"  Frame {frame_num}/{total_frames}")

# Phase 5: Hold reverse at S=0.05
print("Phase 5: Hold reverse at S=0.05")
for _ in range(n_frames_hold2):
    frame_num += 1
    render_frame(frame_num, 0.05, "reverse", total_frames)

print(f"\nGenerated {frame_num} frames")

# Create video with ffmpeg
print("\nCreating video with ffmpeg...")
output_path = "docs/offset_animation.mp4"

cmd = [
    "ffmpeg", "-y",
    "-framerate", "15",
    "-i", str(frames_dir / "frame_%04d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "23",
    output_path
]

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"Video saved to {output_path}")
else:
    print(f"ffmpeg error: {result.stderr}")

# Also create a GIF version
print("\nCreating GIF version...")
gif_path = "docs/offset_animation.gif"

cmd_gif = [
    "ffmpeg", "-y",
    "-framerate", "15",
    "-i", str(frames_dir / "frame_%04d.png"),
    "-vf", "scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
    gif_path
]

result_gif = subprocess.run(cmd_gif, capture_output=True, text=True)

if result_gif.returncode == 0:
    print(f"GIF saved to {gif_path}")
else:
    print(f"GIF error: {result_gif.stderr}")

print("\nDone!")
