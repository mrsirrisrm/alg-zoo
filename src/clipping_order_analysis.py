"""
Clipping Order Analysis

Key question: Does the 2nd_argmax actually clip?
- If argmax comes FIRST: threshold is high, 2nd_argmax may not clip
- If 2nd_argmax comes FIRST: it clips (sets threshold), then argmax clips

Also: How does the gap between max and 2nd_max affect clipping and accuracy?
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def get_full_trajectory(model, x):
    """Compute full trajectory including clipping events."""
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data

    batch_size = x.shape[0]
    clipped = th.zeros(batch_size, 16, 10, dtype=th.bool)
    hidden = th.zeros(batch_size, 16, 10)
    h = th.zeros(batch_size, 16)

    for t in range(10):
        x_t = x[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        clipped[:, :, t] = pre < 0
        h = th.relu(pre)
        hidden[:, :, t] = h

    return clipped, hidden


def analyze_clipping_by_order(model, n_samples=100000):
    """
    Check if 2nd_argmax actually clips, depending on whether it comes before or after argmax.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    clipped, _ = get_full_trajectory(model, x)

    # Does 2nd_argmax come before or after argmax?
    second_before = targets < argmax_pos
    second_after = targets > argmax_pos

    print("=" * 70)
    print("DOES 2ND_ARGMAX ACTUALLY CLIP?")
    print("=" * 70)

    comparators = [1, 6, 7, 8]

    print(f"\nSamples where 2nd_argmax comes BEFORE argmax: {second_before.sum().item()}")
    print(f"Samples where 2nd_argmax comes AFTER argmax: {second_after.sum().item()}")

    print("\n" + "-" * 60)
    print("Clipping rate at 2nd_argmax position:")
    print("-" * 60)
    print(f"{'Neuron':<10} | {'2nd BEFORE argmax':<20} | {'2nd AFTER argmax':<20}")
    print("-" * 60)

    for n in comparators:
        # Check if neuron n clips at the 2nd_argmax position
        clips_at_2nd = th.zeros(n_samples, dtype=th.bool)
        for i in range(n_samples):
            clips_at_2nd[i] = clipped[i, n, targets[i]]

        rate_before = clips_at_2nd[second_before].float().mean().item()
        rate_after = clips_at_2nd[second_after].float().mean().item()

        print(f"n{n:<9} | {rate_before:>18.1%}  | {rate_after:>18.1%}")

    print("\n" + "-" * 60)
    print("Clipping rate at ARGMAX position (should always be high):")
    print("-" * 60)

    for n in comparators:
        clips_at_argmax = th.zeros(n_samples, dtype=th.bool)
        for i in range(n_samples):
            clips_at_argmax[i] = clipped[i, n, argmax_pos[i]]

        rate_before = clips_at_argmax[second_before].float().mean().item()
        rate_after = clips_at_argmax[second_after].float().mean().item()

        print(f"n{n:<9} | {rate_before:>18.1%}  | {rate_after:>18.1%}")

    return second_before, second_after


def analyze_clipping_vs_gap(model, n_samples=100000):
    """
    How does the gap between max and 2nd_max affect clipping at 2nd_argmax?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    top2 = th.topk(x, 2, dim=-1)
    gap = top2.values[:, 0] - top2.values[:, 1]

    clipped, _ = get_full_trajectory(model, x)

    second_before = targets < argmax_pos

    print("\n" + "=" * 70)
    print("CLIPPING AT 2ND_ARGMAX vs GAP SIZE")
    print("=" * 70)

    print("\nWhen 2nd_argmax comes BEFORE argmax:")
    print("-" * 60)

    gap_bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.0)]

    print(f"{'Gap range':<12} | {'n1 clip%':<10} | {'n7 clip%':<10} | {'Accuracy':<10} | {'Count':<8}")
    print("-" * 60)

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    for low, high in gap_bins:
        mask = second_before & (gap >= low) & (gap < high)
        if mask.sum() < 100:
            continue

        # Clip rate at 2nd_argmax position
        clips_n1 = th.zeros(n_samples, dtype=th.bool)
        clips_n7 = th.zeros(n_samples, dtype=th.bool)
        for i in range(n_samples):
            clips_n1[i] = clipped[i, 1, targets[i]]
            clips_n7[i] = clipped[i, 7, targets[i]]

        rate_n1 = clips_n1[mask].float().mean().item()
        rate_n7 = clips_n7[mask].float().mean().item()
        acc = correct[mask].float().mean().item()

        print(f"[{low:.2f}, {high:.2f}) | {rate_n1:>8.1%}  | {rate_n7:>8.1%}  | {acc:>8.1%}  | {mask.sum().item():<8}")

    print("\nWhen 2nd_argmax comes AFTER argmax:")
    print("-" * 60)

    second_after = targets > argmax_pos

    print(f"{'Gap range':<12} | {'n1 clip%':<10} | {'n7 clip%':<10} | {'Accuracy':<10} | {'Count':<8}")
    print("-" * 60)

    for low, high in gap_bins:
        mask = second_after & (gap >= low) & (gap < high)
        if mask.sum() < 100:
            continue

        clips_n1 = th.zeros(n_samples, dtype=th.bool)
        clips_n7 = th.zeros(n_samples, dtype=th.bool)
        for i in range(n_samples):
            clips_n1[i] = clipped[i, 1, targets[i]]
            clips_n7[i] = clipped[i, 7, targets[i]]

        rate_n1 = clips_n1[mask].float().mean().item()
        rate_n7 = clips_n7[mask].float().mean().item()
        acc = correct[mask].float().mean().item()

        print(f"[{low:.2f}, {high:.2f}) | {rate_n1:>8.1%}  | {rate_n7:>8.1%}  | {acc:>8.1%}  | {mask.sum().item():<8}")


def plot_clipping_analysis(model, n_samples=100000, save_path=None):
    """
    Visualize clipping patterns by order and gap.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    top2 = th.topk(x, 2, dim=-1)
    gap = top2.values[:, 0] - top2.values[:, 1]

    clipped, _ = get_full_trajectory(model, x)

    second_before = targets < argmax_pos
    second_after = targets > argmax_pos

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Clip rate at 2nd_argmax by gap (n7)
    ax = axes[0, 0]

    gap_centers = []
    clip_before = []
    clip_after = []

    for low, high in [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
                      (0.20, 0.30), (0.30, 0.50)]:
        gap_centers.append((low + high) / 2)

        # Clip at 2nd_argmax position
        clips_n7 = th.zeros(n_samples, dtype=th.bool)
        for i in range(n_samples):
            clips_n7[i] = clipped[i, 7, targets[i]]

        mask_before = second_before & (gap >= low) & (gap < high)
        mask_after = second_after & (gap >= low) & (gap < high)

        if mask_before.sum() > 50:
            clip_before.append(clips_n7[mask_before].float().mean().item())
        else:
            clip_before.append(np.nan)

        if mask_after.sum() > 50:
            clip_after.append(clips_n7[mask_after].float().mean().item())
        else:
            clip_after.append(np.nan)

    ax.plot(gap_centers, clip_before, 'o-', label='2nd BEFORE argmax', linewidth=2, markersize=8)
    ax.plot(gap_centers, clip_after, 's-', label='2nd AFTER argmax', linewidth=2, markersize=8)
    ax.set_xlabel('Gap (max - 2nd_max)')
    ax.set_ylabel('n7 clip rate at 2nd_argmax position')
    ax.set_title('Does 2nd_argmax clip? (n7)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel 2: Accuracy by gap and order
    ax = axes[0, 1]

    acc_before = []
    acc_after = []

    for low, high in [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
                      (0.20, 0.30), (0.30, 0.50)]:
        mask_before = second_before & (gap >= low) & (gap < high)
        mask_after = second_after & (gap >= low) & (gap < high)

        if mask_before.sum() > 50:
            acc_before.append(correct[mask_before].float().mean().item())
        else:
            acc_before.append(np.nan)

        if mask_after.sum() > 50:
            acc_after.append(correct[mask_after].float().mean().item())
        else:
            acc_after.append(np.nan)

    ax.plot(gap_centers, acc_before, 'o-', label='2nd BEFORE argmax', linewidth=2, markersize=8)
    ax.plot(gap_centers, acc_after, 's-', label='2nd AFTER argmax', linewidth=2, markersize=8)
    ax.set_xlabel('Gap (max - 2nd_max)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Gap and Order')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    # Panel 3: Number of comparators that clip at 2nd_argmax
    ax = axes[1, 0]

    comparators = [1, 6, 7, 8]

    n_clips_before = []
    n_clips_after = []

    for low, high in [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
                      (0.20, 0.30), (0.30, 0.50)]:
        # Count how many comparators clip at 2nd_argmax
        n_clip = th.zeros(n_samples)
        for n in comparators:
            for i in range(n_samples):
                n_clip[i] += clipped[i, n, targets[i]].float()

        mask_before = second_before & (gap >= low) & (gap < high)
        mask_after = second_after & (gap >= low) & (gap < high)

        if mask_before.sum() > 50:
            n_clips_before.append(n_clip[mask_before].mean().item())
        else:
            n_clips_before.append(np.nan)

        if mask_after.sum() > 50:
            n_clips_after.append(n_clip[mask_after].mean().item())
        else:
            n_clips_after.append(np.nan)

    ax.plot(gap_centers, n_clips_before, 'o-', label='2nd BEFORE argmax', linewidth=2, markersize=8)
    ax.plot(gap_centers, n_clips_after, 's-', label='2nd AFTER argmax', linewidth=2, markersize=8)
    ax.set_xlabel('Gap (max - 2nd_max)')
    ax.set_ylabel('# comparators clipping at 2nd_argmax')
    ax.set_title('How many comparators fire at 2nd_argmax?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4)
    ax.axhline(y=4, color='gray', linestyle='--', alpha=0.5)

    # Panel 4: Distribution of gap by order
    ax = axes[1, 1]

    ax.hist(gap[second_before].numpy(), bins=30, alpha=0.5, label='2nd BEFORE argmax', density=True)
    ax.hist(gap[second_after].numpy(), bins=30, alpha=0.5, label='2nd AFTER argmax', density=True)
    ax.set_xlabel('Gap (max - 2nd_max)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of gaps by order')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved clipping analysis to {save_path}')

    return fig


def analyze_two_regimes(model, n_samples=100000):
    """
    Analyze the two different regimes:
    1. 2nd_argmax BEFORE argmax: Both can clip (two impulses)
    2. 2nd_argmax AFTER argmax: Only argmax clips reliably (one impulse)
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    second_before = targets < argmax_pos
    second_after = targets > argmax_pos

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    print("\n" + "=" * 70)
    print("TWO REGIMES ANALYSIS")
    print("=" * 70)

    print(f"\nRegime 1: 2nd_argmax BEFORE argmax")
    print(f"  Count: {second_before.sum().item()} ({second_before.float().mean().item():.1%})")
    print(f"  Accuracy: {correct[second_before].float().mean().item():.1%}")
    print("  Mechanism: BOTH values clip -> two impulses -> Fourier decomposition")

    print(f"\nRegime 2: 2nd_argmax AFTER argmax")
    print(f"  Count: {second_after.sum().item()} ({second_after.float().mean().item():.1%})")
    print(f"  Accuracy: {correct[second_after].float().mean().item():.1%}")
    print("  Mechanism: Only argmax clips -> must use OTHER information")

    print("\n" + "-" * 50)
    print("What information is available in Regime 2?")
    print("-" * 50)
    print("""
When 2nd_argmax comes AFTER argmax:
  - Argmax clips and sets threshold high
  - 2nd_argmax (smaller) typically does NOT clip
  - The comparators only see ONE impulse (argmax)
  - So how does the model know 2nd_argmax position?

Possibilities:
  1. The 2nd_argmax DOES clip sometimes (when gap is small)
  2. Other neurons (not comparators) encode the information
  3. The input values themselves leave traces
  4. Statistical inference from context
""")


def main():
    model = example_2nd_argmax()

    analyze_clipping_by_order(model)
    analyze_clipping_vs_gap(model)
    analyze_two_regimes(model)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    plot_clipping_analysis(model, save_path='docs/figures/clipping_order_analysis.png')

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)


if __name__ == "__main__":
    main()
