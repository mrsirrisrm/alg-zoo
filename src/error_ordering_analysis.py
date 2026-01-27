"""
Error Analysis by Value Ordering

Investigates how model errors depend on:
1. Whether 2nd max comes before or after 3rd max
2. The gap between 2nd and 3rd max values
3. Combined effects of ordering and gap size
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def analyze_ordering_errors(model, n_samples=100000):
    """
    Break down errors by whether 2nd max comes before or after 3rd max.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    with th.no_grad():
        preds = model(x).argmax(dim=-1)

    correct = (preds == targets)

    # Get positions and values of top 3
    top3_vals, top3_pos = th.topk(x, 3, dim=-1)
    max_pos = top3_pos[:, 0]
    second_pos = top3_pos[:, 1]  # This is the target (2nd argmax)
    third_pos = top3_pos[:, 2]

    max_val = top3_vals[:, 0]
    second_val = top3_vals[:, 1]
    third_val = top3_vals[:, 2]

    # Gap between 2nd and 3rd
    gap_23 = second_val - third_val

    # Ordering: does 2nd come before or after 3rd?
    second_before_third = second_pos < third_pos

    print("=" * 70)
    print("ERROR ANALYSIS BY ORDERING")
    print("=" * 70)

    # Overall accuracy
    print(f"\nOverall accuracy: {correct.float().mean().item():.1%}")

    # By ordering
    acc_2nd_first = correct[second_before_third].float().mean().item()
    acc_3rd_first = correct[~second_before_third].float().mean().item()

    n_2nd_first = second_before_third.sum().item()
    n_3rd_first = (~second_before_third).sum().item()

    print(f"\n2nd max comes BEFORE 3rd max:")
    print(f"  Count: {n_2nd_first} ({n_2nd_first/n_samples:.1%})")
    print(f"  Accuracy: {acc_2nd_first:.1%}")

    print(f"\n2nd max comes AFTER 3rd max:")
    print(f"  Count: {n_3rd_first} ({n_3rd_first/n_samples:.1%})")
    print(f"  Accuracy: {acc_3rd_first:.1%}")

    print(f"\nDifference: {abs(acc_2nd_first - acc_3rd_first):.1%}")

    return x, targets, preds, correct, second_before_third, gap_23, second_pos, third_pos


def analyze_gap_by_ordering(x, targets, preds, correct, second_before_third, gap_23):
    """
    Analyze accuracy by gap size, separately for each ordering.
    """
    print("\n" + "=" * 70)
    print("ACCURACY BY GAP SIZE AND ORDERING")
    print("=" * 70)

    gap_bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
                (0.20, 0.30), (0.30, 0.50), (0.50, 1.0)]

    print("\n" + "-" * 60)
    print(f"{'Gap range':<12} | {'2nd before 3rd':^20} | {'3rd before 2nd':^20}")
    print(f"{'':12} | {'Acc':^8} {'Count':^10} | {'Acc':^8} {'Count':^10}")
    print("-" * 60)

    results = {'gap': [], '2nd_first_acc': [], '3rd_first_acc': [],
               '2nd_first_n': [], '3rd_first_n': []}

    for low, high in gap_bins:
        gap_mask = (gap_23 >= low) & (gap_23 < high)

        mask_2nd_first = gap_mask & second_before_third
        mask_3rd_first = gap_mask & (~second_before_third)

        if mask_2nd_first.sum() > 50 and mask_3rd_first.sum() > 50:
            acc_2nd = correct[mask_2nd_first].float().mean().item()
            acc_3rd = correct[mask_3rd_first].float().mean().item()
            n_2nd = mask_2nd_first.sum().item()
            n_3rd = mask_3rd_first.sum().item()

            print(f"[{low:.2f}, {high:.2f}) | {acc_2nd:^8.1%} {n_2nd:^10} | {acc_3rd:^8.1%} {n_3rd:^10}")

            results['gap'].append((low + high) / 2)
            results['2nd_first_acc'].append(acc_2nd)
            results['3rd_first_acc'].append(acc_3rd)
            results['2nd_first_n'].append(n_2nd)
            results['3rd_first_n'].append(n_3rd)

    return results


def analyze_confusion_by_ordering(x, targets, preds, correct, second_before_third, third_pos):
    """
    When the model is wrong, what does it predict?
    """
    print("\n" + "=" * 70)
    print("CONFUSION ANALYSIS BY ORDERING")
    print("=" * 70)

    wrong = ~correct

    # When 2nd comes first and model is wrong
    mask_2nd_wrong = wrong & second_before_third
    # When 3rd comes first and model is wrong
    mask_3rd_wrong = wrong & (~second_before_third)

    print("\nWhen 2nd comes BEFORE 3rd and model is WRONG:")
    print(f"  Total wrong: {mask_2nd_wrong.sum().item()}")

    # What does it predict?
    preds_2nd_wrong = preds[mask_2nd_wrong]
    targets_2nd_wrong = targets[mask_2nd_wrong]
    third_pos_2nd_wrong = third_pos[mask_2nd_wrong]

    # Does it predict the 3rd max position?
    pred_is_3rd = (preds_2nd_wrong == third_pos_2nd_wrong)
    print(f"  Predicts 3rd max position: {pred_is_3rd.float().mean().item():.1%}")

    # What about predicting position 0?
    pred_is_0 = (preds_2nd_wrong == 0)
    print(f"  Predicts position 0: {pred_is_0.float().mean().item():.1%}")

    print("\nWhen 3rd comes BEFORE 2nd and model is WRONG:")
    print(f"  Total wrong: {mask_3rd_wrong.sum().item()}")

    preds_3rd_wrong = preds[mask_3rd_wrong]
    targets_3rd_wrong = targets[mask_3rd_wrong]
    third_pos_3rd_wrong = third_pos[mask_3rd_wrong]

    pred_is_3rd = (preds_3rd_wrong == third_pos_3rd_wrong)
    print(f"  Predicts 3rd max position: {pred_is_3rd.float().mean().item():.1%}")

    pred_is_0 = (preds_3rd_wrong == 0)
    print(f"  Predicts position 0: {pred_is_0.float().mean().item():.1%}")


def analyze_position_combinations(x, targets, preds, correct, second_before_third, gap_23):
    """
    Look at specific position combinations for 2nd and 3rd max.
    """
    print("\n" + "=" * 70)
    print("ACCURACY BY POSITION COMBINATION (2nd_pos, 3rd_pos)")
    print("=" * 70)

    top3_pos = th.topk(x, 3, dim=-1).indices
    second_pos = top3_pos[:, 1]
    third_pos = top3_pos[:, 2]

    # Focus on small gaps where errors are most common
    small_gap = gap_23 < 0.1

    print("\nFor small gaps (< 0.1):")
    print("-" * 50)

    # Create a grid of accuracies
    acc_grid = np.zeros((10, 10))
    count_grid = np.zeros((10, 10))

    for p2 in range(10):
        for p3 in range(10):
            if p2 == p3:
                continue
            mask = (second_pos == p2) & (third_pos == p3) & small_gap
            if mask.sum() > 30:
                acc_grid[p2, p3] = correct[mask].float().mean().item()
                count_grid[p2, p3] = mask.sum().item()

    # Find worst combinations
    worst_combos = []
    for p2 in range(10):
        for p3 in range(10):
            if count_grid[p2, p3] > 30:
                worst_combos.append((p2, p3, acc_grid[p2, p3], count_grid[p2, p3]))

    worst_combos.sort(key=lambda x: x[2])

    print("\nWorst position combinations (small gap):")
    print(f"{'2nd_pos':<8} {'3rd_pos':<8} {'Accuracy':<10} {'Count':<8} {'Order':<15}")
    print("-" * 50)

    for p2, p3, acc, count in worst_combos[:15]:
        order = "2nd first" if p2 < p3 else "3rd first"
        print(f"{p2:<8} {p3:<8} {acc:<10.1%} {int(count):<8} {order:<15}")

    return acc_grid, count_grid


def plot_ordering_analysis(results, save_path=None):
    """
    Visualize the ordering effect on accuracy.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Accuracy by gap, split by ordering
    ax = axes[0]
    gaps = results['gap']
    ax.plot(gaps, results['2nd_first_acc'], 'o-', label='2nd before 3rd',
            color='steelblue', linewidth=2, markersize=8)
    ax.plot(gaps, results['3rd_first_acc'], 's-', label='3rd before 2nd',
            color='coral', linewidth=2, markersize=8)
    ax.set_xlabel('Gap between 2nd and 3rd max values')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Gap Size and Ordering')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    # Panel 2: Difference in accuracy
    ax = axes[1]
    diff = [a - b for a, b in zip(results['2nd_first_acc'], results['3rd_first_acc'])]
    colors = ['steelblue' if d > 0 else 'coral' for d in diff]
    ax.bar(range(len(gaps)), diff, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(range(len(gaps)))
    ax.set_xticklabels([f'{g:.2f}' for g in gaps], rotation=45)
    ax.set_xlabel('Gap (midpoint)')
    ax.set_ylabel('Accuracy difference\n(2nd first - 3rd first)')
    ax.set_title('Ordering Effect by Gap Size')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Sample counts
    ax = axes[2]
    width = 0.35
    x_pos = np.arange(len(gaps))
    ax.bar(x_pos - width/2, results['2nd_first_n'], width, label='2nd before 3rd',
           color='steelblue', alpha=0.7)
    ax.bar(x_pos + width/2, results['3rd_first_n'], width, label='3rd before 2nd',
           color='coral', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{g:.2f}' for g in gaps], rotation=45)
    ax.set_xlabel('Gap (midpoint)')
    ax.set_ylabel('Sample count')
    ax.set_title('Distribution of Cases')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved ordering analysis to {save_path}')

    return fig


def plot_position_heatmap(acc_grid, count_grid, save_path=None):
    """
    Heatmap showing accuracy for each (2nd_pos, 3rd_pos) combination.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mask where count is too low
    acc_masked = np.ma.masked_where(count_grid < 30, acc_grid)

    # Panel 1: Accuracy heatmap
    ax = axes[0]
    im = ax.imshow(acc_masked, cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax.set_xlabel('3rd max position')
    ax.set_ylabel('2nd max position')
    ax.set_title('Accuracy by Position Combination\n(small gap < 0.1)')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax, label='Accuracy')

    # Add diagonal line to show 2nd before/after 3rd
    ax.plot([-0.5, 9.5], [-0.5, 9.5], 'k--', linewidth=1, alpha=0.5)
    ax.text(7, 2, '2nd after 3rd', fontsize=10, alpha=0.7)
    ax.text(2, 7, '2nd before 3rd', fontsize=10, alpha=0.7)

    # Panel 2: Count heatmap
    ax = axes[1]
    im = ax.imshow(count_grid, cmap='Blues')
    ax.set_xlabel('3rd max position')
    ax.set_ylabel('2nd max position')
    ax.set_title('Sample Count by Position Combination\n(small gap < 0.1)')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax, label='Count')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved position heatmap to {save_path}')

    return fig


def main():
    model = example_2nd_argmax()

    # Main ordering analysis
    x, targets, preds, correct, second_before_third, gap_23, second_pos, third_pos = \
        analyze_ordering_errors(model)

    # Gap analysis by ordering
    results = analyze_gap_by_ordering(x, targets, preds, correct, second_before_third, gap_23)

    # Confusion analysis
    analyze_confusion_by_ordering(x, targets, preds, correct, second_before_third, third_pos)

    # Position combinations
    acc_grid, count_grid = analyze_position_combinations(
        x, targets, preds, correct, second_before_third, gap_23)

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_ordering_analysis(results, 'docs/figures/error_ordering_analysis.png')
    plot_position_heatmap(acc_grid, count_grid, 'docs/figures/error_position_heatmap.png')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key findings:

1. ORDERING MATTERS: When 2nd max comes BEFORE 3rd max, the model
   generally has different accuracy than when 3rd comes first.

2. INTERACTION WITH GAP: The ordering effect varies with gap size.
   For very small gaps, the model may struggle more when the values
   are close together regardless of order.

3. CONFUSION PATTERN: When wrong, the model often predicts the 3rd
   max position (confusing 2nd and 3rd) or defaults to position 0.

4. POSITION COMBINATIONS: Certain (2nd_pos, 3rd_pos) combinations
   are harder than others, especially when both are late or both early.
""")


if __name__ == "__main__":
    main()
