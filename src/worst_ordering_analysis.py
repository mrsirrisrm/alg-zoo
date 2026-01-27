"""
Deep dive into the 3-2-1 ordering (worst accuracy at 87.2%).

In this ordering:
- 3rd argmax comes first → clips
- 2nd argmax comes second → clips
- 1st argmax (max) comes last → clips

All three values clip! This creates three impulses and maximum confusion.

Questions:
1. What do the comparators see?
2. How does h_final differ from other orderings?
3. What does n5 do in this regime?
4. Why does the model fail - what's the confusion pattern?
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def get_ordering(pos1, pos2, pos3):
    """Return ordering string."""
    positions = [(pos1, '1'), (pos2, '2'), (pos3, '3')]
    positions.sort(key=lambda x: x[0])
    return '-'.join([p[1] for p in positions])


def get_full_trajectory(model, x):
    """Compute full trajectory with clipping info."""
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data

    batch_size = x.shape[0]
    hidden = th.zeros(batch_size, 16, 10)
    clipped = th.zeros(batch_size, 16, 10, dtype=th.bool)
    h = th.zeros(batch_size, 16)

    for t in range(10):
        x_t = x[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        clipped[:, :, t] = pre < 0
        h = th.relu(pre)
        hidden[:, :, t] = h

    return hidden, clipped


def get_ordering_mask(pos1, pos2, pos3, target_ordering):
    """Get boolean mask for samples with specific ordering."""
    n_samples = pos1.shape[0]
    mask = th.zeros(n_samples, dtype=th.bool)
    for i in range(n_samples):
        if get_ordering(pos1[i].item(), pos2[i].item(), pos3[i].item()) == target_ordering:
            mask[i] = True
    return mask


def analyze_321_clipping_pattern(model, n_samples=200000):
    """
    Detailed clipping analysis for 3-2-1 ordering.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]  # max
    pos2 = top3.indices[:, 1]  # 2nd
    pos3 = top3.indices[:, 2]  # 3rd

    mask_321 = get_ordering_mask(pos1, pos2, pos3, '3-2-1')

    hidden, clipped = get_full_trajectory(model, x)

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    print("=" * 70)
    print("3-2-1 ORDERING: DETAILED CLIPPING ANALYSIS")
    print("=" * 70)

    print(f"\nSamples with 3-2-1 ordering: {mask_321.sum().item()}")
    print(f"Accuracy in this ordering: {correct[mask_321].float().mean().item():.1%}")

    comparators = [1, 6, 7, 8]

    print("\n" + "-" * 70)
    print("Clipping rates at each position:")
    print("-" * 70)
    print(f"{'Neuron':<8} | {'at 3rd (first)':<15} | {'at 2nd (second)':<16} | {'at max (last)':<15}")
    print("-" * 70)

    for n in comparators:
        clips_3rd = []
        clips_2nd = []
        clips_max = []

        for i in range(n_samples):
            if not mask_321[i]:
                continue
            clips_3rd.append(clipped[i, n, pos3[i]].item())
            clips_2nd.append(clipped[i, n, pos2[i]].item())
            clips_max.append(clipped[i, n, pos1[i]].item())

        print(f"n{n:<7} | {np.mean(clips_3rd):<15.1%} | {np.mean(clips_2nd):<16.1%} | {np.mean(clips_max):<15.1%}")

    # Compare with best ordering (2-1-3)
    mask_213 = get_ordering_mask(pos1, pos2, pos3, '2-1-3')

    print("\n" + "-" * 70)
    print("For comparison, 2-1-3 ordering (best accuracy):")
    print("-" * 70)
    print(f"{'Neuron':<8} | {'at 2nd (first)':<15} | {'at max (second)':<16} | {'at 3rd (last)':<15}")
    print("-" * 70)

    for n in comparators:
        clips_2nd = []
        clips_max = []
        clips_3rd = []

        for i in range(n_samples):
            if not mask_213[i]:
                continue
            clips_2nd.append(clipped[i, n, pos2[i]].item())
            clips_max.append(clipped[i, n, pos1[i]].item())
            clips_3rd.append(clipped[i, n, pos3[i]].item())

        print(f"n{n:<7} | {np.mean(clips_2nd):<15.1%} | {np.mean(clips_max):<16.1%} | {np.mean(clips_3rd):<15.1%}")


def analyze_321_h_final(model, n_samples=200000):
    """
    How does h_final differ in 3-2-1 vs other orderings?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]
    pos3 = top3.indices[:, 2]

    mask_321 = get_ordering_mask(pos1, pos2, pos3, '3-2-1')
    mask_213 = get_ordering_mask(pos1, pos2, pos3, '2-1-3')

    hidden, _ = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    print("\n" + "=" * 70)
    print("H_FINAL COMPARISON: 3-2-1 vs 2-1-3")
    print("=" * 70)

    key_neurons = [1, 2, 4, 5, 6, 7, 8, 9]

    print("\nMean h_final by ordering:")
    print("-" * 60)
    print(f"{'Neuron':<8} | {'3-2-1':<12} | {'2-1-3':<12} | {'Difference':<12}")
    print("-" * 60)

    for n in key_neurons:
        mean_321 = h_final[mask_321, n].mean().item()
        mean_213 = h_final[mask_213, n].mean().item()
        diff = mean_321 - mean_213
        print(f"n{n:<7} | {mean_321:<12.3f} | {mean_213:<12.3f} | {diff:<+12.3f}")

    # Within 3-2-1, compare correct vs incorrect
    print("\n" + "-" * 60)
    print("Within 3-2-1: correct vs incorrect predictions")
    print("-" * 60)

    correct_321 = mask_321 & correct
    wrong_321 = mask_321 & ~correct

    print(f"{'Neuron':<8} | {'Correct':<12} | {'Wrong':<12} | {'Difference':<12}")
    print("-" * 60)

    for n in key_neurons:
        mean_correct = h_final[correct_321, n].mean().item()
        mean_wrong = h_final[wrong_321, n].mean().item()
        diff = mean_correct - mean_wrong
        print(f"n{n:<7} | {mean_correct:<12.3f} | {mean_wrong:<12.3f} | {diff:<+12.3f}")


def analyze_321_error_pattern(model, n_samples=200000):
    """
    What exactly goes wrong in 3-2-1?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]
    pos3 = top3.indices[:, 2]
    val2 = top3.values[:, 1]
    val3 = top3.values[:, 2]
    gap_23 = val2 - val3

    mask_321 = get_ordering_mask(pos1, pos2, pos3, '3-2-1')

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    errors_321 = mask_321 & ~correct

    print("\n" + "=" * 70)
    print("3-2-1 ERROR ANALYSIS")
    print("=" * 70)

    print(f"\nTotal errors in 3-2-1: {errors_321.sum().item()}")

    # What does the model predict?
    print("\nWhen wrong, model predicts:")
    print("-" * 40)

    pred_3rd = (preds[errors_321] == pos3[errors_321])
    pred_max = (preds[errors_321] == pos1[errors_321])

    print(f"  3rd argmax position: {pred_3rd.sum().item()} ({pred_3rd.float().mean().item():.1%})")
    print(f"  max position: {pred_max.sum().item()} ({pred_max.float().mean().item():.1%})")
    print(f"  other: {(~pred_3rd & ~pred_max).sum().item()} ({(~pred_3rd & ~pred_max).float().mean().item():.1%})")

    # Error rate by gap
    print("\n" + "-" * 50)
    print("Error rate by gap (2nd - 3rd value):")
    print("-" * 50)

    for low, high in [(0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.20), (0.20, 0.50)]:
        sub_mask = mask_321 & (gap_23 >= low) & (gap_23 < high)
        if sub_mask.sum() > 50:
            acc = correct[sub_mask].float().mean().item()
            err = 1 - acc

            # Of errors, how many predict 3rd?
            sub_errors = sub_mask & ~correct
            if sub_errors.sum() > 10:
                pred_3rd_rate = (preds[sub_errors] == pos3[sub_errors]).float().mean().item()
            else:
                pred_3rd_rate = float('nan')

            print(f"Gap [{low:.2f}, {high:.2f}): error={err:.1%}, predicts_3rd={pred_3rd_rate:.1%}, n={sub_mask.sum().item()}")

    # Error rate by position difference
    print("\n" + "-" * 50)
    print("Error rate by position spread (max_pos - 3rd_pos):")
    print("-" * 50)

    spread = pos1 - pos3  # How far apart are max and 3rd

    for s in range(2, 9):
        sub_mask = mask_321 & (spread == s)
        if sub_mask.sum() > 50:
            acc = correct[sub_mask].float().mean().item()
            print(f"Spread {s}: accuracy={acc:.1%}, n={sub_mask.sum().item()}")


def analyze_321_neuron_contributions(model, n_samples=200000):
    """
    Which neurons help/hurt in 3-2-1?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]
    pos3 = top3.indices[:, 2]

    mask_321 = get_ordering_mask(pos1, pos2, pos3, '3-2-1')

    hidden, _ = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    W_out = model.linear.weight.data

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    print("\n" + "=" * 70)
    print("NEURON CONTRIBUTIONS IN 3-2-1")
    print("=" * 70)

    # For samples in 3-2-1, compute contribution to correct vs predicted class
    correct_321 = mask_321 & correct
    wrong_321 = mask_321 & ~correct

    print("\nContribution to CORRECT class logit:")
    print("-" * 60)
    print(f"{'Neuron':<8} | {'When correct':<14} | {'When wrong':<14} | {'Diff':<10}")
    print("-" * 60)

    for n in range(16):
        # Contribution when correct
        contrib_correct = []
        for i in range(n_samples):
            if correct_321[i]:
                j = targets[i].item()
                contrib_correct.append((h_final[i, n] * W_out[j, n]).item())

        # Contribution when wrong
        contrib_wrong = []
        for i in range(n_samples):
            if wrong_321[i]:
                j = targets[i].item()
                contrib_wrong.append((h_final[i, n] * W_out[j, n]).item())

        if len(contrib_correct) > 0 and len(contrib_wrong) > 0:
            mean_correct = np.mean(contrib_correct)
            mean_wrong = np.mean(contrib_wrong)
            diff = mean_correct - mean_wrong

            if abs(diff) > 0.05:  # Only show significant differences
                print(f"n{n:<7} | {mean_correct:<14.3f} | {mean_wrong:<14.3f} | {diff:<+10.3f}")

    # What about contribution to WRONG class (3rd position)?
    print("\n" + "-" * 60)
    print("Contribution to 3RD POSITION logit (when wrong):")
    print("-" * 60)

    print(f"{'Neuron':<8} | {'Contrib to 3rd':<16} | {'Contrib to 2nd':<16} | {'Diff (3rd-2nd)':<14}")
    print("-" * 60)

    for n in range(16):
        contrib_3rd = []
        contrib_2nd = []

        for i in range(n_samples):
            if wrong_321[i]:
                j_3rd = pos3[i].item()
                j_2nd = targets[i].item()
                contrib_3rd.append((h_final[i, n] * W_out[j_3rd, n]).item())
                contrib_2nd.append((h_final[i, n] * W_out[j_2nd, n]).item())

        if len(contrib_3rd) > 0:
            mean_3rd = np.mean(contrib_3rd)
            mean_2nd = np.mean(contrib_2nd)
            diff = mean_3rd - mean_2nd

            if abs(diff) > 0.05:
                print(f"n{n:<7} | {mean_3rd:<16.3f} | {mean_2nd:<16.3f} | {diff:<+14.3f}")


def analyze_321_timing(model, n_samples=200000):
    """
    How does the timing (position spread) affect 3-2-1 accuracy?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]
    pos3 = top3.indices[:, 2]

    mask_321 = get_ordering_mask(pos1, pos2, pos3, '3-2-1')

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    print("\n" + "=" * 70)
    print("3-2-1 TIMING ANALYSIS")
    print("=" * 70)

    # Gap between 3rd and 2nd positions
    gap_32_pos = pos2 - pos3  # How far apart in time

    print("\nAccuracy by position gap between 3rd and 2nd:")
    print("-" * 50)

    for gap in range(1, 8):
        sub_mask = mask_321 & (gap_32_pos == gap)
        if sub_mask.sum() > 50:
            acc = correct[sub_mask].float().mean().item()
            print(f"Gap {gap} positions: accuracy={acc:.1%}, n={sub_mask.sum().item()}")

    # Gap between 2nd and max positions
    gap_21_pos = pos1 - pos2

    print("\nAccuracy by position gap between 2nd and max:")
    print("-" * 50)

    for gap in range(1, 8):
        sub_mask = mask_321 & (gap_21_pos == gap)
        if sub_mask.sum() > 50:
            acc = correct[sub_mask].float().mean().item()
            print(f"Gap {gap} positions: accuracy={acc:.1%}, n={sub_mask.sum().item()}")


def plot_321_analysis(model, n_samples=100000, save_path=None):
    """
    Visualize 3-2-1 failure patterns.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]
    pos3 = top3.indices[:, 2]
    gap_23 = top3.values[:, 1] - top3.values[:, 2]

    mask_321 = get_ordering_mask(pos1, pos2, pos3, '3-2-1')
    mask_213 = get_ordering_mask(pos1, pos2, pos3, '2-1-3')

    hidden, clipped = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Accuracy by gap for 3-2-1 vs 2-1-3
    ax = axes[0, 0]

    gaps = []
    acc_321 = []
    acc_213 = []

    for low, high in [(0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15),
                      (0.15, 0.20), (0.20, 0.30), (0.30, 0.50)]:
        gaps.append((low + high) / 2)

        sub_321 = mask_321 & (gap_23 >= low) & (gap_23 < high)
        sub_213 = mask_213 & (gap_23 >= low) & (gap_23 < high)

        if sub_321.sum() > 30:
            acc_321.append(correct[sub_321].float().mean().item())
        else:
            acc_321.append(np.nan)

        if sub_213.sum() > 30:
            acc_213.append(correct[sub_213].float().mean().item())
        else:
            acc_213.append(np.nan)

    ax.plot(gaps, acc_321, 'o-', label='3-2-1 (worst)', color='red', linewidth=2, markersize=8)
    ax.plot(gaps, acc_213, 's-', label='2-1-3 (best)', color='green', linewidth=2, markersize=8)
    ax.set_xlabel('Gap (2nd - 3rd value)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by gap: worst vs best ordering')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.6, 1.0)

    # Panel 2: h_final comparison for key neurons
    ax = axes[0, 1]

    neurons = [1, 5, 6, 7, 8, 9]
    h_321 = [h_final[mask_321, n].mean().item() for n in neurons]
    h_213 = [h_final[mask_213, n].mean().item() for n in neurons]

    x_pos = np.arange(len(neurons))
    width = 0.35

    ax.bar(x_pos - width/2, h_321, width, label='3-2-1', color='red', alpha=0.7)
    ax.bar(x_pos + width/2, h_213, width, label='2-1-3', color='green', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'n{n}' for n in neurons])
    ax.set_ylabel('Mean h_final')
    ax.set_title('h_final by ordering (key neurons)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Error breakdown for 3-2-1
    ax = axes[1, 0]

    errors_321 = mask_321 & ~correct
    pred_3rd = (preds[errors_321] == pos3[errors_321]).sum().item()
    pred_max = (preds[errors_321] == pos1[errors_321]).sum().item()
    pred_other = errors_321.sum().item() - pred_3rd - pred_max

    labels = ['Predicts 3rd', 'Predicts max', 'Predicts other']
    sizes = [pred_3rd, pred_max, pred_other]
    colors_pie = ['orange', 'blue', 'gray']

    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax.set_title(f'3-2-1 errors: what does model predict?\n(n={errors_321.sum().item()} errors)')

    # Panel 4: Clipping pattern visualization
    ax = axes[1, 1]

    # For n7, show clipping rates at each position type
    orderings = ['3-2-1', '2-1-3', '1-2-3', '3-1-2']
    positions_labels = ['1st in time', '2nd in time', '3rd in time']

    clip_data = {o: [] for o in orderings}

    for order in orderings:
        mask = get_ordering_mask(pos1, pos2, pos3, order)

        # Get positions in temporal order
        for i in range(n_samples):
            if not mask[i]:
                continue

            positions = [(pos1[i].item(), '1'), (pos2[i].item(), '2'), (pos3[i].item(), '3')]
            positions.sort(key=lambda x: x[0])

            clips = [clipped[i, 7, positions[j][0]].item() for j in range(3)]
            clip_data[order].append(clips)

        if len(clip_data[order]) > 0:
            clip_data[order] = np.mean(clip_data[order], axis=0)

    x_pos = np.arange(len(positions_labels))
    width = 0.2

    for idx, order in enumerate(orderings):
        if len(clip_data[order]) > 0:
            color = 'red' if order == '3-2-1' else 'green' if order == '2-1-3' else 'gray'
            ax.bar(x_pos + idx * width, clip_data[order], width, label=order, alpha=0.7)

    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(positions_labels)
    ax.set_ylabel('n7 clipping rate')
    ax.set_title('n7 clipping by temporal position')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved 3-2-1 analysis to {save_path}')

    return fig


def main():
    model = example_2nd_argmax()

    analyze_321_clipping_pattern(model)
    analyze_321_h_final(model)
    analyze_321_error_pattern(model)
    analyze_321_neuron_contributions(model)
    analyze_321_timing(model)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    plot_321_analysis(model, save_path='docs/figures/321_ordering_analysis.png')

    print("\n" + "=" * 70)
    print("SUMMARY: WHY 3-2-1 FAILS")
    print("=" * 70)
    print("""
THE 3-2-1 FAILURE MECHANISM:

1. THREE IMPULSES:
   - 3rd clips first (98%)
   - 2nd clips second (94%)
   - max clips last (93%)
   All three create impulses in the comparators!

2. INTERFERENCE CONFUSION:
   In 2-1-3 (best case): only 2nd and max create impulses
   → Clean anti-phase interference encodes their difference

   In 3-2-1 (worst case): 3rd, 2nd, AND max all create impulses
   → Three-way interference, hard to decode

3. THE SPECIFIC FAILURE:
   - The model sees impulse patterns from 3rd and 2nd
   - Both come BEFORE max, so both partially decay
   - The difference between 3rd and 2nd is subtle
   - When gap is small, model can't distinguish them

4. WHY 56% OF ERRORS PREDICT 3RD:
   - 3rd clips first, has longest decay time
   - Its signal is strong in h_final
   - Model confuses 3rd's strong early signal for 2nd's signal
""")


if __name__ == "__main__":
    main()
