"""
Deep dive into the single-impulse regime (2nd_argmax AFTER argmax).

Key questions:
1. Is there confusion with 3rd_argmax when it comes BEFORE argmax?
2. What do the comparators actually see/fire in this regime?
3. How does n5 disambiguate?
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax, task_2nd_argmax


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


def analyze_third_argmax_confusion(model, n_samples=100000):
    """
    In the 2nd-after-argmax regime, does the 3rd_argmax (which comes before argmax)
    cause confusion by clipping the comparators?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    # Get 3rd argmax position
    top3 = th.topk(x, 3, dim=-1)
    third_argmax_pos = top3.indices[:, 2]
    third_val = top3.values[:, 2]

    # Focus on 2nd-after-argmax regime
    second_after = targets > argmax_pos

    # Within that, when does 3rd come before argmax?
    third_before_argmax = third_argmax_pos < argmax_pos

    # Combined: 2nd after AND 3rd before
    regime = second_after & third_before_argmax

    hidden, clipped = get_full_trajectory(model, x)

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    print("=" * 70)
    print("SINGLE-IMPULSE REGIME: 3RD ARGMAX CONFUSION ANALYSIS")
    print("=" * 70)

    print(f"\nTotal samples: {n_samples}")
    print(f"2nd AFTER argmax: {second_after.sum().item()} ({second_after.float().mean():.1%})")
    print(f"  Of these, 3rd BEFORE argmax: {regime.sum().item()} ({regime[second_after].float().mean():.1%})")

    # Accuracy comparison
    print("\n" + "-" * 50)
    print("Accuracy by scenario:")
    print("-" * 50)

    # 2nd after, 3rd before (potential confusion)
    if regime.sum() > 100:
        acc = correct[regime].float().mean().item()
        print(f"2nd after + 3rd before argmax: {acc:.1%} (n={regime.sum().item()})")

    # 2nd after, 3rd also after (no confusion)
    third_also_after = second_after & (third_argmax_pos > argmax_pos)
    if third_also_after.sum() > 100:
        acc = correct[third_also_after].float().mean().item()
        print(f"2nd after + 3rd also after:    {acc:.1%} (n={third_also_after.sum().item()})")

    # 2nd before argmax (two-impulse regime for comparison)
    second_before = targets < argmax_pos
    if second_before.sum() > 100:
        acc = correct[second_before].float().mean().item()
        print(f"2nd BEFORE argmax (baseline):   {acc:.1%} (n={second_before.sum().item()})")

    return regime, third_argmax_pos


def analyze_comparator_clipping_in_regime(model, n_samples=100000):
    """
    In the 2nd-after-argmax regime, what do comparators clip at?
    Do they clip at 3rd_argmax position?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    top3 = th.topk(x, 3, dim=-1)
    third_argmax_pos = top3.indices[:, 2]

    second_after = targets > argmax_pos

    hidden, clipped = get_full_trajectory(model, x)

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 70)
    print("COMPARATOR CLIPPING IN 2ND-AFTER-ARGMAX REGIME")
    print("=" * 70)

    print("\nClipping rates at different positions:")
    print("-" * 70)
    print(f"{'Neuron':<8} | {'at argmax':<12} | {'at 2nd_argmax':<14} | {'at 3rd_argmax':<14}")
    print("-" * 70)

    for n in comparators:
        # Compute clipping rates within the regime
        clips_at_argmax = []
        clips_at_2nd = []
        clips_at_3rd = []

        for i in range(n_samples):
            if not second_after[i]:
                continue

            clips_at_argmax.append(clipped[i, n, argmax_pos[i]].item())
            clips_at_2nd.append(clipped[i, n, targets[i]].item())
            clips_at_3rd.append(clipped[i, n, third_argmax_pos[i]].item())

        rate_argmax = np.mean(clips_at_argmax)
        rate_2nd = np.mean(clips_at_2nd)
        rate_3rd = np.mean(clips_at_3rd)

        print(f"n{n:<7} | {rate_argmax:>10.1%}  | {rate_2nd:>12.1%}  | {rate_3rd:>12.1%}")

    # Now break down by whether 3rd comes before argmax
    print("\n" + "-" * 70)
    print("When 3rd comes BEFORE argmax (within 2nd-after regime):")
    print("-" * 70)

    third_before = second_after & (third_argmax_pos < argmax_pos)

    print(f"{'Neuron':<8} | {'at argmax':<12} | {'at 2nd_argmax':<14} | {'at 3rd_argmax':<14}")
    print("-" * 70)

    for n in comparators:
        clips_at_argmax = []
        clips_at_2nd = []
        clips_at_3rd = []

        for i in range(n_samples):
            if not third_before[i]:
                continue

            clips_at_argmax.append(clipped[i, n, argmax_pos[i]].item())
            clips_at_2nd.append(clipped[i, n, targets[i]].item())
            clips_at_3rd.append(clipped[i, n, third_argmax_pos[i]].item())

        rate_argmax = np.mean(clips_at_argmax)
        rate_2nd = np.mean(clips_at_2nd)
        rate_3rd = np.mean(clips_at_3rd)

        print(f"n{n:<7} | {rate_argmax:>10.1%}  | {rate_2nd:>12.1%}  | {rate_3rd:>12.1%}")


def analyze_what_clips_before_argmax(model, n_samples=100000):
    """
    In single-impulse regime, what DOES clip before the argmax?
    Is it always nothing? Sometimes 3rd? Sometimes other values?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    top3 = th.topk(x, 3, dim=-1)
    third_argmax_pos = top3.indices[:, 2]

    second_after = targets > argmax_pos

    hidden, clipped = get_full_trajectory(model, x)

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 70)
    print("WHAT CLIPS BEFORE ARGMAX IN SINGLE-IMPULSE REGIME?")
    print("=" * 70)

    # For n7 (most selective comparator), count clips before argmax
    print("\nFor n7, counting clips at positions BEFORE argmax:")
    print("-" * 50)

    n_clips_before = []
    for i in range(n_samples):
        if not second_after[i]:
            continue

        pos = argmax_pos[i].item()
        clips_before = clipped[i, 7, :pos].sum().item()
        n_clips_before.append(clips_before)

    print(f"Mean clips before argmax (n7): {np.mean(n_clips_before):.2f}")
    print(f"Distribution:")
    for k in range(5):
        count = sum(1 for x in n_clips_before if x == k)
        pct = count / len(n_clips_before)
        print(f"  {k} clips: {pct:.1%}")

    # When there IS a clip before argmax, is it at 3rd position?
    print("\nWhen n7 clips exactly once before argmax, is it at 3rd_argmax?")
    print("-" * 50)

    matches_3rd = 0
    total_one_clip = 0

    for i in range(n_samples):
        if not second_after[i]:
            continue

        pos = argmax_pos[i].item()
        clips_before = clipped[i, 7, :pos]

        if clips_before.sum() == 1:
            total_one_clip += 1
            # Find where it clipped
            clip_pos = clips_before.nonzero()[0].item()
            if clip_pos == third_argmax_pos[i].item():
                matches_3rd += 1

    if total_one_clip > 0:
        print(f"Cases with exactly 1 clip before argmax: {total_one_clip}")
        print(f"Of these, clip at 3rd_argmax: {matches_3rd} ({matches_3rd/total_one_clip:.1%})")


def analyze_errors_in_single_impulse(model, n_samples=100000):
    """
    When the model makes errors in single-impulse regime,
    what does it predict? Does it confuse 2nd with 3rd?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    top3 = th.topk(x, 3, dim=-1)
    third_argmax_pos = top3.indices[:, 2]

    second_after = targets > argmax_pos

    with th.no_grad():
        preds = model(x).argmax(dim=-1)

    errors = second_after & (preds != targets)

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS IN SINGLE-IMPULSE REGIME")
    print("=" * 70)

    print(f"\nTotal errors in 2nd-after regime: {errors.sum().item()}")

    # Does the model predict 3rd_argmax when it's wrong?
    predicts_3rd = (preds == third_argmax_pos)
    errors_predict_3rd = errors & predicts_3rd

    print(f"Errors where prediction = 3rd_argmax: {errors_predict_3rd.sum().item()} "
          f"({errors_predict_3rd.sum().item() / errors.sum().item():.1%} of errors)")

    # What positions does it predict?
    print("\nPredicted position distribution for errors:")
    print("-" * 40)

    error_preds = preds[errors].numpy()
    for pos in range(10):
        count = (error_preds == pos).sum()
        if count > 0:
            print(f"  Position {pos}: {count} ({count/len(error_preds):.1%})")

    # Compare: actual 2nd_argmax for these errors
    print("\nActual 2nd_argmax for errors:")
    print("-" * 40)

    error_targets = targets[errors].numpy()
    for pos in range(10):
        count = (error_targets == pos).sum()
        if count > 0:
            print(f"  Position {pos}: {count} ({count/len(error_targets):.1%})")


def analyze_3rd_vs_2nd_discrimination(model, n_samples=100000):
    """
    How does the model discriminate 2nd from 3rd when both might clip?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    top3 = th.topk(x, 3, dim=-1)
    second_val = top3.values[:, 1]
    third_val = top3.values[:, 2]
    third_argmax_pos = top3.indices[:, 2]

    gap_2_3 = second_val - third_val

    second_after = targets > argmax_pos
    third_before = third_argmax_pos < argmax_pos

    # Focus on the tricky case: 2nd after, 3rd before
    regime = second_after & third_before

    hidden, clipped = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    print("\n" + "=" * 70)
    print("2ND vs 3RD DISCRIMINATION (2nd after, 3rd before argmax)")
    print("=" * 70)

    print("\nAccuracy by gap between 2nd and 3rd values:")
    print("-" * 50)

    for low, high in [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40)]:
        mask = regime & (gap_2_3 >= low) & (gap_2_3 < high)
        if mask.sum() > 50:
            acc = correct[mask].float().mean().item()
            # Also check: does 3rd clip?
            clips_3rd = []
            for i in range(n_samples):
                if mask[i]:
                    clips_3rd.append(clipped[i, 7, third_argmax_pos[i]].item())
            rate_3rd_clips = np.mean(clips_3rd)

            print(f"Gap [{low:.2f}, {high:.2f}): acc={acc:.1%}, "
                  f"n7 clips at 3rd: {rate_3rd_clips:.1%}, n={mask.sum().item()}")

    # Does n5 help discriminate?
    print("\n" + "-" * 50)
    print("n5's role in discrimination:")
    print("-" * 50)

    h5 = h_final[:, 5].numpy()

    # Compare h5 when correct vs incorrect in this regime
    correct_in_regime = regime & correct
    wrong_in_regime = regime & ~correct

    if correct_in_regime.sum() > 50 and wrong_in_regime.sum() > 50:
        h5_correct = h5[correct_in_regime.numpy()].mean()
        h5_wrong = h5[wrong_in_regime.numpy()].mean()
        print(f"Mean h5 when correct: {h5_correct:.3f}")
        print(f"Mean h5 when wrong:   {h5_wrong:.3f}")

    # Correlation of h5 with whether prediction matches 3rd
    predicts_3rd = (preds == third_argmax_pos)
    corr = np.corrcoef(h5[regime.numpy()], predicts_3rd[regime].numpy().astype(float))[0, 1]
    print(f"\nr(h5, predicts_3rd) in regime: {corr:+.3f}")


def plot_single_impulse_analysis(model, n_samples=50000, save_path=None):
    """
    Visualize the single-impulse regime.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    top3 = th.topk(x, 3, dim=-1)
    third_argmax_pos = top3.indices[:, 2]
    gap_2_3 = top3.values[:, 1] - top3.values[:, 2]

    second_after = targets > argmax_pos
    third_before = third_argmax_pos < argmax_pos
    regime = second_after & third_before

    hidden, clipped = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Clipping rates at different positions
    ax = axes[0, 0]
    comparators = [1, 6, 7, 8]
    positions = ['argmax', '2nd_argmax', '3rd_argmax']

    rates = {n: [] for n in comparators}
    for n in comparators:
        clips_argmax = []
        clips_2nd = []
        clips_3rd = []
        for i in range(n_samples):
            if not second_after[i]:
                continue
            clips_argmax.append(clipped[i, n, argmax_pos[i]].item())
            clips_2nd.append(clipped[i, n, targets[i]].item())
            clips_3rd.append(clipped[i, n, third_argmax_pos[i]].item())

        rates[n] = [np.mean(clips_argmax), np.mean(clips_2nd), np.mean(clips_3rd)]

    x_pos = np.arange(len(positions))
    width = 0.2
    for idx, n in enumerate(comparators):
        ax.bar(x_pos + idx * width, rates[n], width, label=f'n{n}', alpha=0.7)

    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(positions)
    ax.set_ylabel('Clipping rate')
    ax.set_title('Comparator clipping in 2nd-after-argmax regime')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Accuracy by gap between 2nd and 3rd
    ax = axes[0, 1]
    gaps = []
    accs = []
    for low, high in [(0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15),
                       (0.15, 0.20), (0.20, 0.30), (0.30, 0.50)]:
        mask = regime & (gap_2_3 >= low) & (gap_2_3 < high)
        if mask.sum() > 30:
            gaps.append((low + high) / 2)
            accs.append(correct[mask].float().mean().item())

    ax.plot(gaps, accs, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Gap (2nd_val - 3rd_val)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy when 2nd after & 3rd before argmax')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    # Panel 3: Error confusion matrix (predicted vs actual)
    ax = axes[1, 0]
    errors = regime & (preds != targets)
    if errors.sum() > 10:
        # Check if errors predict 3rd
        predicts_3rd = (preds[errors] == third_argmax_pos[errors])
        predicts_other = ~predicts_3rd

        labels = ['Predicts 3rd', 'Predicts other']
        sizes = [predicts_3rd.sum().item(), predicts_other.sum().item()]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'When wrong in tricky regime (n={errors.sum().item()})')
    else:
        ax.text(0.5, 0.5, 'Not enough errors', ha='center', va='center')
        ax.set_title('Error analysis')

    # Panel 4: h5 by scenario
    ax = axes[1, 1]
    scenarios = ['2nd before\nargmax', '2nd after,\n3rd also after', '2nd after,\n3rd before']
    h5_means = []

    second_before = targets < argmax_pos
    h5_means.append(h_final[second_before, 5].mean().item())

    third_also_after = second_after & (third_argmax_pos > argmax_pos)
    h5_means.append(h_final[third_also_after, 5].mean().item())

    h5_means.append(h_final[regime, 5].mean().item())

    colors = ['blue', 'green', 'red']
    ax.bar(scenarios, h5_means, color=colors, alpha=0.7)
    ax.set_ylabel('Mean h5')
    ax.set_title('n5 activation by scenario')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved single-impulse analysis to {save_path}')

    return fig


def main():
    model = example_2nd_argmax()

    analyze_third_argmax_confusion(model)
    analyze_comparator_clipping_in_regime(model)
    analyze_what_clips_before_argmax(model)
    analyze_errors_in_single_impulse(model)
    analyze_3rd_vs_2nd_discrimination(model)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    plot_single_impulse_analysis(model, save_path='docs/figures/single_impulse_regime.png')

    print("\n" + "=" * 70)
    print("SUMMARY: SINGLE-IMPULSE REGIME")
    print("=" * 70)
    print("""
KEY FINDINGS:

1. 3RD ARGMAX CONFUSION:
   When 2nd comes after argmax and 3rd comes before, does 3rd clip
   and confuse the comparators? This is the "tricky case".

2. COMPARATOR BEHAVIOR:
   In the 2nd-after regime, comparators clip at:
   - argmax: ~99% (always)
   - 2nd_argmax: varies by gap
   - 3rd_argmax: key question!

3. DISCRIMINATION MECHANISM:
   How does the model know it's seeing 3rd and not 2nd when
   something clips before argmax?
   - Amplitude information (3rd is smaller than 2nd)
   - n5 helps with regime identification

4. ERROR PATTERNS:
   When wrong in this regime, does the model predict 3rd?
""")


if __name__ == "__main__":
    main()
