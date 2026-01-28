"""
Partial Clipping Analysis

When max comes first (1-x-x orderings) with a large gap,
the 2nd argmax may not clip at all, or only some comparators clip.

Questions:
1. Which comparators clip at 2nd position when gap is large?
2. Does the threshold cascade (W_ih ordering) create partial clipping?
3. How does accuracy depend on WHICH comparators clip?
4. What information does partial clipping provide?
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
    pre_activations = th.zeros(batch_size, 16, 10)
    h = th.zeros(batch_size, 16)

    for t in range(10):
        x_t = x[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        pre_activations[:, :, t] = pre
        clipped[:, :, t] = pre < 0
        h = th.relu(pre)
        hidden[:, :, t] = h

    return hidden, clipped, pre_activations


def analyze_partial_clipping_by_gap(model, n_samples=200000):
    """
    For max-first orderings, analyze which comparators clip at 2nd position
    as a function of the gap.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]  # max
    pos2 = top3.indices[:, 1]  # 2nd
    val1 = top3.values[:, 0]
    val2 = top3.values[:, 1]
    gap = val1 - val2

    # Focus on orderings where max comes first
    max_first = pos1 < pos2  # max appears before 2nd in time

    hidden, clipped, _ = get_full_trajectory(model, x)

    comparators = [1, 6, 7, 8]

    print("=" * 70)
    print("PARTIAL CLIPPING: MAX FIRST, BY GAP SIZE")
    print("=" * 70)

    print(f"\nSamples where max comes before 2nd: {max_first.sum().item()}")

    print("\n" + "-" * 80)
    print("Clipping rate at 2ND ARGMAX position by gap:")
    print("-" * 80)
    print(f"{'Gap range':<15} | {'n1':<8} | {'n6':<8} | {'n7':<8} | {'n8':<8} | {'Count':<10}")
    print("-" * 80)

    gap_bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
                (0.20, 0.30), (0.30, 0.40), (0.40, 0.60), (0.60, 1.0)]

    for low, high in gap_bins:
        mask = max_first & (gap >= low) & (gap < high)
        if mask.sum() < 100:
            continue

        rates = {}
        for n in comparators:
            clips = []
            for i in range(n_samples):
                if mask[i]:
                    clips.append(clipped[i, n, pos2[i]].item())
            rates[n] = np.mean(clips)

        print(f"[{low:.2f}, {high:.2f})    | {rates[1]:<8.1%} | {rates[6]:<8.1%} | "
              f"{rates[7]:<8.1%} | {rates[8]:<8.1%} | {mask.sum().item():<10}")

    return gap_bins


def analyze_clipping_patterns(model, n_samples=200000):
    """
    Categorize samples by which comparators clip at 2nd position.
    Pattern: (n1, n6, n7, n8) as binary string.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]
    val2 = top3.values[:, 1]
    gap = top3.values[:, 0] - val2

    max_first = pos1 < pos2

    hidden, clipped, _ = get_full_trajectory(model, x)

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 70)
    print("CLIPPING PATTERNS AT 2ND POSITION (max first only)")
    print("=" * 70)

    # Build pattern for each sample
    patterns = {}

    for i in range(n_samples):
        if not max_first[i]:
            continue

        pattern = tuple(clipped[i, n, pos2[i]].item() for n in comparators)
        pattern_str = ''.join(['1' if p else '0' for p in pattern])

        if pattern_str not in patterns:
            patterns[pattern_str] = {
                'count': 0, 'correct': 0, 'gaps': [], 'val2s': []
            }

        patterns[pattern_str]['count'] += 1
        patterns[pattern_str]['correct'] += correct[i].item()
        patterns[pattern_str]['gaps'].append(gap[i].item())
        patterns[pattern_str]['val2s'].append(val2[i].item())

    print("\nPattern: n1-n6-n7-n8 (1=clips, 0=doesn't clip)")
    print("-" * 80)
    print(f"{'Pattern':<12} | {'Count':<10} | {'Accuracy':<10} | {'Mean gap':<12} | {'Mean 2nd val':<12}")
    print("-" * 80)

    # Sort by count
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['count'], reverse=True)

    for pattern_str, data in sorted_patterns:
        if data['count'] < 50:
            continue
        acc = data['correct'] / data['count']
        mean_gap = np.mean(data['gaps'])
        mean_val2 = np.mean(data['val2s'])
        print(f"{pattern_str:<12} | {data['count']:<10} | {acc:<10.1%} | {mean_gap:<12.3f} | {mean_val2:<12.3f}")

    return patterns


def analyze_threshold_cascade(model):
    """
    Show the W_ih values that create the threshold cascade.
    """
    W_ih = model.rnn.weight_ih_l0.data.squeeze()

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 70)
    print("THRESHOLD CASCADE (W_ih values)")
    print("=" * 70)

    print("\nComparator input weights (determine clipping threshold):")
    print("-" * 50)

    sorted_comp = sorted(comparators, key=lambda n: W_ih[n].item(), reverse=True)

    for n in sorted_comp:
        w = W_ih[n].item()
        # More negative = higher threshold = clips later
        print(f"  n{n}: W_ih = {w:+.2f}  (clips {'first' if n == sorted_comp[0] else 'later'})")

    print("""
Interpretation:
  - More negative W_ih = needs larger input to overcome negative bias
  - n1 (-10.56): Lowest threshold, clips at smallest values
  - n6 (-11.00): Medium-low threshold
  - n8 (-12.31): Medium-high threshold
  - n7 (-13.17): Highest threshold, clips at largest values

When 2nd max is moderate:
  - n1 clips (low threshold)
  - n6 might clip
  - n8 might not clip
  - n7 definitely doesn't clip

This creates PARTIAL CLIPPING patterns that encode amplitude!
""")


def analyze_partial_clipping_information(model, n_samples=200000):
    """
    What information does partial clipping provide?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]
    val2 = top3.values[:, 1]
    gap = top3.values[:, 0] - val2

    max_first = pos1 < pos2

    hidden, clipped, _ = get_full_trajectory(model, x)

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 70)
    print("INFORMATION IN PARTIAL CLIPPING")
    print("=" * 70)

    # Count how many comparators clip at 2nd position
    n_clip_at_2nd = th.zeros(n_samples)
    for i in range(n_samples):
        if max_first[i]:
            for n in comparators:
                n_clip_at_2nd[i] += clipped[i, n, pos2[i]].float()

    print("\nNumber of comparators clipping at 2nd (when max first):")
    print("-" * 60)
    print(f"{'# Clips':<10} | {'Count':<10} | {'Mean gap':<12} | {'Mean 2nd val':<12}")
    print("-" * 60)

    for k in range(5):
        mask = max_first & (n_clip_at_2nd == k)
        if mask.sum() > 50:
            mean_gap = gap[mask].mean().item()
            mean_val2 = val2[mask].mean().item()
            print(f"{k:<10} | {mask.sum().item():<10} | {mean_gap:<12.3f} | {mean_val2:<12.3f}")

    # Correlation between # clips and 2nd value
    valid_mask = max_first & (n_clip_at_2nd >= 0)
    corr_val2 = np.corrcoef(n_clip_at_2nd[valid_mask].numpy(),
                            val2[valid_mask].numpy())[0, 1]
    corr_gap = np.corrcoef(n_clip_at_2nd[valid_mask].numpy(),
                           gap[valid_mask].numpy())[0, 1]

    print(f"\nCorrelation with # comparators clipping:")
    print(f"  r(# clips, 2nd value) = {corr_val2:+.3f}")
    print(f"  r(# clips, gap) = {corr_gap:+.3f}")


def analyze_accuracy_by_clipping_count(model, n_samples=200000):
    """
    How does accuracy depend on how many comparators clip at 2nd?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]
    gap = top3.values[:, 0] - top3.values[:, 1]

    max_first = pos1 < pos2

    hidden, clipped, _ = get_full_trajectory(model, x)

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 70)
    print("ACCURACY BY NUMBER OF COMPARATORS CLIPPING AT 2ND")
    print("=" * 70)

    # Count clips at 2nd
    n_clip_at_2nd = th.zeros(n_samples)
    for i in range(n_samples):
        if max_first[i]:
            for n in comparators:
                n_clip_at_2nd[i] += clipped[i, n, pos2[i]].float()

    print("\n" + "-" * 60)
    print(f"{'# Clips':<10} | {'Count':<10} | {'Accuracy':<10} | {'Mean gap':<12}")
    print("-" * 60)

    for k in range(5):
        mask = max_first & (n_clip_at_2nd == k)
        if mask.sum() > 50:
            acc = correct[mask].float().mean().item()
            mean_gap = gap[mask].mean().item()
            print(f"{k:<10} | {mask.sum().item():<10} | {acc:<10.1%} | {mean_gap:<12.3f}")

    # Within each clipping count, how does gap affect accuracy?
    print("\n" + "-" * 60)
    print("Accuracy by gap, for different clipping counts:")
    print("-" * 60)

    for k in [0, 1, 2, 4]:
        print(f"\n{k} comparators clip at 2nd:")
        for low, high in [(0.0, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.0)]:
            mask = max_first & (n_clip_at_2nd == k) & (gap >= low) & (gap < high)
            if mask.sum() > 30:
                acc = correct[mask].float().mean().item()
                print(f"  Gap [{low:.2f}, {high:.2f}): {acc:.1%} (n={mask.sum().item()})")


def analyze_h_final_by_clipping_pattern(model, n_samples=200000):
    """
    How does h_final vary by clipping pattern?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]

    max_first = pos1 < pos2

    hidden, clipped, _ = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 70)
    print("H_FINAL BY CLIPPING PATTERN")
    print("=" * 70)

    # Group by pattern
    pattern_h = {}

    for i in range(n_samples):
        if not max_first[i]:
            continue

        pattern = tuple(clipped[i, n, pos2[i]].item() for n in comparators)
        pattern_str = ''.join(['1' if p else '0' for p in pattern])

        if pattern_str not in pattern_h:
            pattern_h[pattern_str] = {n: [] for n in range(16)}

        for n in range(16):
            pattern_h[pattern_str][n].append(h_final[i, n].item())

    print("\nMean h_final for comparators by clipping pattern:")
    print("-" * 70)
    print(f"{'Pattern':<12} | {'h1':<10} | {'h6':<10} | {'h7':<10} | {'h8':<10}")
    print("-" * 70)

    for pattern_str in sorted(pattern_h.keys(), key=lambda p: sum(int(c) for c in p)):
        if len(pattern_h[pattern_str][1]) < 50:
            continue
        h1 = np.mean(pattern_h[pattern_str][1])
        h6 = np.mean(pattern_h[pattern_str][6])
        h7 = np.mean(pattern_h[pattern_str][7])
        h8 = np.mean(pattern_h[pattern_str][8])
        print(f"{pattern_str:<12} | {h1:<10.2f} | {h6:<10.2f} | {h7:<10.2f} | {h8:<10.2f}")


def plot_partial_clipping(model, n_samples=100000, save_path=None):
    """
    Visualize partial clipping patterns.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]
    val2 = top3.values[:, 1]
    gap = top3.values[:, 0] - val2

    max_first = pos1 < pos2

    hidden, clipped, _ = get_full_trajectory(model, x)

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    comparators = [1, 6, 7, 8]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Clipping rate at 2nd by gap
    ax = axes[0, 0]

    gaps = []
    rates = {n: [] for n in comparators}

    for low, high in [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
                      (0.20, 0.30), (0.30, 0.40), (0.40, 0.60)]:
        mask = max_first & (gap >= low) & (gap < high)
        if mask.sum() < 50:
            continue

        gaps.append((low + high) / 2)

        for n in comparators:
            clips = []
            for i in range(n_samples):
                if mask[i]:
                    clips.append(clipped[i, n, pos2[i]].item())
            rates[n].append(np.mean(clips))

    colors = {1: 'blue', 6: 'green', 7: 'red', 8: 'purple'}
    for n in comparators:
        ax.plot(gaps, rates[n], 'o-', color=colors[n], label=f'n{n}', linewidth=2, markersize=6)

    ax.set_xlabel('Gap (max - 2nd)')
    ax.set_ylabel('Clipping rate at 2nd position')
    ax.set_title('Threshold cascade: which comparators clip?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel 2: Accuracy by number of clips
    ax = axes[0, 1]

    n_clip_at_2nd = th.zeros(n_samples)
    for i in range(n_samples):
        if max_first[i]:
            for n in comparators:
                n_clip_at_2nd[i] += clipped[i, n, pos2[i]].float()

    clip_counts = []
    accuracies = []
    counts = []

    for k in range(5):
        mask = max_first & (n_clip_at_2nd == k)
        if mask.sum() > 50:
            clip_counts.append(k)
            accuracies.append(correct[mask].float().mean().item())
            counts.append(mask.sum().item())

    bars = ax.bar(clip_counts, accuracies, color='steelblue', alpha=0.7)
    ax.set_xlabel('Number of comparators clipping at 2nd')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by clipping count (max first)')
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    # Panel 3: Distribution of clipping patterns
    ax = axes[1, 0]

    patterns = {}
    for i in range(n_samples):
        if not max_first[i]:
            continue
        pattern = tuple(clipped[i, n, pos2[i]].item() for n in comparators)
        pattern_str = ''.join(['1' if p else '0' for p in pattern])
        patterns[pattern_str] = patterns.get(pattern_str, 0) + 1

    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:8]
    pattern_names = [p[0] for p in sorted_patterns]
    pattern_counts = [p[1] for p in sorted_patterns]

    ax.barh(pattern_names, pattern_counts, color='coral', alpha=0.7)
    ax.set_xlabel('Count')
    ax.set_ylabel('Pattern (n1-n6-n7-n8)')
    ax.set_title('Most common clipping patterns')
    ax.grid(True, alpha=0.3, axis='x')

    # Panel 4: 2nd value distribution by clipping count
    ax = axes[1, 1]

    for k in [0, 1, 2, 3, 4]:
        mask = max_first & (n_clip_at_2nd == k)
        if mask.sum() > 100:
            vals = val2[mask].numpy()
            ax.hist(vals, bins=30, alpha=0.4, label=f'{k} clips', density=True)

    ax.set_xlabel('2nd max value')
    ax.set_ylabel('Density')
    ax.set_title('2nd value distribution by # comparators clipping')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved partial clipping analysis to {save_path}')

    return fig


def main():
    model = example_2nd_argmax()

    analyze_threshold_cascade(model)
    analyze_partial_clipping_by_gap(model)
    analyze_clipping_patterns(model)
    analyze_partial_clipping_information(model)
    analyze_accuracy_by_clipping_count(model)
    analyze_h_final_by_clipping_pattern(model)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    plot_partial_clipping(model, save_path='docs/figures/partial_clipping_analysis.png')

    print("\n" + "=" * 70)
    print("SUMMARY: PARTIAL CLIPPING")
    print("=" * 70)
    print("""
PARTIAL CLIPPING MECHANISM:

1. THRESHOLD CASCADE (W_ih values):
   n1: -10.56 (lowest threshold, clips first)
   n6: -11.00 (medium-low)
   n8: -12.31 (medium-high)
   n7: -13.17 (highest threshold, clips last)

2. PARTIAL CLIPPING PATTERNS:
   When 2nd value is moderate:
   - n1 clips (always)
   - n6 might clip
   - n8 might not clip
   - n7 rarely clips

3. INFORMATION ENCODED:
   The PATTERN of which comparators clip encodes the 2nd value's amplitude.
   More clips = larger 2nd value = smaller gap from max.

4. ACCURACY IMPLICATIONS:
   - 0 clips at 2nd: relies entirely on single-impulse mechanism
   - 4 clips at 2nd: full two-impulse mechanism
   - 1-3 clips: partial information, intermediate accuracy

This is how the model encodes AMPLITUDE information, not just position!
""")


if __name__ == "__main__":
    main()
