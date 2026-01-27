"""
Accuracy breakdown by all possible orderings of 1st, 2nd, 3rd argmax positions.

There are 6 possible orderings:
- 1-2-3: argmax first, then 2nd, then 3rd
- 1-3-2: argmax first, then 3rd, then 2nd
- 2-1-3: 2nd first, then argmax, then 3rd
- 2-3-1: 2nd first, then 3rd, then argmax
- 3-1-2: 3rd first, then argmax, then 2nd
- 3-2-1: 3rd first, then 2nd, then argmax
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


def get_ordering(pos1, pos2, pos3):
    """
    Return ordering string like '1-2-3' or '3-2-1'.
    Numbers refer to rank (1=max, 2=2nd max, 3=3rd max).
    """
    positions = [(pos1, '1'), (pos2, '2'), (pos3, '3')]
    positions.sort(key=lambda x: x[0])  # Sort by position (time)
    return '-'.join([p[1] for p in positions])


def analyze_accuracy_by_ordering(model, n_samples=200000):
    """
    Get accuracy breakdown by all 6 orderings.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]  # argmax position
    pos2 = top3.indices[:, 1]  # 2nd argmax position
    pos3 = top3.indices[:, 2]  # 3rd argmax position

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    # Categorize by ordering
    orderings = {}
    for i in range(n_samples):
        order = get_ordering(pos1[i].item(), pos2[i].item(), pos3[i].item())
        if order not in orderings:
            orderings[order] = {'correct': 0, 'total': 0, 'indices': []}
        orderings[order]['total'] += 1
        orderings[order]['correct'] += correct[i].item()
        orderings[order]['indices'].append(i)

    print("=" * 70)
    print("ACCURACY BY ORDERING OF 1ST, 2ND, 3RD ARGMAX")
    print("=" * 70)
    print("\nOrdering notation: position order in time (1=max, 2=2nd, 3=3rd)")
    print("Example: '3-1-2' means 3rd appears first, then max, then 2nd\n")

    print("-" * 70)
    print(f"{'Ordering':<12} | {'Count':<10} | {'Fraction':<10} | {'Accuracy':<10} | {'Interpretation':<25}")
    print("-" * 70)

    interpretations = {
        '1-2-3': '2nd after max (single impulse)',
        '1-3-2': '2nd after max, 3rd between',
        '2-1-3': '2nd before max (two impulse)',
        '2-3-1': '2nd first, 3rd before max',
        '3-1-2': '3rd first, 2nd after max',
        '3-2-1': '3rd first, then 2nd, then max',
    }

    results = []
    for order in sorted(orderings.keys()):
        data = orderings[order]
        acc = data['correct'] / data['total']
        frac = data['total'] / n_samples
        interp = interpretations.get(order, '')
        results.append((order, data['total'], frac, acc, interp))
        print(f"{order:<12} | {data['total']:<10} | {frac:<10.1%} | {acc:<10.1%} | {interp:<25}")

    print("-" * 70)
    print(f"{'TOTAL':<12} | {n_samples:<10} | {'100%':<10} | {correct.float().mean().item():<10.1%} |")

    return orderings, results


def analyze_clipping_by_ordering(model, n_samples=200000):
    """
    For each ordering, show which positions clip.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]
    pos3 = top3.indices[:, 2]

    hidden, clipped = get_full_trajectory(model, x)

    print("\n" + "=" * 70)
    print("N7 CLIPPING RATES BY ORDERING")
    print("=" * 70)

    print("\n" + "-" * 70)
    print(f"{'Ordering':<12} | {'at 1st (max)':<14} | {'at 2nd':<14} | {'at 3rd':<14}")
    print("-" * 70)

    orderings = {}
    for i in range(n_samples):
        order = get_ordering(pos1[i].item(), pos2[i].item(), pos3[i].item())
        if order not in orderings:
            orderings[order] = {'clip1': [], 'clip2': [], 'clip3': []}

        orderings[order]['clip1'].append(clipped[i, 7, pos1[i]].item())
        orderings[order]['clip2'].append(clipped[i, 7, pos2[i]].item())
        orderings[order]['clip3'].append(clipped[i, 7, pos3[i]].item())

    for order in sorted(orderings.keys()):
        data = orderings[order]
        rate1 = np.mean(data['clip1'])
        rate2 = np.mean(data['clip2'])
        rate3 = np.mean(data['clip3'])
        print(f"{order:<12} | {rate1:<14.1%} | {rate2:<14.1%} | {rate3:<14.1%}")


def analyze_errors_by_ordering(model, n_samples=200000):
    """
    For each ordering, what does the model predict when wrong?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]
    pos3 = top3.indices[:, 2]

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS BY ORDERING")
    print("=" * 70)

    print("\nWhen wrong, does the model predict 3rd argmax position?")
    print("-" * 60)
    print(f"{'Ordering':<12} | {'Errors':<10} | {'Predicts 3rd':<14} | {'Predicts other':<14}")
    print("-" * 60)

    orderings = {}
    for i in range(n_samples):
        order = get_ordering(pos1[i].item(), pos2[i].item(), pos3[i].item())
        if order not in orderings:
            orderings[order] = {'errors': 0, 'pred_3rd': 0, 'pred_other': 0}

        if not correct[i]:
            orderings[order]['errors'] += 1
            if preds[i] == pos3[i]:
                orderings[order]['pred_3rd'] += 1
            else:
                orderings[order]['pred_other'] += 1

    for order in sorted(orderings.keys()):
        data = orderings[order]
        if data['errors'] > 0:
            pred_3rd_rate = data['pred_3rd'] / data['errors']
            pred_other_rate = data['pred_other'] / data['errors']
            print(f"{order:<12} | {data['errors']:<10} | {pred_3rd_rate:<14.1%} | {pred_other_rate:<14.1%}")


def analyze_detailed_ordering(model, n_samples=200000):
    """
    More detailed breakdown including gap analysis.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]
    pos3 = top3.indices[:, 2]
    val1 = top3.values[:, 0]
    val2 = top3.values[:, 1]
    val3 = top3.values[:, 2]

    gap_12 = val1 - val2
    gap_23 = val2 - val3

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    print("\n" + "=" * 70)
    print("ACCURACY BY ORDERING AND GAP SIZE")
    print("=" * 70)

    # Focus on the problematic orderings
    problem_orderings = ['1-3-2', '3-1-2']

    for order_filter in problem_orderings:
        print(f"\n--- Ordering: {order_filter} ---")
        print("-" * 50)

        mask = th.zeros(n_samples, dtype=th.bool)
        for i in range(n_samples):
            order = get_ordering(pos1[i].item(), pos2[i].item(), pos3[i].item())
            if order == order_filter:
                mask[i] = True

        if mask.sum() < 100:
            print("Not enough samples")
            continue

        # By gap_23
        print(f"\nAccuracy by gap (2nd - 3rd value):")
        print(f"{'Gap range':<15} | {'Accuracy':<10} | {'Count':<10}")
        print("-" * 40)

        for low, high in [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50)]:
            sub_mask = mask & (gap_23 >= low) & (gap_23 < high)
            if sub_mask.sum() > 30:
                acc = correct[sub_mask].float().mean().item()
                print(f"[{low:.2f}, {high:.2f})    | {acc:<10.1%} | {sub_mask.sum().item():<10}")


def plot_ordering_analysis(model, n_samples=100000, save_path=None):
    """
    Visualize accuracy by ordering.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    top3 = th.topk(x, 3, dim=-1)
    pos1 = top3.indices[:, 0]
    pos2 = top3.indices[:, 1]
    pos3 = top3.indices[:, 2]

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    correct = (preds == targets)

    # Categorize
    orderings = {}
    for i in range(n_samples):
        order = get_ordering(pos1[i].item(), pos2[i].item(), pos3[i].item())
        if order not in orderings:
            orderings[order] = {'correct': 0, 'total': 0}
        orderings[order]['total'] += 1
        orderings[order]['correct'] += correct[i].item()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Accuracy by ordering
    ax = axes[0, 0]
    orders = sorted(orderings.keys())
    accs = [orderings[o]['correct'] / orderings[o]['total'] for o in orders]
    counts = [orderings[o]['total'] for o in orders]

    colors = ['green' if '2-' in o and o[0] == '2' else
              'orange' if o.startswith('1-') else
              'red' for o in orders]

    bars = ax.bar(orders, accs, color=colors, alpha=0.7)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Ordering (1=max, 2=2nd, 3=3rd)')
    ax.set_title('Accuracy by temporal ordering')
    ax.axhline(y=0.89, color='black', linestyle='--', alpha=0.5, label='Overall avg')
    ax.set_ylim(0.75, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'n={count}', ha='center', va='bottom', fontsize=8)

    # Panel 2: Fraction by ordering
    ax = axes[0, 1]
    fracs = [orderings[o]['total'] / n_samples for o in orders]
    ax.bar(orders, fracs, color=colors, alpha=0.7)
    ax.set_ylabel('Fraction of samples')
    ax.set_xlabel('Ordering')
    ax.set_title('Distribution of orderings')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Error rate (1 - accuracy) on log scale
    ax = axes[1, 0]
    error_rates = [1 - acc for acc in accs]
    ax.bar(orders, error_rates, color=colors, alpha=0.7)
    ax.set_ylabel('Error rate')
    ax.set_xlabel('Ordering')
    ax.set_title('Error rate by ordering')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Regime grouping
    ax = axes[1, 1]

    # Group by regime
    regime_data = {
        '2nd before max\n(two impulse)': {'orders': ['2-1-3', '2-3-1', '3-2-1'], 'correct': 0, 'total': 0},
        '2nd after max\n(single impulse)': {'orders': ['1-2-3', '1-3-2', '3-1-2'], 'correct': 0, 'total': 0},
    }

    for regime, data in regime_data.items():
        for o in data['orders']:
            if o in orderings:
                data['correct'] += orderings[o]['correct']
                data['total'] += orderings[o]['total']

    regimes = list(regime_data.keys())
    regime_accs = [regime_data[r]['correct'] / regime_data[r]['total'] if regime_data[r]['total'] > 0 else 0
                   for r in regimes]
    regime_counts = [regime_data[r]['total'] for r in regimes]

    bars = ax.bar(regimes, regime_accs, color=['green', 'orange'], alpha=0.7)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by regime')
    ax.set_ylim(0.8, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, regime_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'n={count}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved ordering analysis to {save_path}')

    return fig


def main():
    model = example_2nd_argmax()

    analyze_accuracy_by_ordering(model)
    analyze_clipping_by_ordering(model)
    analyze_errors_by_ordering(model)
    analyze_detailed_ordering(model)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    plot_ordering_analysis(model, save_path='docs/figures/ordering_analysis.png')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
KEY INSIGHTS:

ORDERINGS WHERE 2ND COMES BEFORE MAX (two-impulse regime):
  - 2-1-3: 2nd first, then max, then 3rd
  - 2-3-1: 2nd first, then 3rd, then max
  - 3-2-1: 3rd first, then 2nd, then max
  → Both 2nd and max clip, creating interference pattern

ORDERINGS WHERE 2ND COMES AFTER MAX (single-impulse regime):
  - 1-2-3: max first, then 2nd, then 3rd
  - 1-3-2: max first, then 3rd, then 2nd (3rd clips, 2nd may not!)
  - 3-1-2: 3rd first, then max, then 2nd (both 3rd and max clip!)
  → Potential confusion when 3rd clips but 2nd doesn't

PROBLEMATIC CASES:
  - 1-3-2: 3rd clips before 2nd arrives
  - 3-1-2: 3rd clips, then max clips, 2nd comes last
  → Model may confuse 3rd for 2nd
""")


if __name__ == "__main__":
    main()
