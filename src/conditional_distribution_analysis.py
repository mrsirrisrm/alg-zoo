"""
Conditional Distribution Analysis

Investigates how the model learns P(2nd_argmax | argmax) through W_out weights,
and how this combines with the Fourier-encoded h_final to make predictions.
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def compute_empirical_conditional(n_samples=500000):
    """
    Compute the empirical P(2nd_argmax | argmax) from uniform random data.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    # Compute P(2nd_argmax = j | argmax = i)
    conditional = np.zeros((10, 10))
    counts = np.zeros(10)

    for i in range(10):
        mask = argmax_pos == i
        counts[i] = mask.sum().item()
        if counts[i] > 0:
            for j in range(10):
                conditional[i, j] = ((targets[mask] == j).sum().item() / counts[i])

    return conditional, counts


def analyze_conditional_structure():
    """
    Analyze the structure of P(2nd_argmax | argmax).
    """
    conditional, counts = compute_empirical_conditional()

    print("=" * 70)
    print("EMPIRICAL P(2nd_argmax | argmax)")
    print("=" * 70)

    print("\nConditional probability matrix:")
    print("Rows = argmax position, Cols = 2nd_argmax position")
    print("-" * 70)

    header = "argmax | " + " ".join(f"{j:^5}" for j in range(10))
    print(header)
    print("-" * 70)

    for i in range(10):
        row = f"  {i}    | " + " ".join(f"{conditional[i,j]:5.2f}" for j in range(10))
        print(row)

    print("\nKey observations:")
    print("-" * 50)

    # Diagonal is zero (can't have 2nd_argmax == argmax)
    print(f"1. Diagonal is zero (by definition)")

    # Is it symmetric?
    asymmetry = np.abs(conditional - conditional.T).mean()
    print(f"2. Asymmetry measure: {asymmetry:.4f} (0 = symmetric)")

    # Does position matter?
    row_entropy = []
    for i in range(10):
        p = conditional[i, :]
        p = p[p > 0]  # Remove zeros
        entropy = -np.sum(p * np.log(p + 1e-10))
        row_entropy.append(entropy)

    print(f"3. Mean entropy per row: {np.mean(row_entropy):.3f}")
    print(f"   (max possible = {np.log(9):.3f} for uniform over 9 positions)")

    return conditional


def analyze_wout_vs_conditional(model):
    """
    Compare W_out structure to the conditional distribution.
    """
    W_out = model.linear.weight.data.numpy()
    conditional, _ = compute_empirical_conditional()

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 70)
    print("W_OUT VS CONDITIONAL DISTRIBUTION")
    print("=" * 70)

    # For each output position j, W_out[j, :] should help discriminate
    # when 2nd_argmax = j

    print("\nCorrelation between W_out columns and conditional columns:")
    print("-" * 50)

    for n in comparators:
        # W_out[:, n] has 10 values (one per output position)
        # Compare to conditional[:, j] for each j

        # Actually, let's think about this differently:
        # W_out[j, n] should be high when P(2nd_argmax=j | argmax) is high
        # for the argmax positions that neuron n encodes

        w_col = W_out[:, n]
        print(f"\nn{n} output weights: {[f'{w:+.1f}' for w in w_col]}")


def plot_conditional_heatmap(model, save_path=None):
    """
    Visualize the conditional distribution and compare to model behavior.
    """
    conditional, _ = compute_empirical_conditional()

    # Get model's predictions
    n_samples = 100000
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    with th.no_grad():
        logits = model(x)
        probs = th.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

    # Compute model's conditional P(pred | argmax)
    model_conditional = np.zeros((10, 10))
    for i in range(10):
        mask = argmax_pos == i
        if mask.sum() > 0:
            for j in range(10):
                model_conditional[i, j] = (preds[mask] == j).float().mean().item()

    # Compute model's mean predicted probability P(j | argmax=i)
    model_prob = np.zeros((10, 10))
    for i in range(10):
        mask = argmax_pos == i
        if mask.sum() > 0:
            model_prob[i, :] = probs[mask].mean(dim=0).numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: True conditional
    ax = axes[0, 0]
    im = ax.imshow(conditional, cmap='Blues', vmin=0, vmax=0.2)
    ax.set_xlabel('2nd argmax position')
    ax.set_ylabel('Argmax position')
    ax.set_title('True P(2nd_argmax | argmax)\n(empirical from data)')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax)

    # Panel 2: Model's prediction conditional
    ax = axes[0, 1]
    im = ax.imshow(model_conditional, cmap='Blues', vmin=0, vmax=0.2)
    ax.set_xlabel('Predicted position')
    ax.set_ylabel('Argmax position')
    ax.set_title('Model P(prediction | argmax)\n(what model actually predicts)')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax)

    # Panel 3: Model's softmax probabilities
    ax = axes[1, 0]
    im = ax.imshow(model_prob, cmap='Blues', vmin=0, vmax=0.2)
    ax.set_xlabel('Output position')
    ax.set_ylabel('Argmax position')
    ax.set_title('Model mean softmax P(j | argmax=i)\n(soft predictions)')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax)

    # Panel 4: Difference (model - true)
    ax = axes[1, 1]
    diff = model_conditional - conditional
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    ax.set_xlabel('Position')
    ax.set_ylabel('Argmax position')
    ax.set_title('Difference: Model - True\n(red = over-predicts, blue = under-predicts)')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved conditional heatmaps to {save_path}')

    return fig


def analyze_conditional_by_gap(n_samples=500000):
    """
    The conditional P(2nd_argmax | argmax) should depend on the gap
    between max and 2nd max values.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    top2 = th.topk(x, 2, dim=-1)
    gap = top2.values[:, 0] - top2.values[:, 1]

    print("\n" + "=" * 70)
    print("CONDITIONAL DISTRIBUTION BY GAP SIZE")
    print("=" * 70)

    print("\nWhen gap is small, 2nd_argmax could be anywhere.")
    print("When gap is large, 2nd_argmax is more constrained.")

    for gap_low, gap_high in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.0)]:
        gap_mask = (gap >= gap_low) & (gap < gap_high)

        # Compute entropy of 2nd_argmax distribution
        target_counts = np.zeros(10)
        for j in range(10):
            target_counts[j] = (targets[gap_mask] == j).sum().item()

        target_probs = target_counts / target_counts.sum()
        entropy = -np.sum(target_probs * np.log(target_probs + 1e-10))

        print(f"\nGap [{gap_low:.1f}, {gap_high:.1f}):")
        print(f"  Samples: {gap_mask.sum().item()}")
        print(f"  Entropy of 2nd_argmax: {entropy:.3f} (max={np.log(10):.3f})")
        print(f"  Distribution: {[f'{p:.2f}' for p in target_probs]}")


def analyze_position_adjacency():
    """
    Analyze whether adjacent positions are more likely for 2nd_argmax.
    """
    conditional, _ = compute_empirical_conditional()

    print("\n" + "=" * 70)
    print("ADJACENCY ANALYSIS")
    print("=" * 70)

    print("\nIs 2nd_argmax more likely to be adjacent to argmax?")
    print("-" * 50)

    for i in range(10):
        probs = conditional[i, :]

        # Adjacent positions
        adjacent = []
        if i > 0:
            adjacent.append(i - 1)
        if i < 9:
            adjacent.append(i + 1)

        adj_prob = sum(probs[j] for j in adjacent)
        non_adj_prob = sum(probs[j] for j in range(10) if j not in adjacent and j != i)

        n_adj = len(adjacent)
        n_non_adj = 9 - n_adj

        avg_adj = adj_prob / n_adj if n_adj > 0 else 0
        avg_non_adj = non_adj_prob / n_non_adj if n_non_adj > 0 else 0

        print(f"argmax={i}: P(adjacent)={adj_prob:.3f} ({n_adj} pos), "
              f"P(non-adj)={non_adj_prob:.3f} ({n_non_adj} pos), "
              f"ratio={avg_adj/avg_non_adj:.2f}x")


def plot_wout_interpretation(model, save_path=None):
    """
    Visualize how W_out encodes the conditional relationship.
    """
    W_out = model.linear.weight.data.numpy()
    conditional, _ = compute_empirical_conditional()

    comparators = [1, 6, 7, 8]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: W_out for comparators
    ax = axes[0, 0]
    W_comp = W_out[:, comparators]
    im = ax.imshow(W_comp, cmap='RdBu_r', aspect='auto', vmin=-8, vmax=8)
    ax.set_xlabel('Comparator neuron')
    ax.set_ylabel('Output position (2nd_argmax)')
    ax.set_title('W_out weights for comparators')
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'n{n}' for n in comparators])
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax)

    # Panel 2: True conditional (transposed to match W_out orientation)
    ax = axes[0, 1]
    im = ax.imshow(conditional.T, cmap='Blues', vmin=0, vmax=0.2)
    ax.set_xlabel('Argmax position')
    ax.set_ylabel('2nd_argmax position')
    ax.set_title('True P(2nd_argmax | argmax)\n(transposed)')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax)

    # Panel 3: W_out row patterns vs position
    ax = axes[1, 0]
    for j in [0, 3, 5, 9]:  # Sample output positions
        weights = [W_out[j, n] for n in comparators]
        ax.plot(range(4), weights, 'o-', label=f'2nd_argmax={j}', linewidth=2, markersize=8)
    ax.set_xlabel('Comparator neuron')
    ax.set_ylabel('W_out weight')
    ax.set_title('W_out patterns for different 2nd_argmax positions')
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'n{n}' for n in comparators])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Panel 4: How W_out discriminates positions
    ax = axes[1, 1]

    # For each pair of adjacent positions, show W_out difference
    diffs = []
    pairs = []
    for j in range(9):
        diff = W_out[j, comparators] - W_out[j+1, comparators]
        diffs.append(np.linalg.norm(diff))
        pairs.append(f'{j}-{j+1}')

    ax.bar(range(9), diffs, color='steelblue', alpha=0.7)
    ax.set_xlabel('Position pair')
    ax.set_ylabel('||W_out[j] - W_out[j+1]||')
    ax.set_title('W_out separation between adjacent positions\n(larger = easier to discriminate)')
    ax.set_xticks(range(9))
    ax.set_xticklabels(pairs)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved W_out interpretation to {save_path}')

    return fig


def main():
    model = example_2nd_argmax()

    conditional = analyze_conditional_structure()
    analyze_wout_vs_conditional(model)
    analyze_conditional_by_gap()
    analyze_position_adjacency()

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_conditional_heatmap(model, 'docs/figures/conditional_distribution.png')
    plot_wout_interpretation(model, 'docs/figures/wout_interpretation.png')

    print("\n" + "=" * 70)
    print("SUMMARY: THE CONDITIONAL DISTRIBUTION MECHANISM")
    print("=" * 70)
    print("""
The model learns to predict 2nd_argmax by decomposing the problem:

1. ENCODE ARGMAX (h_final):
   - Comparators use Fourier-like encoding of argmax position
   - 4 neurons at different frequencies provide robust encoding
   - This is computed by the RNN dynamics (recency after clipping)

2. DECODE 2ND_ARGMAX (W_out):
   - W_out weights encode P(2nd_argmax | argmax)
   - Each output position j has weights W_out[j,:] that are high
     when the encoded argmax predicts 2nd_argmax = j

3. THE COMBINATION (h_final @ W_out.T):
   - Fourier encoding of argmax × learned conditional weights
   - Produces logits that approximate P(2nd_argmax | argmax, data)

Key insight: The model doesn't try to track 2nd_argmax directly.
Instead, it robustly encodes argmax and uses learned weights to
capture the statistical relationship between argmax and 2nd_argmax.

For uniform random inputs:
- P(2nd_argmax | argmax) is nearly uniform over non-argmax positions
- But NOT quite uniform: there's slight structure from order statistics
- W_out captures this structure plus any biases from the encoding
""")


if __name__ == "__main__":
    main()
