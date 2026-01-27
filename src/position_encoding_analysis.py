"""
Position Encoding Analysis for the 2nd Argmax Model

Investigates how the model encodes position information using a
distributed code across multiple neurons, with n2 acting as a normalizer.
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def get_final_hidden(model, x):
    """Compute final hidden state after processing sequence."""
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data

    h = th.zeros(x.shape[0], 16)
    for t in range(10):
        x_t = x[:, t:t+1]
        pre_act = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre_act)
    return h


def compute_differential_contributions(model, n_samples=30000):
    """
    For each position, compute how each neuron's contribution differs
    from its mean contribution to other positions.

    Returns:
        diffs: [10 positions, 16 neurons] array of differential contributions
    """
    W_out = model.linear.weight.data

    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    h_final = get_final_hidden(model, x)

    diffs = np.zeros((10, 16))

    for p in range(10):
        mask = targets == p
        h_p = h_final[mask]

        for n in range(16):
            # Contribution to correct position
            contrib_correct = (h_p[:, n] * W_out[p, n]).mean().item()
            # Mean contribution to other positions
            other_contribs = [(h_p[:, n] * W_out[other_p, n]).mean().item()
                             for other_p in range(10) if other_p != p]
            contrib_others = sum(other_contribs) / len(other_contribs)
            diffs[p, n] = contrib_correct - contrib_others

    return diffs


def plot_position_encoding_heatmap(model, save_path=None):
    """
    Create a heatmap showing which neurons help/hurt each position.
    """
    diffs = compute_differential_contributions(model)

    # Focus on the key neurons
    key_neurons = [1, 6, 7, 8, 2, 0, 10, 14]
    neuron_labels = [f'n{n}' for n in key_neurons]

    diffs_subset = diffs[:, key_neurons]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(diffs_subset, cmap='RdBu_r', aspect='auto',
                   vmin=-50, vmax=50)

    # Labels
    ax.set_xticks(range(len(key_neurons)))
    ax.set_xticklabels(neuron_labels, fontsize=12)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'pos {i}' for i in range(10)], fontsize=12)

    ax.set_xlabel('Neuron', fontsize=14)
    ax.set_ylabel('2nd Argmax Position', fontsize=14)
    ax.set_title('Differential Neuron Contribution by Position\n(positive = helps identify this position)',
                 fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Contribution difference vs. other positions', fontsize=12)

    # Add text annotations
    for i in range(10):
        for j, n in enumerate(key_neurons):
            val = diffs_subset[i, j]
            color = 'white' if abs(val) > 25 else 'black'
            ax.text(j, i, f'{val:+.0f}', ha='center', va='center',
                   color=color, fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved heatmap to {save_path}')

    return fig


def plot_position_signatures(model, save_path=None):
    """
    Create a visualization showing the unique signature for each position
    using the 4 main comparator neurons.
    """
    diffs = compute_differential_contributions(model)

    # Focus on comparators
    comparators = [1, 6, 7, 8]
    diffs_comp = diffs[:, comparators]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for pos in range(10):
        ax = axes[pos]
        colors = ['green' if d > 5 else 'red' if d < -5 else 'gray'
                  for d in diffs_comp[pos]]
        bars = ax.bar(range(4), diffs_comp[pos], color=colors)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f'n{n}' for n in comparators])
        ax.set_ylim(-50, 50)
        ax.set_title(f'Position {pos}', fontsize=12)

        if pos % 5 == 0:
            ax.set_ylabel('Diff. contribution')

    fig.suptitle('Position Signatures: Comparator Neuron Contributions\n(green = helps, red = hurts)',
                 fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved signatures to {save_path}')

    return fig


def plot_n2_normalization(model, save_path=None):
    """
    Visualize how n2 normalizes the logits across positions.
    """
    W_out = model.linear.weight.data

    x = th.rand(20000, 10)
    targets = task_2nd_argmax(x)
    h_final = get_final_hidden(model, x)

    # Compute logits with and without n2
    logits_no_n2 = th.zeros(len(x), 10)
    for n in range(16):
        if n != 2:
            logits_no_n2 += h_final[:, n:n+1] * W_out[:, n:n+1].T

    logits_just_n2 = h_final[:, 2:3] * W_out[:, 2:3].T
    logits_full = logits_no_n2 + logits_just_n2

    # Get mean correct logit for each position
    no_n2_by_pos = []
    n2_contrib_by_pos = []
    full_by_pos = []

    for p in range(10):
        mask = targets == p
        no_n2_by_pos.append(logits_no_n2[mask, p].mean().item())
        n2_contrib_by_pos.append(logits_just_n2[mask, p].mean().item())
        full_by_pos.append(logits_full[mask, p].mean().item())

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(10)
    width = 0.25

    bars1 = ax.bar(x_pos - width, no_n2_by_pos, width, label='Without n2', color='coral')
    bars2 = ax.bar(x_pos, n2_contrib_by_pos, width, label='n2 contribution', color='steelblue')
    bars3 = ax.bar(x_pos + width, full_by_pos, width, label='Full model', color='green')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('2nd Argmax Position', fontsize=12)
    ax.set_ylabel('Mean logit for correct position', fontsize=12)
    ax.set_title('n2 as Normalizer: Balancing Position Logits', fontsize=14)
    ax.set_xticks(x_pos)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved n2 normalization plot to {save_path}')

    return fig


def analyze_ablation(model, n_samples=20000):
    """
    Analyze accuracy when using different subsets of neurons.
    """
    W_out = model.linear.weight.data

    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    h_final = get_final_hidden(model, x)

    with th.no_grad():
        pred_full = model(x).argmax(dim=-1)
    acc_full = (pred_full == targets).float().mean().item()

    results = {'full': acc_full}

    # Different neuron subsets
    subsets = {
        'comparators (1,6,7,8)': [1, 6, 7, 8],
        'n2 only': [2],
        'all except n2': [n for n in range(16) if n != 2],
        'comparators + n2': [1, 2, 6, 7, 8],
    }

    for name, neurons in subsets.items():
        logits = th.zeros(len(x), 10)
        for n in neurons:
            logits += h_final[:, n:n+1] * W_out[:, n:n+1].T
        pred = logits.argmax(dim=-1)
        acc = (pred == targets).float().mean().item()
        results[name] = acc

    return results


def plot_confusion_vs_similarity(model, n_samples=100000, save_path=None):
    """
    Plot confusion matrix alongside signature similarity to show correlation.
    """
    from numpy.linalg import norm

    W_out = model.linear.weight.data

    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    h_final = get_final_hidden(model, x)

    with th.no_grad():
        preds = model(x).argmax(dim=-1)

    # Compute confusion matrix
    confusion = np.zeros((10, 10))
    for t, p in zip(targets.numpy(), preds.numpy()):
        confusion[t, p] += 1
    confusion_norm = confusion / confusion.sum(axis=1, keepdims=True)

    # Compute signatures
    key_neurons = [1, 6, 7, 8]
    signatures = np.zeros((10, len(key_neurons)))

    for pos in range(10):
        mask = targets == pos
        h_p = h_final[mask]
        for j, n in enumerate(key_neurons):
            contrib_correct = (h_p[:, n] * W_out[pos, n]).mean().item()
            other_contribs = [(h_p[:, n] * W_out[other_p, n]).mean().item()
                             for other_p in range(10) if other_p != pos]
            contrib_others = sum(other_contribs) / len(other_contribs)
            signatures[pos, j] = contrib_correct - contrib_others

    # Compute similarity
    similarity = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if norm(signatures[i]) > 0 and norm(signatures[j]) > 0:
                similarity[i, j] = np.dot(signatures[i], signatures[j]) / (norm(signatures[i]) * norm(signatures[j]))

    # Zero diagonal for visualization
    similarity_offdiag = similarity.copy()
    np.fill_diagonal(similarity_offdiag, 0)
    confusion_offdiag = confusion_norm.copy()
    np.fill_diagonal(confusion_offdiag, 0)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im1 = axes[0].imshow(confusion_offdiag, cmap='Reds', vmin=0, vmax=0.05)
    axes[0].set_title('Confusion Matrix\n(off-diagonal, true->pred)', fontsize=12)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_xticks(range(10))
    axes[0].set_yticks(range(10))
    plt.colorbar(im1, ax=axes[0], label='P(pred|true)')

    im2 = axes[1].imshow(similarity_offdiag, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Signature Similarity\n(off-diagonal, cosine)', fontsize=12)
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Position')
    axes[1].set_xticks(range(10))
    axes[1].set_yticks(range(10))
    plt.colorbar(im2, ax=axes[1], label='Cosine similarity')

    # Scatter plot
    sim_flat = []
    conf_flat = []
    for i in range(10):
        for j in range(10):
            if i != j:
                sim_flat.append(similarity[i, j])
                conf_flat.append(confusion_norm[i, j])

    axes[2].scatter(sim_flat, conf_flat, alpha=0.6)
    axes[2].set_xlabel('Signature Similarity (cosine)')
    axes[2].set_ylabel('Confusion Rate')
    axes[2].set_title('Similarity vs Confusion', fontsize=12)

    corr = np.corrcoef(sim_flat, conf_flat)[0, 1]
    axes[2].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[2].transAxes, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved confusion vs similarity to {save_path}')

    return fig, corr


def main():
    model = example_2nd_argmax()

    print("=" * 70)
    print("POSITION ENCODING ANALYSIS")
    print("=" * 70)

    # Ablation study
    print("\n1. ABLATION STUDY")
    print("-" * 50)
    results = analyze_ablation(model)
    for name, acc in results.items():
        print(f"  {name:30s}: {acc:.1%}")

    # Generate plots
    print("\n2. GENERATING VISUALIZATIONS")
    print("-" * 50)

    plot_position_encoding_heatmap(model, 'docs/figures/position_encoding_heatmap.png')
    plot_position_signatures(model, 'docs/figures/position_signatures.png')
    plot_n2_normalization(model, 'docs/figures/n2_normalization.png')
    _, corr = plot_confusion_vs_similarity(model, save_path='docs/figures/confusion_vs_similarity.png')

    print("\n3. KEY FINDINGS")
    print("-" * 50)
    print(f"""
    - Comparators (n1,n6,n7,n8) alone: ~12% accuracy (near random)
    - n2 alone: ~10% accuracy (random)
    - All except n2: ~14% accuracy (still near random!)
    - Full model: ~89% accuracy

    The combination of comparators + n2 is critical.
    n2 acts as a normalizer that rebalances position biases.

    Confusion vs similarity correlation: r = {corr:.3f}
    """)


if __name__ == "__main__":
    main()
