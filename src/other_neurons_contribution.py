"""
Other Neurons' Contribution Analysis

If comparators (n1,6,7,8) primarily encode ARGMAX position,
what encodes the 2ND_ARGMAX position?

Hypothesis: The other neurons (especially n2, and the integrators)
must carry the 2nd_argmax information.
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def get_full_trajectory(model, x):
    """Compute full trajectory."""
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data

    batch_size = x.shape[0]
    hidden = th.zeros(batch_size, 16, 10)
    h = th.zeros(batch_size, 16)

    for t in range(10):
        x_t = x[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
        hidden[:, :, t] = h

    return hidden


def analyze_all_neuron_correlations(model, n_samples=50000):
    """
    For ALL 16 neurons, check correlation with argmax vs 2nd_argmax.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    W_out = model.linear.weight.data

    print("=" * 70)
    print("ALL NEURONS: CORRELATION WITH ARGMAX vs 2ND_ARGMAX")
    print("=" * 70)

    print(f"\n{'Neuron':<8} | {'r(argmax)':<12} | {'r(2nd_argmax)':<14} | {'|W_out|':<10} | {'Category':<15}")
    print("-" * 75)

    correlations = []

    for n in range(16):
        corr_argmax = np.corrcoef(h_final[:, n].numpy(), argmax_pos.numpy())[0, 1]
        corr_2nd = np.corrcoef(h_final[:, n].numpy(), targets.numpy())[0, 1]
        w_importance = th.abs(W_out[:, n]).sum().item()

        # Categorize
        if n in [1, 6, 7, 8]:
            cat = "Comparator"
        elif n == 4:
            cat = "Inv. Comparator"
        elif n == 2:
            cat = "Max Tracker"
        else:
            cat = "Other"

        correlations.append((n, corr_argmax, corr_2nd, w_importance, cat))
        print(f"n{n:<7} | {corr_argmax:>+10.3f}  | {corr_2nd:>+12.3f}  | {w_importance:>8.1f}  | {cat:<15}")

    # Sort by 2nd_argmax correlation
    print("\n" + "-" * 75)
    print("Sorted by |r(2nd_argmax)|:")
    sorted_by_2nd = sorted(correlations, key=lambda x: abs(x[2]), reverse=True)
    for n, r_arg, r_2nd, w, cat in sorted_by_2nd[:8]:
        print(f"  n{n}: r(2nd_argmax) = {r_2nd:+.3f}, r(argmax) = {r_arg:+.3f}")

    return correlations


def analyze_neuron_by_both_positions(model, n_samples=100000):
    """
    For key neurons, show h_final as function of BOTH argmax and 2nd_argmax.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    print("\n" + "=" * 70)
    print("VARIANCE DECOMPOSITION: ARGMAX vs 2ND_ARGMAX")
    print("=" * 70)

    print(f"\n{'Neuron':<8} | {'Var(by argmax)':<16} | {'Var(by 2nd_argmax)':<18} | {'Ratio':<8}")
    print("-" * 65)

    for n in range(16):
        # Compute variance explained by argmax
        means_by_argmax = []
        for i in range(10):
            mask = argmax_pos == i
            if mask.sum() > 50:
                means_by_argmax.append(h_final[mask, n].mean().item())
        var_argmax = np.var(means_by_argmax)

        # Compute variance explained by 2nd_argmax
        means_by_2nd = []
        for j in range(10):
            mask = targets == j
            if mask.sum() > 50:
                means_by_2nd.append(h_final[mask, n].mean().item())
        var_2nd = np.var(means_by_2nd)

        ratio = var_argmax / var_2nd if var_2nd > 0.001 else float('inf')

        print(f"n{n:<7} | {var_argmax:>14.3f}  | {var_2nd:>16.3f}  | {ratio:>6.1f}x")


def analyze_residual_after_argmax(model, n_samples=50000):
    """
    After controlling for argmax position, what predicts 2nd_argmax?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS: CONTROLLING FOR ARGMAX")
    print("=" * 70)

    # For each neuron, compute residual after removing argmax effect
    print("\nFor each neuron, regress out argmax position, then correlate residual with 2nd_argmax:")
    print("-" * 60)

    for n in range(16):
        h_n = h_final[:, n].numpy()

        # Compute mean h_n for each argmax position
        argmax_means = np.zeros(10)
        for i in range(10):
            mask = (argmax_pos == i).numpy()
            if mask.sum() > 0:
                argmax_means[i] = h_n[mask].mean()

        # Compute residual
        predicted = argmax_means[argmax_pos.numpy()]
        residual = h_n - predicted

        # Correlate residual with 2nd_argmax
        corr_residual = np.corrcoef(residual, targets.numpy())[0, 1]

        if abs(corr_residual) > 0.05:
            print(f"  n{n}: r(residual, 2nd_argmax) = {corr_residual:+.3f}")


def analyze_within_argmax_discrimination(model, n_samples=100000):
    """
    Within samples that have the SAME argmax position,
    which neurons help discriminate different 2nd_argmax positions?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    W_out = model.linear.weight.data

    print("\n" + "=" * 70)
    print("WITHIN-ARGMAX DISCRIMINATION")
    print("=" * 70)
    print("\nWhen argmax is fixed at position 5, which neurons discriminate 2nd_argmax?")

    mask_argmax5 = argmax_pos == 5

    print(f"\nSamples with argmax=5: {mask_argmax5.sum().item()}")

    print(f"\n{'Neuron':<8} | {'Var(h|2nd_argmax)':<18} | {'Discriminability':<15}")
    print("-" * 50)

    discriminabilities = []

    for n in range(16):
        # Variance of h_final[n] explained by 2nd_argmax, within argmax=5
        means = []
        for j in range(10):
            if j == 5:
                continue
            mask = mask_argmax5 & (targets == j)
            if mask.sum() > 30:
                means.append(h_final[mask, n].mean().item())

        var_between = np.var(means) if len(means) > 1 else 0

        # Weight by W_out importance for this neuron
        w_var = th.var(W_out[:, n]).item()

        discriminability = var_between * w_var

        discriminabilities.append((n, var_between, discriminability))
        print(f"n{n:<7} | {var_between:>16.4f}  | {discriminability:>13.4f}")

    # Top discriminators
    print("\nTop discriminators (var × W_out_var):")
    sorted_disc = sorted(discriminabilities, key=lambda x: x[2], reverse=True)
    for n, var_b, disc in sorted_disc[:5]:
        print(f"  n{n}: {disc:.4f}")


def plot_neuron_contributions_by_role(model, save_path=None):
    """
    Visualize which neurons encode argmax vs 2nd_argmax.
    """
    n_samples = 50000
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Correlation with argmax vs 2nd_argmax for all neurons
    ax = axes[0, 0]
    corr_argmax = []
    corr_2nd = []
    for n in range(16):
        corr_argmax.append(np.corrcoef(h_final[:, n].numpy(), argmax_pos.numpy())[0, 1])
        corr_2nd.append(np.corrcoef(h_final[:, n].numpy(), targets.numpy())[0, 1])

    colors = ['red' if n in [1,6,7,8] else 'blue' if n == 2 else 'green' if n == 4 else 'gray'
              for n in range(16)]

    ax.scatter(corr_argmax, corr_2nd, c=colors, s=100)
    for n in range(16):
        ax.annotate(f'n{n}', (corr_argmax[n], corr_2nd[n]), fontsize=9)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Correlation with argmax')
    ax.set_ylabel('Correlation with 2nd_argmax')
    ax.set_title('Which position does each neuron encode?\n(red=comparators, blue=n2, green=n4)')
    ax.grid(True, alpha=0.3)

    # Panel 2: h_final by 2nd_argmax for non-comparator neurons (when argmax=5)
    ax = axes[0, 1]
    mask = argmax_pos == 5
    other_neurons = [0, 2, 3, 5, 9, 10, 11, 12]

    for n in other_neurons:
        means = []
        positions = []
        for j in range(10):
            if j == 5:
                continue
            m = mask & (targets == j)
            if m.sum() > 30:
                positions.append(j)
                means.append(h_final[m, n].mean().item())

        # Normalize for comparison
        if len(means) > 0:
            means_norm = (np.array(means) - np.mean(means)) / (np.std(means) + 1e-6)
            ax.plot(positions, means_norm, 'o-', label=f'n{n}', alpha=0.7)

    ax.set_xlabel('2nd argmax position')
    ax.set_ylabel('Normalized h_final (when argmax=5)')
    ax.set_title('Non-comparator neurons by 2nd_argmax\n(controlling for argmax)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Comparators by 2nd_argmax (when argmax=5)
    ax = axes[1, 0]
    comparators = [1, 6, 7, 8]

    for n in comparators:
        means = []
        positions = []
        for j in range(10):
            if j == 5:
                continue
            m = mask & (targets == j)
            if m.sum() > 30:
                positions.append(j)
                means.append(h_final[m, n].mean().item())

        ax.plot(positions, means, 'o-', label=f'n{n}', linewidth=2, markersize=8)

    ax.set_xlabel('2nd argmax position')
    ax.set_ylabel('Mean h_final (when argmax=5)')
    ax.set_title('Comparators DO vary by 2nd_argmax!\n(when argmax is controlled)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Contribution to correct output by neuron group
    ax = axes[1, 1]
    W_out = model.linear.weight.data

    # For samples with argmax=5, compute contribution to correct output
    contrib_comparators = []
    contrib_n2 = []
    contrib_others = []

    for j in range(10):
        if j == 5:
            continue
        m = mask & (targets == j)
        if m.sum() > 30:
            # Contribution from comparators
            c_comp = sum((h_final[m, n].mean() * W_out[j, n]).item() for n in [1,6,7,8])
            c_n2 = (h_final[m, 2].mean() * W_out[j, 2]).item()
            c_other = sum((h_final[m, n].mean() * W_out[j, n]).item()
                         for n in range(16) if n not in [1,2,6,7,8])

            contrib_comparators.append(c_comp)
            contrib_n2.append(c_n2)
            contrib_others.append(c_other)

    positions = [j for j in range(10) if j != 5]
    x_pos = np.arange(len(positions))
    width = 0.25

    ax.bar(x_pos - width, contrib_comparators, width, label='Comparators', color='red', alpha=0.7)
    ax.bar(x_pos, contrib_n2, width, label='n2', color='blue', alpha=0.7)
    ax.bar(x_pos + width, contrib_others, width, label='Others', color='gray', alpha=0.7)

    ax.set_xlabel('2nd argmax position')
    ax.set_ylabel('Contribution to correct logit')
    ax.set_title('Who contributes to correct prediction?\n(when argmax=5)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(positions)
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved neuron contributions to {save_path}')

    return fig


def main():
    model = example_2nd_argmax()

    analyze_all_neuron_correlations(model)
    analyze_neuron_by_both_positions(model)
    analyze_residual_after_argmax(model)
    analyze_within_argmax_discrimination(model)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    plot_neuron_contributions_by_role(model, 'docs/figures/neuron_contributions_by_role.png')

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The comparators (n1,6,7,8) encode ARGMAX position strongly, but they
ALSO carry information about 2nd_argmax in their fine-grained values.

When we control for argmax (look within samples with same argmax),
the comparators still show variation by 2nd_argmax position!

This is because:
1. Comparators clip at BOTH argmax AND 2nd_argmax (both are "new max" events)
2. The timing of both clips affects the rebuild trajectory
3. h_final encodes a COMBINATION of both positions

The W_out weights then decode this combined signal to extract 2nd_argmax.

It's not that comparators encode ONLY argmax - they encode a mixture,
with argmax being the dominant component but 2nd_argmax information
still present in the residual variation.
""")


if __name__ == "__main__":
    main()
