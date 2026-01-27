"""
Which Position is Encoded?

Clarifies whether the comparator neurons encode:
- Argmax position (via last clip)
- 2nd argmax position (via second-to-last clip)
- Or some combination
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


def analyze_which_position(model, n_samples=50000):
    """
    Compare h_final correlation with:
    1. Steps since last clip (argmax-related)
    2. Steps since 2nd-to-last clip (2nd argmax-related)
    3. Actual argmax position
    4. Actual 2nd argmax position
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    clipped, hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]

    print("=" * 70)
    print("WHICH POSITION IS ENCODED?")
    print("=" * 70)

    print("\nCorrelation of h_final with various position measures:")
    print("-" * 70)
    print(f"{'Neuron':<8} | {'argmax_pos':^12} | {'2nd_argmax':^12} | {'steps_last':^12} | {'steps_2nd':^12}")
    print("-" * 70)

    for n in comparators:
        # Find last and second-to-last clip positions
        last_clip = th.zeros(n_samples, dtype=th.long) - 1
        second_last_clip = th.zeros(n_samples, dtype=th.long) - 1

        for t in range(10):
            mask = clipped[:, n, t]
            # Update second_last before updating last
            second_last_clip[mask & (last_clip >= 0)] = last_clip[mask & (last_clip >= 0)]
            last_clip[mask] = t

        steps_since_last = 9 - last_clip
        steps_since_2nd = 9 - second_last_clip

        # Only compute correlations where we have valid data
        valid_last = last_clip >= 0
        valid_2nd = second_last_clip >= 0

        # Correlations
        corr_argmax = np.corrcoef(h_final[:, n].numpy(), argmax_pos.numpy())[0, 1]
        corr_2nd_argmax = np.corrcoef(h_final[:, n].numpy(), targets.numpy())[0, 1]

        corr_steps_last = np.corrcoef(h_final[valid_last, n].numpy(),
                                       steps_since_last[valid_last].numpy())[0, 1]

        if valid_2nd.sum() > 100:
            corr_steps_2nd = np.corrcoef(h_final[valid_2nd, n].numpy(),
                                          steps_since_2nd[valid_2nd].numpy())[0, 1]
        else:
            corr_steps_2nd = np.nan

        print(f"n{n:<7} | {corr_argmax:^12.3f} | {corr_2nd_argmax:^12.3f} | {corr_steps_last:^12.3f} | {corr_steps_2nd:^12.3f}")

    print("-" * 70)
    print("\nInterpretation:")
    print("  argmax_pos: actual position of maximum value")
    print("  2nd_argmax: actual position of 2nd maximum (target)")
    print("  steps_last: steps since last clip (typically at argmax)")
    print("  steps_2nd: steps since 2nd-to-last clip (may relate to 2nd argmax)")


def plot_h_final_by_both_positions(model, n_samples=50000, save_path=None):
    """
    Create heatmaps showing h_final as function of both argmax and 2nd_argmax positions.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    clipped, hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx, n in enumerate(comparators):
        # Top row: h_final by argmax position
        ax = axes[0, idx]
        means = []
        for pos in range(10):
            mask = argmax_pos == pos
            if mask.sum() > 50:
                means.append(h_final[mask, n].mean().item())
            else:
                means.append(np.nan)
        ax.bar(range(10), means, color='steelblue', alpha=0.7)
        ax.set_xlabel('Argmax position')
        ax.set_ylabel('Mean h_final')
        ax.set_title(f'n{n} by ARGMAX pos')
        ax.set_xticks(range(10))

        # Bottom row: h_final by 2nd argmax position
        ax = axes[1, idx]
        means = []
        for pos in range(10):
            mask = targets == pos
            if mask.sum() > 50:
                means.append(h_final[mask, n].mean().item())
            else:
                means.append(np.nan)
        ax.bar(range(10), means, color='coral', alpha=0.7)
        ax.set_xlabel('2nd Argmax position')
        ax.set_ylabel('Mean h_final')
        ax.set_title(f'n{n} by 2ND ARGMAX pos')
        ax.set_xticks(range(10))

    plt.suptitle('Comparator h_final: Argmax vs 2nd Argmax Position\n'
                 '(Which position do they actually encode?)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved comparison to {save_path}')

    return fig


def plot_h_final_heatmap(model, n_samples=100000, save_path=None):
    """
    2D heatmap: h_final as function of BOTH argmax and 2nd_argmax.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    _, hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    # Focus on n7 as the primary comparator
    n = 7

    # Create 2D grid
    h_grid = np.zeros((10, 10))
    count_grid = np.zeros((10, 10))

    for i in range(10):  # argmax
        for j in range(10):  # 2nd argmax
            if i == j:
                continue
            mask = (argmax_pos == i) & (targets == j)
            if mask.sum() > 30:
                h_grid[i, j] = h_final[mask, n].mean().item()
                count_grid[i, j] = mask.sum().item()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap of h_final
    ax = axes[0]
    h_masked = np.ma.masked_where(count_grid < 30, h_grid)
    im = ax.imshow(h_masked, cmap='viridis', aspect='equal')
    ax.set_xlabel('2nd Argmax position')
    ax.set_ylabel('Argmax position')
    ax.set_title(f'n{n} h_final by (argmax, 2nd_argmax)')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax, label='Mean h_final')

    # Show that it depends primarily on argmax
    ax = axes[1]
    # Compute row means (average over 2nd_argmax) and col means (average over argmax)
    row_means = np.nanmean(np.where(count_grid > 30, h_grid, np.nan), axis=1)
    col_means = np.nanmean(np.where(count_grid > 30, h_grid, np.nan), axis=0)

    ax.plot(range(10), row_means, 'o-', label='By argmax (row mean)', linewidth=2, markersize=8)
    ax.plot(range(10), col_means, 's-', label='By 2nd_argmax (col mean)', linewidth=2, markersize=8)
    ax.set_xlabel('Position')
    ax.set_ylabel('Mean h_final')
    ax.set_title(f'n{n}: Which position matters more?')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add variance annotation
    row_var = np.nanvar(row_means)
    col_var = np.nanvar(col_means)
    ax.text(0.05, 0.95, f'Variance by argmax: {row_var:.2f}\nVariance by 2nd_argmax: {col_var:.2f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved heatmap to {save_path}')

    return fig


def analyze_output_weights(model):
    """
    The output weights W_out encode the 2nd argmax position.
    Show how this combines with h_final (which encodes argmax) to predict 2nd argmax.
    """
    W_out = model.linear.weight.data

    print("\n" + "=" * 70)
    print("OUTPUT WEIGHT ANALYSIS")
    print("=" * 70)

    comparators = [1, 6, 7, 8]

    print("\nW_out weights for comparators (rows = output position):")
    print("-" * 50)
    print(f"{'Pos':<5} | " + " | ".join(f"n{n:^5}" for n in comparators))
    print("-" * 50)

    for pos in range(10):
        weights = [W_out[pos, n].item() for n in comparators]
        print(f"{pos:<5} | " + " | ".join(f"{w:+6.1f}" for w in weights))

    print("\nKey insight:")
    print("  - h_final encodes ARGMAX position (via recency/Fourier)")
    print("  - W_out encodes 2ND ARGMAX position (learned weights)")
    print("  - The combination h_final @ W_out.T decodes 2nd argmax")


def main():
    model = example_2nd_argmax()

    analyze_which_position(model)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_h_final_by_both_positions(model, save_path='docs/figures/h_final_by_positions.png')
    plot_h_final_heatmap(model, save_path='docs/figures/h_final_2d_heatmap.png')

    analyze_output_weights(model)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The comparator neurons' h_final values primarily encode ARGMAX position,
not 2nd argmax position directly.

The mechanism for predicting 2nd argmax:
1. h_final encodes argmax position via recency/Fourier-like dynamics
2. W_out weights encode position-specific patterns
3. The dot product h_final @ W_out.T combines these to predict 2nd argmax

This is analogous to:
  "Given the max is at position X (encoded in h_final),
   what's the most likely 2nd argmax position?"

The model has learned that argmax position constrains 2nd argmax:
- If max is early, 2nd argmax can be anywhere after
- If max is late, 2nd argmax is likely earlier
- Adjacent positions have similar likelihoods

W_out captures these conditional relationships.
""")


if __name__ == "__main__":
    main()
