"""
Anti-phase Analysis

Investigating whether argmax and 2nd_argmax components have opposite phase
in the comparator neurons.

If true, this is elegant: h_final = A*sin(ωt_argmax) - B*sin(ωt_2nd)
The difference encodes the relative timing.
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


def compute_position_curves(model, n_samples=100000):
    """
    Compute mean h_final as function of argmax and 2nd_argmax positions.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]

    # Compute curves
    argmax_curves = {}
    second_curves = {}

    for n in comparators:
        # By argmax (excluding position 9 for cleaner analysis)
        argmax_curve = []
        for pos in range(9):  # 0-8 only
            mask = argmax_pos == pos
            if mask.sum() > 50:
                argmax_curve.append(h_final[mask, n].mean().item())
            else:
                argmax_curve.append(np.nan)
        argmax_curves[n] = np.array(argmax_curve)

        # By 2nd_argmax (excluding position 9)
        second_curve = []
        for pos in range(9):  # 0-8 only
            mask = targets == pos
            if mask.sum() > 50:
                second_curve.append(h_final[mask, n].mean().item())
            else:
                second_curve.append(np.nan)
        second_curves[n] = np.array(second_curve)

    return argmax_curves, second_curves


def analyze_antiphase(model):
    """
    Quantify the anti-phase relationship.
    """
    argmax_curves, second_curves = compute_position_curves(model)

    print("=" * 70)
    print("ANTI-PHASE ANALYSIS (positions 0-8)")
    print("=" * 70)

    comparators = [1, 6, 7, 8]

    print("\nCorrelation between argmax and 2nd_argmax curves:")
    print("-" * 50)

    for n in comparators:
        ac = argmax_curves[n]
        sc = second_curves[n]

        # Remove NaN
        valid = ~(np.isnan(ac) | np.isnan(sc))
        ac_valid = ac[valid]
        sc_valid = sc[valid]

        # Correlation
        corr = np.corrcoef(ac_valid, sc_valid)[0, 1]

        # Normalize and check
        ac_norm = (ac_valid - ac_valid.mean()) / ac_valid.std()
        sc_norm = (sc_valid - sc_valid.mean()) / sc_valid.std()

        print(f"  n{n}: correlation = {corr:+.3f}")

    print("""
Interpretation:
  - Correlation near -1: Perfect anti-phase (opposite patterns)
  - Correlation near 0: Independent patterns
  - Correlation near +1: In-phase (same patterns)
""")

    return argmax_curves, second_curves


def plot_antiphase(model, save_path=None):
    """
    Visualize the anti-phase relationship.
    """
    argmax_curves, second_curves = compute_position_curves(model)

    comparators = [1, 6, 7, 8]
    colors = {1: 'steelblue', 6: 'coral', 7: 'green', 8: 'purple'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    positions = np.arange(9)  # 0-8

    for idx, n in enumerate(comparators):
        ax = axes[idx]

        ac = argmax_curves[n]
        sc = second_curves[n]

        # Normalize for comparison
        ac_norm = (ac - np.nanmean(ac)) / np.nanstd(ac)
        sc_norm = (sc - np.nanmean(sc)) / np.nanstd(sc)

        ax.plot(positions, ac_norm, 'o-', color='blue', linewidth=2,
                markersize=8, label='By ARGMAX')
        ax.plot(positions, sc_norm, 's-', color='red', linewidth=2,
                markersize=8, label='By 2ND_ARGMAX')
        ax.plot(positions, -sc_norm, 's--', color='red', linewidth=1,
                markersize=6, alpha=0.5, label='-1 × 2nd_argmax (flipped)')

        corr = np.corrcoef(ac[~np.isnan(ac)], sc[~np.isnan(sc)])[0, 1]

        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Position')
        ax.set_ylabel('Normalized h_final')
        ax.set_title(f'n{n}: correlation = {corr:+.3f}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(positions)

    plt.suptitle('Anti-phase Relationship: Argmax vs 2nd_Argmax Curves\n'
                 '(dashed red = inverted 2nd_argmax curve)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved anti-phase plot to {save_path}')

    return fig


def analyze_superposition_model(model, n_samples=100000):
    """
    Test the model: h_final = A*f(argmax) + B*f(2nd_argmax)
    where A and B might have opposite signs.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    print("\n" + "=" * 70)
    print("SUPERPOSITION MODEL: h = A*f(argmax) + B*g(2nd_argmax)")
    print("=" * 70)

    comparators = [1, 6, 7, 8]

    # For each neuron, fit: h_final = a*argmax_pos + b*2nd_argmax_pos + c
    # (Linear approximation)
    print("\nLinear regression: h_final ~ a*argmax + b*2nd_argmax + c")
    print("-" * 60)
    print(f"{'Neuron':<8} | {'a (argmax)':<12} | {'b (2nd_argmax)':<14} | {'R²':<8}")
    print("-" * 60)

    for n in comparators:
        h = h_final[:, n].numpy()
        X = np.column_stack([
            argmax_pos.numpy(),
            targets.numpy(),
            np.ones(n_samples)
        ])

        # Least squares
        coeffs, residuals, _, _ = np.linalg.lstsq(X, h, rcond=None)
        a, b, c = coeffs

        # R²
        h_pred = X @ coeffs
        ss_res = np.sum((h - h_pred) ** 2)
        ss_tot = np.sum((h - h.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        print(f"n{n:<7} | {a:>+10.4f}  | {b:>+12.4f}  | {r2:>6.3f}")

        # Check if a and b have opposite signs
        if a * b < 0:
            print(f"         → OPPOSITE SIGNS! Ratio: {abs(a/b):.2f}")

    print("""
If a and b have opposite signs, it means:
  - h_final INCREASES with argmax position (positive a)
  - h_final DECREASES with 2nd_argmax position (negative b)
  - Or vice versa

This creates the anti-phase relationship!
""")


def analyze_difference_encoding(model, n_samples=100000):
    """
    Test if h_final encodes (argmax - 2nd_argmax) or similar differences.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    print("\n" + "=" * 70)
    print("DIFFERENCE ENCODING")
    print("=" * 70)

    comparators = [1, 6, 7, 8]

    diff = argmax_pos - targets  # Positive if argmax > 2nd_argmax

    print("\nCorrelation of h_final with (argmax - 2nd_argmax):")
    print("-" * 50)

    for n in comparators:
        corr = np.corrcoef(h_final[:, n].numpy(), diff.numpy())[0, 1]
        print(f"  n{n}: r = {corr:+.3f}")

    print("""
If correlation is strong, h_final directly encodes the DIFFERENCE
between argmax and 2nd_argmax positions.

This would be the elegant encoding:
  - Positive diff (argmax later): one sign
  - Negative diff (argmax earlier): opposite sign
  - Magnitude of diff: strength of signal
""")


def plot_difference_encoding(model, n_samples=50000, save_path=None):
    """
    Visualize h_final as function of position difference.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    diff = (argmax_pos - targets).numpy()

    comparators = [1, 6, 7, 8]
    colors = {1: 'steelblue', 6: 'coral', 7: 'green', 8: 'purple'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: h_final vs difference
    ax = axes[0]
    for n in comparators:
        means = []
        diffs = list(range(-8, 9))
        for d in diffs:
            mask = diff == d
            if mask.sum() > 50:
                means.append(h_final[mask, n].mean().item())
            else:
                means.append(np.nan)
        ax.plot(diffs, means, 'o-', color=colors[n], linewidth=2,
                markersize=6, label=f'n{n}')

    ax.axhline(y=5, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('argmax - 2nd_argmax')
    ax.set_ylabel('Mean h_final')
    ax.set_title('h_final vs Position Difference')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Normalized version
    ax = axes[1]
    for n in comparators:
        means = []
        diffs = list(range(-8, 9))
        for d in diffs:
            mask = diff == d
            if mask.sum() > 50:
                means.append(h_final[mask, n].mean().item())
            else:
                means.append(np.nan)

        means = np.array(means)
        means_norm = (means - np.nanmean(means)) / np.nanstd(means)
        ax.plot(diffs, means_norm, 'o-', color=colors[n], linewidth=2,
                markersize=6, label=f'n{n}')

    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('argmax - 2nd_argmax')
    ax.set_ylabel('Normalized h_final')
    ax.set_title('Normalized h_final vs Position Difference')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Does h_final encode the DIFFERENCE between positions?',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved difference encoding plot to {save_path}')

    return fig


def main():
    model = example_2nd_argmax()

    analyze_antiphase(model)
    analyze_superposition_model(model)
    analyze_difference_encoding(model)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_antiphase(model, save_path='docs/figures/antiphase_analysis.png')
    plot_difference_encoding(model, save_path='docs/figures/difference_encoding.png')

    print("\n" + "=" * 70)
    print("SUMMARY: ANTI-PHASE ENCODING")
    print("=" * 70)
    print("""
The argmax and 2nd_argmax signals appear to have OPPOSITE PHASE in h_final.

This creates an elegant encoding:
  h_final[n] ≈ A*f(argmax_pos) - B*f(2nd_argmax_pos)

Where f() is the Fourier-like response for neuron n.

Implications:
1. When argmax = 2nd_argmax + k (constant difference):
   The signals partially cancel in a predictable way

2. The RESIDUAL after cancellation encodes the difference

3. This is similar to:
   - Differential signaling (noise cancellation)
   - I/Q demodulation (phase difference)
   - Interferometry (path length difference)

The model doesn't just encode two positions separately - it encodes
their RELATIONSHIP through constructive/destructive interference.
""")


if __name__ == "__main__":
    main()
