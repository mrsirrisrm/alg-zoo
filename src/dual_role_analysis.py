"""
Dual Role Analysis: Comparators as Both Detectors AND Position Encoders

Investigates how neurons 1, 6, 7, 8 serve two purposes:
1. Clipping comparators: detect when x[t] > threshold (≈ running max)
2. Position encoders: their final values encode the 2nd argmax position

Key insight: The VALUE at t=9 depends on WHEN the neuron last clipped,
creating a "time since last clip" signal that encodes position.
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


def analyze_clip_timing(model, n_samples=50000):
    """
    Analyze when each comparator clips and how it relates to argmax/2nd_argmax.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    clipped, _ = get_full_trajectory(model, x)

    comparators = [1, 6, 7, 8]

    print("=" * 70)
    print("CLIP TIMING ANALYSIS")
    print("=" * 70)

    # For each comparator, find its last clip position
    for n in comparators:
        last_clip_pos = th.zeros(n_samples, dtype=th.long) - 1  # -1 if never clips

        for t in range(10):
            mask = clipped[:, n, t]
            last_clip_pos[mask] = t

        # How often does last clip = argmax?
        match_argmax = (last_clip_pos == argmax_pos).float().mean().item()

        # How often does second-to-last clip = 2nd_argmax?
        second_last_clip = th.zeros(n_samples, dtype=th.long) - 1
        for t in range(10):
            mask = clipped[:, n, t] & (t < last_clip_pos)
            second_last_clip[mask] = t

        match_2nd = (second_last_clip == targets).float().mean().item()

        print(f"\nNeuron {n}:")
        print(f"  Last clip = argmax position:       {match_argmax:.1%}")
        print(f"  2nd-to-last clip = 2nd_argmax:     {match_2nd:.1%}")


def analyze_value_by_position(model, n_samples=50000):
    """
    Show how h_final differs by 2nd_argmax position when max is fixed.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    _, hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 70)
    print("H_FINAL BY 2ND_ARGMAX POSITION (when max at position 5)")
    print("=" * 70)

    # Fix max position = 5
    max_mask = argmax_pos == 5

    print("\n2nd_argmax |   h1    |   h6    |   h7    |   h8    |")
    print("-" * 55)

    for second_pos in range(10):
        if second_pos == 5:
            continue  # Can't be same as max

        mask = max_mask & (targets == second_pos)
        if mask.sum() < 50:
            continue

        vals = []
        for n in comparators:
            val = h_final[mask, n].mean().item()
            vals.append(f"{val:7.2f}")

        print(f"    {second_pos}      | {'|'.join(vals)} |")


def analyze_recency_encoding(model, n_samples=50000):
    """
    Show how h_final encodes "steps since last clip" for each comparator.
    """
    x = th.rand(n_samples, 10)

    clipped, hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]
    W_hh = model.rnn.weight_hh_l0.data

    print("\n" + "=" * 70)
    print("RECENCY ENCODING: H_FINAL vs STEPS SINCE LAST CLIP")
    print("=" * 70)

    for n in comparators:
        # Find last clip position for each sample
        last_clip = th.zeros(n_samples, dtype=th.long) - 1
        for t in range(10):
            mask = clipped[:, n, t]
            last_clip[mask] = t

        # Steps since last clip (at t=9)
        steps_since = 9 - last_clip
        steps_since[last_clip == -1] = 10  # Never clipped

        # Correlation
        valid = last_clip >= 0
        corr = np.corrcoef(h_final[valid, n].numpy(),
                          steps_since[valid].numpy())[0, 1]

        # Mean h_final by steps_since
        print(f"\nNeuron {n} (self-recurrence = {W_hh[n, n].item():.2f}):")
        print(f"  Correlation h_final vs steps_since: {corr:.3f}")
        print(f"  Steps since clip | Mean h_final")

        for s in range(1, 10):
            mask = steps_since == s
            if mask.sum() > 100:
                mean_h = h_final[mask, n].mean().item()
                print(f"       {s}           |   {mean_h:.2f}")


def plot_dual_role_visualization(model, save_path=None):
    """
    Create a visualization showing both roles of comparator neurons.
    """
    n_samples = 30000
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    clipped, hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Clip probability by input value (Role 1: Comparator)
    ax = axes[0, 0]
    t = 5
    x_val = x[:, t].numpy()

    for n in comparators:
        clip_probs = []
        x_bins = np.linspace(0, 1, 11)
        x_centers = (x_bins[:-1] + x_bins[1:]) / 2

        for low, high in zip(x_bins[:-1], x_bins[1:]):
            mask = (x_val >= low) & (x_val < high)
            if mask.sum() > 50:
                clip_probs.append(clipped[mask, n, t].float().mean().item())
            else:
                clip_probs.append(np.nan)

        ax.plot(x_centers, clip_probs, 'o-', label=f'n{n}', linewidth=2)

    ax.set_xlabel('Input value x[t=5]')
    ax.set_ylabel('Clip probability')
    ax.set_title('Role 1: Comparator\n(clips when x > threshold)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: h_final by steps since clip (Role 2: Recency encoder)
    ax = axes[0, 1]

    for n in comparators:
        last_clip = th.zeros(n_samples, dtype=th.long) - 1
        for t in range(10):
            mask = clipped[:, n, t]
            last_clip[mask] = t

        steps_since = 9 - last_clip

        mean_h = []
        steps = list(range(1, 9))
        for s in steps:
            mask = steps_since == s
            if mask.sum() > 100:
                mean_h.append(h_final[mask, n].mean().item())
            else:
                mean_h.append(np.nan)

        ax.plot(steps, mean_h, 'o-', label=f'n{n}', linewidth=2)

    ax.set_xlabel('Steps since last clip (at t=9)')
    ax.set_ylabel('Mean h_final')
    ax.set_title('Role 2: Recency Encoder\n(h_final depends on when last clipped)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: h_final by 2nd_argmax position (fixed max=5)
    ax = axes[1, 0]

    max_mask = argmax_pos == 5

    for n in comparators:
        h_by_pos = []
        positions = []
        for second_pos in range(10):
            if second_pos == 5:
                continue
            mask = max_mask & (targets == second_pos)
            if mask.sum() > 50:
                positions.append(second_pos)
                h_by_pos.append(h_final[mask, n].mean().item())

        ax.plot(positions, h_by_pos, 'o-', label=f'n{n}', linewidth=2)

    ax.set_xlabel('2nd argmax position')
    ax.set_ylabel('Mean h_final (when max=5)')
    ax.set_title('h_final varies by 2nd argmax\n(small differences → output via W_out)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Output logit by 2nd_argmax (showing the decoding)
    ax = axes[1, 1]
    W_out = model.linear.weight.data

    for second_pos in range(10):
        if second_pos == 5:
            continue
        mask = max_mask & (targets == second_pos)
        if mask.sum() < 50:
            continue

        # Compute mean contribution from each comparator
        contrib = []
        for n in comparators:
            c = (h_final[mask, n].mean() * W_out[second_pos, n]).item()
            contrib.append(c)

        ax.bar(np.arange(4) + (second_pos - 4.5) * 0.08, contrib,
               width=0.08, label=f'pos {second_pos}', alpha=0.7)

    ax.set_xticks(range(4))
    ax.set_xticklabels([f'n{n}' for n in comparators])
    ax.set_ylabel('Contribution to correct logit')
    ax.set_title('Comparator contributions by position\n(h_final × W_out)')
    ax.legend(ncol=3, fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Dual Role of Comparator Neurons (n1, n6, n7, n8)', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved dual role visualization to {save_path}')

    return fig


def analyze_mechanism(model):
    """
    Explain the full mechanism: how clipping creates position encoding.
    """
    W_hh = model.rnn.weight_hh_l0.data

    print("\n" + "=" * 70)
    print("THE DUAL-ROLE MECHANISM")
    print("=" * 70)

    print("""
When a comparator neuron (n1, n6, n7, n8) clips:
  1. Its hidden state h[n] → 0
  2. On subsequent steps, it rebuilds via: h[n,t] = ReLU(W_hh[n,:] @ h[t-1] + W_ih[n] * x[t])

The REBUILD RATE depends on:
  - Self-recurrence W_hh[n,n]: higher = slower rebuild
  - Other recurrent connections: feed from other neurons
  - Input W_ih[n]: but this is negative, so small x helps rebuild

KEY INSIGHT: The VALUE of h[n] at t=9 encodes TIME SINCE LAST CLIP.
  - Clipped at t=8 → 1 step to rebuild → small h_final
  - Clipped at t=5 → 4 steps to rebuild → larger h_final
  - Clipped at t=2 → 7 steps to rebuild → even larger h_final

This creates the RECENCY ENCODING that distinguishes positions.
""")

    print("Self-recurrence values:")
    for n in [1, 6, 7, 8]:
        print(f"  n{n}: W_hh[{n},{n}] = {W_hh[n,n].item():+.2f}")

    print("""
Since comparators clip when x > running_max:
  - They clip at the ARGMAX position (by definition)
  - They ALSO clip at any position where x was a new max
  - The 2nd argmax was the previous max → comparators clipped there

The PATTERN of h_final values across comparators uniquely identifies
the 2nd argmax position through the W_out weights (position signatures).
""")


def main():
    model = example_2nd_argmax()

    analyze_clip_timing(model)
    analyze_value_by_position(model)
    analyze_recency_encoding(model)
    analyze_mechanism(model)  # No samples needed - just prints explanation

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)
    plot_dual_role_visualization(model, 'docs/figures/dual_role_comparators.png')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Comparator neurons serve TWO simultaneous roles:

ROLE 1: COMPARATORS (detecting large values)
  - Clip (→0) when x[t] > adaptive threshold
  - Threshold ≈ running maximum (tracked by n2)
  - Creates binary "is this a new max?" signal

ROLE 2: POSITION ENCODERS (via recency)
  - After clipping, neurons rebuild over subsequent timesteps
  - Value at t=9 encodes "steps since last clip"
  - Different neurons rebuild at different rates
  - This creates unique signatures for each position

The magic: ONE mechanism (clipping + rebuild) serves BOTH purposes!
The model doesn't need separate position counters - position emerges
from the dynamics of the comparator neurons themselves.
""")


if __name__ == "__main__":
    main()
