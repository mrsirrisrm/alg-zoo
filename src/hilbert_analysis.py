"""
Hilbert Transform / Envelope-Phase Analysis

Hypothesis: The 4 comparators form a system analogous to I/Q demodulation:
- ENVELOPE (magnitude): Total energy across comparators = "how big was the max"
- PHASE (relative differences): Encodes timing information

This is similar to:
- Hilbert transform instantaneous amplitude/phase
- I/Q demodulation in radio
- Complex-valued signal processing
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


def compute_envelope_phase(h_final, comparators=[1, 6, 7, 8]):
    """
    Compute envelope (magnitude) and phase-like quantities from comparator values.
    """
    h_comp = h_final[:, comparators]

    # Envelope: total magnitude
    envelope = th.sqrt((h_comp ** 2).sum(dim=1))

    # Alternative: L1 norm
    envelope_l1 = h_comp.abs().sum(dim=1)

    # Mean activation
    mean_activation = h_comp.mean(dim=1)

    # "Phase" - relative differences
    # Treat as 2D: (n1+n7) vs (n6+n8) or similar pairings
    # Or compute ratios

    # Pairwise ratios (phase-like)
    ratio_17 = h_final[:, 1] / (h_final[:, 7] + 1e-6)
    ratio_68 = h_final[:, 6] / (h_final[:, 8] + 1e-6)

    # Differences (also phase-like)
    diff_17 = h_final[:, 1] - h_final[:, 7]
    diff_68 = h_final[:, 6] - h_final[:, 8]

    return {
        'envelope': envelope,
        'envelope_l1': envelope_l1,
        'mean': mean_activation,
        'ratio_17': ratio_17,
        'ratio_68': ratio_68,
        'diff_17': diff_17,
        'diff_68': diff_68,
    }


def analyze_envelope_phase(model, n_samples=50000):
    """
    Analyze what envelope vs phase encode.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    top2 = th.topk(x, 2, dim=-1)
    max_val = top2.values[:, 0]
    second_val = top2.values[:, 1]
    gap = max_val - second_val

    _, hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    ep = compute_envelope_phase(h_final)

    print("=" * 70)
    print("ENVELOPE vs PHASE: WHAT DO THEY ENCODE?")
    print("=" * 70)

    print("\nCorrelations with various quantities:")
    print("-" * 60)
    print(f"{'Measure':<15} | {'max_val':<10} | {'gap':<10} | {'argmax':<10} | {'2nd_argmax':<10}")
    print("-" * 60)

    for name, values in ep.items():
        v = values.numpy()
        corr_max = np.corrcoef(v, max_val.numpy())[0, 1]
        corr_gap = np.corrcoef(v, gap.numpy())[0, 1]
        corr_argmax = np.corrcoef(v, argmax_pos.numpy())[0, 1]
        corr_2nd = np.corrcoef(v, targets.numpy())[0, 1]

        print(f"{name:<15} | {corr_max:>+8.3f}  | {corr_gap:>+8.3f}  | {corr_argmax:>+8.3f}  | {corr_2nd:>+8.3f}")

    return ep


def analyze_threshold_cascade(model):
    """
    Analyze how the different W_ih values create a threshold cascade.
    """
    W_ih = model.rnn.weight_ih_l0.data.squeeze()

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 70)
    print("THRESHOLD CASCADE (W_ih values)")
    print("=" * 70)

    print("\nComparator W_ih values (determine clipping threshold):")
    print("-" * 40)

    for n in sorted(comparators, key=lambda x: W_ih[x].item(), reverse=True):
        w = W_ih[n].item()
        # Threshold is approximately -h_prev / W_ih (simplified)
        print(f"  n{n}: W_ih = {w:+.2f}")

    print("""
Interpretation:
  - n1 (W_ih = -10.6): Lowest threshold, clips first
  - n6 (W_ih = -11.0): Slightly higher threshold
  - n8 (W_ih = -12.3): Higher threshold
  - n7 (W_ih = -13.2): Highest threshold, clips last

When a value arrives that's between thresholds:
  - Some comparators clip, others don't
  - The PATTERN of which clip encodes the VALUE's magnitude
  - This is amplitude encoding via partial clipping
""")


def analyze_partial_clipping(model, n_samples=100000):
    """
    Analyze the information in partial clipping patterns.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    top2 = th.topk(x, 2, dim=-1)
    second_val = top2.values[:, 1]

    clipped, _ = get_full_trajectory(model, x)

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 70)
    print("PARTIAL CLIPPING PATTERNS")
    print("=" * 70)

    # Count how many comparators clip at 2nd_argmax position
    n_clip_at_2nd = th.zeros(n_samples)
    for n in comparators:
        for i in range(n_samples):
            n_clip_at_2nd[i] += clipped[i, n, targets[i]].float()

    print("\nDistribution of # comparators clipping at 2nd_argmax:")
    for k in range(5):
        mask = n_clip_at_2nd == k
        if mask.sum() > 0:
            mean_val = second_val[mask].mean().item()
            print(f"  {k} comparators clip: {mask.sum().item():>6} samples, mean 2nd_val = {mean_val:.3f}")

    print("\n2nd_max value by clipping pattern:")
    print("-" * 50)

    # Analyze specific patterns
    patterns = {}
    for i in range(n_samples):
        pattern = tuple(clipped[i, n, targets[i]].item() for n in comparators)
        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append(second_val[i].item())

    print(f"{'Pattern (n1,n6,n7,n8)':<25} | {'Count':<8} | {'Mean 2nd_val':<12}")
    print("-" * 50)

    for pattern in sorted(patterns.keys(), key=lambda p: len(patterns[p]), reverse=True)[:10]:
        vals = patterns[pattern]
        pattern_str = ''.join(['1' if p else '0' for p in pattern])
        print(f"  {pattern_str:<23} | {len(vals):<8} | {np.mean(vals):.3f}")


def plot_envelope_phase(model, n_samples=50000, save_path=None):
    """
    Visualize envelope vs phase decomposition.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    top2 = th.topk(x, 2, dim=-1)
    max_val = top2.values[:, 0]

    _, hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]
    h_comp = h_final[:, comparators]

    # Compute envelope and "phase"
    envelope = th.sqrt((h_comp ** 2).sum(dim=1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Envelope vs max_val
    ax = axes[0, 0]
    ax.scatter(max_val.numpy(), envelope.numpy(), alpha=0.1, s=1)
    ax.set_xlabel('max(x)')
    ax.set_ylabel('Envelope (L2 norm of comparators)')
    ax.set_title(f'Envelope encodes max value\nr = {np.corrcoef(max_val.numpy(), envelope.numpy())[0,1]:.3f}')
    ax.grid(True, alpha=0.3)

    # Panel 2: Envelope by argmax position
    ax = axes[0, 1]
    env_by_argmax = []
    for i in range(10):
        mask = argmax_pos == i
        env_by_argmax.append(envelope[mask].mean().item())

    ax.bar(range(10), env_by_argmax, color='steelblue', alpha=0.7)
    ax.set_xlabel('Argmax position')
    ax.set_ylabel('Mean envelope')
    ax.set_title('Envelope by argmax position')
    ax.grid(True, alpha=0.3)

    # Panel 3: "Phase" (h1-h7 vs h6-h8) colored by 2nd_argmax
    ax = axes[1, 0]
    diff_17 = (h_final[:, 1] - h_final[:, 7]).numpy()
    diff_68 = (h_final[:, 6] - h_final[:, 8]).numpy()

    # Subsample for visibility
    idx = np.random.choice(n_samples, 5000, replace=False)

    scatter = ax.scatter(diff_17[idx], diff_68[idx], c=targets.numpy()[idx],
                        cmap='tab10', alpha=0.5, s=10)
    ax.set_xlabel('h1 - h7 (fast vs medium-fast)')
    ax.set_ylabel('h6 - h8 (medium vs slow)')
    ax.set_title('"Phase" space colored by 2nd_argmax')
    plt.colorbar(scatter, ax=ax, label='2nd_argmax position')
    ax.grid(True, alpha=0.3)

    # Panel 4: Same but colored by argmax
    ax = axes[1, 1]
    scatter = ax.scatter(diff_17[idx], diff_68[idx], c=argmax_pos.numpy()[idx],
                        cmap='tab10', alpha=0.5, s=10)
    ax.set_xlabel('h1 - h7')
    ax.set_ylabel('h6 - h8')
    ax.set_title('"Phase" space colored by ARGMAX')
    plt.colorbar(scatter, ax=ax, label='argmax position')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Envelope-Phase Decomposition\n'
                 'Envelope ≈ max value, Phase ≈ timing information',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved envelope-phase plot to {save_path}')

    return fig


def analyze_absence_as_signal(model, n_samples=100000):
    """
    Analyze how the ABSENCE of a second clipping event is informative.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    clipped, _ = get_full_trajectory(model, x)

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 70)
    print("ABSENCE AS SIGNAL")
    print("=" * 70)

    # For late argmax, what are the possible 2nd_argmax positions?
    print("\nWhen argmax is at position 8:")
    mask_argmax8 = argmax_pos == 8

    # Count clips at each position for n7
    print("\nn7 clipping pattern when argmax=8:")
    for pos in range(10):
        if pos == 8:
            continue
        mask = mask_argmax8 & (targets == pos)
        if mask.sum() > 30:
            # Does n7 clip at the 2nd_argmax position?
            clips = th.zeros(mask.sum())
            idx = 0
            for i in range(n_samples):
                if mask[i]:
                    clips[idx] = clipped[i, 7, pos].float()
                    idx += 1
            clip_rate = clips.mean().item()
            print(f"  2nd_argmax={pos}: n7 clips {clip_rate:.1%} of time, n={mask.sum().item()}")

    print("""
Key insight: When argmax=8, the only position AFTER is 9.
If we see n7 clip only once (at argmax), 2nd_argmax must be at 0-7.
If we see n7 NOT clip at positions 0-7, 2nd_argmax must be at 9.

The ABSENCE of clipping at earlier positions constrains possibilities!
""")

    # Analyze "number of clips" as information
    print("\nInformation in clip count:")
    print("-" * 50)

    # For each argmax position, how does # of n7 clips relate to 2nd_argmax?
    for argmax in [3, 5, 7]:
        mask = argmax_pos == argmax
        print(f"\nWhen argmax={argmax}:")

        # Count total n7 clips in sequence
        n7_total_clips = clipped[:, 7, :].sum(dim=1)

        for n_clips in range(1, 5):
            clip_mask = mask & (n7_total_clips == n_clips)
            if clip_mask.sum() > 30:
                mean_2nd = targets[clip_mask].float().mean().item()
                std_2nd = targets[clip_mask].float().std().item()
                print(f"  {n_clips} total n7 clips: mean 2nd_argmax = {mean_2nd:.1f} ± {std_2nd:.1f}")


def main():
    model = example_2nd_argmax()

    analyze_envelope_phase(model)
    analyze_threshold_cascade(model)
    analyze_partial_clipping(model)
    analyze_absence_as_signal(model)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    plot_envelope_phase(model, save_path='docs/figures/envelope_phase.png')

    print("\n" + "=" * 70)
    print("HILBERT-LIKE INTERPRETATION")
    print("=" * 70)
    print("""
The 4 comparators form an envelope-phase system:

ENVELOPE (total magnitude):
  - Encodes max_val (r ≈ 0.4-0.5)
  - Higher when max is large
  - Provides "normalization" context

PHASE (relative differences):
  - Encodes timing (argmax position primarily)
  - Also encodes 2nd_argmax via residual structure
  - The pattern of WHICH comparators clip and by HOW MUCH

PARTIAL CLIPPING:
  - W_ih values create threshold cascade: n1 < n6 < n8 < n7
  - Values near threshold cause partial clipping
  - Pattern encodes amplitude information

ABSENCE AS SIGNAL:
  - Missing second impulse is informative
  - Constrains possibilities based on argmax position
  - Late argmax + single impulse → 2nd must be at position 9

This is a form of AMPLITUDE-PHASE MODULATION:
  - 4 carriers at different frequencies
  - Each modulated by clipping events
  - Demodulation via W_out matched filters
""")


if __name__ == "__main__":
    main()
