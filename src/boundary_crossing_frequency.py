"""
Test hypothesis: neurons that frequently cross ReLU boundaries during S_mag variation
are downweighted in W_out (since boundary crossings clip gradient).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch as th
from alg_zoo.architectures import DistRNN

def load_local_model():
    path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
    state_dict = th.load(path, weights_only=True, map_location='cpu')
    seq_len, hidden_size = state_dict['linear.weight'].shape
    model = DistRNN(hidden_size=hidden_size, seq_len=seq_len, bias=False)
    model.load_state_dict(state_dict)
    return model

model = load_local_model()
W_ih = model.rnn.weight_ih_l0.detach().numpy().squeeze()
W_hh = model.rnn.weight_hh_l0.detach().numpy()
W_out = model.linear.weight.detach().numpy()

COMPS = [1, 6, 7, 8]
WAVES = [0, 10, 11, 12, 14]
BRIDGES = [3, 5, 13, 15]
OTHERS = [2, 4, 9]

NEURON_COLORS = {n: 'red' for n in COMPS}
NEURON_COLORS.update({n: 'blue' for n in WAVES})
NEURON_COLORS.update({n: 'green' for n in BRIDGES})
NEURON_COLORS.update({n: 'gray' for n in OTHERS})

def run_stepwise(impulses):
    h = np.zeros(16)
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
    return h

def get_activation_pattern(h):
    """Return binary pattern of which neurons are active."""
    return tuple((h > 0).astype(int))

def count_boundary_crossings():
    """
    For each position pair, vary S_mag and count how often each neuron
    crosses a ReLU boundary.
    """
    M_val = 1.0
    S_mag_values = np.linspace(0.1, 0.95, 50)  # Vary S_mag

    # Track crossings per neuron
    crossing_counts = np.zeros(16)
    total_transitions = 0

    # Also track per-pair crossing frequency
    pair_crossings = []

    for m_pos in range(10):
        for s_pos in range(10):
            if m_pos == s_pos:
                continue

            # Forward direction: M first, then S
            prev_pattern = None
            pair_neuron_crossings = np.zeros(16)

            for S_val in S_mag_values:
                h_fwd = run_stepwise([(m_pos, M_val), (s_pos, S_val)])
                pattern = get_activation_pattern(h_fwd)

                if prev_pattern is not None:
                    # Count which neurons changed state
                    for n in range(16):
                        if pattern[n] != prev_pattern[n]:
                            crossing_counts[n] += 1
                            pair_neuron_crossings[n] += 1
                    total_transitions += 1

                prev_pattern = pattern

            # Reverse direction: S first, then M
            prev_pattern = None
            for S_val in S_mag_values:
                h_rev = run_stepwise([(m_pos, S_val), (s_pos, M_val)])
                pattern = get_activation_pattern(h_rev)

                if prev_pattern is not None:
                    for n in range(16):
                        if pattern[n] != prev_pattern[n]:
                            crossing_counts[n] += 1
                            pair_neuron_crossings[n] += 1
                    total_transitions += 1

                prev_pattern = pattern

            pair_crossings.append(pair_neuron_crossings)

    return crossing_counts, total_transitions, pair_crossings

def analyze_crossing_vs_output():
    print("=" * 70)
    print("BOUNDARY CROSSING FREQUENCY VS OUTPUT WEIGHT")
    print("=" * 70)

    crossing_counts, total_transitions, pair_crossings = count_boundary_crossings()

    # Normalize to get frequency
    crossing_freq = crossing_counts / total_transitions

    # Output weight norms
    w_out_norm = np.linalg.norm(W_out, axis=0)

    print(f"\nTotal S_mag transitions analyzed: {total_transitions}")
    print(f"(90 pairs × 2 directions × 49 transitions)")

    print("\n" + "-" * 50)
    print("NEURON CROSSING FREQUENCY VS OUTPUT WEIGHT")
    print("-" * 50)
    print(f"{'Neuron':<8} {'Crossings':>10} {'Freq':>10} {'||W_out||':>10} {'Category':<8}")

    for n in range(16):
        cat = 'comp' if n in COMPS else 'wave' if n in WAVES else 'bridge' if n in BRIDGES else 'other'
        print(f"n{n:<6} {crossing_counts[n]:>10.0f} {crossing_freq[n]:>10.3f} {w_out_norm[n]:>10.2f} {cat:<8}")

    # Correlation
    corr = np.corrcoef(crossing_freq, w_out_norm)[0, 1]
    print(f"\nCorrelation(crossing_freq, ||W_out||): {corr:.3f}")

    # Spearman
    from scipy.stats import spearmanr, pearsonr
    spear, spear_p = spearmanr(crossing_freq, w_out_norm)
    pears, pears_p = pearsonr(crossing_freq, w_out_norm)
    print(f"Spearman correlation: {spear:.3f} (p={spear_p:.4f})")
    print(f"Pearson correlation:  {pears:.3f} (p={pears_p:.4f})")

    # By category
    print("\n" + "-" * 50)
    print("BY CATEGORY")
    print("-" * 50)

    for cat, neurons in [('Comparators', COMPS), ('Waves', WAVES), ('Bridges', BRIDGES), ('Others', OTHERS)]:
        mean_cross = np.mean([crossing_freq[n] for n in neurons])
        mean_wout = np.mean([w_out_norm[n] for n in neurons])
        print(f"{cat:12s}: mean_crossing={mean_cross:.3f}, mean_||W_out||={mean_wout:.2f}")

    # Create visualization
    create_visualization(crossing_freq, w_out_norm, crossing_counts)

    return crossing_freq, w_out_norm

def create_visualization(crossing_freq, w_out_norm, crossing_counts):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    colors = [NEURON_COLORS[n] for n in range(16)]

    # 1. Scatter: crossing frequency vs output weight
    ax = axes[0, 0]
    ax.scatter(crossing_freq, w_out_norm, c=colors, s=200, edgecolors='black', alpha=0.8)
    for n in range(16):
        ax.annotate(f'n{n}', (crossing_freq[n], w_out_norm[n]), fontsize=9, ha='center', va='bottom')

    # Fit line
    z = np.polyfit(crossing_freq, w_out_norm, 1)
    p = np.poly1d(z)
    x_line = np.linspace(crossing_freq.min(), crossing_freq.max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.5)

    corr = np.corrcoef(crossing_freq, w_out_norm)[0, 1]
    ax.set_xlabel('Boundary crossing frequency', fontsize=12)
    ax.set_ylabel('||W_out[:, n]||', fontsize=12)
    ax.set_title(f'Crossing frequency vs output weight\nr = {corr:.3f}', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 2. Bar chart: crossing counts by neuron
    ax = axes[0, 1]
    ax.bar(range(16), crossing_counts, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Neuron', fontsize=12)
    ax.set_ylabel('Total boundary crossings', fontsize=12)
    ax.set_title('Boundary crossings during S_mag variation\n(across all 90 pairs × 2 directions)', fontsize=12)
    ax.set_xticks(range(16))
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Bar chart: output weight by neuron (sorted by crossing freq)
    ax = axes[1, 0]
    sorted_idx = np.argsort(-crossing_freq)
    ax.bar(range(16), w_out_norm[sorted_idx], color=[colors[i] for i in sorted_idx], edgecolor='black', alpha=0.8)
    ax.set_xlabel('Neuron (sorted by crossing frequency, high to low)', fontsize=12)
    ax.set_ylabel('||W_out[:, n]||', fontsize=12)
    ax.set_title('Output weights sorted by crossing frequency', fontsize=12)
    ax.set_xticks(range(16))
    ax.set_xticklabels([f'n{sorted_idx[i]}' for i in range(16)])
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Summary
    ax = axes[1, 1]
    ax.axis('off')

    corr = np.corrcoef(crossing_freq, w_out_norm)[0, 1]
    from scipy.stats import spearmanr
    spear, spear_p = spearmanr(crossing_freq, w_out_norm)

    # Category means
    comp_cross = np.mean([crossing_freq[n] for n in COMPS])
    comp_wout = np.mean([w_out_norm[n] for n in COMPS])
    wave_cross = np.mean([crossing_freq[n] for n in WAVES])
    wave_wout = np.mean([w_out_norm[n] for n in WAVES])
    bridge_cross = np.mean([crossing_freq[n] for n in BRIDGES])
    bridge_wout = np.mean([w_out_norm[n] for n in BRIDGES])

    summary = f"""HYPOTHESIS TEST

Do frequently-crossing neurons have lower W_out?

CORRELATIONS:
  Pearson r  = {corr:.3f}
  Spearman ρ = {spear:.3f} (p={spear_p:.4f})

CATEGORY ANALYSIS:
                  Crossing    ||W_out||
  Comparators:    {comp_cross:.3f}       {comp_wout:.1f}
  Waves:          {wave_cross:.3f}       {wave_wout:.1f}
  Bridges:        {bridge_cross:.3f}       {bridge_wout:.1f}

INTERPRETATION:

{"CONFIRMED" if corr < -0.3 else "WEAK/NO SUPPORT" if corr < 0 else "OPPOSITE"}:
Correlation is {corr:.3f}

{"Neurons that cross boundaries more often ARE" if corr < -0.3 else "Neurons that cross boundaries are NOT clearly"}
downweighted in W_out.

Legend: Red=comp, Blue=wave, Green=bridge, Gray=other
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            ha='left', va='top', fontsize=11, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig('docs/boundary_crossing_vs_wout.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved to docs/boundary_crossing_vs_wout.png")

if __name__ == "__main__":
    analyze_crossing_vs_output()
