"""
Analyze how tropical dominance relates to actual RNN dynamics.

Questions:
1. Do tropically dominant neurons tend to be active at t=10?
2. How does the activation pattern (tropical cell) relate to tropical eigenvector?
3. Can we predict which neurons will be active based on tropical structure?
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch as th
from alg_zoo.architectures import DistRNN
from tropical_eigenvector_analysis import compute_tropical_eigenvalue, NEURON_COLORS, NEURON_CATEGORY


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


def run_stepwise(impulses):
    h = np.zeros(16)
    states = [h.copy()]
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        states.append(h.copy())
    return states


def analyze_tropical_dominance_vs_activation():
    """Analyze relationship between tropical eigenvector and final activations."""

    print("=" * 70)
    print("TROPICAL DOMINANCE VS ACTUAL ACTIVATIONS")
    print("=" * 70)

    # Get tropical eigenvector
    eigenvalue, eigenvector, _ = compute_tropical_eigenvalue(W_hh)

    # Rank neurons by tropical dominance (higher = more dominant)
    trop_rank = np.argsort(-eigenvector)  # Descending order

    # Collect activation statistics across all clean pairs
    all_final_states = []
    active_counts = np.zeros(16)
    n_cases = 0

    M_val = 1.0
    S_val = 0.8

    for m_pos in range(10):
        for s_pos in range(10):
            if m_pos == s_pos:
                continue

            # Forward
            states_fwd = run_stepwise([(m_pos, M_val), (s_pos, S_val)])
            final_fwd = states_fwd[-1]
            all_final_states.append(final_fwd)
            active_counts += (final_fwd > 0).astype(int)
            n_cases += 1

            # Reverse
            states_rev = run_stepwise([(m_pos, S_val), (s_pos, M_val)])
            final_rev = states_rev[-1]
            all_final_states.append(final_rev)
            active_counts += (final_rev > 0).astype(int)
            n_cases += 1

    active_freq = active_counts / n_cases
    mean_activation = np.mean(all_final_states, axis=0)

    print(f"\nAnalyzed {n_cases} cases (90 pairs × 2 orderings)")

    # Compare tropical rank to activation frequency
    print("\n" + "-" * 50)
    print("NEURONS RANKED BY TROPICAL EIGENVECTOR")
    print("-" * 50)
    print(f"{'Rank':<5} {'Neuron':<8} {'Category':<8} {'Trop_v':>10} {'Active%':>10} {'Mean_act':>10}")

    for rank, n in enumerate(trop_rank):
        print(f"{rank+1:<5} n{n:<6} {NEURON_CATEGORY[n]:<8} {eigenvector[n]:>10.3f} {active_freq[n]*100:>9.1f}% {mean_activation[n]:>10.2f}")

    # Correlation analysis
    print("\n" + "-" * 50)
    print("CORRELATION ANALYSIS")
    print("-" * 50)

    corr_freq = np.corrcoef(eigenvector, active_freq)[0, 1]
    corr_mean = np.corrcoef(eigenvector, mean_activation)[0, 1]

    print(f"Corr(tropical_v, active_frequency): {corr_freq:.3f}")
    print(f"Corr(tropical_v, mean_activation):  {corr_mean:.3f}")

    # Spearman rank correlation
    from scipy.stats import spearmanr
    spear_freq, _ = spearmanr(eigenvector, active_freq)
    spear_mean, _ = spearmanr(eigenvector, mean_activation)
    print(f"Spearman(tropical_v, active_frequency): {spear_freq:.3f}")
    print(f"Spearman(tropical_v, mean_activation):  {spear_mean:.3f}")

    # Analyze which neurons switch on/off
    print("\n" + "-" * 50)
    print("ALWAYS/SOMETIMES/NEVER ACTIVE")
    print("-" * 50)

    always_active = [n for n in range(16) if active_freq[n] == 1.0]
    never_active = [n for n in range(16) if active_freq[n] == 0.0]
    sometimes = [n for n in range(16) if 0 < active_freq[n] < 1.0]

    print(f"Always active (100%): {always_active}")
    print(f"  Categories: {[NEURON_CATEGORY[n] for n in always_active]}")
    print(f"  Trop_v: {[f'{eigenvector[n]:.3f}' for n in always_active]}")

    print(f"\nNever active (0%): {never_active}")
    print(f"  Categories: {[NEURON_CATEGORY[n] for n in never_active]}")
    print(f"  Trop_v: {[f'{eigenvector[n]:.3f}' for n in never_active]}")

    print(f"\nSometimes active: {sometimes}")
    for n in sometimes:
        print(f"  n{n} ({NEURON_CATEGORY[n]}): {active_freq[n]*100:.1f}%, trop_v={eigenvector[n]:.3f}")

    # W_out analysis: which neurons matter for readout?
    print("\n" + "-" * 50)
    print("OUTPUT INFLUENCE VS TROPICAL DOMINANCE")
    print("-" * 50)

    w_out_norm = np.linalg.norm(W_out, axis=0)
    w_out_rank = np.argsort(-w_out_norm)

    print(f"{'Neuron':<8} {'Trop_v':>10} {'||W_out||':>10} {'Category':<8}")
    for n in w_out_rank:
        print(f"n{n:<6} {eigenvector[n]:>10.3f} {w_out_norm[n]:>10.2f}   {NEURON_CATEGORY[n]:<8}")

    corr_out = np.corrcoef(eigenvector, w_out_norm)[0, 1]
    print(f"\nCorr(tropical_v, ||W_out||): {corr_out:.3f}")

    # Create visualization
    create_dominance_visualization(eigenvector, active_freq, mean_activation, w_out_norm)

    return eigenvector, active_freq, mean_activation


def create_dominance_visualization(eigenvector, active_freq, mean_activation, w_out_norm):
    """Visualize relationship between tropical dominance and activation."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    colors = [NEURON_COLORS[n] for n in range(16)]

    # 1. Tropical eigenvector vs activation frequency
    ax = axes[0, 0]
    ax.scatter(eigenvector, active_freq * 100, c=colors, s=200, edgecolors='black', alpha=0.8)
    for n in range(16):
        ax.annotate(f'n{n}', (eigenvector[n], active_freq[n] * 100),
                   fontsize=9, ha='center', va='bottom')
    ax.set_xlabel('Tropical eigenvector v', fontsize=12)
    ax.set_ylabel('Active at t=10 (%)', fontsize=12)
    ax.set_title('Tropical dominance vs activation frequency\n(over all 180 fwd/rev cases)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Fit line
    z = np.polyfit(eigenvector, active_freq * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(eigenvector.min(), eigenvector.max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.5, label=f'Linear fit')
    ax.legend()

    # 2. Tropical eigenvector vs mean activation
    ax = axes[0, 1]
    ax.scatter(eigenvector, mean_activation, c=colors, s=200, edgecolors='black', alpha=0.8)
    for n in range(16):
        ax.annotate(f'n{n}', (eigenvector[n], mean_activation[n]),
                   fontsize=9, ha='center', va='bottom')
    ax.set_xlabel('Tropical eigenvector v', fontsize=12)
    ax.set_ylabel('Mean activation at t=10', fontsize=12)
    ax.set_title('Tropical dominance vs mean activation', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 3. Output weight norm vs tropical eigenvector
    ax = axes[1, 0]
    ax.scatter(eigenvector, w_out_norm, c=colors, s=200, edgecolors='black', alpha=0.8)
    for n in range(16):
        ax.annotate(f'n{n}', (eigenvector[n], w_out_norm[n]),
                   fontsize=9, ha='center', va='bottom')
    ax.set_xlabel('Tropical eigenvector v', fontsize=12)
    ax.set_ylabel('||W_out[:, n]||', fontsize=12)
    ax.set_title('Tropical dominance vs output influence', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 4. Summary bar chart
    ax = axes[1, 1]

    # Sort by tropical eigenvector
    sorted_idx = np.argsort(-eigenvector)

    x = np.arange(16)
    width = 0.25

    # Normalize for comparison
    norm_trop = (eigenvector - eigenvector.min()) / (eigenvector.max() - eigenvector.min())
    norm_freq = active_freq
    norm_out = (w_out_norm - w_out_norm.min()) / (w_out_norm.max() - w_out_norm.min())

    ax.bar(x - width, norm_trop[sorted_idx], width, label='Tropical v (normalized)', color='purple', alpha=0.7)
    ax.bar(x, norm_freq[sorted_idx], width, label='Active frequency', color='green', alpha=0.7)
    ax.bar(x + width, norm_out[sorted_idx], width, label='Output influence', color='orange', alpha=0.7)

    ax.set_xlabel('Neuron (sorted by tropical dominance)', fontsize=12)
    ax.set_ylabel('Normalized value', fontsize=12)
    ax.set_title('Comparison: Tropical dominance vs frequency vs output', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'n{sorted_idx[i]}' for i in range(16)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('docs/tropical_dominance_vs_activation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved to docs/tropical_dominance_vs_activation.png")


if __name__ == "__main__":
    analyze_tropical_dominance_vs_activation()
