"""
Analyze the PCA spectrum of hidden states.
How much variance is captured by PC1,2? What's in the higher components?
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch as th
from sklearn.decomposition import PCA
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

def run_stepwise(impulses):
    h = np.zeros(16)
    states = [h.copy()]
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        states.append(h.copy())
    return states

def collect_hidden_states():
    """Collect all hidden states for PCA."""
    all_states = []
    final_states = []

    M_val = 1.0
    S_val = 0.8

    for m_pos in range(10):
        for s_pos in range(10):
            if m_pos == s_pos:
                continue

            # Forward
            states_fwd = run_stepwise([(m_pos, M_val), (s_pos, S_val)])
            all_states.extend(states_fwd[1:])  # Skip t=0 (zeros)
            final_states.append(states_fwd[-1])

            # Reverse
            states_rev = run_stepwise([(m_pos, S_val), (s_pos, M_val)])
            all_states.extend(states_rev[1:])
            final_states.append(states_rev[-1])

    return np.array(all_states), np.array(final_states)

def analyze_pca_spectrum():
    print("=" * 70)
    print("PCA SPECTRUM ANALYSIS")
    print("=" * 70)

    all_states, final_states = collect_hidden_states()

    print(f"\nCollected {len(all_states)} hidden states (all timesteps)")
    print(f"Collected {len(final_states)} final states (t=10 only)")

    # PCA on all states
    pca_all = PCA()
    pca_all.fit(all_states)

    # PCA on final states only
    pca_final = PCA()
    pca_final.fit(final_states)

    print("\n" + "-" * 50)
    print("EXPLAINED VARIANCE RATIO (ALL TIMESTEPS)")
    print("-" * 50)

    cumsum = np.cumsum(pca_all.explained_variance_ratio_)
    for i in range(16):
        print(f"  PC{i+1:2d}: {pca_all.explained_variance_ratio_[i]*100:6.2f}%  (cumulative: {cumsum[i]*100:6.2f}%)")

    print("\n" + "-" * 50)
    print("EXPLAINED VARIANCE RATIO (FINAL STATES ONLY)")
    print("-" * 50)

    cumsum_final = np.cumsum(pca_final.explained_variance_ratio_)
    for i in range(16):
        print(f"  PC{i+1:2d}: {pca_final.explained_variance_ratio_[i]*100:6.2f}%  (cumulative: {cumsum_final[i]*100:6.2f}%)")

    # How many PCs needed for 90%, 95%, 99%?
    for threshold in [0.9, 0.95, 0.99]:
        n_all = np.searchsorted(cumsum, threshold) + 1
        n_final = np.searchsorted(cumsum_final, threshold) + 1
        print(f"\nPCs needed for {threshold*100:.0f}% variance: all={n_all}, final={n_final}")

    # Analyze what each PC captures
    print("\n" + "-" * 50)
    print("PC LOADINGS (FINAL STATES) - TOP NEURONS PER PC")
    print("-" * 50)

    for pc_idx in range(6):
        loadings = pca_final.components_[pc_idx]
        sorted_neurons = np.argsort(-np.abs(loadings))
        top_3 = sorted_neurons[:3]

        print(f"\n  PC{pc_idx+1} ({pca_final.explained_variance_ratio_[pc_idx]*100:.1f}%):")
        for n in top_3:
            cat = 'comp' if n in COMPS else 'wave' if n in WAVES else 'bridge' if n in BRIDGES else 'other'
            sign = '+' if loadings[n] > 0 else '-'
            print(f"    n{n} ({cat}): {sign}{abs(loadings[n]):.3f}")

    # Check if offset lives primarily in PC1,2
    print("\n" + "-" * 50)
    print("WHERE DOES THE OFFSET LIVE?")
    print("-" * 50)

    offsets = []
    for m_pos in range(10):
        for s_pos in range(10):
            if m_pos == s_pos:
                continue

            M_val, S_val = 1.0, 0.8
            h_fwd = run_stepwise([(m_pos, M_val), (s_pos, S_val)])[-1]
            h_rev = run_stepwise([(m_pos, S_val), (s_pos, M_val)])[-1]
            offsets.append(h_fwd - h_rev)

    offsets = np.array(offsets)

    # Project offsets onto each PC
    offset_projected = pca_final.transform(offsets + pca_final.mean_) - pca_final.transform(np.zeros((1, 16)) + pca_final.mean_)
    # Actually, for offsets we want: pca.transform(offset) but offset is a difference
    # Better: project offset directly onto PC components
    offset_pc_variance = []
    for pc_idx in range(16):
        pc_component = pca_final.components_[pc_idx]
        projections = offsets @ pc_component
        variance = np.var(projections)
        offset_pc_variance.append(variance)

    total_offset_var = np.sum(offset_pc_variance)
    offset_pc_ratio = np.array(offset_pc_variance) / total_offset_var

    print("\nOffset variance by PC:")
    cumsum_offset = np.cumsum(offset_pc_ratio)
    for i in range(10):
        print(f"  PC{i+1:2d}: {offset_pc_ratio[i]*100:6.2f}%  (cumulative: {cumsum_offset[i]*100:6.2f}%)")

    # Compare hidden state PCA vs offset PCA
    print("\n" + "-" * 50)
    print("COMPARISON: HIDDEN STATE VS OFFSET VARIANCE BY PC")
    print("-" * 50)
    print(f"{'PC':<5} {'Hidden State':>15} {'Offset':>15}")
    for i in range(8):
        print(f"PC{i+1:<3} {pca_final.explained_variance_ratio_[i]*100:>14.1f}% {offset_pc_ratio[i]*100:>14.1f}%")

    # Create visualization
    create_visualization(pca_all, pca_final, offset_pc_ratio, final_states, offsets)

    return pca_all, pca_final

def create_visualization(pca_all, pca_final, offset_pc_ratio, final_states, offsets):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Scree plot - all states
    ax = axes[0, 0]
    ax.bar(range(1, 17), pca_all.explained_variance_ratio_ * 100, color='steelblue', alpha=0.7, label='Individual')
    ax.plot(range(1, 17), np.cumsum(pca_all.explained_variance_ratio_) * 100, 'ro-', label='Cumulative')
    ax.axhline(90, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_title('PCA Spectrum (All Timesteps)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 17))

    # 2. Scree plot - final states
    ax = axes[0, 1]
    ax.bar(range(1, 17), pca_final.explained_variance_ratio_ * 100, color='darkgreen', alpha=0.7, label='Individual')
    ax.plot(range(1, 17), np.cumsum(pca_final.explained_variance_ratio_) * 100, 'ro-', label='Cumulative')
    ax.axhline(90, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_title('PCA Spectrum (Final States t=10)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 17))

    # 3. Offset variance by PC
    ax = axes[0, 2]
    ax.bar(range(1, 17), offset_pc_ratio * 100, color='darkorange', alpha=0.7, label='Individual')
    ax.plot(range(1, 17), np.cumsum(offset_pc_ratio) * 100, 'ro-', label='Cumulative')
    ax.axhline(90, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Offset Variance (%)', fontsize=12)
    ax.set_title('Where Does Offset Live?', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 17))

    # 4. PC1 vs PC2 (what we've been looking at)
    ax = axes[1, 0]
    projected = pca_final.transform(final_states)
    ax.scatter(projected[:, 0], projected[:, 1], c='steelblue', alpha=0.5, s=30)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    var12 = (pca_final.explained_variance_ratio_[0] + pca_final.explained_variance_ratio_[1]) * 100
    ax.set_title(f'PC1 vs PC2 ({var12:.1f}% variance)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 5. PC3 vs PC4 (what we've been missing)
    ax = axes[1, 1]
    ax.scatter(projected[:, 2], projected[:, 3], c='darkgreen', alpha=0.5, s=30)
    ax.set_xlabel('PC3', fontsize=12)
    ax.set_ylabel('PC4', fontsize=12)
    var34 = (pca_final.explained_variance_ratio_[2] + pca_final.explained_variance_ratio_[3]) * 100
    ax.set_title(f'PC3 vs PC4 ({var34:.1f}% variance)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 6. Loadings heatmap for first 6 PCs
    ax = axes[1, 2]
    loadings = pca_final.components_[:6, :]
    im = ax.imshow(loadings, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax.set_xlabel('Neuron', fontsize=12)
    ax.set_ylabel('Principal Component', fontsize=12)
    ax.set_title('PC Loadings (which neurons define each PC)', fontsize=12)
    ax.set_xticks(range(16))
    ax.set_yticks(range(6))
    ax.set_yticklabels([f'PC{i+1}' for i in range(6)])
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('docs/pca_spectrum_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved to docs/pca_spectrum_analysis.png")

if __name__ == "__main__":
    analyze_pca_spectrum()
