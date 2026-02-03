"""
UMAP and t-SNE visualizations of hidden states and offsets.
These nonlinear methods can capture the ~9 important dimensions
that linear PCA misses in a 2D projection.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch as th
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from alg_zoo.architectures import DistRNN

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

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


def collect_data():
    """Collect final hidden states with metadata."""
    M_val = 1.0
    S_val = 0.8

    final_fwd = []
    final_rev = []
    m_positions = []
    s_positions = []
    gaps = []

    for m_pos in range(10):
        for s_pos in range(10):
            if m_pos == s_pos:
                continue

            h_fwd = run_stepwise([(m_pos, M_val), (s_pos, S_val)])[-1]
            h_rev = run_stepwise([(m_pos, S_val), (s_pos, M_val)])[-1]

            final_fwd.append(h_fwd)
            final_rev.append(h_rev)
            m_positions.append(m_pos)
            s_positions.append(s_pos)
            gaps.append(abs(m_pos - s_pos))

    return (np.array(final_fwd), np.array(final_rev),
            np.array(m_positions), np.array(s_positions), np.array(gaps))


def collect_trajectories():
    """Collect full trajectories for a few example pairs."""
    M_val = 1.0
    S_val = 0.8

    pairs = [(0, 5), (2, 7), (4, 5), (1, 8), (3, 9)]
    trajs = {}

    for m_pos, s_pos in pairs:
        states_fwd = run_stepwise([(m_pos, M_val), (s_pos, S_val)])
        states_rev = run_stepwise([(m_pos, S_val), (s_pos, M_val)])
        trajs[(m_pos, s_pos)] = (states_fwd, states_rev)

    return trajs


def run_analysis():
    print("Collecting data...")
    final_fwd, final_rev, m_pos, s_pos, gaps = collect_data()
    trajs = collect_trajectories()

    offsets = final_fwd - final_rev
    n_pairs = len(final_fwd)

    # Combine fwd and rev for joint embedding
    all_final = np.vstack([final_fwd, final_rev])
    labels = np.array(['fwd'] * n_pairs + ['rev'] * n_pairs)
    all_gaps = np.concatenate([gaps, gaps])
    all_m_pos = np.concatenate([m_pos, m_pos])
    all_s_pos = np.concatenate([s_pos, s_pos])

    # Also collect trajectory states for embedding
    traj_states = []
    traj_labels = []
    traj_times = []
    traj_pair_ids = []

    for pair_id, ((mp, sp), (sf, sr)) in enumerate(trajs.items()):
        for t in range(1, 11):  # skip t=0 (zeros)
            traj_states.append(sf[t])
            traj_labels.append('fwd')
            traj_times.append(t)
            traj_pair_ids.append(pair_id)

            traj_states.append(sr[t])
            traj_labels.append('rev')
            traj_times.append(t)
            traj_pair_ids.append(pair_id)

    traj_states = np.array(traj_states)
    traj_times = np.array(traj_times)
    traj_pair_ids = np.array(traj_pair_ids)

    # --- t-SNE ---
    print("Running t-SNE on final states...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=2000)
    tsne_final = tsne.fit_transform(all_final)

    print("Running t-SNE on offsets...")
    tsne_offset = TSNE(n_components=2, perplexity=20, random_state=42, max_iter=2000)
    tsne_off = tsne_offset.fit_transform(offsets)

    print("Running t-SNE on trajectories...")
    tsne_traj = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=2000)
    tsne_tr = tsne_traj.fit_transform(traj_states)

    # --- UMAP ---
    umap_final = None
    umap_off = None
    umap_tr = None

    if HAS_UMAP:
        print("Running UMAP on final states...")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        umap_final = reducer.fit_transform(all_final)

        print("Running UMAP on offsets...")
        reducer_off = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        umap_off = reducer_off.fit_transform(offsets)

        print("Running UMAP on trajectories...")
        reducer_traj = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        umap_tr = reducer_traj.fit_transform(traj_states)
    else:
        print("UMAP not available, skipping. Install with: uv add umap-learn")

    # --- Visualize ---
    create_final_state_plots(tsne_final, umap_final, labels, all_gaps, all_m_pos, all_s_pos, n_pairs)
    create_offset_plots(tsne_off, umap_off, offsets, gaps, m_pos, s_pos)
    create_trajectory_plots(tsne_tr, umap_tr, traj_labels, traj_times, traj_pair_ids, trajs)


def create_final_state_plots(tsne_emb, umap_emb, labels, gaps, m_pos, s_pos, n_pairs):
    n_methods = 2 if umap_emb is not None else 1
    fig, axes = plt.subplots(n_methods, 3, figsize=(18, 6 * n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    for row, (emb, method) in enumerate([(tsne_emb, 't-SNE'), (umap_emb, 'UMAP')]):
        if emb is None:
            continue

        # 1. Color by fwd/rev
        ax = axes[row, 0]
        fwd_mask = labels == 'fwd'
        rev_mask = labels == 'rev'
        ax.scatter(emb[fwd_mask, 0], emb[fwd_mask, 1], c='blue', alpha=0.6, s=40, label='Fwd')
        ax.scatter(emb[rev_mask, 0], emb[rev_mask, 1], c='red', alpha=0.6, s=40, label='Rev')

        # Draw offset arrows for some pairs
        for i in range(0, n_pairs, 5):
            ax.annotate('', xy=(emb[i, 0], emb[i, 1]),
                       xytext=(emb[i + n_pairs, 0], emb[i + n_pairs, 1]),
                       arrowprops=dict(arrowstyle='->', color='purple', alpha=0.3, lw=1))

        ax.set_title(f'{method}: Fwd vs Rev', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Color by gap
        ax = axes[row, 1]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=gaps, cmap='viridis', alpha=0.7, s=40)
        plt.colorbar(sc, ax=ax, label='|m_pos - s_pos|')
        ax.set_title(f'{method}: Color by gap', fontsize=12)
        ax.grid(True, alpha=0.3)

        # 3. Color by m_pos
        ax = axes[row, 2]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=m_pos, cmap='tab10', alpha=0.7, s=40)
        plt.colorbar(sc, ax=ax, label='M position')
        ax.set_title(f'{method}: Color by M position', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Final Hidden States (all 90 pairs × 2 directions)', fontsize=14)
    plt.tight_layout()
    plt.savefig('docs/umap_tsne_final_states.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved docs/umap_tsne_final_states.png")


def create_offset_plots(tsne_emb, umap_emb, offsets, gaps, m_pos, s_pos):
    n_methods = 2 if umap_emb is not None else 1
    fig, axes = plt.subplots(n_methods, 3, figsize=(18, 6 * n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    # Compute margin for each pair
    margins = []
    M_val = 1.0
    S_val = 0.8
    for i in range(len(m_pos)):
        h_fwd = offsets[i] + np.zeros(16)  # dummy, we need actual logits
        # Recompute
        h_fwd_full = run_stepwise([(m_pos[i], M_val), (s_pos[i], S_val)])[-1]
        h_rev_full = run_stepwise([(m_pos[i], S_val), (s_pos[i], M_val)])[-1]
        logits_fwd = W_out @ h_fwd_full
        logits_rev = W_out @ h_rev_full
        # Correct answer is s_pos[i]
        margin_fwd = logits_fwd[s_pos[i]] - np.max(np.delete(logits_fwd, s_pos[i]))
        margin_rev = logits_rev[s_pos[i]] - np.max(np.delete(logits_rev, s_pos[i]))
        margins.append(min(margin_fwd, margin_rev))

    margins = np.array(margins)
    correct = margins > 0

    for row, (emb, method) in enumerate([(tsne_emb, 't-SNE'), (umap_emb, 'UMAP')]):
        if emb is None:
            continue

        # 1. Color by gap
        ax = axes[row, 0]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=gaps, cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(sc, ax=ax, label='|m_pos - s_pos|')
        ax.set_title(f'{method} of offsets: color by gap', fontsize=12)
        ax.grid(True, alpha=0.3)

        # 2. Color by margin
        ax = axes[row, 1]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=margins, cmap='RdYlGn', alpha=0.7, s=50)
        plt.colorbar(sc, ax=ax, label='Min margin')
        ax.set_title(f'{method} of offsets: color by margin', fontsize=12)
        ax.grid(True, alpha=0.3)

        # 3. Color by s_pos (the answer)
        ax = axes[row, 2]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=s_pos, cmap='tab10', alpha=0.7, s=50)
        plt.colorbar(sc, ax=ax, label='S position (answer)')
        ax.set_title(f'{method} of offsets: color by answer', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Offset Vectors (h_fwd - h_rev) in 16D → 2D', fontsize=14)
    plt.tight_layout()
    plt.savefig('docs/umap_tsne_offsets.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved docs/umap_tsne_offsets.png")


def create_trajectory_plots(tsne_emb, umap_emb, traj_labels, traj_times, traj_pair_ids, trajs):
    n_methods = 2 if umap_emb is not None else 1
    fig, axes = plt.subplots(n_methods, 2, figsize=(14, 6 * n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    pair_list = list(trajs.keys())
    pair_colors = plt.cm.Set1(np.linspace(0, 1, len(pair_list)))

    for row, (emb, method) in enumerate([(tsne_emb, 't-SNE'), (umap_emb, 'UMAP')]):
        if emb is None:
            continue

        # 1. Color by time
        ax = axes[row, 0]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=traj_times, cmap='plasma', alpha=0.6, s=40)
        plt.colorbar(sc, ax=ax, label='Timestep')
        ax.set_title(f'{method}: trajectories colored by time', fontsize=12)
        ax.grid(True, alpha=0.3)

        # 2. Color by pair, with fwd/rev distinguished
        ax = axes[row, 1]
        traj_labels_arr = np.array(traj_labels)

        for pair_id in range(len(pair_list)):
            mask = traj_pair_ids == pair_id
            fwd_mask = mask & (traj_labels_arr == 'fwd')
            rev_mask = mask & (traj_labels_arr == 'rev')

            color = pair_colors[pair_id]
            mp, sp = pair_list[pair_id]

            # Sort by time for line drawing
            fwd_idx = np.where(fwd_mask)[0]
            rev_idx = np.where(rev_mask)[0]
            fwd_times = traj_times[fwd_idx]
            rev_times = traj_times[rev_idx]
            fwd_order = np.argsort(fwd_times)
            rev_order = np.argsort(rev_times)

            ax.plot(emb[fwd_idx[fwd_order], 0], emb[fwd_idx[fwd_order], 1],
                   '-', color=color, linewidth=2, alpha=0.7)
            ax.plot(emb[rev_idx[rev_order], 0], emb[rev_idx[rev_order], 1],
                   '--', color=color, linewidth=2, alpha=0.7)

            ax.scatter(emb[fwd_idx, 0], emb[fwd_idx, 1], c=[color], s=30, marker='o', alpha=0.8)
            ax.scatter(emb[rev_idx, 0], emb[rev_idx, 1], c=[color], s=30, marker='s', alpha=0.8)

            # Label final states
            fwd_final = fwd_idx[fwd_order[-1]]
            rev_final = rev_idx[rev_order[-1]]
            ax.annotate(f'{mp},{sp}F', (emb[fwd_final, 0], emb[fwd_final, 1]), fontsize=7)
            ax.annotate(f'{mp},{sp}R', (emb[rev_final, 0], emb[rev_final, 1]), fontsize=7)

        ax.set_title(f'{method}: trajectories by pair (solid=fwd, dashed=rev)', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Trajectory Embeddings (5 example pairs × 10 timesteps × 2 directions)', fontsize=14)
    plt.tight_layout()
    plt.savefig('docs/umap_tsne_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved docs/umap_tsne_trajectories.png")


if __name__ == "__main__":
    run_analysis()
