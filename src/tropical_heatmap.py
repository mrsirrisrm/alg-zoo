"""
Heatmap of number of active neurons at t=9 for each (Mt, St) pair.
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo.architectures import DistRNN


def load_local_model():
    import os
    path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
    state_dict = th.load(path, weights_only=True, map_location='cpu')
    seq_len, hidden_size = state_dict['linear.weight'].shape
    model = DistRNN(hidden_size=hidden_size, seq_len=seq_len, bias=False)
    model.load_state_dict(state_dict)
    return model


def run_trace_full(W_ih, W_hh, x_single):
    h = th.zeros(1, 16)
    for t in range(10):
        x_t = x_single[t:t+1].unsqueeze(0)
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
    return pre[0].detach().numpy()


def main():
    model = load_local_model()
    W_ih = model.rnn.weight_ih_l0.data.squeeze().clone()
    W_hh = model.rnn.weight_hh_l0.data.clone()

    # Count active neurons at t=9 for each pair
    grid = np.full((10, 10), np.nan)

    for mt in range(10):
        for st in range(10):
            if mt == st:
                continue
            x = th.zeros(10)
            x[mt] = 1.0
            x[st] = 0.8
            pre_final = run_trace_full(W_ih, W_hh, x)
            n_active = (pre_final > 0).sum()
            grid[mt, st] = n_active

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap='viridis', origin='lower', vmin=9, vmax=15)

    # Annotate each cell
    for mt in range(10):
        for st in range(10):
            if mt == st:
                continue
            val = int(grid[mt, st])
            color = 'white' if val < 12 else 'black'
            ax.text(st, mt, str(val), ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    # Diagonal
    for i in range(10):
        ax.text(i, i, '—', ha='center', va='center', fontsize=9, color='gray')

    ax.set_xlabel('St (2nd largest position)')
    ax.set_ylabel('Mt (max position)')
    ax.set_title('Active neurons at t=9 (of 16)')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax, label='# active neurons')

    # Add forward/reversed labels
    ax.plot([-0.5, 9.5], [-0.5, 9.5], 'w--', alpha=0.3, linewidth=1)
    ax.text(8, 2, 'Forward\n(St > Mt)', ha='center', va='center',
            fontsize=10, color='white', alpha=0.7)
    ax.text(2, 8, 'Reversed\n(St < Mt)', ha='center', va='center',
            fontsize=10, color='white', alpha=0.7)

    plt.tight_layout()
    plt.savefig('docs/tropical_active_neurons_heatmap.png', dpi=150)
    print("Saved to docs/tropical_active_neurons_heatmap.png")

    # Print summary stats
    fwd_vals = [grid[mt, st] for mt in range(10) for st in range(mt+1, 10)]
    rev_vals = [grid[mt, st] for mt in range(10) for st in range(0, mt)]
    print(f"\nForward:  mean={np.mean(fwd_vals):.1f}, range={int(min(fwd_vals))}-{int(max(fwd_vals))}")
    print(f"Reversed: mean={np.mean(rev_vals):.1f}, range={int(min(rev_vals))}-{int(max(rev_vals))}")


if __name__ == "__main__":
    main()
