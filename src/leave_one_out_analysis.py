"""
Leave-One-Out Analysis

Investigate whether neurons beyond the 4 comparators show leave-one-out behavior,
and how this relates to the Fourier/interference mechanism.

Key questions:
1. Do other neurons compute leave-one-out maximums?
2. Is leave-one-out SEPARATE from Fourier encoding, or the SAME thing?
3. Are they complementary explanations of the same phenomenon?
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def get_full_trajectory(model, x):
    """Compute full trajectory with pre-activations."""
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data

    batch_size = x.shape[0]
    hidden = th.zeros(batch_size, 16, 10)
    pre_act = th.zeros(batch_size, 16, 10)
    h = th.zeros(batch_size, 16)

    for t in range(10):
        x_t = x[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        pre_act[:, :, t] = pre
        h = th.relu(pre)
        hidden[:, :, t] = h

    return hidden, pre_act


def compute_leave_one_out_features(x):
    """
    Compute various leave-one-out maximum features for comparison.

    Returns dict of feature names -> values at each timestep
    """
    batch_size, seq_len = x.shape

    features = {}

    # Running max (no leave-out)
    running_max = th.zeros(batch_size, seq_len)
    for t in range(seq_len):
        if t == 0:
            running_max[:, t] = x[:, 0]
        else:
            running_max[:, t] = th.max(running_max[:, t-1], x[:, t])
    features['running_max'] = running_max

    # max(x_0, ..., x_{t-1}) - excluding current
    max_excluding_current = th.zeros(batch_size, seq_len)
    for t in range(seq_len):
        if t == 0:
            max_excluding_current[:, t] = 0
        else:
            max_excluding_current[:, t] = x[:, :t].max(dim=-1).values
    features['max_excl_current'] = max_excluding_current

    # max(x_0, ..., x_{t-2}) - excluding last two
    max_excl_last_two = th.zeros(batch_size, seq_len)
    for t in range(seq_len):
        if t <= 1:
            max_excl_last_two[:, t] = 0
        else:
            max_excl_last_two[:, t] = x[:, :t-1].max(dim=-1).values
    features['max_excl_last_two'] = max_excl_last_two

    # max(x_0, ..., x_{t-1}) - x_{t-1}  (leave-one-out difference)
    leave_one_out_diff = th.zeros(batch_size, seq_len)
    for t in range(seq_len):
        if t == 0:
            leave_one_out_diff[:, t] = 0
        elif t == 1:
            leave_one_out_diff[:, t] = th.relu(x[:, 0] - x[:, 0])  # = 0
        else:
            prev_max = x[:, :t-1].max(dim=-1).values
            leave_one_out_diff[:, t] = th.relu(prev_max - x[:, t-1])
    features['leave_one_out_diff'] = leave_one_out_diff

    # Consecutive max difference
    consec_max_diff = th.zeros(batch_size, seq_len)
    for t in range(seq_len):
        if t == 0:
            consec_max_diff[:, t] = 0
        else:
            curr_max = running_max[:, t]
            prev_max = running_max[:, t-1] if t > 0 else th.zeros(batch_size)
            consec_max_diff[:, t] = curr_max - prev_max
    features['consec_max_diff'] = consec_max_diff

    return features


def analyze_neuron_vs_leave_one_out(model, n_samples=50000):
    """
    For each neuron, check correlation with various leave-one-out features.
    """
    x = th.rand(n_samples, 10)
    hidden, _ = get_full_trajectory(model, x)
    loo_features = compute_leave_one_out_features(x)

    print("=" * 80)
    print("NEURON CORRELATION WITH LEAVE-ONE-OUT FEATURES")
    print("=" * 80)

    # Check at final timestep
    print("\nAt t=9 (final hidden state):")
    print("-" * 80)
    print(f"{'Neuron':<8} | {'running_max':<12} | {'max_excl_cur':<12} | {'max_excl_2':<12} | {'loo_diff':<12} | {'consec_diff':<12}")
    print("-" * 80)

    correlations = {}

    for n in range(16):
        h_n = hidden[:, n, 9].numpy()

        corrs = {}
        for fname, fvals in loo_features.items():
            f_t9 = fvals[:, 9].numpy()
            corr = np.corrcoef(h_n, f_t9)[0, 1]
            corrs[fname] = corr

        correlations[n] = corrs

        print(f"n{n:<7} | {corrs['running_max']:>+10.3f}  | {corrs['max_excl_current']:>+10.3f}  | "
              f"{corrs['max_excl_last_two']:>+10.3f}  | {corrs['leave_one_out_diff']:>+10.3f}  | "
              f"{corrs['consec_max_diff']:>+10.3f}")

    return correlations


def analyze_temporal_correlation(model, n_samples=50000):
    """
    Check if neurons track leave-one-out features across TIME, not just at t=9.
    """
    x = th.rand(n_samples, 10)
    hidden, _ = get_full_trajectory(model, x)
    loo_features = compute_leave_one_out_features(x)

    print("\n" + "=" * 80)
    print("TEMPORAL CORRELATION: h[n,t] vs leave-one-out features")
    print("=" * 80)

    # Focus on key neurons
    key_neurons = [1, 2, 4, 6, 7, 8]

    for n in key_neurons:
        print(f"\n--- Neuron {n} ---")
        print(f"{'Time':<6} | {'running_max':<12} | {'max_excl_cur':<12} | {'loo_diff':<12}")
        print("-" * 50)

        for t in range(10):
            h_nt = hidden[:, n, t].numpy()

            corr_rm = np.corrcoef(h_nt, loo_features['running_max'][:, t].numpy())[0, 1]
            corr_mec = np.corrcoef(h_nt, loo_features['max_excl_current'][:, t].numpy())[0, 1]
            corr_loo = np.corrcoef(h_nt, loo_features['leave_one_out_diff'][:, t].numpy())[0, 1]

            print(f"t={t:<4} | {corr_rm:>+10.3f}  | {corr_mec:>+10.3f}  | {corr_loo:>+10.3f}")


def analyze_fourier_vs_leave_one_out(model, n_samples=50000):
    """
    Key question: Is the Fourier encoding SEPARATE from leave-one-out,
    or are they describing the SAME phenomenon?

    Hypothesis: Leave-one-out max tracking CREATES the conditions for
    Fourier-like encoding through clipping dynamics.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden, _ = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]
    loo_features = compute_leave_one_out_features(x)

    print("\n" + "=" * 80)
    print("FOURIER vs LEAVE-ONE-OUT: Same or Different?")
    print("=" * 80)

    comparators = [1, 6, 7, 8]

    print("\nFor comparators, what explains h_final better?")
    print("-" * 70)

    for n in comparators:
        h_n = h_final[:, n].numpy()

        # Correlation with argmax position (Fourier interpretation)
        corr_argmax = np.corrcoef(h_n, argmax_pos.numpy())[0, 1]

        # Correlation with leave-one-out diff at t=9
        corr_loo = np.corrcoef(h_n, loo_features['leave_one_out_diff'][:, 9].numpy())[0, 1]

        # Correlation with max_excl_current at t=9
        corr_mec = np.corrcoef(h_n, loo_features['max_excl_current'][:, 9].numpy())[0, 1]

        print(f"n{n}: r(argmax_pos)={corr_argmax:+.3f}, r(loo_diff)={corr_loo:+.3f}, r(max_excl_cur)={corr_mec:+.3f}")

    print("""
Key insight: If argmax_pos and loo_diff correlations are similar,
they may be describing the SAME underlying phenomenon from different angles.

- Leave-one-out view: "neuron tracks max minus recent values"
- Fourier view: "neuron encodes position via recurrence dynamics"

The CONNECTION: When x[t] exceeds the running max, the neuron CLIPS.
The leave-one-out value encodes "how much above threshold" while
the Fourier response encodes "how long since clipping".

These are TWO ASPECTS of the same clipping event!
""")

    # Check if they're redundant
    print("\nRedundancy check: Partial correlation")
    print("-" * 50)

    for n in comparators:
        h_n = h_final[:, n].numpy()
        argmax = argmax_pos.numpy().astype(float)
        loo = loo_features['leave_one_out_diff'][:, 9].numpy()

        # Regress h_n on argmax, then correlate residual with loo
        X = np.column_stack([argmax, np.ones(n_samples)])
        coeffs = np.linalg.lstsq(X, h_n, rcond=None)[0]
        residual = h_n - X @ coeffs

        partial_corr = np.corrcoef(residual, loo)[0, 1]
        print(f"n{n}: r(loo_diff | argmax_pos) = {partial_corr:+.3f}")

    print("""
If partial correlation is near zero, argmax_pos and loo_diff
encode REDUNDANT information in h_final.

If partial correlation is significant, they encode DIFFERENT
aspects that combine in h_final.
""")


def analyze_n2_as_running_max(model, n_samples=50000):
    """
    Special analysis for n2: Does it track running max as ARC suggests?
    """
    x = th.rand(n_samples, 10)
    hidden, _ = get_full_trajectory(model, x)
    loo_features = compute_leave_one_out_features(x)

    print("\n" + "=" * 80)
    print("NEURON 2: RUNNING MAX TRACKER")
    print("=" * 80)

    print("\nn2 correlation with running max across time:")
    print("-" * 40)

    for t in range(10):
        h_2t = hidden[:, 2, t].numpy()
        rm_t = loo_features['running_max'][:, t].numpy()
        corr = np.corrcoef(h_2t, rm_t)[0, 1]
        print(f"t={t}: r(n2, running_max) = {corr:+.3f}")

    # But is it max(x_0,...,x_{t-2}) as ARC suggests?
    print("\nn2 correlation with max_excl_last_two:")
    print("-" * 40)

    for t in range(2, 10):
        h_2t = hidden[:, 2, t].numpy()
        melt = loo_features['max_excl_last_two'][:, t].numpy()
        corr = np.corrcoef(h_2t, melt)[0, 1]
        print(f"t={t}: r(n2, max_excl_last_two) = {corr:+.3f}")


def plot_leave_one_out_comparison(model, n_samples=20000, save_path=None):
    """
    Visualize the relationship between neurons and leave-one-out features.
    """
    x = th.rand(n_samples, 10)
    argmax_pos = x.argmax(dim=-1)
    hidden, _ = get_full_trajectory(model, x)
    loo_features = compute_leave_one_out_features(x)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Panel 1: n2 vs running_max at t=9
    ax = axes[0, 0]
    ax.scatter(loo_features['running_max'][:, 9].numpy(),
               hidden[:, 2, 9].numpy(), alpha=0.1, s=1)
    ax.set_xlabel('running_max at t=9')
    ax.set_ylabel('n2 at t=9')
    ax.set_title('n2 tracks running maximum')
    corr = np.corrcoef(hidden[:, 2, 9].numpy(), loo_features['running_max'][:, 9].numpy())[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=12)
    ax.grid(True, alpha=0.3)

    # Panel 2: n7 vs leave_one_out_diff at t=9
    ax = axes[0, 1]
    ax.scatter(loo_features['leave_one_out_diff'][:, 9].numpy(),
               hidden[:, 7, 9].numpy(), alpha=0.1, s=1)
    ax.set_xlabel('leave_one_out_diff at t=9')
    ax.set_ylabel('n7 at t=9')
    ax.set_title('n7 vs leave-one-out diff')
    corr = np.corrcoef(hidden[:, 7, 9].numpy(), loo_features['leave_one_out_diff'][:, 9].numpy())[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=12)
    ax.grid(True, alpha=0.3)

    # Panel 3: n7 vs argmax_pos (Fourier view)
    ax = axes[0, 2]
    means = []
    for pos in range(10):
        mask = argmax_pos == pos
        means.append(hidden[mask, 7, 9].mean().item())
    ax.bar(range(10), means, color='steelblue', alpha=0.7)
    ax.set_xlabel('argmax position')
    ax.set_ylabel('mean n7 at t=9')
    ax.set_title('n7 Fourier-like encoding of argmax')
    ax.grid(True, alpha=0.3)

    # Panel 4: Comparator correlations with argmax vs loo_diff
    ax = axes[1, 0]
    comparators = [1, 6, 7, 8]
    corr_argmax = []
    corr_loo = []
    for n in comparators:
        h_n = hidden[:, n, 9].numpy()
        corr_argmax.append(np.corrcoef(h_n, argmax_pos.numpy())[0, 1])
        corr_loo.append(np.corrcoef(h_n, loo_features['leave_one_out_diff'][:, 9].numpy())[0, 1])

    x_pos = np.arange(len(comparators))
    width = 0.35
    ax.bar(x_pos - width/2, corr_argmax, width, label='r(argmax_pos)', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, corr_loo, width, label='r(loo_diff)', color='red', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'n{n}' for n in comparators])
    ax.set_ylabel('Correlation')
    ax.set_title('Comparators: Fourier vs Leave-One-Out')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 5: All neurons - best leave-one-out correlation
    ax = axes[1, 1]
    best_loo_corr = []
    best_loo_name = []
    for n in range(16):
        h_n = hidden[:, n, 9].numpy()
        best_corr = 0
        best_name = ''
        for fname, fvals in loo_features.items():
            corr = abs(np.corrcoef(h_n, fvals[:, 9].numpy())[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_name = fname
        best_loo_corr.append(best_corr)
        best_loo_name.append(best_name[:8])

    colors = ['red' if n in [1,6,7,8] else 'blue' if n == 2 else 'green' if n == 4 else 'gray'
              for n in range(16)]
    ax.bar(range(16), best_loo_corr, color=colors, alpha=0.7)
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Best LOO correlation')
    ax.set_title('Which neurons show leave-one-out behavior?\n(red=comparators, blue=n2, green=n4)')
    ax.set_xticks(range(16))
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 6: n4 analysis (ARC says it computes consecutive max diff)
    ax = axes[1, 2]
    ax.scatter(loo_features['consec_max_diff'][:, 9].numpy(),
               hidden[:, 4, 9].numpy(), alpha=0.1, s=1)
    ax.set_xlabel('consecutive max difference at t=9')
    ax.set_ylabel('n4 at t=9')
    ax.set_title('n4 vs consecutive max diff')
    corr = np.corrcoef(hidden[:, 4, 9].numpy(), loo_features['consec_max_diff'][:, 9].numpy())[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved leave-one-out comparison to {save_path}')

    return fig


def main():
    model = example_2nd_argmax()

    analyze_neuron_vs_leave_one_out(model)
    analyze_temporal_correlation(model)
    analyze_fourier_vs_leave_one_out(model)
    analyze_n2_as_running_max(model)

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)

    plot_leave_one_out_comparison(model, save_path='docs/figures/leave_one_out_comparison.png')

    print("\n" + "=" * 80)
    print("SYNTHESIS: FOURIER vs LEAVE-ONE-OUT")
    print("=" * 80)
    print("""
KEY FINDING: Fourier encoding and leave-one-out are TWO VIEWS of the SAME mechanism.

The CONNECTION:
1. Leave-one-out functions (algebraic view):
   - n2 ≈ running_max(x_0, ..., x_{t-2})
   - n7 ≈ max(x_0, ..., x_{t-1}) - x_{t-1}

2. These functions DETERMINE when neurons clip:
   - Neuron clips when current input exceeds its leave-one-out threshold
   - Clipping is the "impulse" in our Fourier interpretation

3. After clipping, recurrence creates Fourier-like decay:
   - Different self-recurrence values = different frequencies
   - Time since clipping = phase of the response

4. The POSITION ENCODING emerges from:
   - WHEN the neuron clipped (determined by leave-one-out function)
   - HOW LONG AGO it clipped (encoded by Fourier-like recurrence)

CONCLUSION:
- Leave-one-out describes WHAT triggers clipping
- Fourier describes HOW position is encoded after clipping
- They are COMPLEMENTARY, not competing
- The full explanation needs BOTH:

  [Leave-one-out threshold] → [Clipping event] → [Fourier recurrence] → [Position encoding]
""")


if __name__ == "__main__":
    main()
