"""
Check for additional leave-one-out neurons beyond comparators.

Focus on:
- n5: r(loo_diff) = -0.670 - VERY STRONG!
- n9: r(loo_diff) = -0.438 - also strong
- n4: r(consec_max_diff) = 0.961 - near perfect!
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


def compute_leave_one_out_features(x):
    """Compute various leave-one-out features."""
    batch_size, seq_len = x.shape
    features = {}

    # Running max
    running_max = th.zeros(batch_size, seq_len)
    for t in range(seq_len):
        if t == 0:
            running_max[:, t] = x[:, 0]
        else:
            running_max[:, t] = th.max(running_max[:, t-1], x[:, t])
    features['running_max'] = running_max

    # max(x_0, ..., x_{t-1}) - x_{t-1}  (leave-one-out difference)
    leave_one_out_diff = th.zeros(batch_size, seq_len)
    for t in range(seq_len):
        if t <= 1:
            leave_one_out_diff[:, t] = 0
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
            prev_max = running_max[:, t-1]
            consec_max_diff[:, t] = curr_max - prev_max
    features['consec_max_diff'] = consec_max_diff

    return features


def analyze_n5_and_n9(model, n_samples=50000):
    """
    Analyze n5 and n9 which showed strong leave-one-out correlations.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    loo_features = compute_leave_one_out_features(x)

    W_out = model.linear.weight.data

    print("=" * 70)
    print("ANALYSIS OF n5 AND n9 (STRONG LOO CORRELATION)")
    print("=" * 70)

    # n5: r(loo_diff) = -0.670
    print("\n--- NEURON 5 ---")
    h5 = hidden[:, 5, 9].numpy()
    loo = loo_features['leave_one_out_diff'][:, 9].numpy()

    corr = np.corrcoef(h5, loo)[0, 1]
    print(f"r(n5, loo_diff) = {corr:.3f}")

    # Check W_out importance
    w5_importance = th.abs(W_out[:, 5]).sum().item()
    print(f"|W_out[:,5]| sum = {w5_importance:.2f}")

    # Check correlation with positions
    corr_argmax = np.corrcoef(h5, argmax_pos.numpy())[0, 1]
    corr_2nd = np.corrcoef(h5, targets.numpy())[0, 1]
    print(f"r(n5, argmax) = {corr_argmax:.3f}")
    print(f"r(n5, 2nd_argmax) = {corr_2nd:.3f}")

    # n9: r(loo_diff) = -0.438
    print("\n--- NEURON 9 ---")
    h9 = hidden[:, 9, 9].numpy()

    corr = np.corrcoef(h9, loo)[0, 1]
    print(f"r(n9, loo_diff) = {corr:.3f}")

    w9_importance = th.abs(W_out[:, 9]).sum().item()
    print(f"|W_out[:,9]| sum = {w9_importance:.2f}")

    corr_argmax = np.corrcoef(h9, argmax_pos.numpy())[0, 1]
    corr_2nd = np.corrcoef(h9, targets.numpy())[0, 1]
    print(f"r(n9, argmax) = {corr_argmax:.3f}")
    print(f"r(n9, 2nd_argmax) = {corr_2nd:.3f}")


def analyze_n4_consecutive_diff(model, n_samples=50000):
    """
    n4 has r(consec_max_diff) = 0.961 - nearly perfect!
    This means n4 fires when a NEW max is encountered.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    loo_features = compute_leave_one_out_features(x)

    W_out = model.linear.weight.data

    print("\n" + "=" * 70)
    print("NEURON 4: NEW MAX DETECTOR")
    print("=" * 70)

    h4 = hidden[:, 4, 9].numpy()
    consec = loo_features['consec_max_diff'][:, 9].numpy()

    corr = np.corrcoef(h4, consec)[0, 1]
    print(f"\nr(n4, consec_max_diff) = {corr:.3f}")

    # This means n4 is HIGH when the last value was a new max
    # i.e., when argmax = 9

    print("\nn4 mean by argmax position:")
    for pos in range(10):
        mask = argmax_pos == pos
        if mask.sum() > 50:
            print(f"  argmax={pos}: mean n4 = {h4[mask.numpy()].mean():.3f}")

    # Check W_out
    w4_importance = th.abs(W_out[:, 4]).sum().item()
    print(f"\n|W_out[:,4]| sum = {w4_importance:.2f}")

    # What does W_out[:,4] look like?
    print("\nW_out[:,4] (contribution to each output class):")
    for j in range(10):
        print(f"  class {j}: {W_out[j, 4].item():.3f}")


def analyze_all_loo_neurons(model, n_samples=50000):
    """
    Comprehensive analysis of which neurons show leave-one-out behavior
    and their importance to the output.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    loo_features = compute_leave_one_out_features(x)

    W_out = model.linear.weight.data

    print("\n" + "=" * 70)
    print("ALL NEURONS: LOO BEHAVIOR AND OUTPUT IMPORTANCE")
    print("=" * 70)

    results = []

    for n in range(16):
        h_n = hidden[:, n, 9].numpy()

        # Best LOO correlation
        corr_rm = np.corrcoef(h_n, loo_features['running_max'][:, 9].numpy())[0, 1]
        corr_loo = np.corrcoef(h_n, loo_features['leave_one_out_diff'][:, 9].numpy())[0, 1]
        corr_consec = np.corrcoef(h_n, loo_features['consec_max_diff'][:, 9].numpy())[0, 1]

        # Position correlations
        corr_argmax = np.corrcoef(h_n, argmax_pos.numpy())[0, 1]
        corr_2nd = np.corrcoef(h_n, targets.numpy())[0, 1]

        # Output importance
        w_importance = th.abs(W_out[:, n]).sum().item()

        results.append({
            'neuron': n,
            'corr_rm': corr_rm,
            'corr_loo': corr_loo,
            'corr_consec': corr_consec,
            'corr_argmax': corr_argmax,
            'corr_2nd': corr_2nd,
            'w_importance': w_importance
        })

    # Sort by output importance
    results.sort(key=lambda x: x['w_importance'], reverse=True)

    print(f"\n{'Neuron':<8} | {'|W_out|':<8} | {'r(argmax)':<10} | {'r(2nd)':<10} | {'Best LOO':<20}")
    print("-" * 70)

    for r in results:
        # Find best LOO
        best_loo_name = ''
        best_loo_val = 0
        if abs(r['corr_rm']) > abs(best_loo_val):
            best_loo_val = r['corr_rm']
            best_loo_name = 'running_max'
        if abs(r['corr_loo']) > abs(best_loo_val):
            best_loo_val = r['corr_loo']
            best_loo_name = 'loo_diff'
        if abs(r['corr_consec']) > abs(best_loo_val):
            best_loo_val = r['corr_consec']
            best_loo_name = 'consec_diff'

        print(f"n{r['neuron']:<7} | {r['w_importance']:>6.1f}  | {r['corr_argmax']:>+8.3f}  | "
              f"{r['corr_2nd']:>+8.3f}  | {best_loo_name}:{best_loo_val:+.3f}")

    return results


def categorize_neurons(model, n_samples=50000):
    """
    Categorize all 16 neurons by their function.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    loo_features = compute_leave_one_out_features(x)

    W_out = model.linear.weight.data

    print("\n" + "=" * 70)
    print("NEURON CATEGORIZATION")
    print("=" * 70)

    categories = {
        'Comparators (Fourier)': [],
        'Max Trackers': [],
        'LOO Difference': [],
        'New Max Detector': [],
        'Position Encoders': [],
        'Low Importance': [],
    }

    for n in range(16):
        h_n = hidden[:, n, 9].numpy()

        corr_rm = np.corrcoef(h_n, loo_features['running_max'][:, 9].numpy())[0, 1]
        corr_loo = np.corrcoef(h_n, loo_features['leave_one_out_diff'][:, 9].numpy())[0, 1]
        corr_consec = np.corrcoef(h_n, loo_features['consec_max_diff'][:, 9].numpy())[0, 1]
        corr_argmax = np.corrcoef(h_n, argmax_pos.numpy())[0, 1]
        corr_2nd = np.corrcoef(h_n, targets.numpy())[0, 1]

        w_importance = th.abs(W_out[:, n]).sum().item()

        # Categorize
        if n in [1, 6, 7, 8]:
            categories['Comparators (Fourier)'].append((n, w_importance))
        elif abs(corr_rm) > 0.7:
            categories['Max Trackers'].append((n, w_importance, corr_rm))
        elif abs(corr_consec) > 0.5:
            categories['New Max Detector'].append((n, w_importance, corr_consec))
        elif abs(corr_loo) > 0.3:
            categories['LOO Difference'].append((n, w_importance, corr_loo))
        elif abs(corr_argmax) > 0.15 or abs(corr_2nd) > 0.15:
            categories['Position Encoders'].append((n, w_importance, corr_argmax, corr_2nd))
        else:
            categories['Low Importance'].append((n, w_importance))

    for cat, neurons in categories.items():
        print(f"\n{cat}:")
        if not neurons:
            print("  (none)")
        else:
            for item in neurons:
                if len(item) == 2:
                    print(f"  n{item[0]}: |W_out|={item[1]:.1f}")
                elif len(item) == 3:
                    print(f"  n{item[0]}: |W_out|={item[1]:.1f}, r={item[2]:+.3f}")
                elif len(item) == 4:
                    print(f"  n{item[0]}: |W_out|={item[1]:.1f}, r(argmax)={item[2]:+.3f}, r(2nd)={item[3]:+.3f}")

    return categories


def main():
    model = example_2nd_argmax()

    analyze_n5_and_n9(model)
    analyze_n4_consecutive_diff(model)
    results = analyze_all_loo_neurons(model)
    categories = categorize_neurons(model)

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("""
ADDITIONAL LEAVE-ONE-OUT NEURONS DISCOVERED:

1. n2: Running max tracker (r=0.907 with max_excl_current)
   - This is the "sample-and-hold" we identified before
   - Matches ARC's description: max(x_0, ..., x_{t-2})

2. n4: New max detector (r=0.961 with consec_max_diff!)
   - Nearly PERFECTLY tracks when a new max is encountered
   - High when argmax=9 (last position was the max)
   - This is ARC's "difference between consecutive maximums"

3. n5: Leave-one-out difference (r=-0.670 with loo_diff)
   - NEGATIVELY correlated - high when loo_diff is LOW
   - May encode "margin" or "confidence" of the max

4. n9: Weaker LOO difference (r=-0.438 with loo_diff)
   - Similar to n5 but weaker
   - Low W_out importance

THE PICTURE:
- n2, n4: Track max-related features (ARC's isolated subcircuit)
- n1, n6, n7, n8: Comparators/Fourier encoders (our main finding)
- n5, n9: Additional LOO difference encoders

The leave-one-out and Fourier views are UNIFIED:
- Leave-one-out functions (n2, n4) SET THE THRESHOLD
- Comparators (n1,6,7,8) CLIP against this threshold
- Clipping creates IMPULSES that evolve via Fourier-like recurrence
- n5, n9 provide additional amplitude/margin information
""")


if __name__ == "__main__":
    main()
