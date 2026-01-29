"""
Clean Impulse Analysis

Create clean dataset:
- Only max and 2nd_max are non-zero
- All other positions are 0
- Remove duplicate (max_pos, 2nd_pos) pairs

This isolates the pure two-impulse signal.
"""

import torch as th
import numpy as np
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def get_full_trajectory(model, x):
    """Compute full trajectory."""
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


def create_clean_dataset():
    """
    Create all unique (max_pos, 2nd_pos) pairs with clean impulses.

    For each pair, create input with:
    - 1.0 at max_pos
    - 0.8 at 2nd_pos (so it's clearly 2nd)
    - 0.0 everywhere else
    """
    samples = []
    labels = []

    for max_pos in range(10):
        for sec_pos in range(10):
            if max_pos == sec_pos:
                continue

            x = th.zeros(10)
            x[max_pos] = 1.0
            x[sec_pos] = 0.8

            samples.append(x)
            labels.append(sec_pos)

    X = th.stack(samples)  # 90 x 10
    y = th.tensor(labels)

    return X, y


def analyze_clean_dataset(model):
    """Analyze model behavior on clean dataset."""
    X, y = create_clean_dataset()

    hidden, pre_act = get_full_trajectory(model, X)
    h_final = hidden[:, :, 9]

    W_out = model.linear.weight.data
    logits = h_final @ W_out.T
    preds = logits.argmax(dim=-1)

    # Accuracy
    acc = (preds == y).float().mean().item() * 100

    print("=" * 80)
    print("CLEAN IMPULSE DATASET ANALYSIS")
    print("=" * 80)
    print(f"\nDataset: 90 unique (max_pos, 2nd_pos) pairs")
    print(f"Input: 1.0 at max, 0.8 at 2nd, 0.0 elsewhere")
    print(f"Accuracy: {acc:.1f}%")

    # Get max_pos and 2nd_pos for each sample
    max_pos = X.argmax(dim=-1)
    sec_pos = y

    return X, y, h_final, pre_act, max_pos, sec_pos, preds


def analyze_h_final_structure(h_final, max_pos, sec_pos):
    """Analyze h_final as function of positions."""

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("H_FINAL STRUCTURE (CLEAN DATA)")
    print("=" * 80)

    # Build matrices: h_final[max_pos, 2nd_pos] for each neuron
    print("\n1. H_FINAL BY (MAX_POS, 2ND_POS) FOR EACH COMPARATOR")
    print("-" * 70)

    for n in comparators:
        print(f"\nn{n}:")
        print("      2nd: ", end="")
        for s in range(10):
            print(f"{s:>6}", end="")
        print()
        print("max")

        for m in range(10):
            print(f" {m}       ", end="")
            for s in range(10):
                if m == s:
                    print("     -", end="")
                else:
                    # Find this sample
                    mask = (max_pos == m) & (sec_pos == s)
                    if mask.sum() > 0:
                        val = h_final[mask, n].item()
                        print(f"{val:>6.2f}", end="")
                    else:
                        print("     ?", end="")
            print()


def analyze_marginal_curves(h_final, max_pos, sec_pos):
    """Extract marginal curves: h(max_pos) and h(2nd_pos)."""

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("MARGINAL CURVES (AVERAGING OVER OTHER POSITION)")
    print("=" * 80)

    for n in comparators:
        # h by max_pos (averaging over all 2nd_pos)
        h_by_max = []
        for m in range(10):
            mask = max_pos == m
            h_by_max.append(h_final[mask, n].mean().item())

        # h by 2nd_pos (averaging over all max_pos)
        h_by_sec = []
        for s in range(10):
            mask = sec_pos == s
            h_by_sec.append(h_final[mask, n].mean().item())

        h_by_max = np.array(h_by_max)
        h_by_sec = np.array(h_by_sec)

        # Correlation
        corr = np.corrcoef(h_by_max, h_by_sec)[0, 1]

        print(f"\nn{n}:")
        print(f"  h by max_pos: {' '.join([f'{v:.2f}' for v in h_by_max])}")
        print(f"  h by 2nd_pos: {' '.join([f'{v:.2f}' for v in h_by_sec])}")
        print(f"  Correlation: {corr:+.3f}")

        # DFT analysis
        fft_max = np.fft.fft(h_by_max - h_by_max.mean())
        fft_sec = np.fft.fft(h_by_sec - h_by_sec.mean())

        energy_max = np.abs(fft_max) ** 2
        energy_sec = np.abs(fft_sec) ** 2

        print(f"  DFT (max): k=1: {energy_max[1]/energy_max.sum()*100:.0f}%, "
              f"k=2: {energy_max[2]/energy_max.sum()*100:.0f}%")
        print(f"  DFT (2nd): k=1: {energy_sec[1]/energy_sec.sum()*100:.0f}%, "
              f"k=2: {energy_sec[2]/energy_sec.sum()*100:.0f}%")

        # Phase at k=1
        phase_max = np.angle(fft_max[1])
        phase_sec = np.angle(fft_sec[1])
        print(f"  Phase at k=1: max={np.degrees(phase_max):.0f}°, "
              f"2nd={np.degrees(phase_sec):.0f}°, "
              f"diff={np.degrees(phase_max - phase_sec):.0f}°")


def test_superposition_clean(h_final, max_pos, sec_pos):
    """
    Test superposition model on clean data.

    Model: h[n] = A * f(max_pos) + B * g(2nd_pos) + C

    With clean data, this should have MUCH higher R².
    """
    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("SUPERPOSITION MODEL TEST (CLEAN DATA)")
    print("=" * 80)

    # First, get the marginal curves
    for n in comparators:
        h_by_max = []
        for m in range(10):
            mask = max_pos == m
            h_by_max.append(h_final[mask, n].mean().item())

        h_by_sec = []
        for s in range(10):
            mask = sec_pos == s
            h_by_sec.append(h_final[mask, n].mean().item())

        h_by_max = np.array(h_by_max)
        h_by_sec = np.array(h_by_sec)

        # For each sample, get f(max) and g(2nd)
        f_max = np.array([h_by_max[m] for m in max_pos.numpy()])
        g_sec = np.array([h_by_sec[s] for s in sec_pos.numpy()])
        h_actual = h_final[:, n].numpy()

        # Fit: h = A*f_max + B*g_sec + C
        X = np.column_stack([f_max, g_sec, np.ones(len(h_actual))])
        coeffs, _, _, _ = np.linalg.lstsq(X, h_actual, rcond=None)
        A, B, C = coeffs

        h_pred = X @ coeffs
        ss_res = np.sum((h_actual - h_pred) ** 2)
        ss_tot = np.sum((h_actual - h_actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        print(f"\nn{n}:")
        print(f"  h = {A:.3f} * f(max) + {B:.3f} * g(2nd) + {C:.3f}")
        print(f"  R² = {r2:.4f}")
        print(f"  A + B = {A + B:.3f}")


def test_additive_model(h_final, max_pos, sec_pos):
    """
    Test if h_final is additive in positions.

    If h = f(max) + g(2nd), then:
    h[max=3, 2nd=7] - h[max=3, 2nd=5] = g(7) - g(5)
    h[max=4, 2nd=7] - h[max=4, 2nd=5] = g(7) - g(5)  (same!)

    This is the ANOVA-style decomposition.
    """
    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("ADDITIVE MODEL TEST")
    print("=" * 80)
    print("\nIf h = f(max) + g(2nd), then varying max while holding 2nd fixed")
    print("should give the SAME differences as varying max with different 2nd.")

    for n in comparators:
        # Build the full matrix
        h_matrix = np.full((10, 10), np.nan)
        for i in range(len(max_pos)):
            m = max_pos[i].item()
            s = sec_pos[i].item()
            h_matrix[m, s] = h_final[i, n].item()

        # Check additivity: compare row differences
        # h[m1, s] - h[m2, s] should be constant across s
        print(f"\nn{n}:")

        # Compare rows 2 and 5
        m1, m2 = 2, 5
        diffs = []
        for s in range(10):
            if s == m1 or s == m2:
                continue
            if not np.isnan(h_matrix[m1, s]) and not np.isnan(h_matrix[m2, s]):
                diffs.append(h_matrix[m1, s] - h_matrix[m2, s])

        diffs = np.array(diffs)
        print(f"  h[max=2, s] - h[max=5, s] for all s: {diffs}")
        print(f"  Mean: {diffs.mean():.3f}, Std: {diffs.std():.3f}")

        # Compute total explained variance by additive model
        # Using ANOVA decomposition
        grand_mean = np.nanmean(h_matrix)
        row_means = np.nanmean(h_matrix, axis=1)
        col_means = np.nanmean(h_matrix, axis=0)

        ss_total = 0
        ss_additive = 0
        ss_residual = 0

        for m in range(10):
            for s in range(10):
                if np.isnan(h_matrix[m, s]):
                    continue
                val = h_matrix[m, s]
                pred_additive = row_means[m] + col_means[s] - grand_mean
                ss_total += (val - grand_mean) ** 2
                ss_additive += (pred_additive - grand_mean) ** 2
                ss_residual += (val - pred_additive) ** 2

        r2_additive = 1 - ss_residual / ss_total
        print(f"  Additive model R² = {r2_additive:.4f}")


def analyze_single_vs_double_impulse(model):
    """
    Compare single impulse (only max) vs double impulse (max + 2nd).
    """
    print("\n" + "=" * 80)
    print("SINGLE VS DOUBLE IMPULSE COMPARISON")
    print("=" * 80)

    comparators = [1, 6, 7, 8]

    # Single impulse at each position
    single_h = []
    for pos in range(10):
        x = th.zeros(1, 10)
        x[0, pos] = 1.0
        hidden, _ = get_full_trajectory(model, x)
        single_h.append(hidden[0, :, 9])
    single_h = th.stack(single_h)  # 10 x 16

    print("\nSingle impulse h_final (by impulse position):")
    print("-" * 70)
    print(f"{'pos':<5}", end="")
    for n in comparators:
        print(f"{'n'+str(n):<8}", end="")
    print()

    for pos in range(10):
        print(f"{pos:<5}", end="")
        for n in comparators:
            print(f"{single_h[pos, n].item():<8.2f}", end="")
        print()

    # Now create double impulse and check if it's sum of singles
    print("\n\nDouble impulse vs sum of singles:")
    print("-" * 70)

    test_pairs = [(2, 6), (3, 7), (1, 8), (4, 9)]

    for max_p, sec_p in test_pairs:
        x = th.zeros(1, 10)
        x[0, max_p] = 1.0
        x[0, sec_p] = 0.8
        hidden, _ = get_full_trajectory(model, x)
        h_double = hidden[0, :, 9]

        # Predicted if linear: single[max] + 0.8 * single[sec]
        h_sum = single_h[max_p] + 0.8 * single_h[sec_p]

        print(f"\nmax={max_p}, 2nd={sec_p}:")
        print(f"  {'Neuron':<8} {'Actual':<10} {'Sum':<10} {'Diff':<10} {'Ratio':<10}")
        for n in comparators:
            actual = h_double[n].item()
            predicted = h_sum[n].item()
            diff = actual - predicted
            ratio = actual / predicted if predicted != 0 else float('nan')
            print(f"  n{n:<7} {actual:<10.2f} {predicted:<10.2f} {diff:<+10.2f} {ratio:<10.2f}")


def analyze_impulse_response_by_position(model):
    """
    For each (max_pos, 2nd_pos) pair, trace what happens.
    """
    print("\n" + "=" * 80)
    print("DETAILED IMPULSE RESPONSE TRACES")
    print("=" * 80)

    n = 7  # Focus on n7

    # Single impulse at position 3
    x_single = th.zeros(1, 10)
    x_single[0, 3] = 1.0
    hidden_single, pre_single = get_full_trajectory(model, x_single)

    # Double impulse: max at 3, 2nd at 7
    x_double = th.zeros(1, 10)
    x_double[0, 3] = 1.0
    x_double[0, 7] = 0.8
    hidden_double, pre_double = get_full_trajectory(model, x_double)

    print(f"\nNeuron n{n} trajectory:")
    print("-" * 70)
    print(f"{'t':<4} {'Single(3)':<12} {'Double(3,7)':<12} {'Diff':<12}")
    print("-" * 70)

    for t in range(10):
        h_s = hidden_single[0, n, t].item()
        h_d = hidden_double[0, n, t].item()
        diff = h_d - h_s
        marker = " <-- 2nd impulse" if t == 7 else ""
        print(f"{t:<4} {h_s:<12.3f} {h_d:<12.3f} {diff:<+12.3f}{marker}")

    print(f"\nAt t=9: Single={hidden_single[0, n, 9]:.3f}, "
          f"Double={hidden_double[0, n, 9]:.3f}, "
          f"Diff={hidden_double[0, n, 9] - hidden_single[0, n, 9]:.3f}")


def main():
    model = example_2nd_argmax()

    X, y, h_final, pre_act, max_pos, sec_pos, preds = analyze_clean_dataset(model)

    analyze_h_final_structure(h_final, max_pos, sec_pos)
    analyze_marginal_curves(h_final, max_pos, sec_pos)
    test_superposition_clean(h_final, max_pos, sec_pos)
    test_additive_model(h_final, max_pos, sec_pos)
    analyze_single_vs_double_impulse(model)
    analyze_impulse_response_by_position(model)

    print("\n" + "=" * 80)
    print("CONCLUSIONS FROM CLEAN DATA")
    print("=" * 80)
    print("""
With clean data (only max and 2nd non-zero):

1. We can see exactly how h_final depends on both positions
2. The additive model h = f(max) + g(2nd) can be tested precisely
3. Comparison with single impulse reveals nonlinear interactions
4. The anti-phase structure should be much clearer
""")


if __name__ == "__main__":
    main()
