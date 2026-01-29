"""
Readout Mechanism Investigation

Two competing hypotheses:
  A) Suppression model: Max is suppressed, 2nd emerges as winner
  B) Delta model: We read max position, then find delta to 2nd

Three regimes to analyze separately:
  1) Position 9 special case (max or 2nd at position 9)
  2) 2nd_argmax BEFORE argmax (two-impulse regime)
  3) 2nd_argmax AFTER argmax (single-impulse regime)

Key questions:
  - Which comparators clip for 2nd argmax (1001 pattern)?
  - Does the model confuse 3rd with 2nd in 3-2-1 ordering?
  - What does W_out actually decode?
"""

import torch as th
import numpy as np
from collections import defaultdict
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def get_full_trajectory(model, x):
    """Compute full trajectory including clipping info."""
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


def analyze_three_regimes(model, n_samples=50000):
    """Analyze the three regimes separately."""
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    # Get predictions
    hidden, pre_act = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]
    W_out = model.linear.weight.data  # 10 x 16
    logits = h_final @ W_out.T
    preds = logits.argmax(dim=-1)

    print("=" * 80)
    print("THREE-REGIME ANALYSIS")
    print("=" * 80)

    # Define regimes
    # Regime A: Position 9 involved (either max or 2nd at pos 9)
    regime_A = (argmax_pos == 9) | (targets == 9)

    # Regime B: 2nd before max (excluding pos 9 cases)
    regime_B = (targets < argmax_pos) & ~regime_A

    # Regime C: 2nd after max (excluding pos 9 cases)
    regime_C = (targets > argmax_pos) & ~regime_A

    regimes = {
        'A (pos 9 involved)': regime_A,
        'B (2nd before max)': regime_B,
        'C (2nd after max)': regime_C,
    }

    print("\nRegime breakdown:")
    print("-" * 50)
    for name, mask in regimes.items():
        count = mask.sum().item()
        acc = (preds[mask] == targets[mask]).float().mean().item() * 100
        print(f"  {name}: {count:,} samples ({count/n_samples*100:.1f}%), accuracy = {acc:.1f}%")

    return regimes, x, targets, argmax_pos, preds, h_final, logits


def analyze_logit_contributions(model, regimes, x, targets, argmax_pos, h_final):
    """
    For each regime, analyze what contributes to the logits.

    Key question: Is max SUPPRESSED, or is delta ENCODED?
    """
    W_out = model.linear.weight.data  # 10 x 16

    print("\n" + "=" * 80)
    print("LOGIT CONTRIBUTION ANALYSIS")
    print("=" * 80)

    comparators = [1, 6, 7, 8]
    wave_neurons = [10, 11, 12]
    other_neurons = [0, 2, 3, 4, 5, 9, 13, 14, 15]

    for regime_name, mask in regimes.items():
        if mask.sum() < 100:
            continue

        print(f"\n--- {regime_name} ---")

        h_regime = h_final[mask]
        targets_regime = targets[mask]
        argmax_regime = argmax_pos[mask]

        n_regime = mask.sum().item()

        # For each sample, compute contribution to:
        # 1. Logit at max position
        # 2. Logit at 2nd position
        # 3. Logit at 3rd position (to check confusion)

        # Get 3rd argmax position
        x_regime = x[mask]
        vals_sorted = x_regime.sort(dim=-1, descending=True)
        third_argmax = vals_sorted.indices[:, 2]

        logit_at_max = th.zeros(n_regime)
        logit_at_2nd = th.zeros(n_regime)
        logit_at_3rd = th.zeros(n_regime)

        # Contribution by neuron group
        contrib_at_max = {'comparators': th.zeros(n_regime),
                          'wave': th.zeros(n_regime),
                          'other': th.zeros(n_regime)}
        contrib_at_2nd = {'comparators': th.zeros(n_regime),
                          'wave': th.zeros(n_regime),
                          'other': th.zeros(n_regime)}

        for i in range(n_regime):
            max_p = argmax_regime[i].item()
            sec_p = targets_regime[i].item()
            thi_p = third_argmax[i].item()

            logit_at_max[i] = (h_regime[i] * W_out[max_p]).sum()
            logit_at_2nd[i] = (h_regime[i] * W_out[sec_p]).sum()
            logit_at_3rd[i] = (h_regime[i] * W_out[thi_p]).sum()

            for n in comparators:
                contrib_at_max['comparators'][i] += h_regime[i, n] * W_out[max_p, n]
                contrib_at_2nd['comparators'][i] += h_regime[i, n] * W_out[sec_p, n]
            for n in wave_neurons:
                contrib_at_max['wave'][i] += h_regime[i, n] * W_out[max_p, n]
                contrib_at_2nd['wave'][i] += h_regime[i, n] * W_out[sec_p, n]
            for n in other_neurons:
                contrib_at_max['other'][i] += h_regime[i, n] * W_out[max_p, n]
                contrib_at_2nd['other'][i] += h_regime[i, n] * W_out[sec_p, n]

        print(f"\n  Mean logit at MAX position:  {logit_at_max.mean():.2f}")
        print(f"  Mean logit at 2ND position:  {logit_at_2nd.mean():.2f}")
        print(f"  Mean logit at 3RD position:  {logit_at_3rd.mean():.2f}")
        print(f"  Gap (2nd - max):             {(logit_at_2nd - logit_at_max).mean():.2f}")
        print(f"  Gap (2nd - 3rd):             {(logit_at_2nd - logit_at_3rd).mean():.2f}")

        print(f"\n  Contribution breakdown (to max vs 2nd):")
        print(f"  {'Group':<12} | {'At MAX':<10} | {'At 2ND':<10} | {'Diff':<10}")
        print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        for group in ['comparators', 'wave', 'other']:
            m_max = contrib_at_max[group].mean().item()
            m_2nd = contrib_at_2nd[group].mean().item()
            diff = m_2nd - m_max
            print(f"  {group:<12} | {m_max:>+9.2f} | {m_2nd:>+9.2f} | {diff:>+9.2f}")


def analyze_ordering_confusion(model, n_samples=100000):
    """
    Specifically test the 3-2-1 ordering prediction.
    Does the model confuse 2nd with 3rd?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    # Get ordering
    vals_sorted = x.sort(dim=-1, descending=True)
    pos_1st = vals_sorted.indices[:, 0]  # max position
    pos_2nd = vals_sorted.indices[:, 1]  # 2nd max position
    pos_3rd = vals_sorted.indices[:, 2]  # 3rd max position

    # Determine temporal order
    # 3-2-1 means: 3rd appears first, then 2nd, then max
    ordering_321 = (pos_3rd < pos_2nd) & (pos_2nd < pos_1st)
    ordering_213 = (pos_2nd < pos_1st) & (pos_1st < pos_3rd)
    ordering_123 = (pos_1st < pos_2nd) & (pos_2nd < pos_3rd)

    # Get predictions
    hidden, _ = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]
    W_out = model.linear.weight.data
    logits = h_final @ W_out.T
    preds = logits.argmax(dim=-1)

    print("\n" + "=" * 80)
    print("ORDERING CONFUSION ANALYSIS")
    print("=" * 80)

    orderings = {
        '3-2-1 (worst)': ordering_321,
        '2-1-3 (best)': ordering_213,
        '1-2-3': ordering_123,
    }

    for name, mask in orderings.items():
        if mask.sum() < 100:
            continue

        n = mask.sum().item()
        correct = (preds[mask] == targets[mask])
        acc = correct.float().mean().item() * 100

        # Among errors, what does model predict?
        errors = ~correct
        if errors.sum() > 0:
            error_preds = preds[mask][errors]
            error_targets = targets[mask][errors]
            error_3rd = pos_3rd[mask][errors]
            error_max = pos_1st[mask][errors]

            pred_is_3rd = (error_preds == error_3rd).float().mean().item() * 100
            pred_is_max = (error_preds == error_max).float().mean().item() * 100
            pred_is_other = 100 - pred_is_3rd - pred_is_max

            print(f"\n{name}:")
            print(f"  Count: {n:,}, Accuracy: {acc:.1f}%")
            print(f"  When wrong, predicts:")
            print(f"    - 3rd position: {pred_is_3rd:.1f}%")
            print(f"    - max position: {pred_is_max:.1f}%")
            print(f"    - other:        {pred_is_other:.1f}%")
        else:
            print(f"\n{name}:")
            print(f"  Count: {n:,}, Accuracy: {acc:.1f}%")


def analyze_comparator_clipping_patterns(model, n_samples=50000):
    """
    Analyze which comparators clip for 2nd argmax.
    Focus on the 1001 pattern mentioned.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden, pre_act = get_full_trajectory(model, x)

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("COMPARATOR CLIPPING PATTERNS AT 2ND ARGMAX POSITION")
    print("=" * 80)

    # For samples where 2nd comes BEFORE max (so it might clip)
    before_max = targets < argmax_pos

    # Build clipping pattern at 2nd_argmax timestep
    patterns = defaultdict(int)
    pattern_correct = defaultdict(int)

    # Get predictions
    h_final = hidden[:, :, 9]
    W_out = model.linear.weight.data
    preds = (h_final @ W_out.T).argmax(dim=-1)

    for i in range(n_samples):
        if not before_max[i]:
            continue

        t = targets[i].item()
        pattern = ""
        for n in comparators:
            clipped = pre_act[i, n, t].item() < 0
            pattern += "1" if clipped else "0"

        patterns[pattern] += 1
        if preds[i] == targets[i]:
            pattern_correct[pattern] += 1

    print(f"\n2nd BEFORE max samples: {before_max.sum().item():,}")
    print("\nClipping patterns (n1, n6, n7, n8):")
    print("-" * 60)
    print(f"{'Pattern':<10} | {'Count':<10} | {'Accuracy':<10} | {'Interpretation'}")
    print("-" * 60)

    for pattern in sorted(patterns.keys(), key=lambda p: -patterns[p]):
        count = patterns[pattern]
        correct = pattern_correct[pattern]
        acc = correct / count * 100 if count > 0 else 0

        # Interpret pattern
        if pattern == "1111":
            interp = "All clip (very large 2nd)"
        elif pattern == "0000":
            interp = "None clip (small 2nd)"
        elif pattern == "1001":
            interp = "n1+n8 clip (threshold cascade)"
        elif pattern == "0001":
            interp = "Only n8 clips"
        elif pattern == "1000":
            interp = "Only n1 clips"
        else:
            interp = ""

        if count >= 50:
            print(f"{pattern:<10} | {count:<10,} | {acc:>8.1f}% | {interp}")


def analyze_wout_structure(model):
    """
    Analyze what W_out actually encodes.

    Hypothesis A: Sinusoidal templates for position
    Hypothesis B: Max-suppression + delta encoding
    """
    W_out = model.linear.weight.data.numpy()  # 10 x 16

    print("\n" + "=" * 80)
    print("W_OUT STRUCTURE ANALYSIS")
    print("=" * 80)

    comparators = [1, 6, 7, 8]

    # For each comparator, look at its W_out column
    print("\nComparator W_out columns (contribution to each output position):")
    print("-" * 70)

    for n in comparators:
        col = W_out[:, n]
        print(f"\nn{n}: ", end="")
        for pos in range(10):
            print(f"{col[pos]:+5.1f}", end=" ")
        print()

        # DFT analysis
        fft = np.fft.fft(col)
        energy = np.abs(fft) ** 2
        total_energy = energy.sum()

        print(f"     DFT energy: k=0: {energy[0]/total_energy*100:.0f}%, "
              f"k=1: {energy[1]/total_energy*100:.0f}%, "
              f"k=2: {energy[2]/total_energy*100:.0f}%")

    print("\n" + "-" * 70)
    print("\nMean W_out by position (all comparators summed):")
    comparator_sum = W_out[:, comparators].sum(axis=1)
    for pos in range(10):
        bar = "█" * int(abs(comparator_sum[pos]) * 2)
        sign = "+" if comparator_sum[pos] > 0 else "-"
        print(f"  pos {pos}: {comparator_sum[pos]:+6.1f} {sign}{bar}")


def test_suppression_vs_delta_hypothesis(model, n_samples=50000):
    """
    Direct test of the two hypotheses.

    Hypothesis A (Suppression): The model suppresses max position and
    2nd emerges as winner.

    Hypothesis B (Delta): The model encodes (max_pos, delta) and reads
    out 2nd = max_pos + delta.

    Test: If we manually set the logit at max position to -infinity,
    does accuracy improve or stay the same?
    - If A is correct: accuracy stays same (max already suppressed)
    - If B is correct: accuracy drops (we're breaking the delta encoding)
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden, _ = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]
    W_out = model.linear.weight.data
    logits = h_final @ W_out.T

    print("\n" + "=" * 80)
    print("SUPPRESSION VS DELTA HYPOTHESIS TEST")
    print("=" * 80)

    # Normal prediction
    preds_normal = logits.argmax(dim=-1)
    acc_normal = (preds_normal == targets).float().mean().item() * 100

    # Force-suppress max position
    logits_suppressed = logits.clone()
    for i in range(n_samples):
        logits_suppressed[i, argmax_pos[i]] = -1000

    preds_suppressed = logits_suppressed.argmax(dim=-1)
    acc_suppressed = (preds_suppressed == targets).float().mean().item() * 100

    print(f"\nNormal accuracy:             {acc_normal:.1f}%")
    print(f"With max forcibly suppressed: {acc_suppressed:.1f}%")
    print(f"Difference:                  {acc_suppressed - acc_normal:+.1f}%")

    # How often does normal prediction equal max position?
    pred_is_max = (preds_normal == argmax_pos).float().mean().item() * 100
    print(f"\nNormal model predicts MAX position: {pred_is_max:.1f}% of time")

    # Break down by regime
    regime_B = targets < argmax_pos  # 2nd before max
    regime_C = targets > argmax_pos  # 2nd after max

    for name, mask in [("2nd before max", regime_B), ("2nd after max", regime_C)]:
        if mask.sum() < 100:
            continue
        acc_n = (preds_normal[mask] == targets[mask]).float().mean().item() * 100
        acc_s = (preds_suppressed[mask] == targets[mask]).float().mean().item() * 100
        pred_max = (preds_normal[mask] == argmax_pos[mask]).float().mean().item() * 100
        print(f"\n  {name}:")
        print(f"    Normal: {acc_n:.1f}%, Suppressed: {acc_s:.1f}%, Delta: {acc_s-acc_n:+.1f}%")
        print(f"    Predicts max: {pred_max:.1f}%")

    print("""
Interpretation:
  - If accuracy INCREASES with forced suppression: model sometimes
    fails to suppress max, so we're helping it
  - If accuracy STAYS SAME: model already suppresses max perfectly
  - If accuracy DECREASES: we're breaking something (delta encoding?)
""")


def analyze_position_9_special_case(model, n_samples=50000):
    """
    Analyze position 9 as a special case.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden, _ = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]
    W_out = model.linear.weight.data
    logits = h_final @ W_out.T
    preds = logits.argmax(dim=-1)

    print("\n" + "=" * 80)
    print("POSITION 9 SPECIAL CASE")
    print("=" * 80)

    # Case 1: Max at position 9
    max_at_9 = argmax_pos == 9
    if max_at_9.sum() > 0:
        acc = (preds[max_at_9] == targets[max_at_9]).float().mean().item() * 100
        print(f"\nMax at position 9:")
        print(f"  Count: {max_at_9.sum().item():,}")
        print(f"  Accuracy: {acc:.1f}%")

        # What does it predict?
        pred_dist = preds[max_at_9].bincount(minlength=10).float()
        pred_dist = pred_dist / pred_dist.sum() * 100
        print(f"  Prediction distribution: ", end="")
        for p in range(10):
            if pred_dist[p] > 1:
                print(f"pos{p}:{pred_dist[p]:.0f}% ", end="")
        print()

        # Comparator h_final
        comp_h = h_final[max_at_9][:, [1, 6, 7, 8]].mean(dim=0)
        print(f"  Mean comparator h_final: n1={comp_h[0]:.2f}, n6={comp_h[1]:.2f}, "
              f"n7={comp_h[2]:.2f}, n8={comp_h[3]:.2f}")

    # Case 2: 2nd at position 9
    sec_at_9 = targets == 9
    if sec_at_9.sum() > 0:
        acc = (preds[sec_at_9] == targets[sec_at_9]).float().mean().item() * 100
        print(f"\n2nd at position 9:")
        print(f"  Count: {sec_at_9.sum().item():,}")
        print(f"  Accuracy: {acc:.1f}%")

    # Case 3: Neither at 9
    neither_at_9 = ~max_at_9 & ~sec_at_9
    if neither_at_9.sum() > 0:
        acc = (preds[neither_at_9] == targets[neither_at_9]).float().mean().item() * 100
        print(f"\nNeither at position 9:")
        print(f"  Count: {neither_at_9.sum().item():,}")
        print(f"  Accuracy: {acc:.1f}%")


def main():
    model = example_2nd_argmax()

    # Main analysis
    regimes, x, targets, argmax_pos, preds, h_final, logits = analyze_three_regimes(model)

    analyze_logit_contributions(model, regimes, x, targets, argmax_pos, h_final)

    analyze_ordering_confusion(model)

    analyze_comparator_clipping_patterns(model)

    analyze_wout_structure(model)

    test_suppression_vs_delta_hypothesis(model)

    analyze_position_9_special_case(model)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key questions answered:
1. Is max SUPPRESSED or is DELTA encoded?
2. Are the three regimes handled differently?
3. Does 3-2-1 ordering cause 3rd vs 2nd confusion?
4. What clipping patterns occur at 2nd argmax?
5. How is position 9 special?
""")


if __name__ == "__main__":
    main()
