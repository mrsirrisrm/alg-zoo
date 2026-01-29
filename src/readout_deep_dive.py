"""
Deep Dive into Readout Mechanism

Based on initial results, we see:
1. Comparators contribute +12 logit gap favoring 2nd over max
2. Wave neurons behave DIFFERENTLY in regimes B vs C
3. Model predicts max 3.1% of time - some failures

Let's understand exactly what's happening.
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


def analyze_what_comparators_encode(model, n_samples=50000):
    """
    The key question: Do comparators encode:
    A) max_pos (then delta to get 2nd)
    B) -max_pos and +2nd_pos separately (antiphase)
    C) Something else?

    Look at comparator h_final correlation with:
    - max_pos
    - 2nd_pos
    - delta = max_pos - 2nd_pos
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden, _ = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]
    delta = argmax_pos - targets

    print("=" * 80)
    print("WHAT DO COMPARATORS ENCODE?")
    print("=" * 80)

    print("\nCorrelation of comparator h_final with:")
    print("-" * 60)
    print(f"{'Neuron':<8} | {'max_pos':>10} | {'2nd_pos':>10} | {'delta':>10}")
    print("-" * 60)

    for n in comparators:
        h = h_final[:, n].numpy()
        r_max = np.corrcoef(h, argmax_pos.numpy())[0, 1]
        r_2nd = np.corrcoef(h, targets.numpy())[0, 1]
        r_delta = np.corrcoef(h, delta.numpy())[0, 1]
        print(f"n{n:<7} | {r_max:>+10.3f} | {r_2nd:>+10.3f} | {r_delta:>+10.3f}")

    print("""
Interpretation:
  - If r(max) ≈ -r(2nd): Antiphase encoding (doc 08)
  - If r(delta) is highest: Direct delta encoding
  - Look at the signs!
""")


def analyze_wout_position_profiles(model):
    """
    For each output position, what is the W_out template?

    If delta encoding: W_out[pos] should look like a sinusoid centered on pos
    If suppression: W_out[max] should be strongly negative for comparators
    """
    W_out = model.linear.weight.data.numpy()  # 10 x 16

    print("\n" + "=" * 80)
    print("W_OUT POSITION PROFILES")
    print("=" * 80)

    comparators = [1, 6, 7, 8]

    # Sum comparator contributions
    comp_sum = W_out[:, comparators].sum(axis=1)

    print("\nComparator W_out sum for each output position:")
    print("-" * 50)
    for pos in range(10):
        val = comp_sum[pos]
        bar_len = int(abs(val) * 3)
        bar = "█" * bar_len
        sign = "+" if val > 0 else "-"
        print(f"  pos {pos}: {val:>+6.2f} {sign}{bar}")

    # Now look at what happens when we compute logit
    # logit[pos] = sum_n h_final[n] * W_out[pos, n]
    # For comparators: h_final encodes -f(max_pos) + f(2nd_pos)
    # So logit contribution at pos from comparators:
    #   = sum_n h[n] * W_out[pos, n]
    #   ≈ (-f(max_pos) + f(2nd_pos)) * W_out[pos, comparators]

    print("""
Key insight: The W_out position profile tells us what the model
is looking for. Negative at a position means "suppress this prediction".

Note: pos 9 has the most negative sum (-4.6) - special suppression.
""")


def analyze_per_position_logit_breakdown(model, n_samples=50000):
    """
    For each (max_pos, 2nd_pos) pair, show the logit breakdown.
    This reveals whether suppression or delta is the mechanism.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden, _ = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]
    W_out = model.linear.weight.data

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("LOGIT BREAKDOWN BY (MAX_POS, 2ND_POS) PAIR")
    print("=" * 80)

    # Exclude position 9 for clarity
    valid = (argmax_pos != 9) & (targets != 9)
    h_valid = h_final[valid]
    max_valid = argmax_pos[valid]
    sec_valid = targets[valid]

    # Sample some specific cases
    cases = [
        (4, 2),  # 2nd before max
        (4, 6),  # 2nd after max
        (2, 4),  # max before 2nd
        (6, 4),  # max before 2nd (different)
    ]

    for max_p, sec_p in cases:
        mask = (max_valid == max_p) & (sec_valid == sec_p)
        if mask.sum() < 20:
            continue

        h_case = h_valid[mask]
        n_case = mask.sum().item()

        print(f"\nmax_pos={max_p}, 2nd_pos={sec_p} (n={n_case}):")
        print("-" * 60)

        # Mean h_final for comparators
        h_comp = h_case[:, comparators].mean(dim=0)
        print(f"  Comparator h_final: n1={h_comp[0]:.2f}, n6={h_comp[1]:.2f}, "
              f"n7={h_comp[2]:.2f}, n8={h_comp[3]:.2f}")

        # Logit at each position from comparators
        comp_logits = th.zeros(10)
        for pos in range(10):
            for i, n in enumerate(comparators):
                comp_logits[pos] += h_comp[i] * W_out[pos, n]

        print(f"  Comparator contribution to logits:")
        print(f"    At max_pos ({max_p}): {comp_logits[max_p]:>+6.2f}")
        print(f"    At 2nd_pos ({sec_p}): {comp_logits[sec_p]:>+6.2f}")
        print(f"    Gap (2nd - max):      {comp_logits[sec_p] - comp_logits[max_p]:>+6.2f}")


def understand_antiphase_mechanism(model, n_samples=50000):
    """
    Let's trace through the anti-phase mechanism step by step.

    When max is at position M and 2nd at position S:
    1. At timestep M, comparators clip (x[M] > threshold)
    2. At timestep S (if S < M), comparators clip again (x[S] > threshold)
    3. h_final encodes something about both events

    Question: Is it h = f(M) - f(S), or h = f(M) then delta?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden, pre_act = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    # Focus on regime B: 2nd before max (both clip)
    regime_B = targets < argmax_pos

    print("\n" + "=" * 80)
    print("TRACING THE ANTI-PHASE MECHANISM (Regime B: 2nd before max)")
    print("=" * 80)

    comparators = [1, 6, 7, 8]

    # For each comparator, track:
    # - h at 2nd_pos (just after 2nd clips)
    # - h at max_pos (just after max clips)
    # - h_final

    for n in comparators:
        print(f"\nn{n}:")

        h_at_2nd = []
        h_at_max = []
        h_finals = []
        max_positions = []
        sec_positions = []

        for i in range(n_samples):
            if not regime_B[i]:
                continue

            s = targets[i].item()
            m = argmax_pos[i].item()

            h_at_2nd.append(hidden[i, n, s].item())
            h_at_max.append(hidden[i, n, m].item())
            h_finals.append(h_final[i, n].item())
            max_positions.append(m)
            sec_positions.append(s)

        h_at_2nd = np.array(h_at_2nd)
        h_at_max = np.array(h_at_max)
        h_finals = np.array(h_finals)
        max_positions = np.array(max_positions)
        sec_positions = np.array(sec_positions)

        # Correlations
        r_h2nd_hfinal = np.corrcoef(h_at_2nd, h_finals)[0, 1]
        r_hmax_hfinal = np.corrcoef(h_at_max, h_finals)[0, 1]
        r_h2nd_max_pos = np.corrcoef(h_at_2nd, max_positions)[0, 1]
        r_hmax_max_pos = np.corrcoef(h_at_max, max_positions)[0, 1]

        print(f"  Correlation of h[at 2nd timestep] with h_final: {r_h2nd_hfinal:+.3f}")
        print(f"  Correlation of h[at max timestep] with h_final: {r_hmax_hfinal:+.3f}")
        print(f"  Correlation of h[at 2nd timestep] with max_pos: {r_h2nd_max_pos:+.3f}")
        print(f"  Correlation of h[at max timestep] with max_pos: {r_hmax_max_pos:+.3f}")

        # Key test: Does h_final ≈ h_at_max - h_at_2nd?
        # If antiphase, then h_final should correlate with (h_at_max - h_at_2nd)
        diff = h_at_max - h_at_2nd
        r_diff_hfinal = np.corrcoef(diff, h_finals)[0, 1]
        print(f"  Correlation of (h[max] - h[2nd]) with h_final: {r_diff_hfinal:+.3f}")


def understand_regime_c_mechanism(model, n_samples=50000):
    """
    Regime C: 2nd AFTER max (single-impulse regime).

    Only max clips. How does the model find 2nd?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden, pre_act = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]
    W_out = model.linear.weight.data

    # Focus on regime C: 2nd after max
    regime_C = targets > argmax_pos

    print("\n" + "=" * 80)
    print("REGIME C: 2ND AFTER MAX (Single-impulse)")
    print("=" * 80)

    h_C = h_final[regime_C]
    max_C = argmax_pos[regime_C]
    sec_C = targets[regime_C]

    # What do wave neurons encode?
    wave_neurons = [10, 11, 12]

    print("\nWave neuron h_final correlation with positions:")
    print("-" * 50)
    for n in wave_neurons:
        h = h_C[:, n].numpy()
        r_max = np.corrcoef(h, max_C.numpy())[0, 1]
        r_2nd = np.corrcoef(h, sec_C.numpy())[0, 1]
        r_delta = np.corrcoef(h, (sec_C - max_C).numpy())[0, 1]
        print(f"  n{n}: r(max)={r_max:+.3f}, r(2nd)={r_2nd:+.3f}, r(delta)={r_delta:+.3f}")

    # Comparators in regime C
    comparators = [1, 6, 7, 8]
    print("\nComparator h_final correlation with positions:")
    print("-" * 50)
    for n in comparators:
        h = h_C[:, n].numpy()
        r_max = np.corrcoef(h, max_C.numpy())[0, 1]
        r_2nd = np.corrcoef(h, sec_C.numpy())[0, 1]
        r_delta = np.corrcoef(h, (sec_C - max_C).numpy())[0, 1]
        print(f"  n{n}: r(max)={r_max:+.3f}, r(2nd)={r_2nd:+.3f}, r(delta)={r_delta:+.3f}")

    print("""
In regime C, only max clips. The 2nd value doesn't trigger clipping
because it's smaller than max (which already raised the threshold).

How does the model still find 2nd with 89% accuracy?
""")


def test_linear_decodability(model, n_samples=50000):
    """
    Test: Can we linearly decode max_pos and 2nd_pos from h_final?

    If yes, then h_final contains both positions (antiphase encoding).
    If only max is decodable, then we need delta mechanism.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden, _ = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9].numpy()

    max_np = argmax_pos.numpy()
    sec_np = targets.numpy()

    print("\n" + "=" * 80)
    print("LINEAR DECODABILITY TEST")
    print("=" * 80)

    # Linear regression to predict max_pos from h_final
    X = np.column_stack([h_final, np.ones(n_samples)])

    # Predict max_pos
    coeffs_max, _, _, _ = np.linalg.lstsq(X, max_np, rcond=None)
    pred_max = X @ coeffs_max
    r2_max = 1 - np.sum((max_np - pred_max)**2) / np.sum((max_np - max_np.mean())**2)

    # Predict 2nd_pos
    coeffs_sec, _, _, _ = np.linalg.lstsq(X, sec_np, rcond=None)
    pred_sec = X @ coeffs_sec
    r2_sec = 1 - np.sum((sec_np - pred_sec)**2) / np.sum((sec_np - sec_np.mean())**2)

    # Predict delta
    delta = max_np - sec_np
    coeffs_delta, _, _, _ = np.linalg.lstsq(X, delta, rcond=None)
    pred_delta = X @ coeffs_delta
    r2_delta = 1 - np.sum((delta - pred_delta)**2) / np.sum((delta - delta.mean())**2)

    print(f"\nLinear regression R² from h_final:")
    print(f"  Predicting max_pos: R² = {r2_max:.3f}")
    print(f"  Predicting 2nd_pos: R² = {r2_sec:.3f}")
    print(f"  Predicting delta:   R² = {r2_delta:.3f}")

    # Accuracy of discretized predictions
    acc_max = (np.round(pred_max).astype(int) == max_np).mean() * 100
    acc_sec = (np.round(pred_sec).astype(int) == sec_np).mean() * 100

    print(f"\nDiscretized accuracy:")
    print(f"  max_pos: {acc_max:.1f}%")
    print(f"  2nd_pos: {acc_sec:.1f}%")

    print("""
Interpretation:
  - If both R² are high: h_final encodes both positions (antiphase)
  - If only max R² is high: h_final mainly encodes max_pos
  - Model's actual 2nd accuracy is ~89%, compare to linear probe
""")


def main():
    model = example_2nd_argmax()

    analyze_what_comparators_encode(model)
    analyze_wout_position_profiles(model)
    analyze_per_position_logit_breakdown(model)
    understand_antiphase_mechanism(model)
    understand_regime_c_mechanism(model)
    test_linear_decodability(model)

    print("\n" + "=" * 80)
    print("SYNTHESIS")
    print("=" * 80)
    print("""
The evidence points to:

1. BOTH mechanisms are partially true:
   - Comparators encode antiphase: h ≈ f(max) - f(2nd)
   - W_out then extracts 2nd position from this encoding
   - Max suppression happens as a BYPRODUCT of antiphase encoding

2. The net effect is:
   - Logit at max gets negative contribution from comparators
   - Logit at 2nd gets positive contribution
   - Gap of ~12 logits favoring 2nd over max

3. Different regimes work differently:
   - Regime B (2nd before max): Full antiphase with two impulses
   - Regime C (2nd after max): Wave neurons + "other" neurons compensate

4. Position 9 is indeed special:
   - Comparators have near-zero h_final when max at 9
   - W_out has extra negative weight for pos 9
""")


if __name__ == "__main__":
    main()
