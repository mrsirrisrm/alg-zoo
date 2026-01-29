"""
Final Readout Analysis

Key finding from previous analysis:
- Comparator h_final has LOW correlation with both max_pos and 2nd_pos
- But the LOGIT GAP is huge (+12 between 2nd and max)
- This suggests W_out is doing the heavy lifting

Let's understand exactly what's happening.
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
    h = th.zeros(batch_size, 16)

    for t in range(10):
        x_t = x[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
        hidden[:, :, t] = h

    return hidden


def analyze_position_conditional_encoding(model, n_samples=100000):
    """
    The key insight: h_final correlation with max_pos is low when averaged
    over all samples. But what about CONDITIONAL on a specific max_pos?

    This is what doc 08 was measuring: the CURVES h_final(max_pos) and
    h_final(2nd_pos) are sinusoidal and anti-phase.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]

    print("=" * 80)
    print("POSITION-CONDITIONAL ENCODING")
    print("=" * 80)

    print("\nMean h_final for each comparator, BY MAX POSITION:")
    print("-" * 70)
    print(f"{'max_pos':<8}", end="")
    for n in comparators:
        print(f"n{n:<7}", end="")
    print()
    print("-" * 70)

    h_by_max = {n: [] for n in comparators}
    for pos in range(10):
        mask = argmax_pos == pos
        print(f"{pos:<8}", end="")
        for n in comparators:
            mean_h = h_final[mask, n].mean().item()
            h_by_max[n].append(mean_h)
            print(f"{mean_h:<8.2f}", end="")
        print()

    print("\n" + "-" * 70)
    print("\nMean h_final for each comparator, BY 2ND POSITION:")
    print("-" * 70)
    print(f"{'2nd_pos':<8}", end="")
    for n in comparators:
        print(f"n{n:<7}", end="")
    print()
    print("-" * 70)

    h_by_2nd = {n: [] for n in comparators}
    for pos in range(10):
        mask = targets == pos
        print(f"{pos:<8}", end="")
        for n in comparators:
            mean_h = h_final[mask, n].mean().item()
            h_by_2nd[n].append(mean_h)
            print(f"{mean_h:<8.2f}", end="")
        print()

    # Check anti-phase relationship
    print("\n" + "=" * 80)
    print("ANTI-PHASE CHECK: Correlation between h(max_pos) and h(2nd_pos) curves")
    print("-" * 70)
    for n in comparators:
        curve_max = np.array(h_by_max[n][:9])  # Exclude pos 9
        curve_2nd = np.array(h_by_2nd[n][:9])
        corr = np.corrcoef(curve_max, curve_2nd)[0, 1]
        print(f"  n{n}: r = {corr:+.3f}")

    return h_by_max, h_by_2nd


def analyze_wout_action(model, n_samples=100000):
    """
    Given the h_final curves, how does W_out create the gap?

    logit[pos] = sum_n h_final[n] * W_out[pos, n]

    If h_final varies with max_pos and 2nd_pos in anti-phase,
    and W_out is sinusoidal, the product creates selective amplification.
    """
    W_out = model.linear.weight.data.numpy()

    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("W_OUT SELECTIVITY ANALYSIS")
    print("=" * 80)

    # For a specific (max_pos, 2nd_pos) pair, trace the logit contribution
    test_pairs = [(4, 2), (2, 4), (6, 3), (3, 6)]

    for max_p, sec_p in test_pairs:
        mask = (argmax_pos == max_p) & (targets == sec_p)
        if mask.sum() < 50:
            continue

        h_case = h_final[mask].mean(dim=0).numpy()  # Mean h_final for this case

        print(f"\nmax={max_p}, 2nd={sec_p}:")
        print("-" * 60)

        # Compute logit at each output position from comparators
        logits_comp = np.zeros(10)
        for pos in range(10):
            for n in comparators:
                logits_comp[pos] += h_case[n] * W_out[pos, n]

        # Show key positions
        print(f"  Comparator logit at max ({max_p}):   {logits_comp[max_p]:>+6.2f}")
        print(f"  Comparator logit at 2nd ({sec_p}):   {logits_comp[sec_p]:>+6.2f}")
        print(f"  Gap:                                {logits_comp[sec_p] - logits_comp[max_p]:>+6.2f}")

        # Break down by neuron
        print(f"\n  Per-neuron breakdown (at max vs 2nd):")
        for n in comparators:
            contrib_max = h_case[n] * W_out[max_p, n]
            contrib_2nd = h_case[n] * W_out[sec_p, n]
            diff = contrib_2nd - contrib_max
            print(f"    n{n}: h={h_case[n]:.2f}, W[max]={W_out[max_p,n]:+.2f}, "
                  f"W[2nd]={W_out[sec_p,n]:+.2f} → contrib_diff={diff:+.2f}")


def analyze_the_sinusoidal_trick(model, n_samples=100000):
    """
    The core mechanism:

    1. h_final[n] varies sinusoidally with max_pos (call it f_max)
    2. h_final[n] varies sinusoidally with 2nd_pos (call it f_2nd)
    3. These are ANTI-PHASE (doc 08 showed r ≈ -0.9)

    4. W_out[pos, n] is also sinusoidal in pos

    5. logit[pos] = sum_n h_final[n] * W_out[pos, n]
                  = sum_n (A*f_max(max_pos) - B*f_2nd(2nd_pos)) * sin(pos)

    When pos = max_pos:
        - The max term has f_max(max_pos) * sin(max_pos) = self-product
        - The 2nd term has f_2nd(2nd_pos) * sin(max_pos) = cross-product

    When pos = 2nd_pos:
        - The max term has f_max(max_pos) * sin(2nd_pos) = cross-product
        - The 2nd term has f_2nd(2nd_pos) * sin(2nd_pos) = self-product

    The anti-phase means these work out to suppress max and boost 2nd!
    """
    W_out = model.linear.weight.data.numpy()

    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    print("\n" + "=" * 80)
    print("THE SINUSOIDAL TRICK: WHY ANTI-PHASE + SINUSOIDAL W_OUT WORKS")
    print("=" * 80)

    # For one comparator, show the mechanism
    n = 7  # Most important comparator

    # Get h_final curve by max_pos
    h_vs_max = []
    for pos in range(10):
        mask = argmax_pos == pos
        h_vs_max.append(h_final[mask, n].mean().item())
    h_vs_max = np.array(h_vs_max)

    # Get h_final curve by 2nd_pos
    h_vs_2nd = []
    for pos in range(10):
        mask = targets == pos
        h_vs_2nd.append(h_final[mask, n].mean().item())
    h_vs_2nd = np.array(h_vs_2nd)

    # W_out for n7
    w = W_out[:, n]

    print(f"\nNeuron n{n}:")
    print("-" * 60)
    print(f"{'pos':<6}{'h(max)':<10}{'h(2nd)':<10}{'W_out':<10}{'h*W(max)':<12}{'h*W(2nd)':<12}")
    print("-" * 60)

    for pos in range(10):
        h_m = h_vs_max[pos]
        h_s = h_vs_2nd[pos]
        w_p = w[pos]
        prod_m = h_m * w_p
        prod_s = h_s * w_p
        print(f"{pos:<6}{h_m:<10.2f}{h_s:<10.2f}{w_p:<+10.2f}{prod_m:<+12.2f}{prod_s:<+12.2f}")

    print("""
Key insight: When computing logit[pos]:
- The product h_final * W_out[pos] varies based on BOTH max_pos AND 2nd_pos
- Anti-phase curves mean when max_pos = pos, h_final is LOW → negative contribution
- But W_out is tuned so that 2nd_pos = pos gets positive contribution

The sinusoidal W_out acts as a MATCHED FILTER that:
1. Suppresses the max position (negative product)
2. Boosts the 2nd position (positive product)

This is NOT simple max suppression - it's INTERFERENCE-BASED position encoding!
""")


def analyze_full_logit_decomposition(model, n_samples=50000):
    """
    Full decomposition of logits to understand the mechanism.
    """
    W_out = model.linear.weight.data

    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    logits = h_final @ W_out.T  # n_samples x 10
    preds = logits.argmax(dim=-1)

    print("\n" + "=" * 80)
    print("FULL LOGIT DECOMPOSITION")
    print("=" * 80)

    # For correct predictions, what's the logit ranking?
    correct = preds == targets

    # Get logit at: max_pos, 2nd_pos, 3rd_pos
    vals_sorted = x.sort(dim=-1, descending=True)
    pos_3rd = vals_sorted.indices[:, 2]

    logit_at_max = th.zeros(n_samples)
    logit_at_2nd = th.zeros(n_samples)
    logit_at_3rd = th.zeros(n_samples)

    for i in range(n_samples):
        logit_at_max[i] = logits[i, argmax_pos[i]]
        logit_at_2nd[i] = logits[i, targets[i]]
        logit_at_3rd[i] = logits[i, pos_3rd[i]]

    print(f"\nCorrect predictions (n={correct.sum().item()}):")
    print(f"  Mean logit at MAX:  {logit_at_max[correct].mean():.2f}")
    print(f"  Mean logit at 2ND:  {logit_at_2nd[correct].mean():.2f}")
    print(f"  Mean logit at 3RD:  {logit_at_3rd[correct].mean():.2f}")
    print(f"  Gap (2nd - max):    {(logit_at_2nd[correct] - logit_at_max[correct]).mean():.2f}")

    print(f"\nIncorrect predictions (n={(~correct).sum().item()}):")
    print(f"  Mean logit at MAX:  {logit_at_max[~correct].mean():.2f}")
    print(f"  Mean logit at 2ND:  {logit_at_2nd[~correct].mean():.2f}")
    print(f"  Mean logit at 3RD:  {logit_at_3rd[~correct].mean():.2f}")
    print(f"  Gap (2nd - max):    {(logit_at_2nd[~correct] - logit_at_max[~correct]).mean():.2f}")

    # When wrong, what wins?
    print(f"\nWhen wrong, prediction is:")
    error_pred = preds[~correct]
    error_max = argmax_pos[~correct]
    error_3rd = pos_3rd[~correct]
    error_2nd = targets[~correct]

    print(f"  - MAX position: {(error_pred == error_max).float().mean()*100:.1f}%")
    print(f"  - 3RD position: {(error_pred == error_3rd).float().mean()*100:.1f}%")
    print(f"  - Other:        {((error_pred != error_max) & (error_pred != error_3rd) & (error_pred != error_2nd)).float().mean()*100:.1f}%")


def analyze_regime_specific_mechanism(model, n_samples=100000):
    """
    Are regimes B and C using DIFFERENT mechanisms?
    """
    W_out = model.linear.weight.data

    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    # Define regimes (excluding pos 9)
    regime_B = (targets < argmax_pos) & (argmax_pos != 9) & (targets != 9)
    regime_C = (targets > argmax_pos) & (argmax_pos != 9) & (targets != 9)

    comparators = [1, 6, 7, 8]
    wave_neurons = [10, 11, 12]

    print("\n" + "=" * 80)
    print("REGIME-SPECIFIC MECHANISM ANALYSIS")
    print("=" * 80)

    for name, mask in [("Regime B (2nd before max)", regime_B),
                       ("Regime C (2nd after max)", regime_C)]:

        print(f"\n{name} (n={mask.sum().item()}):")
        print("-" * 60)

        h = h_final[mask]
        max_p = argmax_pos[mask]
        sec_p = targets[mask]

        # Mean h_final by neuron
        comp_h = h[:, comparators].mean(dim=0)
        wave_h = h[:, wave_neurons].mean(dim=0)

        print(f"  Mean comparator h_final: n1={comp_h[0]:.2f}, n6={comp_h[1]:.2f}, "
              f"n7={comp_h[2]:.2f}, n8={comp_h[3]:.2f}")
        print(f"  Mean wave h_final: n10={wave_h[0]:.2f}, n11={wave_h[1]:.2f}, "
              f"n12={wave_h[2]:.2f}")

        # For each neuron, correlation with 2nd_pos (given this regime)
        print(f"\n  Correlation with 2nd_pos:")
        for n in comparators:
            r = np.corrcoef(h[:, n].numpy(), sec_p.numpy())[0, 1]
            print(f"    n{n}: {r:+.3f}")
        for n in wave_neurons:
            r = np.corrcoef(h[:, n].numpy(), sec_p.numpy())[0, 1]
            print(f"    n{n}: {r:+.3f}")


def main():
    model = example_2nd_argmax()

    h_by_max, h_by_2nd = analyze_position_conditional_encoding(model)
    analyze_wout_action(model)
    analyze_the_sinusoidal_trick(model)
    analyze_full_logit_decomposition(model)
    analyze_regime_specific_mechanism(model)

    print("\n" + "=" * 80)
    print("FINAL SYNTHESIS: SUPPRESSION VS DELTA")
    print("=" * 80)
    print("""
THE ANSWER: It's NEITHER pure suppression NOR pure delta encoding.
It's INTERFERENCE-BASED encoding:

1. h_final varies SINUSOIDALLY with both max_pos and 2nd_pos
   - These variations are ANTI-PHASE (r ≈ -0.9)
   - This means: h_final ≈ A*sin(ω*max) - B*sin(ω*2nd)

2. W_out is SINUSOIDAL (matched filter)
   - When computing logit[pos], it multiplies h_final by sin(pos)

3. The product creates SELECTIVE AMPLIFICATION:
   - When pos = max_pos: the "max term" is large negative
   - When pos = 2nd_pos: the "2nd term" is large positive
   - The anti-phase relationship ensures max is suppressed and 2nd boosted

4. This is fundamentally DIFFERENT from:
   - "Suppress max, then pick remaining winner" (pure suppression)
   - "Encode max, encode delta, add to get 2nd" (pure delta)

   Instead: The SAME h_final representation simultaneously encodes
   BOTH the max position AND the 2nd position through interference,
   and the sinusoidal W_out decodes the 2nd position by exploiting
   the anti-phase relationship.

5. The key insight: The model doesn't separately encode max and delta.
   It encodes their DIFFERENCE directly in h_final as an interference
   pattern, which W_out then reads out as the 2nd position.
""")


if __name__ == "__main__":
    main()
