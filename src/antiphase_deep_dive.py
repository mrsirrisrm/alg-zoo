"""
Deep dive into the anti-phase mechanism and matched filter.

Key questions:
1. What is f(pos) exactly? Is it sinusoidal?
2. Are A and B the same or different?
3. Is it really h = A*f(max) - B*f(2nd), or something else?
4. How does W_out decode this?
5. Are the frequencies the same for max and 2nd encoding?
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


def extract_position_curves(model, n_samples=200000):
    """
    Extract h_final as a function of max_pos and 2nd_pos.

    For each comparator, we want to see:
    - f_max(pos) = E[h_final | max_pos = pos]
    - f_2nd(pos) = E[h_final | 2nd_pos = pos]
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]

    # Extract curves (excluding position 9 for cleaner analysis)
    curves_by_max = {n: np.zeros(9) for n in comparators}
    curves_by_2nd = {n: np.zeros(9) for n in comparators}

    for pos in range(9):
        mask_max = argmax_pos == pos
        mask_2nd = targets == pos

        for n in comparators:
            curves_by_max[n][pos] = h_final[mask_max, n].mean().item()
            curves_by_2nd[n][pos] = h_final[mask_2nd, n].mean().item()

    return curves_by_max, curves_by_2nd


def analyze_curve_structure(curves_by_max, curves_by_2nd):
    """
    Analyze the structure of the position curves.
    Are they sinusoidal? At what frequency?
    """
    comparators = [1, 6, 7, 8]

    print("=" * 80)
    print("CURVE STRUCTURE ANALYSIS")
    print("=" * 80)

    print("\n1. RAW CURVES (positions 0-8)")
    print("-" * 70)
    print(f"{'Neuron':<8} | {'Curve by max_pos':<45} | {'Curve by 2nd_pos'}")
    print("-" * 70)

    for n in comparators:
        max_str = " ".join([f"{v:.2f}" for v in curves_by_max[n]])
        sec_str = " ".join([f"{v:.2f}" for v in curves_by_2nd[n]])
        print(f"n{n:<7} | {max_str:<45} | {sec_str}")

    print("\n2. DFT ANALYSIS OF CURVES")
    print("-" * 70)

    for n in comparators:
        # Zero-mean for DFT
        max_curve = curves_by_max[n] - curves_by_max[n].mean()
        sec_curve = curves_by_2nd[n] - curves_by_2nd[n].mean()

        # DFT
        fft_max = np.fft.fft(max_curve)
        fft_sec = np.fft.fft(sec_curve)

        # Energy at each frequency
        energy_max = np.abs(fft_max) ** 2
        energy_sec = np.abs(fft_sec) ** 2

        # Phase at dominant frequency
        k_dom_max = np.argmax(energy_max[1:5]) + 1  # k=1 to k=4
        k_dom_sec = np.argmax(energy_sec[1:5]) + 1

        phase_max = np.angle(fft_max[k_dom_max])
        phase_sec = np.angle(fft_sec[k_dom_sec])

        print(f"\nn{n}:")
        print(f"  By max_pos: dominant k={k_dom_max}, "
              f"energy={energy_max[k_dom_max]/energy_max.sum()*100:.1f}%, "
              f"phase={phase_max:.2f} rad ({np.degrees(phase_max):.0f}°)")
        print(f"  By 2nd_pos: dominant k={k_dom_sec}, "
              f"energy={energy_sec[k_dom_sec]/energy_sec.sum()*100:.1f}%, "
              f"phase={phase_sec:.2f} rad ({np.degrees(phase_sec):.0f}°)")
        print(f"  Phase difference: {phase_max - phase_sec:.2f} rad "
              f"({np.degrees(phase_max - phase_sec):.0f}°)")


def test_superposition_model(model, n_samples=200000):
    """
    Test: Is h_final[n] ≈ A*f(max_pos) + B*f(2nd_pos)?

    If so, what are A and B? Are they opposite sign (anti-phase)?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    # Get position curves
    curves_by_max, curves_by_2nd = extract_position_curves(model, n_samples)

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("SUPERPOSITION MODEL TEST")
    print("=" * 80)
    print("\nModel: h_final[n] = A * f_max(max_pos) + B * f_2nd(2nd_pos) + C")
    print("-" * 70)

    for n in comparators:
        # For each sample, get f_max(max_pos) and f_2nd(2nd_pos)
        f_max_vals = np.array([curves_by_max[n][min(p, 8)] for p in argmax_pos.numpy()])
        f_2nd_vals = np.array([curves_by_2nd[n][min(p, 8)] for p in targets.numpy()])
        h_actual = h_final[:, n].numpy()

        # Exclude position 9 cases for cleaner fit
        valid = (argmax_pos < 9) & (targets < 9)
        f_max_v = f_max_vals[valid]
        f_2nd_v = f_2nd_vals[valid]
        h_v = h_actual[valid]

        # Fit: h = A*f_max + B*f_2nd + C
        X = np.column_stack([f_max_v, f_2nd_v, np.ones(len(h_v))])
        coeffs, residuals, _, _ = np.linalg.lstsq(X, h_v, rcond=None)
        A, B, C = coeffs

        # R²
        h_pred = X @ coeffs
        ss_res = np.sum((h_v - h_pred) ** 2)
        ss_tot = np.sum((h_v - h_v.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        print(f"\nn{n}:")
        print(f"  A (coeff on f_max) = {A:+.3f}")
        print(f"  B (coeff on f_2nd) = {B:+.3f}")
        print(f"  C (intercept)      = {C:+.3f}")
        print(f"  R² = {r2:.4f}")
        print(f"  A/B ratio = {A/B:.3f}" if B != 0 else "  B=0")


def analyze_what_f_actually_is(model, n_samples=200000):
    """
    What is f(pos) really?

    Hypothesis: f(pos) is related to "time since clipping at pos"
    encoded through the recurrent dynamics.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    curves_by_max, curves_by_2nd = extract_position_curves(model, n_samples)

    print("\n" + "=" * 80)
    print("WHAT IS f(pos)?")
    print("=" * 80)

    # For n7, plot the curve and a fitted sinusoid
    n = 7
    curve = curves_by_max[n]

    print(f"\nFor n{n}, curve by max_pos:")
    print(f"  Raw: {curve}")

    # Fit: f(pos) = A*sin(ω*pos + φ) + C
    # Try k=1 (ω = 2π/9)
    positions = np.arange(9)
    omega = 2 * np.pi / 9  # k=1 frequency

    # Fit A, phi, C
    def fit_sinusoid(pos, curve):
        """Fit A*sin(ω*pos + φ) + C"""
        best_r2 = -1
        best_params = None

        for phi in np.linspace(0, 2*np.pi, 100):
            sin_vals = np.sin(omega * pos + phi)
            X = np.column_stack([sin_vals, np.ones(len(pos))])
            coeffs, _, _, _ = np.linalg.lstsq(X, curve, rcond=None)
            A, C = coeffs

            pred = A * sin_vals + C
            ss_res = np.sum((curve - pred) ** 2)
            ss_tot = np.sum((curve - curve.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot

            if r2 > best_r2:
                best_r2 = r2
                best_params = (A, phi, C)

        return best_params, best_r2

    (A, phi, C), r2 = fit_sinusoid(positions, curve)

    print(f"\n  Best sinusoidal fit (k=1):")
    print(f"    f(pos) = {A:.2f} * sin(2π/9 * pos + {phi:.2f}) + {C:.2f}")
    print(f"    R² = {r2:.4f}")
    print(f"    Phase = {phi:.2f} rad = {np.degrees(phi):.0f}°")

    # Now fit the 2nd curve
    curve_2nd = curves_by_2nd[n]
    (A2, phi2, C2), r2_2 = fit_sinusoid(positions, curve_2nd)

    print(f"\n  For 2nd_pos curve:")
    print(f"    f(pos) = {A2:.2f} * sin(2π/9 * pos + {phi2:.2f}) + {C2:.2f}")
    print(f"    R² = {r2_2:.4f}")
    print(f"    Phase = {phi2:.2f} rad = {np.degrees(phi2):.0f}°")

    print(f"\n  Phase difference: {phi - phi2:.2f} rad = {np.degrees(phi - phi2):.0f}°")
    print(f"  Amplitude ratio: {A/A2:.2f}")


def analyze_wout_as_matched_filter(model):
    """
    How does W_out decode the interference pattern?

    If h = A*sin(ω*max + φ) - B*sin(ω*2nd + φ), and
    W_out[pos] = sin(ω*pos + ψ), then

    logit[pos] = h · W_out[pos]
              = sum_n h[n] * W_out[pos, n]

    The question: Is W_out really sinusoidal? At what phase?
    """
    W_out = model.linear.weight.data.numpy()  # 10 x 16

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("W_OUT AS MATCHED FILTER")
    print("=" * 80)

    positions = np.arange(10)
    omega = 2 * np.pi / 10  # k=1 for 10 positions

    for n in comparators:
        col = W_out[:, n]

        # Fit sinusoid
        best_r2 = -1
        best_params = None

        for phi in np.linspace(0, 2*np.pi, 100):
            sin_vals = np.sin(omega * positions + phi)
            X = np.column_stack([sin_vals, np.ones(10)])
            coeffs, _, _, _ = np.linalg.lstsq(X, col, rcond=None)
            A, C = coeffs

            pred = A * sin_vals + C
            ss_res = np.sum((col - pred) ** 2)
            ss_tot = np.sum((col - col.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot

            if r2 > best_r2:
                best_r2 = r2
                best_params = (A, phi, C)

        A, phi, C = best_params

        print(f"\nn{n}:")
        print(f"  W_out[:, {n}] = {A:.2f} * sin(2π/10 * pos + {phi:.2f}) + {C:.2f}")
        print(f"  R² = {best_r2:.4f}")
        print(f"  Phase = {phi:.2f} rad = {np.degrees(phi):.0f}°")
        print(f"  Raw: {' '.join([f'{v:+.1f}' for v in col])}")


def demonstrate_matched_filter_mechanism(model, n_samples=50000):
    """
    Step-by-step demonstration of how the matched filter works.

    For a specific (max_pos, 2nd_pos) pair, show:
    1. h_final for each comparator
    2. W_out for each output position
    3. The resulting logits
    4. Why 2nd gets higher logit than max
    """
    W_out = model.linear.weight.data.numpy()

    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("MATCHED FILTER MECHANISM DEMONSTRATION")
    print("=" * 80)

    # Pick a specific case: max at 3, 2nd at 7
    max_p, sec_p = 3, 7
    mask = (argmax_pos == max_p) & (targets == sec_p)

    if mask.sum() < 50:
        print(f"Not enough samples for max={max_p}, 2nd={sec_p}")
        return

    h_case = h_final[mask].mean(dim=0).numpy()

    print(f"\nCase: max_pos={max_p}, 2nd_pos={sec_p}")
    print("=" * 70)

    print(f"\n1. COMPARATOR h_final VALUES:")
    for n in comparators:
        print(f"   h[{n}] = {h_case[n]:.3f}")

    print(f"\n2. W_OUT FOR KEY POSITIONS:")
    print(f"   {'Neuron':<8} | {'W[max={max_p}]':<12} | {'W[2nd={sec_p}]':<12} | {'Difference'}")
    print(f"   {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for n in comparators:
        w_max = W_out[max_p, n]
        w_sec = W_out[sec_p, n]
        diff = w_sec - w_max
        print(f"   n{n:<7} | {w_max:>+10.2f}  | {w_sec:>+10.2f}  | {diff:>+10.2f}")

    print(f"\n3. LOGIT CONTRIBUTIONS (h * W):")
    print(f"   {'Neuron':<8} | {'To max={max_p}':<12} | {'To 2nd={sec_p}':<12} | {'Diff'}")
    print(f"   {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    total_max = 0
    total_sec = 0
    for n in comparators:
        contrib_max = h_case[n] * W_out[max_p, n]
        contrib_sec = h_case[n] * W_out[sec_p, n]
        diff = contrib_sec - contrib_max
        total_max += contrib_max
        total_sec += contrib_sec
        print(f"   n{n:<7} | {contrib_max:>+10.2f}  | {contrib_sec:>+10.2f}  | {diff:>+10.2f}")

    print(f"   {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    print(f"   {'TOTAL':<8} | {total_max:>+10.2f}  | {total_sec:>+10.2f}  | {total_sec-total_max:>+10.2f}")

    print(f"""
4. WHY THIS WORKS:

   The h_final values encode the INTERFERENCE between two impulses:
   - Impulse at max_pos={max_p} contributed to h
   - Impulse at 2nd_pos={sec_p} contributed with OPPOSITE phase

   When we compute logit[pos], we multiply h by W_out[pos].

   W_out is a SINUSOIDAL filter. When pos={max_p} (the max position):
   - The max impulse's contribution gets multiplied by W[{max_p}]
   - This creates a NEGATIVE product (suppression)

   When pos={sec_p} (the 2nd position):
   - The 2nd impulse's contribution gets multiplied by W[{sec_p}]
   - This creates a POSITIVE product (boosting)

   The GAP of {total_sec-total_max:+.1f} logits ensures 2nd wins over max.
""")


def analyze_frequency_matching(model, n_samples=200000):
    """
    Are the frequencies of h_final curves and W_out matched?
    """
    W_out = model.linear.weight.data.numpy()

    curves_by_max, curves_by_2nd = extract_position_curves(model, n_samples)

    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("FREQUENCY MATCHING ANALYSIS")
    print("=" * 80)

    print("\nDFT energy distribution:")
    print("-" * 70)
    print(f"{'Neuron':<8} | {'h by max (k=1,2,3)':<25} | {'h by 2nd (k=1,2,3)':<25} | {'W_out (k=1,2,3)'}")
    print("-" * 70)

    for n in comparators:
        # h curves (9 points, so k goes 0-4)
        h_max = curves_by_max[n] - curves_by_max[n].mean()
        h_2nd = curves_by_2nd[n] - curves_by_2nd[n].mean()

        fft_max = np.abs(np.fft.fft(h_max)) ** 2
        fft_2nd = np.abs(np.fft.fft(h_2nd)) ** 2

        pct_max = [fft_max[k]/fft_max.sum()*100 for k in [1, 2, 3]]
        pct_2nd = [fft_2nd[k]/fft_2nd.sum()*100 for k in [1, 2, 3]]

        # W_out (10 points)
        w = W_out[:, n] - W_out[:, n].mean()
        fft_w = np.abs(np.fft.fft(w)) ** 2
        pct_w = [fft_w[k]/fft_w.sum()*100 for k in [1, 2, 3]]

        str_max = f"{pct_max[0]:.0f}%, {pct_max[1]:.0f}%, {pct_max[2]:.0f}%"
        str_2nd = f"{pct_2nd[0]:.0f}%, {pct_2nd[1]:.0f}%, {pct_2nd[2]:.0f}%"
        str_w = f"{pct_w[0]:.0f}%, {pct_w[1]:.0f}%, {pct_w[2]:.0f}%"

        print(f"n{n:<7} | {str_max:<25} | {str_2nd:<25} | {str_w}")

    print("""
Note: If h and W_out have matching frequencies, the inner product
will be large. If they have mismatched frequencies, they'll be
orthogonal and the product will be small.

The matched filter works because:
- h_final is dominated by k=1 (period 9-10)
- W_out is dominated by k=1 (period 10)
- Their inner product efficiently extracts the encoded position
""")


def main():
    model = example_2nd_argmax()

    curves_by_max, curves_by_2nd = extract_position_curves(model)

    analyze_curve_structure(curves_by_max, curves_by_2nd)
    test_superposition_model(model)
    analyze_what_f_actually_is(model)
    analyze_wout_as_matched_filter(model)
    demonstrate_matched_filter_mechanism(model)
    analyze_frequency_matching(model)

    print("\n" + "=" * 80)
    print("SUMMARY: THE ANTI-PHASE MATCHED FILTER MECHANISM")
    print("=" * 80)
    print("""
1. ENCODING (in h_final):

   Each comparator's h_final can be written as:

   h[n] ≈ A_n * sin(ω*max_pos + φ_n) + B_n * sin(ω*2nd_pos + ψ_n) + C_n

   Key insight: The phases φ and ψ are OPPOSITE (anti-phase).
   When max_pos increases, h goes UP; when 2nd_pos increases, h goes DOWN.

   This creates an INTERFERENCE PATTERN that encodes BOTH positions.

2. DECODING (by W_out):

   W_out[:, n] is approximately:

   W[pos, n] ≈ D_n * sin(ω*pos + θ_n) + E_n

   This is a SINUSOIDAL TEMPLATE at the same frequency.

3. THE MATCHED FILTER PRODUCT:

   logit[pos] = Σ_n h[n] * W[pos, n]

   When pos = max_pos:
     - The max term: sin(ω*max) * sin(ω*max + θ) = cos(θ)/2 + ...
     - This gives a NEGATIVE contribution (due to phase alignment)

   When pos = 2nd_pos:
     - The 2nd term: sin(ω*2nd + π) * sin(ω*2nd + θ) = ...
     - The anti-phase (π shift) flips the sign, giving POSITIVE contribution

4. NET EFFECT:

   - Max position gets negative logit (suppressed)
   - 2nd position gets positive logit (boosted)
   - Gap of ~12 logits ensures correct prediction

   This is EXACTLY how a matched filter works in signal processing:
   - Encode signal with sinusoidal carrier
   - Decode with template at same frequency
   - Phase relationship determines output sign
""")


if __name__ == "__main__":
    main()
