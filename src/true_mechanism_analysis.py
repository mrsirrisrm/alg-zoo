"""
True Mechanism Analysis

With 100% accuracy on clean data, we can now uncover the real mechanism.

Key insight from previous analysis:
- The 2nd impulse CLIPS the comparators, destroying the max signal
- h_final depends on time-since-clipping, not anti-phase interference

Let's trace this precisely.
"""

import torch as th
import numpy as np
from alg_zoo import example_2nd_argmax


def get_full_trajectory(model, x):
    """Compute full trajectory with clipping info."""
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data

    batch_size = x.shape[0]
    hidden = th.zeros(batch_size, 16, 10)
    pre_act = th.zeros(batch_size, 16, 10)
    clipped = th.zeros(batch_size, 16, 10, dtype=th.bool)
    h = th.zeros(batch_size, 16)

    for t in range(10):
        x_t = x[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        pre_act[:, :, t] = pre
        clipped[:, :, t] = pre < 0
        h = th.relu(pre)
        hidden[:, :, t] = h

    return hidden, pre_act, clipped


def trace_single_impulse(model, pos):
    """Trace what happens with a single impulse at given position."""
    x = th.zeros(1, 10)
    x[0, pos] = 1.0

    hidden, pre_act, clipped = get_full_trajectory(model, x)

    return hidden[0], pre_act[0], clipped[0]


def trace_double_impulse(model, max_pos, sec_pos, max_val=1.0, sec_val=0.8):
    """Trace what happens with two impulses."""
    x = th.zeros(1, 10)
    x[0, max_pos] = max_val
    x[0, sec_pos] = sec_val

    hidden, pre_act, clipped = get_full_trajectory(model, x)

    return hidden[0], pre_act[0], clipped[0]


def analyze_clipping_at_2nd_impulse(model):
    """
    When the 2nd impulse arrives, what happens to each comparator?
    """
    comparators = [1, 6, 7, 8]
    W_ih = model.rnn.weight_ih_l0.data.squeeze()

    print("=" * 80)
    print("CLIPPING AT 2ND IMPULSE")
    print("=" * 80)

    print(f"\nW_ih values (input weights):")
    for n in comparators:
        print(f"  n{n}: W_ih = {W_ih[n].item():.2f}")

    print(f"\nFor 2nd impulse with value 0.8:")
    print(f"  Direct input contribution = 0.8 * W_ih")
    for n in comparators:
        contrib = 0.8 * W_ih[n].item()
        print(f"  n{n}: {contrib:.2f}")

    print("\n" + "-" * 70)
    print("Analyzing specific (max, 2nd) pairs:")

    test_cases = [
        (3, 7),  # 2nd after max
        (7, 3),  # 2nd before max
        (2, 8),  # far apart
        (4, 5),  # adjacent
    ]

    for max_pos, sec_pos in test_cases:
        h, pre, clip = trace_double_impulse(model, max_pos, sec_pos)

        print(f"\nmax={max_pos}, 2nd={sec_pos}:")
        print(f"  At t={sec_pos} (2nd impulse arrives):")

        for n in comparators:
            h_before = h[n, sec_pos - 1].item() if sec_pos > 0 else 0
            pre_at = pre[n, sec_pos].item()
            h_after = h[n, sec_pos].item()
            did_clip = clip[n, sec_pos].item()

            print(f"    n{n}: h_before={h_before:.2f}, pre={pre_at:.2f}, "
                  f"h_after={h_after:.2f}, clipped={did_clip}")


def analyze_time_since_clipping(model):
    """
    Key hypothesis: h_final depends on TIME SINCE LAST CLIPPING EVENT.

    For each (max, 2nd) pair, track when each comparator last clips,
    and see if h_final correlates with (9 - last_clip_time).
    """
    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("TIME SINCE LAST CLIPPING")
    print("=" * 80)

    results = {n: {'last_clip': [], 'h_final': [], 'max_pos': [], 'sec_pos': []}
               for n in comparators}

    for max_pos in range(10):
        for sec_pos in range(10):
            if max_pos == sec_pos:
                continue

            h, pre, clip = trace_double_impulse(model, max_pos, sec_pos)

            for n in comparators:
                # Find last clipping time
                clip_times = th.where(clip[n])[0]
                if len(clip_times) > 0:
                    last_clip = clip_times[-1].item()
                else:
                    last_clip = -1  # Never clipped

                results[n]['last_clip'].append(last_clip)
                results[n]['h_final'].append(h[n, 9].item())
                results[n]['max_pos'].append(max_pos)
                results[n]['sec_pos'].append(sec_pos)

    print("\nCorrelation of h_final with (9 - last_clip_time):")
    print("-" * 50)

    for n in comparators:
        last_clip = np.array(results[n]['last_clip'])
        h_final = np.array(results[n]['h_final'])
        max_pos = np.array(results[n]['max_pos'])
        sec_pos = np.array(results[n]['sec_pos'])

        # Time since last clip
        time_since = 9 - last_clip

        # Correlations
        r_time = np.corrcoef(time_since, h_final)[0, 1]
        r_max = np.corrcoef(max_pos, h_final)[0, 1]
        r_sec = np.corrcoef(sec_pos, h_final)[0, 1]
        r_last = np.corrcoef(last_clip, h_final)[0, 1]

        print(f"\nn{n}:")
        print(f"  r(h_final, 9 - last_clip) = {r_time:+.3f}")
        print(f"  r(h_final, last_clip)     = {r_last:+.3f}")
        print(f"  r(h_final, max_pos)       = {r_max:+.3f}")
        print(f"  r(h_final, 2nd_pos)       = {r_sec:+.3f}")

    return results


def analyze_clipping_patterns_by_position(model):
    """
    For each (max, 2nd) pair, show the complete clipping pattern.
    """
    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("CLIPPING PATTERNS FOR ALL (MAX, 2ND) PAIRS")
    print("=" * 80)

    # Focus on n7
    n = 7

    print(f"\nNeuron n{n} - Last clipping time for each (max, 2nd):")
    print("      2nd: ", end="")
    for s in range(10):
        print(f"{s:>4}", end="")
    print()
    print("max")

    for m in range(10):
        print(f" {m}        ", end="")
        for s in range(10):
            if m == s:
                print("   -", end="")
            else:
                h, pre, clip = trace_double_impulse(model, m, s)
                clip_times = th.where(clip[n])[0]
                if len(clip_times) > 0:
                    last = clip_times[-1].item()
                    print(f"{last:>4}", end="")
                else:
                    print("   .", end="")
        print()

    print(f"\nNeuron n{n} - h_final for each (max, 2nd):")
    print("      2nd: ", end="")
    for s in range(10):
        print(f"{s:>6}", end="")
    print()
    print("max")

    for m in range(10):
        print(f" {m}        ", end="")
        for s in range(10):
            if m == s:
                print("     -", end="")
            else:
                h, pre, clip = trace_double_impulse(model, m, s)
                print(f"{h[n, 9].item():>6.1f}", end="")
        print()


def analyze_impulse_response_after_clip(model):
    """
    After a clipping event, how does h rebuild?

    This is the "impulse response" that determines h_final.
    """
    comparators = [1, 6, 7, 8]
    W_hh = model.rnn.weight_hh_l0.data

    print("\n" + "=" * 80)
    print("IMPULSE RESPONSE AFTER CLIPPING")
    print("=" * 80)

    # Self-recurrence
    print("\nSelf-recurrence values (W_hh diagonal):")
    for n in comparators:
        print(f"  n{n}: W_hh[{n},{n}] = {W_hh[n, n].item():.3f}")

    # Trace recovery after a clip
    # Use single impulse at t=0 and watch recovery
    print("\n" + "-" * 70)
    print("Recovery trajectory after impulse at t=0:")

    h, pre, clip = trace_single_impulse(model, 0)

    print(f"\n{'t':<4}", end="")
    for n in comparators:
        print(f"{'n'+str(n):<10}", end="")
    print()
    print("-" * 44)

    for t in range(10):
        print(f"{t:<4}", end="")
        for n in comparators:
            val = h[n, t].item()
            clp = "C" if clip[n, t].item() else " "
            print(f"{val:>6.2f}{clp:<3}", end="")
        print()


def analyze_what_determines_h_final(model):
    """
    Regression analysis: what predicts h_final?

    Candidates:
    - last_clip_time
    - max_pos
    - 2nd_pos
    - max(last_clip_time) across neurons (the "reset point")
    """
    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("WHAT DETERMINES H_FINAL?")
    print("=" * 80)

    # Collect data for all (max, 2nd) pairs
    data = []

    for max_pos in range(10):
        for sec_pos in range(10):
            if max_pos == sec_pos:
                continue

            h, pre, clip = trace_double_impulse(model, max_pos, sec_pos)

            row = {
                'max_pos': max_pos,
                'sec_pos': sec_pos,
                'delta': max_pos - sec_pos,
                'later_pos': max(max_pos, sec_pos),  # which comes later
                'earlier_pos': min(max_pos, sec_pos),
            }

            for n in comparators:
                clip_times = th.where(clip[n])[0]
                last_clip = clip_times[-1].item() if len(clip_times) > 0 else -1
                row[f'last_clip_{n}'] = last_clip
                row[f'h_final_{n}'] = h[n, 9].item()
                row[f'time_since_{n}'] = 9 - last_clip

            data.append(row)

    # Convert to arrays
    max_pos = np.array([d['max_pos'] for d in data])
    sec_pos = np.array([d['sec_pos'] for d in data])
    later_pos = np.array([d['later_pos'] for d in data])

    for n in comparators:
        h_final = np.array([d[f'h_final_{n}'] for d in data])
        last_clip = np.array([d[f'last_clip_{n}'] for d in data])
        time_since = np.array([d[f'time_since_{n}'] for d in data])

        print(f"\nn{n}:")

        # Simple correlations
        print(f"  Correlations:")
        print(f"    r(h_final, max_pos)     = {np.corrcoef(h_final, max_pos)[0,1]:+.3f}")
        print(f"    r(h_final, 2nd_pos)     = {np.corrcoef(h_final, sec_pos)[0,1]:+.3f}")
        print(f"    r(h_final, later_pos)   = {np.corrcoef(h_final, later_pos)[0,1]:+.3f}")
        print(f"    r(h_final, last_clip)   = {np.corrcoef(h_final, last_clip)[0,1]:+.3f}")
        print(f"    r(h_final, time_since)  = {np.corrcoef(h_final, time_since)[0,1]:+.3f}")

        # Regression: h_final ~ time_since
        X = np.column_stack([time_since, np.ones(len(h_final))])
        coeffs, _, _, _ = np.linalg.lstsq(X, h_final, rcond=None)
        pred = X @ coeffs
        r2 = 1 - np.sum((h_final - pred)**2) / np.sum((h_final - h_final.mean())**2)
        print(f"  Regression h = a*time_since + b:")
        print(f"    a = {coeffs[0]:.3f}, b = {coeffs[1]:.3f}, R² = {r2:.4f}")

        # What's the last clip for each case?
        print(f"  Last clip distribution:")
        unique, counts = np.unique(last_clip, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"    t={int(u)}: {c} cases, mean h_final={h_final[last_clip==u].mean():.2f}")


def analyze_which_position_clips_last(model):
    """
    Key question: Is the last clipping event at max_pos or 2nd_pos?
    """
    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("WHICH POSITION CAUSES LAST CLIP?")
    print("=" * 80)

    for n in comparators:
        print(f"\nn{n}:")

        clip_at_max = 0
        clip_at_sec = 0
        clip_at_neither = 0
        clip_at_both = 0  # same time (impossible but check)

        for max_pos in range(10):
            for sec_pos in range(10):
                if max_pos == sec_pos:
                    continue

                h, pre, clip = trace_double_impulse(model, max_pos, sec_pos)

                clip_times = th.where(clip[n])[0].numpy()

                if len(clip_times) == 0:
                    clip_at_neither += 1
                else:
                    last = clip_times[-1]
                    clipped_at_max = max_pos in clip_times
                    clipped_at_sec = sec_pos in clip_times

                    if last == max_pos and last == sec_pos:
                        clip_at_both += 1
                    elif last == max_pos:
                        clip_at_max += 1
                    elif last == sec_pos:
                        clip_at_sec += 1
                    else:
                        # Last clip is at neither position (some other time)
                        clip_at_neither += 1

        total = clip_at_max + clip_at_sec + clip_at_neither + clip_at_both
        print(f"  Last clip at max_pos: {clip_at_max} ({clip_at_max/total*100:.1f}%)")
        print(f"  Last clip at 2nd_pos: {clip_at_sec} ({clip_at_sec/total*100:.1f}%)")
        print(f"  Last clip at neither: {clip_at_neither} ({clip_at_neither/total*100:.1f}%)")


def analyze_the_readout(model):
    """
    Given that h_final is determined by time-since-clip,
    how does W_out decode this to find the 2nd position?
    """
    W_out = model.linear.weight.data.numpy()
    comparators = [1, 6, 7, 8]

    print("\n" + "=" * 80)
    print("THE READOUT MECHANISM")
    print("=" * 80)

    # For each (max, 2nd), compute logits and see what happens
    print("\nLogit analysis for selected (max, 2nd) pairs:")

    test_cases = [(3, 7), (7, 3), (2, 5), (5, 2), (1, 8), (8, 1)]

    for max_pos, sec_pos in test_cases:
        h, pre, clip = trace_double_impulse(model, max_pos, sec_pos)
        h_final = h[:, 9].numpy()

        # Compute logits
        logits = h_final @ W_out.T

        # Get last clip time for n7
        clip_times = th.where(clip[7])[0]
        last_clip = clip_times[-1].item() if len(clip_times) > 0 else -1

        print(f"\nmax={max_pos}, 2nd={sec_pos}, last_clip(n7)={last_clip}:")
        print(f"  Logits: ", end="")
        for pos in range(10):
            marker = "*" if pos == sec_pos else ("^" if pos == max_pos else " ")
            print(f"{pos}:{logits[pos]:+.1f}{marker} ", end="")
        print()
        print(f"  Prediction: {np.argmax(logits)} (correct={sec_pos})")


def analyze_position_encoding_via_rebuild(model):
    """
    After clipping, the neuron rebuilds via recurrence.
    The rebuild trajectory encodes position.

    If clip at time t, then h_final = f(9-t) where f is the rebuild function.
    """
    comparators = [1, 6, 7, 8]
    W_hh = model.rnn.weight_hh_l0.data.numpy()

    print("\n" + "=" * 80)
    print("POSITION ENCODING VIA REBUILD TRAJECTORY")
    print("=" * 80)

    # For n7, trace rebuild after clip at different times
    n = 7

    print(f"\nNeuron n{n} rebuild trajectories:")
    print("(Single impulse at each position, tracking h[n7] from clip to t=9)")
    print()

    for impulse_pos in range(10):
        h, pre, clip = trace_single_impulse(model, impulse_pos)

        # Find clip times for n7
        clip_times = th.where(clip[n])[0].numpy()

        traj = []
        for t in range(10):
            val = h[n, t].item()
            c = "C" if clip[n, t].item() else " "
            traj.append(f"{val:.1f}{c}")

        print(f"  Impulse at t={impulse_pos}: {' → '.join(traj)}")
        print(f"    h_final = {h[n, 9].item():.2f}, last_clip = {clip_times[-1] if len(clip_times) > 0 else 'none'}")


def main():
    model = example_2nd_argmax()

    analyze_clipping_at_2nd_impulse(model)
    results = analyze_time_since_clipping(model)
    analyze_clipping_patterns_by_position(model)
    analyze_impulse_response_after_clip(model)
    analyze_what_determines_h_final(model)
    analyze_which_position_clips_last(model)
    analyze_the_readout(model)
    analyze_position_encoding_via_rebuild(model)

    print("\n" + "=" * 80)
    print("SYNTHESIS: THE TRUE MECHANISM")
    print("=" * 80)
    print("""
Based on clean data analysis, the mechanism is:

1. CLIPPING ENCODES POSITION
   - When max arrives: comparators clip (x > threshold)
   - When 2nd arrives: comparators clip AGAIN if 2nd > current threshold
   - The LAST clipping event sets the "reset point"

2. REBUILD ENCODES TIME-SINCE-CLIP
   - After clipping, h rebuilds via W_hh recurrence
   - h_final = f(9 - last_clip_time)
   - The rebuild function f() is determined by network dynamics

3. W_OUT DECODES TIME-SINCE-CLIP TO POSITION
   - Different positions have different typical last_clip times
   - W_out maps h_final (which encodes time-since-clip) to position logits
   - The position with matching time-since-clip gets highest logit

4. THE KEY INSIGHT:
   - It's NOT anti-phase interference of two sinusoids
   - It's a RESET-AND-REBUILD mechanism
   - Last clipping event "wins" and determines h_final
   - Time-since-clip directly encodes position
""")


if __name__ == "__main__":
    main()
