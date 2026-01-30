"""
2AM BEFORE 1AM analysis: What happens when the 2nd-largest value appears
at an earlier position than the max?

Focus on all 4 comparator activations through time.
"""

import torch as th
import numpy as np
from alg_zoo.architectures import DistRNN


def load_local_model():
    import os
    path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
    state_dict = th.load(path, weights_only=True, map_location='cpu')
    seq_len, hidden_size = state_dict['linear.weight'].shape
    model = DistRNN(hidden_size=hidden_size, seq_len=seq_len, bias=False)
    model.load_state_dict(state_dict)
    return model


def get_full_trajectory(model, x):
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


def section(title):
    print(f"\n{'='*80}")
    print(title)
    print('='*80)


# =============================================================================
# PART 1: Full comp trajectory for 2AM-before-1AM vs 2AM-after-1AM
# =============================================================================
def comp_trajectories(model):
    section("PART 1: COMP TRAJECTORIES — 2AM BEFORE vs AFTER 1AM")
    print("Compare the full temporal activation of all 4 comparators.")
    print("Case A: 2AM at pos 3, 1AM at pos 7 (2AM BEFORE 1AM)")
    print("Case B: 1AM at pos 3, 2AM at pos 7 (1AM BEFORE 2AM — previous analysis)")
    print("Case C: 1AM at pos 7 only (baseline for case A)")
    print("Case D: 1AM at pos 3 only (baseline for case B)")

    comps = [1, 6, 7, 8]
    W_ih = model.rnn.weight_ih_l0.data.squeeze().numpy()

    print(f"\nW_ih for comps: ", end="")
    for n in comps:
        print(f"n{n}={W_ih[n]:.2f}  ", end="")
    print()

    cases = {
        'A: 2AM=3,1AM=7': lambda: _make_input(3, 0.8, 7, 1.0),
        'B: 1AM=3,2AM=7': lambda: _make_input(3, 1.0, 7, 0.8),
        'C: 1AM=7 only':  lambda: _make_input_single(7, 1.0),
        'D: 1AM=3 only':  lambda: _make_input_single(3, 1.0),
    }

    trajectories = {}
    pre_trajectories = {}
    for name, make_x in cases.items():
        x = make_x()
        hidden, pre_act = get_full_trajectory(model, x)
        trajectories[name] = hidden[0].numpy()   # 16 x 10
        pre_trajectories[name] = pre_act[0].numpy()

    for n in comps:
        print(f"\n--- Neuron n{n} (W_ih={W_ih[n]:.2f}) ---")
        print(f"{'case':<20}", end="")
        for t in range(10):
            print(f"{'t='+str(t):>8}", end="")
        print(f"  h_final")
        print("-" * (20 + 8*10 + 8))

        for name in cases:
            h = trajectories[name]
            print(f"{name:<20}", end="")
            for t in range(10):
                print(f"{h[n, t]:>8.2f}", end="")
            print(f"  {h[n, 9]:>6.2f}")

        # Also show pre-activations to see clipping
        print(f"\n  Pre-activations:")
        for name in cases:
            pre = pre_trajectories[name]
            print(f"  {name:<20}", end="")
            for t in range(10):
                val = pre[n, t]
                clip_marker = "*" if val < 0 else " "
                print(f"{val:>7.1f}{clip_marker}", end="")
            print()


def _make_input(pos1, val1, pos2, val2):
    x = th.zeros(1, 10)
    x[0, pos1] = val1
    x[0, pos2] = val2
    return x


def _make_input_single(pos, val):
    x = th.zeros(1, 10)
    x[0, pos] = val
    return x


# =============================================================================
# PART 2: Magnitude sweep for 2AM-before-1AM
# =============================================================================
def magnitude_sweep_reversed(model):
    section("PART 2: MAGNITUDE SWEEP — 2AM BEFORE 1AM")
    print("Fix 1AM=7, sweep 2AM magnitude at pos 3 from 0 to 0.99.")
    print("Compare with forward case: 1AM=3, 2AM=7.\n")

    comps = [1, 6, 7, 8]
    waves = [0, 10, 11, 12, 14]
    W_out = model.linear.weight.data.numpy()

    mags = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    # Reversed: 2AM at pos 3, 1AM at pos 7
    m_1am, m_2am = 7, 3  # 1AM position, 2AM position
    target = m_2am  # correct answer is the 2AM position

    # Baseline: 1AM only
    x_base = th.zeros(1, 10)
    x_base[0, m_1am] = 1.0
    h_base, _ = get_full_trajectory(model, x_base)
    baseline = h_base[0, :, 9].numpy()
    base_logit = baseline @ W_out.T

    print(f"Baseline (1AM={m_1am} only): logit argmax = {base_logit.argmax()}")
    print(f"Target answer: position {target}\n")

    print(f"{'mag':>5} {'top_delta_n':>11} {'delta_val':>10} {'logit_peak':>11} {'correct':>8}")
    print("-" * 50)

    for mag in mags:
        x = th.zeros(1, 10)
        x[0, m_1am] = 1.0
        x[0, m_2am] = mag
        h, _ = get_full_trajectory(model, x)
        h_final = h[0, :, 9].numpy()

        delta = h_final - baseline
        logit = h_final @ W_out.T
        pred = logit.argmax()

        # Most disrupted neuron
        abs_delta = np.abs(delta)
        top_n = abs_delta.argmax()

        print(f"{mag:>5.2f} {'n'+str(top_n):>11} {delta[top_n]:>+10.2f} "
              f"{'p='+str(pred):>11} {'YES' if pred == target else 'no':>8}")

    # Detailed delta by group at mag=0.8
    mag = 0.8
    x = th.zeros(1, 10)
    x[0, m_1am] = 1.0
    x[0, m_2am] = mag
    h, _ = get_full_trajectory(model, x)
    delta = h[0, :, 9].numpy() - baseline

    print(f"\n--- Delta h_final at mag={mag} (2AM={m_2am} before 1AM={m_1am}) ---")
    print(f"{'neuron':>8} {'delta':>8} {'group':>8}")
    print("-" * 28)
    for n in range(16):
        group = 'comp' if n in comps else ('wave' if n in waves else 'other')
        if abs(delta[n]) > 0.1:
            print(f"{'n'+str(n):>8} {delta[n]:>+8.2f} {group:>8}")

    # Delta logit profile by group
    print(f"\n--- Delta logit profile by group (2AM={m_2am} before 1AM={m_1am}) ---")
    groups = {'comps': comps, 'waves': waves, 'other': [2,3,4,5,9,13,15]}
    print(f"{'group':<8}", end="")
    for p in range(10):
        marker = "^" if p == m_1am else ("*" if p == m_2am else " ")
        print(f" {p}{marker}    ", end="")
    print(f"  peak")
    print("-" * (8 + 7*10 + 8))

    for gname, neurons in groups.items():
        g_logit = np.zeros(10)
        for n in neurons:
            g_logit += delta[n] * W_out[:, n]
        peak = np.argmax(g_logit)
        print(f"{gname:<8}", end="")
        for p in range(10):
            print(f"{g_logit[p]:>+7.1f}", end="")
        print(f"  p={peak}")

    total = delta @ W_out.T
    peak = np.argmax(total)
    print(f"{'TOTAL':<8}", end="")
    for p in range(10):
        print(f"{total[p]:>+7.1f}", end="")
    print(f"  p={peak}")


# =============================================================================
# PART 3: Compare forward vs reversed for multiple pairs
# =============================================================================
def forward_vs_reversed(model):
    section("PART 3: FORWARD (1AM<2AM) vs REVERSED (2AM<1AM) — MULTIPLE PAIRS")
    print("For each gap, compare forward vs reversed orientation.")
    print("Which neurons carry the signal? Does the mechanism differ?\n")

    comps = [1, 6, 7, 8]
    waves = [0, 10, 11, 12, 14]
    W_out = model.linear.weight.data.numpy()
    groups = {'comps': comps, 'waves': waves, 'other': [2,3,4,5,9,13,15]}

    pairs = [
        # (1AM_pos, 2AM_pos, label)
        (3, 7, "FWD: 1AM=3, 2AM=7"),
        (7, 3, "REV: 1AM=7, 2AM=3"),
        (2, 6, "FWD: 1AM=2, 2AM=6"),
        (6, 2, "REV: 1AM=6, 2AM=2"),
        (4, 8, "FWD: 1AM=4, 2AM=8"),
        (8, 4, "REV: 1AM=8, 2AM=4"),
        (1, 5, "FWD: 1AM=1, 2AM=5"),
        (5, 1, "REV: 1AM=5, 2AM=1"),
    ]

    print(f"{'case':<25} {'target':>7} {'comp_pk':>8} {'wave_pk':>8} {'total_pk':>9} {'correct':>8} {'top_Δn':>7}")
    print("-" * 75)

    for m_1am, m_2am, label in pairs:
        target = m_2am

        # Baseline: 1AM only
        x_base = th.zeros(1, 10)
        x_base[0, m_1am] = 1.0
        h_base, _ = get_full_trajectory(model, x_base)
        baseline = h_base[0, :, 9].numpy()

        # With 2AM
        x = th.zeros(1, 10)
        x[0, m_1am] = 1.0
        x[0, m_2am] = 0.8
        h, _ = get_full_trajectory(model, x)
        h_final = h[0, :, 9].numpy()

        delta = h_final - baseline
        logit = h_final @ W_out.T
        pred = logit.argmax()

        # Group peaks
        peaks = {}
        for gname, neurons in groups.items():
            g_logit = np.zeros(10)
            for n in neurons:
                g_logit += delta[n] * W_out[:, n]
            peaks[gname] = np.argmax(g_logit)

        total_delta = delta @ W_out.T
        total_peak = np.argmax(total_delta)
        correct = "YES" if pred == target else "no"

        # Top disrupted neuron
        top_n = np.argmax(np.abs(delta))

        print(f"{label:<25} {'p='+str(target):>7} {'p='+str(peaks['comps']):>8} "
              f"{'p='+str(peaks['waves']):>8} {'p='+str(total_peak):>9} {correct:>8} "
              f"{'n'+str(top_n):>7}")

        # Show separator between forward/reversed pair
        if label.startswith("REV"):
            print()


# =============================================================================
# PART 4: Detailed comp activation comparison for one reversed pair
# =============================================================================
def reversed_comp_detail(model):
    section("PART 4: DETAILED COMP ANALYSIS — REVERSED PAIR (1AM=7, 2AM=3)")
    print("In the reversed case, 2AM (0.8) arrives FIRST at t=3.")
    print("Then 1AM (1.0) arrives at t=7.")
    print("Question: does 2AM clip the comps? Or is it too weak?\n")

    comps = [1, 6, 7, 8]
    W_ih = model.rnn.weight_ih_l0.data.squeeze().numpy()

    # Three cases
    cases = [
        ("2AM=3 only (0.8)", _make_input_single(3, 0.8)),
        ("1AM=7 only (1.0)", _make_input_single(7, 1.0)),
        ("Both: 2AM=3, 1AM=7", _make_input(3, 0.8, 7, 1.0)),
    ]

    for case_name, x in cases:
        hidden, pre_act = get_full_trajectory(model, x)
        print(f"\n--- {case_name} ---")

        for n in comps:
            print(f"  n{n} (W_ih={W_ih[n]:.2f}):")
            print(f"    h:   ", end="")
            for t in range(10):
                print(f"{hidden[0, n, t].item():>7.2f}", end="")
            print()
            print(f"    pre: ", end="")
            clips = []
            for t in range(10):
                val = pre_act[0, n, t].item()
                clipped = val < 0
                if clipped:
                    clips.append(t)
                marker = "*" if clipped else " "
                print(f"{val:>6.1f}{marker}", end="")
            print(f"  clips@{clips}")

    # Now show: what is the LAST clip time for each comp in each case?
    print(f"\n--- Last clip time for each comp ---")
    print(f"{'case':<25}", end="")
    for n in comps:
        print(f" {'n'+str(n):>5}", end="")
    print()
    print("-" * (25 + 6*4))

    for case_name, x in cases:
        _, pre_act = get_full_trajectory(model, x)
        print(f"{case_name:<25}", end="")
        for n in comps:
            last_clip = -1
            for t in range(10):
                if pre_act[0, n, t].item() < 0:
                    last_clip = t
            print(f" {'t='+str(last_clip) if last_clip >= 0 else 'none':>5}", end="")
        print()


# =============================================================================
# PART 5: Sweep all reversed pairs at fixed gap
# =============================================================================
def reversed_gap_sweep(model, gap):
    section(f"PART 5: ALL REVERSED PAIRS AT GAP={gap}")
    print(f"2AM comes {gap} positions BEFORE 1AM.")
    print("For each pair, show: comp clips, delta profile, answer.\n")

    comps = [1, 6, 7, 8]
    waves = [0, 10, 11, 12, 14]
    W_out = model.linear.weight.data.numpy()
    groups = {'comps': comps, 'waves': waves, 'other': [2,3,4,5,9,13,15]}

    # Show last clip times
    print(f"--- Last clip times (reversed: 2AM before 1AM, gap={gap}) ---")
    print(f"{'pair':<18} {'target':>7}", end="")
    for n in comps:
        print(f" {'n'+str(n):>5}", end="")
    print(f"  {'last_clip_from':>15}")
    print("-" * 70)

    for s_2am in range(10 - gap):
        s_1am = s_2am + gap
        target = s_2am

        x = th.zeros(1, 10)
        x[0, s_1am] = 1.0
        x[0, s_2am] = 0.8
        _, pre_act = get_full_trajectory(model, x)

        print(f"1AM={s_1am},2AM={s_2am}    {'p='+str(target):>7}", end="")
        last_clips = []
        for n in comps:
            last_clip = -1
            for t in range(10):
                if pre_act[0, n, t].item() < 0:
                    last_clip = t
            last_clips.append(last_clip)
            print(f" {'t='+str(last_clip) if last_clip >= 0 else 'none':>5}", end="")

        # Which impulse caused the last clip?
        from_1am = sum(1 for lc in last_clips if lc == s_1am)
        from_2am = sum(1 for lc in last_clips if lc == s_2am)
        from_other = sum(1 for lc in last_clips if lc >= 0 and lc != s_1am and lc != s_2am)
        print(f"  1AM:{from_1am} 2AM:{from_2am} other:{from_other}")

    # Show delta logit profiles
    print(f"\n--- Delta logit group peaks (reversed: 2AM before 1AM, gap={gap}) ---")
    print(f"{'pair':<18} {'target':>7} {'comp_pk':>8} {'wave_pk':>8} {'total_pk':>9} {'correct':>8}")
    print("-" * 60)

    for s_2am in range(10 - gap):
        s_1am = s_2am + gap
        target = s_2am

        # Baseline
        x_base = th.zeros(1, 10)
        x_base[0, s_1am] = 1.0
        h_base, _ = get_full_trajectory(model, x_base)
        baseline = h_base[0, :, 9].numpy()

        # With 2AM
        x = th.zeros(1, 10)
        x[0, s_1am] = 1.0
        x[0, s_2am] = 0.8
        h, _ = get_full_trajectory(model, x)
        delta = h[0, :, 9].numpy() - baseline

        logit = h[0, :, 9].numpy() @ W_out.T
        pred = logit.argmax()

        peaks = {}
        for gname, neurons in groups.items():
            g_logit = np.zeros(10)
            for n in neurons:
                g_logit += delta[n] * W_out[:, n]
            peaks[gname] = np.argmax(g_logit)

        total_delta = delta @ W_out.T
        total_peak = np.argmax(total_delta)
        correct = "YES" if pred == target else "no"

        print(f"1AM={s_1am},2AM={s_2am}    {'p='+str(target):>7} {'p='+str(peaks['comps']):>8} "
              f"{'p='+str(peaks['waves']):>8} {'p='+str(total_peak):>9} {correct:>8}")


# =============================================================================
# PART 6: Key question — in reversed case, does 2AM even disrupt anything?
# =============================================================================
def reversed_disruption_analysis(model):
    section("PART 6: DOES 2AM EVEN DISRUPT WHEN IT COMES FIRST?")
    print("In the forward case, 2AM disrupts comps which propagate to waves.")
    print("In the reversed case, 2AM arrives first (weaker, 0.8).")
    print("Then 1AM arrives (stronger, 1.0) and clips everything.")
    print("Does 1AM erase the trace of 2AM? Or does 2AM leave a mark?\n")

    comps = [1, 6, 7, 8]
    waves = [0, 10, 11, 12, 14]
    W_out = model.linear.weight.data.numpy()

    # Compare: h_final with both vs h_final with 1AM only
    # The delta should show what 2AM contributed
    print(f"--- Delta h_final (both - 1AM_only) for reversed pairs ---")
    print(f"  If 2AM has NO effect, delta ≈ 0 everywhere.")
    print(f"  If 2AM affects the answer, delta should be structured.\n")

    print(f"{'pair':<18} {'Σ|Δcomp|':>10} {'Σ|Δwave|':>10} {'Σ|Δother|':>11} {'top_Δ':>8} {'Δ_val':>8}")
    print("-" * 68)

    for s_2am in range(8):
        s_1am = s_2am + 3  # gap=3
        if s_1am >= 10:
            continue

        x_base = th.zeros(1, 10)
        x_base[0, s_1am] = 1.0
        h_base, _ = get_full_trajectory(model, x_base)

        x_both = th.zeros(1, 10)
        x_both[0, s_1am] = 1.0
        x_both[0, s_2am] = 0.8
        h_both, _ = get_full_trajectory(model, x_both)

        delta = h_both[0, :, 9].numpy() - h_base[0, :, 9].numpy()

        comp_d = sum(abs(delta[n]) for n in comps)
        wave_d = sum(abs(delta[n]) for n in waves)
        other_d = sum(abs(delta[n]) for n in [2,3,4,5,9,13,15])
        top_n = np.argmax(np.abs(delta))

        print(f"1AM={s_1am},2AM={s_2am}    {comp_d:>10.2f} {wave_d:>10.2f} "
              f"{other_d:>11.2f} {'n'+str(top_n):>8} {delta[top_n]:>+8.2f}")

    # Now do the forward comparison
    print(f"\n--- Same metric for FORWARD pairs (gap=3) for comparison ---")
    print(f"{'pair':<18} {'Σ|Δcomp|':>10} {'Σ|Δwave|':>10} {'Σ|Δother|':>11} {'top_Δ':>8} {'Δ_val':>8}")
    print("-" * 68)

    for m in range(7):
        s = m + 3

        x_base = th.zeros(1, 10)
        x_base[0, m] = 1.0
        h_base, _ = get_full_trajectory(model, x_base)

        x_both = th.zeros(1, 10)
        x_both[0, m] = 1.0
        x_both[0, s] = 0.8
        h_both, _ = get_full_trajectory(model, x_both)

        delta = h_both[0, :, 9].numpy() - h_base[0, :, 9].numpy()

        comp_d = sum(abs(delta[n]) for n in comps)
        wave_d = sum(abs(delta[n]) for n in waves)
        other_d = sum(abs(delta[n]) for n in [2,3,4,5,9,13,15])
        top_n = np.argmax(np.abs(delta))

        print(f"1AM={m},2AM={s}      {comp_d:>10.2f} {wave_d:>10.2f} "
              f"{other_d:>11.2f} {'n'+str(top_n):>8} {delta[top_n]:>+8.2f}")


# =============================================================================
# PART 7: Full temporal trace — where does 2AM leave its mark?
# =============================================================================
def reversed_temporal_trace(model):
    section("PART 7: TEMPORAL TRACE — WHERE DOES 2AM LEAVE ITS MARK? (REVERSED)")
    print("For reversed pair 1AM=7, 2AM=3: trace all 4 comps through time.")
    print("Show h[t] for: 1AM only, 2AM only (0.8), and both.")
    print("The DIFFERENCE between 'both' and '1AM only' at each timestep")
    print("reveals when and how 2AM's trace survives.\n")

    comps = [1, 6, 7, 8]

    x_1am = th.zeros(1, 10); x_1am[0, 7] = 1.0
    x_2am = th.zeros(1, 10); x_2am[0, 3] = 0.8
    x_both = th.zeros(1, 10); x_both[0, 7] = 1.0; x_both[0, 3] = 0.8

    h_1am, pre_1am = get_full_trajectory(model, x_1am)
    h_2am, pre_2am = get_full_trajectory(model, x_2am)
    h_both, pre_both = get_full_trajectory(model, x_both)

    for n in comps:
        print(f"\n--- n{n} ---")
        print(f"{'case':<16}", end="")
        for t in range(10):
            print(f"{'t='+str(t):>8}", end="")
        print()
        print("-" * (16 + 8*10))

        for label, h in [("1AM=7 only", h_1am), ("2AM=3 only", h_2am), ("Both", h_both)]:
            print(f"{label:<16}", end="")
            for t in range(10):
                print(f"{h[0, n, t].item():>8.2f}", end="")
            print()

        # Delta = both - 1AM_only
        print(f"{'Δ(both-1AM)':>16}", end="")
        for t in range(10):
            d = h_both[0, n, t].item() - h_1am[0, n, t].item()
            print(f"{d:>+8.2f}", end="")
        print()

        # Pre-act clips for 'both'
        print(f"{'clips(both)':>16}", end="")
        for t in range(10):
            val = pre_both[0, n, t].item()
            if val < 0:
                print(f"  CLIP  ", end="")
            else:
                print(f"{'':>8}", end="")
        print()


def main():
    model = load_local_model()

    comp_trajectories(model)
    magnitude_sweep_reversed(model)
    forward_vs_reversed(model)
    reversed_comp_detail(model)
    reversed_gap_sweep(model, gap=3)
    reversed_disruption_analysis(model)
    reversed_temporal_trace(model)


if __name__ == "__main__":
    main()
