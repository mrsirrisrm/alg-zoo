"""
Why doesn't 1AM restart the n4→n2→comp cascade?

1AM DOES fire n4 again. So why doesn't the second n4 kick corrupt n2's memory?
Trace the exact values through n4→n2→comps for both impulses.
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
# PART 1: n4 fires TWICE — trace both firings
# =============================================================================
def n4_fires_twice(model):
    section("PART 1: n4 FIRES TWICE — TRACE BOTH IMPULSES")
    print("Reversed pair: 2AM=3 (mag 0.8), 1AM=7 (mag 1.0)")
    print("n4 has W_ih=+10.16 → fires on BOTH inputs.")
    print("Question: what does n4 look like at each firing?\n")

    W_ih = model.rnn.weight_ih_l0.data.squeeze().numpy()
    W_hh = model.rnn.weight_hh_l0.data.numpy()

    # Three cases: 2AM only, 1AM only, both
    cases = {
        '2AM only (0.8@3)': [(3, 0.8)],
        '1AM only (1.0@7)': [(7, 1.0)],
        'Both (0.8@3, 1.0@7)': [(3, 0.8), (7, 1.0)],
    }

    trajs = {}
    pre_trajs = {}
    for name, inputs in cases.items():
        x = th.zeros(1, 10)
        for pos, val in inputs:
            x[0, pos] = val
        h, pre = get_full_trajectory(model, x)
        trajs[name] = h[0].numpy()
        pre_trajs[name] = pre[0].numpy()

    # n4 trajectory
    print("--- n4 activation (h) trajectory ---")
    print(f"{'t':>3}", end="")
    for name in cases:
        print(f"  {name:>22}", end="")
    print()
    print("-" * 75)
    for t in range(10):
        print(f"{t:>3}", end="")
        for name in cases:
            print(f"  {trajs[name][4, t]:>22.3f}", end="")
        print(f"  {'<-- 2AM' if t==3 else '<-- 1AM' if t==7 else ''}")

    # n4 pre-activation at both impulse times
    print(f"\n--- n4 pre-activation decomposition ---")
    for t_imp, label, mag in [(3, '2AM', 0.8), (7, '1AM', 1.0)]:
        print(f"\nAt t={t_imp} ({label} arrives, x={mag}):")
        for name in cases:
            if t_imp > 0:
                h_prev = trajs[name][:, t_imp-1]
            else:
                h_prev = np.zeros(16)
            recur = h_prev @ W_hh[4, :]
            inp = mag * W_ih[4] if any(pos == t_imp for pos, _ in cases[name]) else 0
            # Correct: get actual input at that time
            x_test = th.zeros(1, 10)
            for pos, val in cases[name]:
                x_test[0, pos] = val
            actual_inp = x_test[0, t_imp].item() * W_ih[4]
            actual_pre = pre_trajs[name][4, t_imp]
            h_val = trajs[name][4, t_imp]
            print(f"  {name:<25} recurrent={recur:>+8.2f}  input={actual_inp:>+8.2f}  pre={actual_pre:>+8.2f}  h={h_val:>8.2f}")


# =============================================================================
# PART 2: What does n4's second firing do to n2?
# =============================================================================
def n4_second_kick_to_n2(model):
    section("PART 2: n4's SECOND FIRING → WHAT HAPPENS TO n2?")
    print("When 1AM fires n4 again at t=7, n4 passes signal to n2 at t=8.")
    print("But n2 already holds the 2AM memory. Does the new n4 kick add/overwrite?\n")

    W_ih = model.rnn.weight_ih_l0.data.squeeze().numpy()
    W_hh = model.rnn.weight_hh_l0.data.numpy()

    cases = {
        '2AM only': [(3, 0.8)],
        '1AM only': [(7, 1.0)],
        'Both': [(3, 0.8), (7, 1.0)],
    }

    trajs = {}
    pre_trajs = {}
    for name, inputs in cases.items():
        x = th.zeros(1, 10)
        for pos, val in inputs:
            x[0, pos] = val
        h, pre = get_full_trajectory(model, x)
        trajs[name] = h[0].numpy()
        pre_trajs[name] = pre[0].numpy()

    # n2 trajectory
    print("--- n2 activation trajectory ---")
    print(f"{'t':>3}  {'2AM only':>10}  {'1AM only':>10}  {'Both':>10}  {'Sum':>10}  {'Both-Sum':>10}")
    print("-" * 65)
    for t in range(10):
        v_2am = trajs['2AM only'][2, t]
        v_1am = trajs['1AM only'][2, t]
        v_both = trajs['Both'][2, t]
        v_sum = v_2am + v_1am
        diff = v_both - v_sum
        marker = " <-- 2AM" if t == 3 else (" <-- 1AM" if t == 7 else "")
        print(f"{t:>3}  {v_2am:>10.3f}  {v_1am:>10.3f}  {v_both:>10.3f}  {v_sum:>10.3f}  {diff:>+10.3f}{marker}")

    # n2 pre-activation decomposition at t=8 (first step after 1AM)
    print(f"\n--- n2 pre-activation decomposition at t=8 (step after 1AM) ---")
    for name in cases:
        h7 = trajs[name][:, 7]
        self_term = h7[2] * W_hh[2, 2]
        from_n4 = h7[4] * W_hh[2, 4]
        inp = 0  # no input at t=8
        other_recur = pre_trajs[name][2, 8] - self_term - from_n4 - inp
        total = pre_trajs[name][2, 8]
        h_val = trajs[name][2, 8]
        print(f"  {name:<12} self(n2)={self_term:>+8.2f}  from_n4={from_n4:>+8.2f}  other_recur={other_recur:>+8.2f}  total_pre={total:>+8.2f}  h={h_val:>8.2f}")

    print(f"\n  W_hh[2,4] = {W_hh[2,4]:+.4f} (n4 → n2 connection)")


# =============================================================================
# PART 3: n4 state at 1AM — WHY is n4 different in 'both' vs '1AM only'?
# =============================================================================
def n4_at_1am_decomposed(model):
    section("PART 3: n4 STATE AT t=7 — WHY DIFFERENT IN 'BOTH' VS '1AM ONLY'?")
    print("In '1AM only', n4 at t=7 starts from h[6]=0 (nothing happened before).")
    print("In 'both', n4 at t=7 starts from h[6] that carries 2AM cascade residue.")
    print("Show n4's pre-act decomposition at t=7 for both cases.\n")

    W_ih = model.rnn.weight_ih_l0.data.squeeze().numpy()
    W_hh = model.rnn.weight_hh_l0.data.numpy()

    cases = {
        '1AM only': [(7, 1.0)],
        'Both': [(3, 0.8), (7, 1.0)],
    }

    trajs = {}
    pre_trajs = {}
    for name, inputs in cases.items():
        x = th.zeros(1, 10)
        for pos, val in inputs:
            x[0, pos] = val
        h, pre = get_full_trajectory(model, x)
        trajs[name] = h[0].numpy()
        pre_trajs[name] = pre[0].numpy()

    for name in cases:
        h6 = trajs[name][:, 6]
        inp = 1.0 * W_ih[4]
        recur = h6 @ W_hh[4, :]
        total = recur + inp
        h_val = max(0, total)

        print(f"{name}:")
        print(f"  input term:     1.0 * W_ih[4] = {inp:+.2f}")
        print(f"  recurrent term: h[6] @ W_hh[4,:] = {recur:+.2f}")
        print(f"  total pre:      {total:+.2f}")
        print(f"  h[4,7]:         {h_val:.2f}")
        print()

        # Break down recurrent by source neurons
        print(f"  Top recurrent contributions to n4 at t=7:")
        contribs = [(j, h6[j] * W_hh[4, j]) for j in range(16)]
        contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        for j, c in contribs[:8]:
            if abs(c) > 0.01:
                grp = 'comp' if j in [1,6,7,8] else ('wave' if j in [0,10,11,12,14] else 'other')
                print(f"    n{j:<3} ({grp:>5}): h={h6[j]:>8.2f} × W_hh[4,{j}]={W_hh[4,j]:>+7.3f} = {c:>+8.2f}")
        print()

    # The key difference
    diff_h6 = trajs['Both'][:, 6] - trajs['1AM only'][:, 6]
    diff_recur = diff_h6 @ W_hh[4, :]
    print(f"Difference in n4's recurrent term: {diff_recur:+.2f}")
    print(f"This comes from the 2AM cascade still echoing in h[6].")


# =============================================================================
# PART 4: The critical question — does the 1AM n4 kick ADD to n2?
# =============================================================================
def does_1am_add_to_n2(model):
    section("PART 4: THE CRITICAL QUESTION — DOES 1AM's n4 KICK ADD TO n2?")
    print("Compare n2 values at t=8 and t=9 (after 1AM) across all three cases.")
    print("If 1AM adds a SECOND dose to n2, we should see n2(both) > n2(2AM only).\n")

    W_ih = model.rnn.weight_ih_l0.data.squeeze().numpy()
    W_hh = model.rnn.weight_hh_l0.data.numpy()

    cases = {
        '2AM only': [(3, 0.8)],
        '1AM only': [(7, 1.0)],
        'Both': [(3, 0.8), (7, 1.0)],
    }

    trajs = {}
    for name, inputs in cases.items():
        x = th.zeros(1, 10)
        for pos, val in inputs:
            x[0, pos] = val
        h, pre = get_full_trajectory(model, x)
        trajs[name] = h[0].numpy()

    print("--- n2 values at key timesteps ---")
    print(f"{'':>12} {'t=6':>8} {'t=7':>8} {'t=8':>8} {'t=9':>8}")
    print("-" * 46)
    for name in cases:
        print(f"{name:>12}", end="")
        for t in [6, 7, 8, 9]:
            print(f" {trajs[name][2, t]:>7.2f}", end="")
        print()

    # Does n2(both) = n2(2AM) + n2(1AM)?
    print(f"\n--- Is n2 additive? ---")
    for t in [7, 8, 9]:
        n2_2am = trajs['2AM only'][2, t]
        n2_1am = trajs['1AM only'][2, t]
        n2_both = trajs['Both'][2, t]
        print(f"  t={t}: n2(both)={n2_both:.2f}, n2(2AM)+n2(1AM)={n2_2am+n2_1am:.2f}, "
              f"diff={n2_both - n2_2am - n2_1am:+.2f}")

    # The key: does the 1AM cause n2 to GROW beyond the 2AM-only case?
    print(f"\n--- Does 1AM make n2 bigger than 2AM-only? ---")
    for t in [7, 8, 9]:
        n2_2am = trajs['2AM only'][2, t]
        n2_both = trajs['Both'][2, t]
        print(f"  t={t}: n2(both)={n2_both:.2f} vs n2(2AM only)={n2_2am:.2f}, "
              f"Δ={n2_both - n2_2am:+.2f}")


# =============================================================================
# PART 5: n4→n2→comp chain for BOTH impulses, step by step
# =============================================================================
def full_chain_both_impulses(model):
    section("PART 5: FULL n4→n2→COMP CHAIN — BOTH IMPULSES")
    print("Trace n4, n2, and n7 step by step for 'both' case.")
    print("Show: which impulse drives which signal through the chain.\n")

    W_ih = model.rnn.weight_ih_l0.data.squeeze().numpy()
    W_hh = model.rnn.weight_hh_l0.data.numpy()

    x = th.zeros(1, 10); x[0, 3] = 0.8; x[0, 7] = 1.0
    h, pre = get_full_trajectory(model, x)
    h = h[0].numpy()
    pre = pre[0].numpy()

    print(f"{'t':>3} {'x':>5} {'n4':>8} {'n2':>8} {'n7':>8} {'n7_pre':>8} {'n7 from n2':>11} {'n7 from n4':>11} {'event'}")
    print("-" * 80)

    for t in range(10):
        x_t = x[0, t].item()
        n4_h = h[4, t]
        n2_h = h[2, t]
        n7_h = h[7, t]
        n7_pre = pre[7, t]

        # n7's contributions from n2 and n4 at this timestep
        if t > 0:
            n7_from_n2 = h[2, t-1] * W_hh[7, 2]
            n7_from_n4 = h[4, t-1] * W_hh[7, 4]
        else:
            n7_from_n2 = 0
            n7_from_n4 = 0

        event = ""
        if t == 3: event = "2AM arrives"
        elif t == 4: event = "1st n4→n2 pass"
        elif t == 7: event = "1AM arrives, clips n7"
        elif t == 8: event = "2nd n4→n2 pass + comp rebuild"

        print(f"{t:>3} {x_t:>5.1f} {n4_h:>8.2f} {n2_h:>8.2f} {n7_h:>8.2f} {n7_pre:>+8.2f} {n7_from_n2:>+11.2f} {n7_from_n4:>+11.2f}  {event}")


# =============================================================================
# PART 6: Compare n4 values at both firings — is 2nd firing different?
# =============================================================================
def n4_both_firings(model):
    section("PART 6: n4 AT BOTH FIRINGS — HOW ARE THEY DIFFERENT?")
    print("First firing (t=3, 2AM=0.8): n4 starts from h=0, gets kicked by input.")
    print("Second firing (t=7, 1AM=1.0): n4 starts from h[6] which has cascade residue.")
    print()
    print("If n4 at t=7 is SMALLER or DIFFERENT than a fresh firing, that's the key.\n")

    W_ih = model.rnn.weight_ih_l0.data.squeeze().numpy()
    W_hh = model.rnn.weight_hh_l0.data.numpy()

    x = th.zeros(1, 10); x[0, 3] = 0.8; x[0, 7] = 1.0
    h, pre = get_full_trajectory(model, x)
    h = h[0].numpy()
    pre = pre[0].numpy()

    # First firing: t=3
    print("FIRST FIRING (t=3, 2AM=0.8):")
    h_prev = h[:, 2] if 3 > 0 else np.zeros(16)  # h at t=2
    h_prev = np.zeros(16)  # actually t=2 has nothing yet... let me check
    # Before 2AM, everything is 0 (no input before t=3)
    if True:
        h_prev_3 = h[:, 2]  # h at t=2
    recur_3 = h_prev_3 @ W_hh[4, :]
    inp_3 = 0.8 * W_ih[4]
    print(f"  h_prev[4] (at t=2) = {h_prev_3[4]:.4f}")
    print(f"  recurrent = {recur_3:+.2f}")
    print(f"  input = 0.8 × {W_ih[4]:.2f} = {inp_3:+.2f}")
    print(f"  pre = {pre[4, 3]:+.2f}")
    print(f"  h[4,3] = {h[4, 3]:.2f}")

    # Second firing: t=7
    print(f"\nSECOND FIRING (t=7, 1AM=1.0):")
    h_prev_7 = h[:, 6]
    recur_7 = h_prev_7 @ W_hh[4, :]
    inp_7 = 1.0 * W_ih[4]
    print(f"  h_prev[4] (at t=6) = {h_prev_7[4]:.4f}")
    print(f"  recurrent = {recur_7:+.2f}")
    print(f"  input = 1.0 × {W_ih[4]:.2f} = {inp_7:+.2f}")
    print(f"  pre = {pre[4, 7]:+.2f}")
    print(f"  h[4,7] = {h[4, 7]:.2f}")

    # Fresh 1AM-only firing
    x_1am = th.zeros(1, 10); x_1am[0, 7] = 1.0
    h_1am, pre_1am = get_full_trajectory(model, x_1am)
    h_1am = h_1am[0].numpy()
    pre_1am = pre_1am[0].numpy()

    print(f"\nFRESH 1AM-ONLY FIRING (t=7, 1.0):")
    h_prev_7f = h_1am[:, 6]
    recur_7f = h_prev_7f @ W_hh[4, :]
    inp_7f = 1.0 * W_ih[4]
    print(f"  h_prev[4] (at t=6) = {h_prev_7f[4]:.4f}")
    print(f"  recurrent = {recur_7f:+.2f}")
    print(f"  input = {inp_7f:+.2f}")
    print(f"  pre = {pre_1am[4, 7]:+.2f}")
    print(f"  h[4,7] = {h_1am[4, 7]:.2f}")

    # Compare
    print(f"\nCOMPARISON:")
    print(f"  n4(both)@t=7  = {h[4,7]:.2f}")
    print(f"  n4(fresh)@t=7 = {h_1am[4,7]:.2f}")
    print(f"  Difference     = {h[4,7] - h_1am[4,7]:+.2f}")
    print(f"\n  The 2AM cascade residue in h[6] CHANGES n4's recurrent term.")
    print(f"  recurrent(both) = {recur_7:+.2f} vs recurrent(fresh) = {recur_7f:+.2f}")
    print(f"  Δrecurrent = {recur_7 - recur_7f:+.2f}")


# =============================================================================
# PART 7: What n2 actually gets from n4 at t=8 — both firings contribute
# =============================================================================
def n2_receives_from_n4(model):
    section("PART 7: WHAT n2 RECEIVES FROM n4 AT EACH STEP")
    print("n2 at t+1 gets: self × 0.97 + n4[t] × 1.73 + other contributions")
    print("Track the n4 contribution to n2 at EVERY timestep.\n")

    W_ih = model.rnn.weight_ih_l0.data.squeeze().numpy()
    W_hh = model.rnn.weight_hh_l0.data.numpy()

    x = th.zeros(1, 10); x[0, 3] = 0.8; x[0, 7] = 1.0
    h, pre = get_full_trajectory(model, x)
    h = h[0].numpy()
    pre = pre[0].numpy()

    print(f"{'t':>3} {'n4[t]':>8} {'→n2 via n4':>11} {'n2[t]':>8} {'n2_self':>8} {'n2_inp':>8} {'n2_other':>9} {'n2_total':>9}")
    print("-" * 80)

    for t in range(10):
        n4_t = h[4, t]
        n2_t = h[2, t]

        if t < 9:
            # What n4[t] contributes to n2 at t+1
            n4_to_n2 = n4_t * W_hh[2, 4]
        else:
            n4_to_n2 = 0  # no next step

        # n2's pre-act decomposition at this t
        if t > 0:
            n2_self = h[2, t-1] * W_hh[2, 2]
            n2_from_n4 = h[4, t-1] * W_hh[2, 4]
            n2_inp = x[0, t].item() * W_ih[2]
            n2_other = pre[2, t] - n2_self - n2_from_n4 - n2_inp
        else:
            n2_self = 0
            n2_from_n4 = 0
            n2_inp = x[0, t].item() * W_ih[2]
            n2_other = pre[2, t] - n2_inp

        event = ""
        if t == 3: event = " <-- 2AM"
        elif t == 4: event = " <-- 1st n4→n2"
        elif t == 7: event = " <-- 1AM"
        elif t == 8: event = " <-- 2nd n4→n2"

        print(f"{t:>3} {n4_t:>8.2f} {n4_to_n2:>+11.2f} {n2_t:>8.2f} {n2_self:>+8.2f} {n2_inp:>+8.2f} {n2_other:>+9.2f} {pre[2,t]:>+9.2f}{event}")

    print(f"\nKey: n4→n2 via W_hh[2,4] = {W_hh[2,4]:+.4f}")
    print(f"     n2 self-recurrence   = {W_hh[2,2]:+.4f}")


# =============================================================================
# PART 8: The bottom line — does the 2nd kick HELP or HURT the answer?
# =============================================================================
def bottom_line(model):
    section("PART 8: THE BOTTOM LINE — DOES THE 2ND KICK HELP OR HURT?")
    print("Compare the final logits for three cases.")
    print("If the 2nd n4 kick corrupts things, 'both' should be worse than '2AM only'.\n")

    W_ih = model.rnn.weight_ih_l0.data.squeeze().numpy()
    W_hh = model.rnn.weight_hh_l0.data.numpy()
    W_out = model.linear.weight.data.numpy()

    cases = {
        '2AM only (0.8@3)': [(3, 0.8)],
        '1AM only (1.0@7)': [(7, 1.0)],
        'Both (0.8@3, 1.0@7)': [(3, 0.8), (7, 1.0)],
    }

    h_finals = {}
    for name, inputs in cases.items():
        x = th.zeros(1, 10)
        for pos, val in inputs:
            x[0, pos] = val
        h, _ = get_full_trajectory(model, x)
        h_finals[name] = h[0, :, 9].numpy()

    print("--- h_final for key neurons ---")
    key_neurons = [2, 4, 7, 8, 0, 10, 11]
    print(f"{'':>22}", end="")
    for n in key_neurons:
        grp = 'comp' if n in [1,6,7,8] else ('wave' if n in [0,10,11,12,14] else 'other')
        print(f"  n{n}({grp[:1]})", end="")
    print()
    for name in cases:
        print(f"{name:>22}", end="")
        for n in key_neurons:
            print(f"  {h_finals[name][n]:>7.2f}", end="")
        print()

    # Logits
    print(f"\n--- Logits (target = position 3, the 2AM position) ---")
    print(f"{'':>22}", end="")
    for p in range(10):
        print(f"  pos{p}", end="")
    print()
    for name in cases:
        logits = h_finals[name] @ W_out.T
        pred = np.argmax(logits)
        print(f"{name:>22}", end="")
        for p in range(10):
            marker = "*" if p == pred else " "
            print(f" {logits[p]:>5.1f}{marker}", end="")
        print()

    print(f"\nTarget answer: position 3 (the 2AM position)")


def main():
    model = load_local_model()

    n4_fires_twice(model)
    n4_second_kick_to_n2(model)
    n4_at_1am_decomposed(model)
    does_1am_add_to_n2(model)
    full_chain_both_impulses(model)
    n4_both_firings(model)
    n2_receives_from_n4(model)
    bottom_line(model)


if __name__ == "__main__":
    main()
