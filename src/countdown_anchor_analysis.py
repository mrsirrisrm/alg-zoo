"""
The countdown is off by 1 from the simple formula.
Let's figure out what the actual anchor is.

Forward: predictions are 9,8,7,6,5,4 (landing on 4)
Reverse: predictions are 8,7,6,5,4,3 (landing on 3)

The countdowns are DIFFERENT - they differ by 1 at every step!
This is exactly the gap between the S positions (4 vs 3).
"""

import numpy as np
from alg_zoo import example_2nd_argmax

model = example_2nd_argmax()
model.eval()

W_ih = model.rnn.weight_ih_l0.detach().numpy().squeeze()
W_hh = model.rnn.weight_hh_l0.detach().numpy()
W_out = model.linear.weight.detach().numpy()

M_val, S_val = 1.0, 0.8

def run_stepwise(impulses):
    h = np.zeros(16)
    states = [h.copy()]
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        states.append(h.copy())
    return states

print("=" * 70)
print("COUNTDOWN ANCHOR ANALYSIS")
print("=" * 70)

# Test multiple gap sizes to find the pattern
print("\nTesting different gaps to find the countdown pattern...")
print("\n" + "-" * 70)

for gap in [1, 2, 3, 5]:
    first_pos = 2
    second_pos = first_pos + gap

    if second_pos >= 10:
        continue

    print(f"\nGAP = {gap}: positions {first_pos} and {second_pos}")

    # Forward: M@first, S@second -> target = second
    states_fwd = run_stepwise([(first_pos, M_val), (second_pos, S_val)])

    # Reverse: S@first, M@second -> target = first
    states_rev = run_stepwise([(first_pos, S_val), (second_pos, M_val)])

    print(f"\n  Forward (M@{first_pos}, S@{second_pos}) target={second_pos}:")
    print(f"    t: ", end="")
    for t in range(second_pos + 1, 11):
        print(f"{t:3d}", end=" ")
    print()
    print(f"    p: ", end="")
    for t in range(second_pos + 1, 11):
        pred = np.argmax(W_out @ states_fwd[t])
        print(f"{pred:3d}", end=" ")
    print()

    # What's the formula? pred = ? + (9 - t) landing on second_pos at t=10
    # At t=10: pred = second_pos, so ? = second_pos
    # At t=9: pred = second_pos + 1? Let's check
    print(f"    Formula check: pred = {second_pos} + (10-t)?")
    print(f"    exp:", end="")
    for t in range(second_pos + 1, 11):
        expected = second_pos + (10 - t)
        print(f"{expected:3d}", end=" ")
    print()

    print(f"\n  Reverse (S@{first_pos}, M@{second_pos}) target={first_pos}:")
    print(f"    t: ", end="")
    for t in range(second_pos + 1, 11):
        print(f"{t:3d}", end=" ")
    print()
    print(f"    p: ", end="")
    for t in range(second_pos + 1, 11):
        pred = np.argmax(W_out @ states_rev[t])
        print(f"{pred:3d}", end=" ")
    print()
    print(f"    Formula check: pred = {first_pos} + (10-t)?")
    print(f"    exp:", end="")
    for t in range(second_pos + 1, 11):
        expected = first_pos + (10 - t)
        print(f"{expected:3d}", end=" ")
    print()

print("\n" + "=" * 70)
print("INSIGHT: The countdown formula is pred = S_pos + (10 - t)")
print("=" * 70)

print("""
The countdown lands on S_pos at t=10 (not t=9!).

This makes sense: the RNN processes 10 timesteps (0-9), and we read
out at t=9 which gives us the state at index 10 in our states array.

The formula: pred(t) = S_pos + (10 - t)

At t=10 (final readout): pred = S_pos + 0 = S_pos ✓

Now the key question: how does each direction encode S_pos?
""")

print("\n" + "=" * 70)
print("COMPARING THE PHASE AT 2ND IMPULSE")
print("=" * 70)

# For gap=1: positions 3,4
first_pos, second_pos = 3, 4

states_fwd = run_stepwise([(first_pos, M_val), (second_pos, S_val)])
states_rev = run_stepwise([(first_pos, S_val), (second_pos, M_val)])

# Right after 2nd impulse (t = second_pos + 1 = 5)
t_after = second_pos + 1

h_fwd = states_fwd[t_after]
h_rev = states_rev[t_after]

pred_fwd = np.argmax(W_out @ h_fwd)
pred_rev = np.argmax(W_out @ h_rev)

print(f"\nRight after 2nd impulse (t={t_after}):")
print(f"  Forward: pred = {pred_fwd}")
print(f"  Reverse: pred = {pred_rev}")
print(f"  Difference: {pred_fwd - pred_rev}")

# According to formula: pred = S_pos + (10 - t)
# Forward S_pos = 4: pred = 4 + (10 - 5) = 9 ✓
# Reverse S_pos = 3: pred = 3 + (10 - 5) = 8 ✓

print(f"\nFormula check:")
print(f"  Forward: S_pos=4, expected = 4 + (10-{t_after}) = {4 + (10-t_after)}, actual = {pred_fwd}")
print(f"  Reverse: S_pos=3, expected = 3 + (10-{t_after}) = {3 + (10-t_after)}, actual = {pred_rev}")

print("\n" + "=" * 70)
print("THE PHASE DIFFERENCE IS ESTABLISHED IMMEDIATELY")
print("=" * 70)

print("""
Right after the 2nd impulse arrives at position 4:
  - Forward predicts 9 (countdown to S_pos=4)
  - Reverse predicts 8 (countdown to S_pos=3)

The phase difference of 1 appears IMMEDIATELY at the 2nd impulse!

This means the first impulse's spiral (M vs S) creates a phase
offset that persists when the second impulse arrives.

Forward: M@3 creates a spiral, S@4 arrives into it
  -> The countdown anchors to position 4 (where S arrived)

Reverse: S@3 creates a spiral, M@4 arrives into it
  -> The countdown anchors to position 3 (where S was!)
  -> M's arrival at position 4 does NOT reset the anchor to 4
""")

print("\n" + "=" * 70)
print("QUANTIFYING THE PHASE OFFSET")
print("=" * 70)

# The hidden state encodes phase. Let's see how they differ.
offset = h_fwd - h_rev
print(f"\n||h_fwd|| = {np.linalg.norm(h_fwd):.2f}")
print(f"||h_rev|| = {np.linalg.norm(h_rev):.2f}")
print(f"||offset|| = {np.linalg.norm(offset):.2f}")

# Project onto W_out to see the readout difference
logits_fwd = W_out @ h_fwd
logits_rev = W_out @ h_rev
logits_offset = W_out @ offset

print(f"\nLogit differences (fwd - rev) at key positions:")
for pos in [3, 4, 8, 9]:
    print(f"  Position {pos}: {logits_offset[pos]:+.2f}")

print(f"\nThe offset shifts the logits so that:")
print(f"  Forward's peak is 1 position higher than Reverse's")

print("\n" + "=" * 70)
print("SUMMARY: How does the first impulse set the phase?")
print("=" * 70)

# Let's trace back: what's different before the 2nd impulse?
h_fwd_before = states_fwd[second_pos]  # M-spiral
h_rev_before = states_rev[second_pos]  # S-spiral

print(f"\nBefore 2nd impulse (t={second_pos}):")
print(f"  Forward (M-spiral): ||h|| = {np.linalg.norm(h_fwd_before):.2f}")
print(f"  Reverse (S-spiral): ||h|| = {np.linalg.norm(h_rev_before):.2f}")
print(f"  Ratio: {np.linalg.norm(h_fwd_before) / np.linalg.norm(h_rev_before):.2f}x")

# The M-spiral is just 1.25x the S-spiral (same shape, different scale)
# But after the 2nd impulse, the phases diverge

print(f"\nThe M-spiral and S-spiral have the same SHAPE but different SCALE.")
print(f"(Both started from the same W_ih, just scaled by M=1.0 vs S=0.8)")

# Verify: are they proportional?
ratio = np.linalg.norm(h_fwd_before) / np.linalg.norm(h_rev_before)
h_fwd_normalized = h_fwd_before / np.linalg.norm(h_fwd_before)
h_rev_normalized = h_rev_before / np.linalg.norm(h_rev_before)
cos_sim = h_fwd_normalized.flatten() @ h_rev_normalized.flatten()
print(f"  Cosine similarity: {cos_sim:.4f} (1.0 = identical shape)")

print("""
The spirals ARE nearly identical in shape! The difference is just scale.

So how does the 2nd impulse create a phase offset of 1?

The answer: the SCALE difference interacts with ReLU nonlinearly.

When the 2nd impulse arrives:
  Forward: W_ih * S + W_hh @ (larger M-spiral)
  Reverse: W_ih * M + W_hh @ (smaller S-spiral)

The W_hh contributions differ by the 1.25x ratio.
The W_ih contributions differ by M/S = 1.25x in the OPPOSITE direction.

These don't cancel because:
1. They're in different directions (W_ih vs W_hh @ h)
2. ReLU clips them differently

The net effect: a phase offset of exactly 1 position.
""")

# Let's verify this is exactly 1 for all cases
print("\n" + "=" * 70)
print("VERIFICATION: Phase offset = gap = 1 for all pairs")
print("=" * 70)

print("\nFor each (first_pos, second_pos) with gap=1:")
print("first | second | fwd_pred@t=second+1 | rev_pred | diff")
print("------|--------|---------------------|----------|-----")

for first in range(9):
    second = first + 1
    states_f = run_stepwise([(first, M_val), (second, S_val)])
    states_r = run_stepwise([(first, S_val), (second, M_val)])

    pred_f = np.argmax(W_out @ states_f[second + 1])
    pred_r = np.argmax(W_out @ states_r[second + 1])

    print(f"  {first}   |   {second}    |         {pred_f}           |    {pred_r}     |  {pred_f - pred_r}")
