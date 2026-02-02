"""
Let's step back and look at fwd vs rev more directly.

The question: how does forward "let S take over" while reverse "protects S"?

Maybe the offset framing is obscuring what's actually happening.
Let's look at each trajectory independently.
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
print("DIRECT COMPARISON: What does each trajectory actually do?")
print("=" * 70)

# Gap = 1 case
pos1, pos2 = 3, 4

print(f"\nPositions: {pos1} and {pos2}")
print(f"Forward: M@{pos1}, S@{pos2} -> target = {pos2}")
print(f"Reverse: S@{pos1}, M@{pos2} -> target = {pos1}")

states_fwd = run_stepwise([(pos1, M_val), (pos2, S_val)])
states_rev = run_stepwise([(pos1, S_val), (pos2, M_val)])

print("\n" + "=" * 70)
print("FORWARD TRAJECTORY: M@3 first, then S@4")
print("=" * 70)

print("\nThe question: How does S@4 'take over' from M@3?")
print("\nt  | event      | ||h|| | pred | what's encoded?")
print("---|------------|-------|------|----------------")

for t in range(11):
    h = states_fwd[t]
    pred = np.argmax(W_out @ h) if t > 0 else "-"

    if t <= 3:
        event = "zeros" if t < 3 else "zeros"
        encoded = "nothing"
    elif t == 4:
        event = "after M@3"
        encoded = "M's position (3)? or just energy?"
    elif t == 5:
        event = "after S@4"
        encoded = "S takes over? Both?"
    else:
        event = "evolving"
        # Check what position the state is pointing to
        logits = W_out @ h
        top2 = np.argsort(logits)[-2:][::-1]
        encoded = f"pointing to {pred}, runner-up {top2[1]}"

    print(f"{t:2d} | {event:10s} | {np.linalg.norm(h):5.2f} | {str(pred):4s} | {encoded}")

print("\n" + "=" * 70)
print("REVERSE TRAJECTORY: S@3 first, then M@4")
print("=" * 70)

print("\nThe question: How does S@3 'survive' M@4?")
print("\nt  | event      | ||h|| | pred | what's encoded?")
print("---|------------|-------|------|----------------")

for t in range(11):
    h = states_rev[t]
    pred = np.argmax(W_out @ h) if t > 0 else "-"

    if t <= 3:
        event = "zeros"
        encoded = "nothing"
    elif t == 4:
        event = "after S@3"
        encoded = "S's position (3)? or just energy?"
    elif t == 5:
        event = "after M@4"
        encoded = "Does M overwrite? Or blend?"
    else:
        event = "evolving"
        logits = W_out @ h
        top2 = np.argsort(logits)[-2:][::-1]
        encoded = f"pointing to {pred}, runner-up {top2[1]}"

    print(f"{t:2d} | {event:10s} | {np.linalg.norm(h):5.2f} | {str(pred):4s} | {encoded}")

print("\n" + "=" * 70)
print("THE CRITICAL COMPARISON: Right after 2nd impulse (t=5)")
print("=" * 70)

h_fwd_5 = states_fwd[5]
h_rev_5 = states_rev[5]

logits_fwd = W_out @ h_fwd_5
logits_rev = W_out @ h_rev_5

print("\nForward (after S@4 into M-spiral):")
print(f"  Target: position 4")
print(f"  Prediction: {np.argmax(logits_fwd)}")
print(f"  Logits: pos3={logits_fwd[3]:.1f}, pos4={logits_fwd[4]:.1f}, gap={logits_fwd[4]-logits_fwd[3]:.1f}")

print("\nReverse (after M@4 into S-spiral):")
print(f"  Target: position 3")
print(f"  Prediction: {np.argmax(logits_rev)}")
print(f"  Logits: pos3={logits_rev[3]:.1f}, pos4={logits_rev[4]:.1f}, gap={logits_rev[3]-logits_rev[4]:.1f}")

print("\n" + "=" * 70)
print("HYPOTHESIS: It's about the COUNTDOWN, not the offset")
print("=" * 70)

print("""
The phase wheel mechanism says: pred(t) = target + (9 - t)

For FORWARD (target=4):
  At t=5 (right after S), pred should be 4 + (9-5) = 8

For REVERSE (target=3):
  At t=5 (right after M), pred should be 3 + (9-5) = 7

Let's check if this countdown is what's actually happening:
""")

print("\nFORWARD countdown check:")
print("t  | pred | expected (4 + 9-t) | match?")
print("---|------|-------------------|-------")
for t in range(5, 11):
    pred = np.argmax(W_out @ states_fwd[t])
    expected = 4 + (9 - t)
    match = "✓" if pred == expected else "✗"
    print(f"{t:2d} | {pred:4d} | {expected:17d} | {match}")

print("\nREVERSE countdown check:")
print("t  | pred | expected (3 + 9-t) | match?")
print("---|------|-------------------|-------")
for t in range(5, 11):
    pred = np.argmax(W_out @ states_rev[t])
    expected = 3 + (9 - t)
    match = "✓" if pred == expected else "✗"
    print(f"{t:2d} | {pred:4d} | {expected:17d} | {match}")

print("\n" + "=" * 70)
print("KEY QUESTION: How does each trajectory 'know' its target?")
print("=" * 70)

print("""
Both trajectories run the same countdown mechanism.
The difference is WHERE the countdown is anchored:
  - Forward anchors to position 4 (where S arrived)
  - Reverse anchors to position 3 (where S arrived)

The countdown always anchors to S's position, regardless of:
  - Whether S arrived first or second
  - Whether M (larger) arrived after S

So the question becomes:
  HOW does the 2nd impulse get the countdown to anchor correctly?
""")

print("\n" + "=" * 70)
print("DECOMPOSITION: What determines the countdown anchor?")
print("=" * 70)

# The state right after the 2nd impulse is:
# h = ReLU(W_ih * x + W_hh @ h_prev)
#
# For forward: x = S = 0.8, h_prev = M-spiral after 1 step
# For reverse: x = M = 1.0, h_prev = S-spiral after 1 step

print("\nAt the 2nd impulse:")
print("  Forward: h_5 = ReLU(W_ih * 0.8 + W_hh @ h_M_spiral)")
print("  Reverse: h_5 = ReLU(W_ih * 1.0 + W_hh @ h_S_spiral)")

h_M_spiral = states_fwd[4]  # M-spiral after 1 step
h_S_spiral = states_rev[4]  # S-spiral after 1 step

# Decompose the contributions
contrib_ih_fwd = W_ih * S_val
contrib_hh_fwd = W_hh @ h_M_spiral

contrib_ih_rev = W_ih * M_val
contrib_hh_rev = W_hh @ h_S_spiral

print(f"\nForward contributions to pre-ReLU:")
print(f"  W_ih * S: ||{np.linalg.norm(contrib_ih_fwd):.2f}||, encodes... position 4? (where S is)")
print(f"  W_hh @ h_M: ||{np.linalg.norm(contrib_hh_fwd):.2f}||, encodes... position 3? (where M was)")

print(f"\nReverse contributions to pre-ReLU:")
print(f"  W_ih * M: ||{np.linalg.norm(contrib_ih_rev):.2f}||, encodes... position 4? (where M is)")
print(f"  W_hh @ h_S: ||{np.linalg.norm(contrib_hh_rev):.2f}||, encodes... position 3? (where S was)")

# But wait - W_ih doesn't encode position! It's the same regardless of position.
# The position information comes from WHEN the impulse arrives (the timestep).

print("\n" + "=" * 70)
print("REALIZATION: W_ih doesn't encode position!")
print("=" * 70)

print("""
W_ih * val is the SAME regardless of which position the impulse is at.
Position is encoded by WHEN the impulse arrives (which timestep).

So the countdown anchor is determined by:
  1. The timestep of the 2nd impulse (this sets where in the countdown we start)
  2. The accumulated spiral from the 1st impulse (this biases the phase)

The 2nd impulse's position IS encoded by the timestep it arrives at.
The 1st impulse's position is encoded in the spiral's phase.

For FORWARD:
  - 2nd impulse at t=4 says "start countdown from here"
  - S is at position 4, so countdown → 4 ✓

For REVERSE:
  - 2nd impulse at t=4 says "start countdown from here"
  - But we need countdown → 3, not 4!
  - The S-spiral (from position 3) must SHIFT the phase

This is where the offset matters: the S-spiral shifts the countdown target.
""")

print("\n" + "=" * 70)
print("TESTING: Single impulse at different positions")
print("=" * 70)

print("\nWhat does a single impulse at position p predict at t=9?")
print("pos | pred@t=9 | expected (countdown from pos)")
print("----|----------|------------------------------")

for pos in range(1, 9):
    states_single = run_stepwise([(pos, M_val)])
    pred = np.argmax(W_out @ states_single[10])
    expected = pos  # If countdown worked: pos + (9-9) = pos... but single impulse stalls!
    print(f"  {pos} | {pred:8d} | {expected} (but single impulse stalls at 9)")

print("\n" + "=" * 70)
print("FINAL PICTURE")
print("=" * 70)

print("""
The countdown mechanism requires TWO impulses to work properly.

With a single impulse: the spiral stalls and predicts 9.

With two impulses: the ReLU interaction between them creates a
proper countdown that lands on S's position.

The key asymmetry:
  - FORWARD: S's impulse (2nd) directly encodes S's position via timing
  - REVERSE: S's impulse (1st) encodes S's position in the spiral's phase,
             which persists through M's arrival and shifts the countdown

Both work because the network learned to:
1. Use the 2nd impulse timing as the countdown start
2. Use the 1st impulse's spiral to bias which position the countdown lands on
3. Combine these so that S's position (not M's) determines the final answer

In forward: S arrives 2nd, so S's timing directly sets the target.
In reverse: S arrives 1st, so S's spiral biases the target despite M arriving later.
""")
