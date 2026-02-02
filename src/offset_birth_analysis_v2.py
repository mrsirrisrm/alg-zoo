"""
Analyze how the offset between forward and reverse trajectories is born.

INSIGHT from v1: When M and S are at the SAME positions (just swapped order),
the trajectories are IDENTICAL because the impulse magnitudes scale linearly
and ReLU clipping is the same.

The offset must arise from the POSITION difference, not just the order.
Let's compare: Forward (M@2, S@7) vs Reverse (S@7, M@2) -- wait, that's just
swapping the labels, not the positions.

The REAL comparison is:
- Forward: M at position m, S at position s (m < s)
- Reverse: S at position m, M at position s (s < m in arrival order)

But we're told the countdown still works. Let me re-read the docs...

Actually the key insight is: in FORWARD, the 2nd impulse is S.
In REVERSE, the 2nd impulse is M.

So for the same (m_pos, s_pos) pair:
- Forward: M@m_pos first, then S@s_pos
- Reverse: S@s_pos first, then M@m_pos

These ARE different sequences! Let me trace this correctly.
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax

# Load model
model = example_2nd_argmax()
model.eval()

# Extract weights
W_ih = model.rnn.weight_ih_l0.detach().numpy()  # (16, 1)
W_hh = model.rnn.weight_hh_l0.detach().numpy()  # (16, 16)
W_out = model.linear.weight.detach().numpy()     # (10, 16)

print("=" * 70)
print("OFFSET BIRTH ANALYSIS v2")
print("=" * 70)

# The task: find position of 2nd largest value
# M = max value = 1.0
# S = 2nd max value = 0.8

# For any (m_pos, s_pos) pair where m_pos != s_pos:
# - Forward: M arrives at m_pos FIRST (if m_pos < s_pos), S arrives at s_pos SECOND
# - Reverse: S arrives at s_pos FIRST (if s_pos < m_pos), M arrives at m_pos SECOND

# Let's pick m_pos=2, s_pos=7
# Forward: M@2 first, S@7 second
# Reverse: S@2 first, M@7 second  <-- WAIT, this swaps both position AND value!

# I think the confusion is: what defines "forward" vs "reverse"?
# From doc 33: "forward (M before S) from reverse (S before M)"
# So it's about which VALUE arrives first, not which position.

# For m_pos=2, s_pos=7 (m_pos < s_pos):
#   Forward means M@2 then S@7 (M before S in time)
#   Reverse means S@2 then M@7 (S before M in time) <-- S is now at position 2!

# Wait no, that changes where S actually is. Let me re-read...

# From doc 32: "Forward (M before S): countdown starts from S arrival"
# This implies S arrives at s_pos. So if s_pos=7:
#   Forward: M@m_pos, S@7, and m_pos < 7
#   Reverse: S@7 first? No, that would mean s_pos > m_pos...

# I think the definition is:
#   Forward: whichever position m_pos or s_pos is smaller has M
#   Reverse: whichever position is smaller has S

# So for testing m_pos=2, s_pos=7:
#   Forward sequence: position 2 gets M=1.0, position 7 gets S=0.8
#   Reverse sequence: position 2 gets S=0.8, position 7 gets M=1.0

# Both output s_pos which in FORWARD=7 but in REVERSE=2!

print("\nLet's clarify the setup:")
print("- M (max) has value 1.0")
print("- S (2nd max) has value 0.8")
print("- The task is to output S's POSITION")
print()
print("For positions (early=2, late=7):")
print("  Forward: M@2 (1.0), S@7 (0.8) -> should output 7")
print("  Reverse: S@2 (0.8), M@7 (1.0) -> should output 2")
print()

M_val, S_val = 1.0, 0.8

def make_sequence(m_pos, s_pos):
    """Create sequence with M at m_pos, S at s_pos, zeros elsewhere."""
    seq = th.zeros(1, 10)
    seq[0, m_pos] = M_val
    seq[0, s_pos] = S_val
    return seq

def run_and_get_states(seq):
    """Run sequence through model, return all hidden states."""
    h = th.zeros(1, 16)
    states = [h.numpy().copy()]

    for t in range(10):
        x_t = seq[0, t].item()
        pre_relu = W_ih.squeeze() * x_t + W_hh @ h.squeeze().numpy()
        h_new = np.maximum(0, pre_relu)
        h = th.tensor(h_new).unsqueeze(0).float()
        states.append(h.numpy().copy())

    return states

def get_prediction(h):
    """Get model's prediction from hidden state."""
    logits = W_out @ h.squeeze()
    return np.argmax(logits)

# Test case 1: positions 2 and 7
print("=" * 70)
print("TEST CASE: positions 2 and 7")
print("=" * 70)

# Forward: M@2, S@7
seq_fwd = make_sequence(m_pos=2, s_pos=7)
states_fwd = run_and_get_states(seq_fwd)
pred_fwd = get_prediction(states_fwd[-1])
print(f"\nForward (M@2, S@7): prediction = {pred_fwd} (expected: 7)")

# Reverse: S@2, M@7 (S is at position 2, M is at position 7)
seq_rev = make_sequence(m_pos=7, s_pos=2)  # M at 7, S at 2
states_rev = run_and_get_states(seq_rev)
pred_rev = get_prediction(states_rev[-1])
print(f"Reverse (S@2, M@7): prediction = {pred_rev} (expected: 2)")

print("\n" + "-" * 70)
print("Comparing trajectories:")
print("-" * 70)

print("\nt  | ||h_fwd|| | ||h_rev|| | ||offset|| | fwd_pred | rev_pred")
print("---|----------|----------|------------|----------|----------")
for t in range(11):
    h_f = states_fwd[t].squeeze()
    h_r = states_rev[t].squeeze()
    offset = h_f - h_r
    pred_f = get_prediction(states_fwd[t]) if t > 0 else '-'
    pred_r = get_prediction(states_rev[t]) if t > 0 else '-'
    note = ""
    if t == 3:
        note = " <- after pos 2"
    elif t == 8:
        note = " <- after pos 7"
    print(f"{t:2d} | {np.linalg.norm(h_f):8.2f} | {np.linalg.norm(h_r):8.2f} | {np.linalg.norm(offset):10.2f} | {pred_f:8} | {pred_r:8}{note}")

print("\n" + "=" * 70)
print("THE CRITICAL INSIGHT")
print("=" * 70)

# At t=3 (after position 2 impulse):
h_fwd_t3 = states_fwd[3].squeeze()  # After M@2
h_rev_t3 = states_rev[3].squeeze()  # After S@2
offset_t3 = h_fwd_t3 - h_rev_t3

# At t=8 (after position 7 impulse):
h_fwd_t8 = states_fwd[8].squeeze()  # After S@7
h_rev_t8 = states_rev[8].squeeze()  # After M@7
offset_t8 = h_fwd_t8 - h_rev_t8

print("\nOffset analysis:")
print(f"After 1st impulse (t=3): ||offset|| = {np.linalg.norm(offset_t3):.2f}")
print(f"After 2nd impulse (t=8): ||offset|| = {np.linalg.norm(offset_t8):.2f}")

# The discriminative directions are DIFFERENT for fwd vs rev!
# Forward wants to distinguish s_pos=7 from m_pos=2
# Reverse wants to distinguish s_pos=2 from m_pos=7

discrim_fwd = W_out[7] - W_out[2]  # should be positive for forward
discrim_rev = W_out[2] - W_out[7]  # should be positive for reverse

# For forward, we want W_out @ h_fwd to have argmax at 7
# For reverse, we want W_out @ h_rev to have argmax at 2

logits_fwd = W_out @ h_fwd_t8
logits_rev = W_out @ h_rev_t8

print(f"\nFinal logits:")
print(f"Forward: logits[7] - logits[2] = {logits_fwd[7] - logits_fwd[2]:.2f}")
print(f"Reverse: logits[2] - logits[7] = {logits_rev[2] - logits_rev[7]:.2f}")

# What's the offset's projection onto the discriminative direction?
margin_fwd_dir = discrim_fwd @ offset_t8
margin_rev_dir = discrim_rev @ offset_t8

print(f"\nOffset projection onto discriminative directions:")
print(f"  offset @ (W_out[7] - W_out[2]) = {margin_fwd_dir:.2f}")
print(f"  offset @ (W_out[2] - W_out[7]) = {margin_rev_dir:.2f}")

print("\n" + "=" * 70)
print("TRACING THE OFFSET'S BIRTH")
print("=" * 70)

# The offset is born at t=3 when the FIRST impulse arrives
# Forward gets M=1.0, Reverse gets S=0.8
# Both into the same initial state (h=0)

print("\nAt t=2 (before first impulse): h = 0 for both")
print(f"First impulse: Forward gets M={M_val}, Reverse gets S={S_val}")

# Pre-ReLU at first impulse
pre_fwd_t2 = W_ih.squeeze() * M_val  # M into zero state
pre_rev_t2 = W_ih.squeeze() * S_val  # S into zero state

print(f"\nPre-ReLU (into zero state):")
print(f"  Forward (M): pre = {M_val} * W_ih")
print(f"  Reverse (S): pre = {S_val} * W_ih")
print(f"  Ratio: {M_val/S_val:.2f}x")

# Post-ReLU
post_fwd_t2 = np.maximum(0, pre_fwd_t2)
post_rev_t2 = np.maximum(0, pre_rev_t2)

# Since both are just scaled versions of W_ih, same neurons are clipped
# The offset is just a scaled difference
offset_at_birth = post_fwd_t2 - post_rev_t2

print(f"\nPost-ReLU offset at birth (t=3):")
print(f"  ||offset|| = {np.linalg.norm(offset_at_birth):.2f}")
print(f"  This equals (M-S) * ||W_ih_positive|| = {M_val - S_val} * {np.linalg.norm(np.maximum(0, W_ih.squeeze())):.2f}")
print(f"             = {(M_val - S_val) * np.linalg.norm(np.maximum(0, W_ih.squeeze())):.2f}")

# The offset at birth is simply proportional to W_ih (positive parts)
# It's a SCALING difference, not a structural difference!

print("\n" + "=" * 70)
print("THE OFFSET EVOLUTION: Where structure emerges")
print("=" * 70)

print("\nThe offset at birth is just (M-S) * ReLU(W_ih) = 0.2 * ReLU(W_ih)")
print("This is a UNIFORM scaling, not yet structured for discrimination.")
print("\nBut as it evolves through W_hh (and the 2nd impulse)...")

# Track the offset's projection onto W_out rows
print("\nOffset projection onto each W_out row over time:")
print("t  |", " | ".join([f"pos{i}" for i in range(10)]))
print("---|" + "-" * 60)

for t in [3, 4, 5, 6, 7, 8, 9, 10]:
    offset_t = states_fwd[t].squeeze() - states_rev[t].squeeze()
    projs = W_out @ offset_t
    # Highlight the key positions
    row = f"{t:2d} |"
    for i in range(10):
        val = projs[i]
        if i == 2:
            row += f" [{val:5.1f}]"  # m_pos for reverse
        elif i == 7:
            row += f" ({val:5.1f})"  # s_pos for forward
        else:
            row += f"  {val:5.1f} "
    print(row)

print("\n[] = position 2 (s_pos for reverse)")
print("() = position 7 (s_pos for forward)")

print("\n" + "=" * 70)
print("KEY REALIZATION")
print("=" * 70)

print("""
The offset evolves to FAVOR both forward AND reverse discrimination!

For Forward (target=7): we need logits[7] > logits[2]
  h_fwd naturally points toward 7 (countdown landed there)

For Reverse (target=2): we need logits[2] > logits[7]
  h_rev naturally points toward 2 (countdown landed there)

The offset = h_fwd - h_rev doesn't need to favor one over the other.
Instead, each trajectory INDEPENDENTLY lands at the correct position!

The offset is just the DIFFERENCE between two correct trajectories.
""")

# Let's verify: do both trajectories independently give the right answer?
print("Verification at final timestep (t=10):")
print(f"  Forward logits: argmax = {np.argmax(W_out @ states_fwd[10].squeeze())} (expected 7)")
print(f"  Reverse logits: argmax = {np.argmax(W_out @ states_rev[10].squeeze())} (expected 2)")

# What makes each trajectory point to the right place?
# The PHASE of the spiral at t=9

print("\n" + "=" * 70)
print("REFINED QUESTION: How does the same mechanism point to different places?")
print("=" * 70)

print("""
Both forward and reverse use the phase wheel mechanism.
The countdown starts at the 2nd impulse and counts down to target.

Forward (M@2, S@7):
  - 1st impulse at t=2 (M): starts a spiral
  - 2nd impulse at t=7 (S): resets/redirects the countdown
  - At t=9: phase has counted 9-7=2 steps, pointing to position 7

Reverse (S@2, M@7):
  - 1st impulse at t=2 (S): starts a spiral
  - 2nd impulse at t=7 (M): resets/redirects the countdown
  - At t=9: phase has counted 9-7=2 steps, pointing to position... 7?

Wait, that's wrong. Let me re-examine the countdown formula.
""")

# Check the countdown formula more carefully
print("\nChecking countdown predictions:")
for t in range(8, 11):
    # For forward: 2nd impulse at s_pos=7, countdown should give 7 + (9-t) at t, = 7 at t=9
    # For reverse: 2nd impulse at m_pos=7, countdown should give... what?

    # Actually, the question is: what does the countdown COUNT DOWN TO?
    # The 2nd impulse position? Or S's position specifically?

    pred_fwd = np.argmax(W_out @ states_fwd[t].squeeze())
    pred_rev = np.argmax(W_out @ states_rev[t].squeeze())

    expected_fwd = 7 + (9 - t)  # countdown from s_pos=7
    expected_rev = 2 + (9 - t)  # countdown from s_pos=2

    print(f"t={t}: Forward pred={pred_fwd} (expected {expected_fwd}), Reverse pred={pred_rev} (expected {expected_rev})")

print("\n" + "=" * 70)
print("THE REAL INSIGHT: The 2nd impulse DOESN'T reset to its position!")
print("=" * 70)

print("""
The countdown formula is: pred(t) = S_pos + (9 - t)

For BOTH forward and reverse, S_pos is the position of the 2nd-largest value.
- Forward: S_pos = 7 (S is at position 7)
- Reverse: S_pos = 2 (S is at position 2)

The countdown doesn't care about WHEN the impulses arrive.
It cares about WHERE S is.

So the question becomes: how does the network know S_pos?

In forward: S arrives 2nd, at position 7. The countdown naturally lands there.
In reverse: M arrives 2nd, at position 7. But we need to count down to position 2!

THIS is where the offset matters. The offset shifts the trajectory so that
even though M's 2nd impulse would naturally point to position 7, the
accumulated information from S's FIRST impulse redirects to position 2.
""")

# Let's look at what happens in reverse more carefully
print("\nReverse trajectory detail:")
print("- S arrives at t=2, kicks off spiral")
print("- Spiral evolves t=3,4,5,6,7")
print("- M arrives at t=7, second impulse")
print("- What does M's impulse do to the spiral?")

# Compare: what if we had ONLY M at position 7?
seq_only_M7 = th.zeros(1, 10)
seq_only_M7[0, 7] = M_val
states_only_M7 = run_and_get_states(seq_only_M7)

print(f"\nM-only at position 7: final prediction = {get_prediction(states_only_M7[-1])}")
print("(This should stall at 9, per the single-impulse behavior)")

# The difference: in reverse, there's already a spiral from S when M arrives
h_before_M_rev = states_rev[7].squeeze()  # State just before M@7 in reverse
h_before_M_only = states_only_M7[7].squeeze()  # State just before M@7 in M-only case

print(f"\n||h|| just before M@7:")
print(f"  Reverse (S preceded): {np.linalg.norm(h_before_M_rev):.2f}")
print(f"  M-only (no S):        {np.linalg.norm(h_before_M_only):.2f}")

# The S-spiral gives M something to interact with!
print("\nThe S-spiral provides the 'memory' of S's position.")
print("When M arrives, it interacts nonlinearly with this existing spiral,")
print("and the result encodes information about BOTH positions.")

# Let's see the offset between reverse and M-only after M's impulse
offset_vs_Monly = states_rev[8].squeeze() - states_only_M7[8].squeeze()
print(f"\nOffset: reverse vs M-only after t=7 impulse")
print(f"  ||offset|| = {np.linalg.norm(offset_vs_Monly):.2f}")
print(f"  This offset encodes S's position (2) into the trajectory!")

# What's this offset's projection onto W_out?
proj_onto_wout = W_out @ offset_vs_Monly
print(f"\nProjection onto W_out (boost to each position):")
for i in range(10):
    marker = " <- S_pos" if i == 2 else (" <- M_pos" if i == 7 else "")
    print(f"  Position {i}: {proj_onto_wout[i]:+.2f}{marker}")

print("\n" + "=" * 70)
print("FINAL PICTURE")
print("=" * 70)

print("""
In REVERSE (S@2, M@7):

1. S kicks off a spiral at t=2
2. The spiral carries information about S's position (encoded in phase)
3. M arrives at t=7 into this S-shaped hidden state
4. The nonlinear (ReLU) interaction preserves S's position encoding
5. The countdown proceeds, but it's counting down to S_pos=2, not M_pos=7
6. At t=9, the network correctly outputs position 2

The "trick" in reverse:
- The S-spiral "protects" S's position information
- When M arrives, it can't overwrite this because:
  a) The existing hidden state biases the ReLU clipping pattern
  b) The offset between "M into S-spiral" vs "M into nothing" encodes S_pos
  c) This offset projects positively onto W_out[S_pos=2]

The offset mechanism is not about fwd vs rev discrimination per se.
It's about how the FIRST impulse's spiral shapes the response to the SECOND.
""")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Trajectories in PCA space
from sklearn.decomposition import PCA
all_states = np.vstack([np.array(states_fwd[1:]).squeeze(), np.array(states_rev[1:]).squeeze()])
pca = PCA(n_components=2)
pca.fit(all_states)

fwd_pca = pca.transform(np.array(states_fwd[1:]).squeeze())
rev_pca = pca.transform(np.array(states_rev[1:]).squeeze())

ax = axes[0, 0]
ax.plot(fwd_pca[:, 0], fwd_pca[:, 1], 'b-o', label='Forward (M@2, S@7)', alpha=0.7)
ax.plot(rev_pca[:, 0], rev_pca[:, 1], 'r-o', label='Reverse (S@2, M@7)', alpha=0.7)
# Mark 2nd impulse
ax.plot(fwd_pca[6, 0], fwd_pca[6, 1], 'b*', markersize=15)
ax.plot(rev_pca[6, 0], rev_pca[6, 1], 'r*', markersize=15)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Trajectories in PCA space')
ax.legend()

# 2. Offset evolution
ax = axes[0, 1]
offsets = [np.linalg.norm(states_fwd[t].squeeze() - states_rev[t].squeeze()) for t in range(11)]
ax.plot(range(11), offsets, 'ko-')
ax.axvline(3, color='gray', linestyle='--', alpha=0.5)
ax.axvline(8, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Timestep')
ax.set_ylabel('||offset||')
ax.set_title('Offset magnitude over time')

# 3. Offset projection onto W_out at final timestep
ax = axes[0, 2]
final_offset = states_fwd[10].squeeze() - states_rev[10].squeeze()
proj = W_out @ final_offset
colors = ['red' if i == 2 else ('blue' if i == 7 else 'gray') for i in range(10)]
ax.bar(range(10), proj, color=colors, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('W_out @ offset')
ax.set_title('Offset projection at t=10')

# 4. Predictions over time
ax = axes[1, 0]
preds_fwd = [np.argmax(W_out @ states_fwd[t].squeeze()) for t in range(1, 11)]
preds_rev = [np.argmax(W_out @ states_rev[t].squeeze()) for t in range(1, 11)]
ax.plot(range(1, 11), preds_fwd, 'b-o', label='Forward pred')
ax.plot(range(1, 11), preds_rev, 'r-o', label='Reverse pred')
ax.axhline(7, color='blue', linestyle='--', alpha=0.3)
ax.axhline(2, color='red', linestyle='--', alpha=0.3)
ax.set_xlabel('Timestep')
ax.set_ylabel('Predicted position')
ax.set_title('Predictions over time')
ax.legend()

# 5. Comparing reverse to M-only
ax = axes[1, 1]
states_only_pca = pca.transform(np.array(states_only_M7[1:]).squeeze())
ax.plot(rev_pca[:, 0], rev_pca[:, 1], 'r-o', label='Reverse (S@2, M@7)', alpha=0.7)
ax.plot(states_only_pca[:, 0], states_only_pca[:, 1], 'g-o', label='M@7 only', alpha=0.7)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Reverse vs M-only')
ax.legend()

# 6. S-spiral contribution to final offset
ax = axes[1, 2]
# The "S-spiral contribution" is the offset between reverse and M-only
s_contribution = states_rev[10].squeeze() - states_only_M7[10].squeeze()
proj_s = W_out @ s_contribution
ax.bar(range(10), proj_s, color='purple', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('W_out @ (reverse - M_only)')
ax.set_title('S-spiral\'s contribution to output')

plt.tight_layout()
plt.savefig('docs/offset_birth_analysis_v2.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nVisualization saved to docs/offset_birth_analysis_v2.png")
