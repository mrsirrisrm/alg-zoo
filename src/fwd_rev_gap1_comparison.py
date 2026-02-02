"""
Compare forward vs reverse with gap=1.

Forward: M@3, S@4 - we WANT S to "take over"
Reverse: S@3, M@4 - we want S to "survive"

How do these differ mechanistically?
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
print("FORWARD vs REVERSE with gap=1")
print("=" * 70)

# Forward: M@3, S@4 -> should output 4
# Reverse: S@3, M@4 -> should output 3

m_pos_fwd, s_pos_fwd = 3, 4  # Forward: M first
s_pos_rev, m_pos_rev = 3, 4  # Reverse: S first (so s_pos=3, m_pos=4)

states_fwd = run_stepwise([(m_pos_fwd, M_val), (s_pos_fwd, S_val)])
states_rev = run_stepwise([(s_pos_rev, S_val), (m_pos_rev, M_val)])

print(f"\nForward: M@{m_pos_fwd}, S@{s_pos_fwd} -> target={s_pos_fwd}")
print(f"Reverse: S@{s_pos_rev}, M@{m_pos_rev} -> target={s_pos_rev}")

# Final predictions
pred_fwd = np.argmax(W_out @ states_fwd[-1])
pred_rev = np.argmax(W_out @ states_rev[-1])
print(f"\nFinal predictions: fwd={pred_fwd}, rev={pred_rev}")

print("\n" + "=" * 70)
print("TRAJECTORY COMPARISON")
print("=" * 70)

print("\nt  | ||h_fwd|| | ||h_rev|| | ||offset|| | pred_f | pred_r")
print("---|----------|----------|------------|--------|--------")
for t in range(11):
    h_f = states_fwd[t]
    h_r = states_rev[t]
    offset = h_f - h_r
    p_f = np.argmax(W_out @ h_f) if t > 0 else "-"
    p_r = np.argmax(W_out @ h_r) if t > 0 else "-"
    marker = ""
    if t == 4:
        marker = " <- 1st impulse"
    elif t == 5:
        marker = " <- 2nd impulse"
    print(f"{t:2d} | {np.linalg.norm(h_f):8.2f} | {np.linalg.norm(h_r):8.2f} | {np.linalg.norm(offset):10.2f} | {p_f:6} | {p_r:6}{marker}")

print("\n" + "=" * 70)
print("AT THE 2ND IMPULSE (t=4)")
print("=" * 70)

# State before 2nd impulse
h_fwd_before = states_fwd[4]  # After M@3, before S@4
h_rev_before = states_rev[4]  # After S@3, before M@4

print(f"\nState before 2nd impulse arrives:")
print(f"  Forward (M-spiral): ||h|| = {np.linalg.norm(h_fwd_before):.2f}")
print(f"  Reverse (S-spiral): ||h|| = {np.linalg.norm(h_rev_before):.2f}")
print(f"  Ratio: {np.linalg.norm(h_fwd_before) / np.linalg.norm(h_rev_before):.2f}")

# The offset at this point
offset_before = h_fwd_before - h_rev_before
print(f"\nOffset before 2nd impulse: ||offset|| = {np.linalg.norm(offset_before):.2f}")

# Decompose: offset = (M-S) * W_ih passed through one W_hh step
# After 1st impulse: h_fwd = ReLU(M * W_ih), h_rev = ReLU(S * W_ih)
# Since same neurons clipped: h_fwd - h_rev = (M-S) * ReLU(W_ih) = 0.2 * ReLU(W_ih)
impulse_diff = (M_val - S_val) * np.maximum(0, W_ih)
print(f"\nInitial offset (right after 1st impulse):")
print(f"  (M-S) * ReLU(W_ih) = 0.2 * ReLU(W_ih)")
print(f"  ||initial offset|| = {np.linalg.norm(impulse_diff):.2f}")

# After W_hh evolution:
evolved_offset = W_hh @ impulse_diff
print(f"\nAfter W_hh: ||W_hh @ initial_offset|| = {np.linalg.norm(evolved_offset):.2f}")

# But ReLU modifies this! Let's check actual vs linear prediction
actual_offset_t4 = states_fwd[4] - states_rev[4]
print(f"Actual offset at t=4: {np.linalg.norm(actual_offset_t4):.2f}")

print("\n" + "=" * 70)
print("WHAT HAPPENS WHEN 2ND IMPULSE ARRIVES")
print("=" * 70)

# Forward: S (0.8) into M-spiral
# Reverse: M (1.0) into S-spiral

print("\n--- FORWARD: S (0.8) into M-spiral ---")
pre_fwd = W_ih * S_val + W_hh @ h_fwd_before
whh_fwd = W_hh @ h_fwd_before
wih_fwd = W_ih * S_val
print(f"  ||W_hh @ h_M|| = {np.linalg.norm(whh_fwd):.2f}")
print(f"  ||W_ih * S||   = {np.linalg.norm(wih_fwd):.2f}")
print(f"  Ratio (M-spiral / S-input): {np.linalg.norm(whh_fwd) / np.linalg.norm(wih_fwd):.2f}")

print("\n--- REVERSE: M (1.0) into S-spiral ---")
pre_rev = W_ih * M_val + W_hh @ h_rev_before
whh_rev = W_hh @ h_rev_before
wih_rev = W_ih * M_val
print(f"  ||W_hh @ h_S|| = {np.linalg.norm(whh_rev):.2f}")
print(f"  ||W_ih * M||   = {np.linalg.norm(wih_rev):.2f}")
print(f"  Ratio (S-spiral / M-input): {np.linalg.norm(whh_rev) / np.linalg.norm(wih_rev):.2f}")

print("\n" + "=" * 70)
print("THE OFFSET DECOMPOSITION AT 2ND IMPULSE")
print("=" * 70)

# pre_fwd = W_ih * S + W_hh @ h_M
# pre_rev = W_ih * M + W_hh @ h_S
#
# offset_pre = pre_fwd - pre_rev
#            = W_ih * (S - M) + W_hh @ (h_M - h_S)
#            = W_ih * (-0.2) + W_hh @ offset_before

offset_from_impulse = W_ih * (S_val - M_val)  # -0.2 * W_ih
offset_from_spiral = W_hh @ offset_before      # W_hh @ (h_M - h_S)
total_offset_pre = offset_from_impulse + offset_from_spiral

print("\npre_fwd - pre_rev = W_ih*(S-M) + W_hh @ (h_M - h_S)")
print(f"\n  W_ih * (S-M) = W_ih * (-0.2):")
print(f"    ||offset_from_impulse|| = {np.linalg.norm(offset_from_impulse):.2f}")
print(f"\n  W_hh @ (h_M - h_S):")
print(f"    ||offset_from_spiral|| = {np.linalg.norm(offset_from_spiral):.2f}")
print(f"\n  Total pre-ReLU offset:")
print(f"    ||total|| = {np.linalg.norm(total_offset_pre):.2f}")

# These two components point in DIFFERENT directions!
cos_sim = (offset_from_impulse @ offset_from_spiral) / (np.linalg.norm(offset_from_impulse) * np.linalg.norm(offset_from_spiral))
print(f"\n  Cosine similarity between components: {cos_sim:.3f}")

print("\n" + "=" * 70)
print("DIRECTION ANALYSIS: Which way does each component push?")
print("=" * 70)

# For forward, we want s_pos=4 to win
# For reverse, we want s_pos=3 to win
# The offset = h_fwd - h_rev

# If offset projects positively onto W_out[4] - W_out[3], it helps forward
# If offset projects negatively, it helps reverse

discrim = W_out[4] - W_out[3]  # positive = helps forward (target=4)

proj_impulse = discrim @ offset_from_impulse
proj_spiral = discrim @ offset_from_spiral
proj_total_pre = discrim @ total_offset_pre

print(f"\nDiscriminative direction: W_out[4] - W_out[3]")
print(f"  (positive projection helps forward, negative helps reverse)")
print(f"\n  Impulse component (S-M) projection: {proj_impulse:+.2f}")
print(f"  Spiral component (M-S spiral diff) projection: {proj_spiral:+.2f}")
print(f"  Total pre-ReLU projection: {proj_total_pre:+.2f}")

# After ReLU
post_fwd = np.maximum(0, pre_fwd)
post_rev = np.maximum(0, pre_rev)
actual_offset_post = post_fwd - post_rev
proj_actual = discrim @ actual_offset_post

print(f"\n  Actual post-ReLU projection: {proj_actual:+.2f}")
print(f"  ReLU modification: {proj_actual - proj_total_pre:+.2f}")

print("\n" + "=" * 70)
print("KEY INSIGHT: The two offset components OPPOSE each other!")
print("=" * 70)

print(f"""
The offset has two sources at the 2nd impulse:

1. IMPULSE DIFFERENCE: W_ih * (S - M) = W_ih * (-0.2)
   - Forward gets smaller impulse (S=0.8)
   - Reverse gets larger impulse (M=1.0)
   - This component is NEGATIVE (points toward reverse)
   - Projection onto discriminative direction: {proj_impulse:+.2f}

2. SPIRAL DIFFERENCE: W_hh @ (h_M - h_S)
   - Forward has M-spiral (larger, from M=1.0)
   - Reverse has S-spiral (smaller, from S=0.8)
   - This component is POSITIVE (points toward forward)
   - Projection onto discriminative direction: {proj_spiral:+.2f}

The spiral difference is {abs(proj_spiral/proj_impulse):.1f}x larger than the impulse difference!

So even though forward receives the SMALLER 2nd impulse (S < M),
the LARGER prior spiral (M > S) more than compensates.
""")

print("\n" + "=" * 70)
print("CHECKING: Does this hold for all gap=1 pairs?")
print("=" * 70)

print("\nFor each gap=1 pair, decomposing the offset at 2nd impulse:")
print("s/m_pos | impulse_proj | spiral_proj | total | ReLU_mod | final_margin")
print("--------|--------------|-------------|-------|----------|-------------")

for pos in range(9):
    # Forward: M@pos, S@pos+1
    # Reverse: S@pos, M@pos+1
    states_f = run_stepwise([(pos, M_val), (pos+1, S_val)])
    states_r = run_stepwise([(pos, S_val), (pos+1, M_val)])

    h_f_before = states_f[pos+1]
    h_r_before = states_r[pos+1]

    offset_imp = W_ih * (S_val - M_val)
    offset_spi = W_hh @ (h_f_before - h_r_before)

    discrim = W_out[pos+1] - W_out[pos]  # fwd target - rev target

    proj_i = discrim @ offset_imp
    proj_s = discrim @ offset_spi
    proj_t = proj_i + proj_s

    # After ReLU
    pre_f = W_ih * S_val + W_hh @ h_f_before
    pre_r = W_ih * M_val + W_hh @ h_r_before
    post_f = np.maximum(0, pre_f)
    post_r = np.maximum(0, pre_r)
    proj_actual = discrim @ (post_f - post_r)
    relu_mod = proj_actual - proj_t

    # Final margin for forward (should be positive for correct fwd)
    final_f = states_f[-1]
    logits_f = W_out @ final_f
    margin_f = logits_f[pos+1] - logits_f[pos]

    print(f"  {pos}/{pos+1}   | {proj_i:+11.2f} | {proj_s:+10.2f} | {proj_t:+5.1f} | {relu_mod:+8.1f} | {margin_f:+11.1f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
The offset at the 2nd impulse has TWO components that work AGAINST each other:

  offset = W_ih × (S-M) + W_hh @ (h_M - h_S)
              ↑                    ↑
         NEGATIVE             POSITIVE
      (helps reverse)       (helps forward)

In forward: S=0.8 arrives, M=1.0 preceded
  - Impulse term: forward gets -0.2 × W_ih (disadvantage)
  - Spiral term: forward has larger prior spiral (advantage)
  - Net: spiral wins because ||W_hh @ h|| >> ||W_ih × val||

In reverse: M=1.0 arrives, S=0.8 preceded
  - Impulse term: reverse gets +0.2 × W_ih (advantage from offset perspective)
  - Spiral term: reverse has smaller prior spiral (disadvantage)
  - Net: the smaller spiral still provides enough differential gating

The key is that W_hh AMPLIFIES the spiral difference more than the raw
impulse difference matters. Both directions work because the FIRST
impulse's spiral dominates over the SECOND impulse's direct contribution.
""")
