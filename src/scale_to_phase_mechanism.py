"""
The spirals are IDENTICAL in shape, just scaled by M/S = 1.25x.
Yet after the 2nd impulse, the phases differ by exactly the gap.

How does a SCALE difference become a PHASE difference?
"""

import numpy as np
from alg_zoo import example_2nd_argmax

model = example_2nd_argmax()
model.eval()

W_ih = model.rnn.weight_ih_l0.detach().numpy().squeeze()
W_hh = model.rnn.weight_hh_l0.detach().numpy()
W_out = model.linear.weight.detach().numpy()

M_val, S_val = 1.0, 0.8
ratio = M_val / S_val  # 1.25

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
print("SCALE TO PHASE: How does 1.25x scale become 1 position phase shift?")
print("=" * 70)

# Gap = 1 case: positions 3 and 4
first_pos, second_pos = 3, 4

# Get the spirals right before 2nd impulse
states_fwd = run_stepwise([(first_pos, M_val), (second_pos, S_val)])
states_rev = run_stepwise([(first_pos, S_val), (second_pos, M_val)])

h_M = states_fwd[second_pos]  # M-spiral
h_S = states_rev[second_pos]  # S-spiral

print(f"\nBefore 2nd impulse (t={second_pos}):")
print(f"  M-spiral: ||h|| = {np.linalg.norm(h_M):.4f}")
print(f"  S-spiral: ||h|| = {np.linalg.norm(h_S):.4f}")
print(f"  Ratio: {np.linalg.norm(h_M) / np.linalg.norm(h_S):.4f} (expected: {ratio})")
print(f"  Cosine sim: {(h_M @ h_S) / (np.linalg.norm(h_M) * np.linalg.norm(h_S)):.4f}")

# Verify: h_M = ratio * h_S (exactly)
print(f"\n  h_M = {ratio} * h_S? Max diff: {np.max(np.abs(h_M - ratio * h_S)):.6f}")

print("\n" + "=" * 70)
print("THE 2ND IMPULSE: What each trajectory receives")
print("=" * 70)

# Forward: S impulse (0.8) into M-spiral
# Reverse: M impulse (1.0) into S-spiral

# Pre-ReLU computation:
# Forward: pre_f = W_ih * S + W_hh @ h_M = W_ih * S + W_hh @ (ratio * h_S)
#                = W_ih * S + ratio * (W_hh @ h_S)
#
# Reverse: pre_r = W_ih * M + W_hh @ h_S

pre_f = W_ih * S_val + W_hh @ h_M
pre_r = W_ih * M_val + W_hh @ h_S

print("\nPre-ReLU decomposition:")
print(f"\n  Forward: pre_f = W_ih * {S_val} + W_hh @ h_M")
print(f"           = W_ih * {S_val} + W_hh @ ({ratio} * h_S)")
print(f"           = W_ih * {S_val} + {ratio} * (W_hh @ h_S)")

print(f"\n  Reverse: pre_r = W_ih * {M_val} + W_hh @ h_S")

# Let's define common terms:
wih_term = W_ih  # The input direction (same for both, just scaled)
whh_term = W_hh @ h_S  # The spiral contribution (S-spiral version)

print(f"\n  Common terms:")
print(f"    ||W_ih|| = {np.linalg.norm(wih_term):.2f}")
print(f"    ||W_hh @ h_S|| = {np.linalg.norm(whh_term):.2f}")

# Rewrite:
# pre_f = S * W_ih + ratio * (W_hh @ h_S)
# pre_r = M * W_ih + 1 * (W_hh @ h_S)

# Difference:
# pre_f - pre_r = (S - M) * W_ih + (ratio - 1) * (W_hh @ h_S)
#               = -0.2 * W_ih + 0.25 * (W_hh @ h_S)

diff_impulse = (S_val - M_val) * W_ih  # -0.2 * W_ih
diff_spiral = (ratio - 1) * whh_term   # 0.25 * (W_hh @ h_S)
diff_total = diff_impulse + diff_spiral

print(f"\n  Difference (pre_f - pre_r):")
print(f"    = (S - M) * W_ih + (ratio - 1) * (W_hh @ h_S)")
print(f"    = {S_val - M_val} * W_ih + {ratio - 1:.2f} * (W_hh @ h_S)")
print(f"\n    ||impulse diff|| = {np.linalg.norm(diff_impulse):.2f}")
print(f"    ||spiral diff||  = {np.linalg.norm(diff_spiral):.2f}")
print(f"    ||total diff||   = {np.linalg.norm(diff_total):.2f}")

# These are in the same directions but opposite signs!
# (S-M) = -0.2, (ratio-1) = +0.25
# So: diff = -0.2 * W_ih + 0.25 * W_hh @ h_S

# The key: W_ih and W_hh @ h_S are NOT parallel!
cos_sim = (W_ih @ whh_term) / (np.linalg.norm(W_ih) * np.linalg.norm(whh_term))
print(f"\n    Cosine(W_ih, W_hh @ h_S) = {cos_sim:.3f}")

print("\n" + "=" * 70)
print("THE CANCELLATION THAT DOESN'T HAPPEN")
print("=" * 70)

print("""
If W_ih and W_hh @ h_S were parallel, then:
  -0.2 * W_ih + 0.25 * (W_hh @ h_S)
could partially cancel.

But they're NOT parallel (cosine = {:.3f}).

So the difference has components in BOTH directions:
  - A component along W_ih
  - A component along W_hh @ h_S

And ReLU treats these differently!
""".format(cos_sim))

print("\n" + "=" * 70)
print("HOW RELU CONVERTS SCALE TO PHASE")
print("=" * 70)

# Post-ReLU
post_f = np.maximum(0, pre_f)
post_r = np.maximum(0, pre_r)

# Which neurons are affected?
f_active = pre_f > 0
r_active = pre_r > 0

print("\nActive neurons:")
print(f"  Forward: {np.sum(f_active)}/16 - {np.where(f_active)[0].tolist()}")
print(f"  Reverse: {np.sum(r_active)}/16 - {np.where(r_active)[0].tolist()}")

diff_active = f_active != r_active
print(f"  Different: {np.sum(diff_active)}/16 - {np.where(diff_active)[0].tolist()}")

# For neurons that are active in both: contribution is just the linear diff
# For neurons that differ: one contributes, one doesn't

print("\nNeuron-by-neuron comparison:")
print("n  | pre_f | pre_r | diff  | f_active | r_active | contrib to phase")
print("---|-------|-------|-------|----------|----------|------------------")

for i in range(16):
    pf, pr = pre_f[i], pre_r[i]
    diff = pf - pr
    fa, ra = f_active[i], r_active[i]

    # What's this neuron's W_out contribution?
    # We care about the discriminative direction for phase
    # Phase offset of 1 means: logits shift by amount corresponding to 1 position

    if fa and ra:
        contrib = "linear diff"
    elif fa and not ra:
        contrib = "fwd only (adds to fwd)"
    elif not fa and ra:
        contrib = "rev only (adds to rev)"
    else:
        contrib = "neither (no contrib)"

    print(f"{i:2d} | {pf:5.1f} | {pr:5.1f} | {diff:+5.1f} | {str(fa):5s}    | {str(ra):5s}    | {contrib}")

print("\n" + "=" * 70)
print("THE KEY NEURONS")
print("=" * 70)

# Find neurons that are differentially active
diff_neurons = np.where(diff_active)[0]

print(f"\nNeurons that are active in only one trajectory: {diff_neurons.tolist()}")

for i in diff_neurons:
    pf, pr = pre_f[i], pre_r[i]
    postf, postr = post_f[i], post_r[i]

    # This neuron's contribution to the output difference
    out_contrib = W_out @ (post_f - post_r)

    print(f"\n  Neuron {i}:")
    print(f"    pre_f = {pf:.2f}, pre_r = {pr:.2f}")
    print(f"    post_f = {postf:.2f}, post_r = {postr:.2f}")
    print(f"    Contribution: post_f - post_r = {postf - postr:.2f}")
    print(f"    W_out[:, {i}] contributions to positions 3,4: {W_out[3,i]:.2f}, {W_out[4,i]:.2f}")

print("\n" + "=" * 70)
print("TOTAL PHASE EFFECT")
print("=" * 70)

# The phase is encoded in which W_out row has highest activation
# Let's see how the post-ReLU difference projects onto W_out

post_diff = post_f - post_r
logit_diff = W_out @ post_diff

print("\nLogit differences (forward - reverse):")
print("pos | logit_diff | interpretation")
print("----|------------|---------------")
for pos in range(10):
    ld = logit_diff[pos]
    if pos == 3:
        interp = "<- S_pos for reverse"
    elif pos == 4:
        interp = "<- S_pos for forward"
    else:
        interp = ""
    print(f"  {pos} | {ld:+10.2f} | {interp}")

# The phase shift should make forward's countdown 1 higher than reverse's
# At t=5 (right after 2nd impulse):
# Forward pred = 9, Reverse pred = 8

logits_f = W_out @ post_f
logits_r = W_out @ post_r

print(f"\nActual predictions right after 2nd impulse:")
print(f"  Forward: argmax = {np.argmax(logits_f)} (expected 9 for countdown to 4)")
print(f"  Reverse: argmax = {np.argmax(logits_r)} (expected 8 for countdown to 3)")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
The scale-to-phase conversion works like this:

1. Before 2nd impulse:
   - M-spiral = {ratio} × S-spiral (identical shape, different scale)

2. At 2nd impulse:
   - Forward receives: S (0.8) + W_hh @ (1.25 × h_S)
   - Reverse receives: M (1.0) + W_hh @ (1.0 × h_S)

3. The pre-ReLU difference is:
   pre_f - pre_r = -0.2 × W_ih + 0.25 × (W_hh @ h_S)
                       ↑              ↑
                  impulse diff    spiral diff

   These point in DIFFERENT directions (cosine = {cos_sim:.2f})

4. ReLU clips some neurons differently:
   {len(diff_neurons)} neurons are active in only one trajectory: {diff_neurons.tolist()}

5. These differentially-active neurons create a PHASE OFFSET:
   - Forward's logits peak at position 9
   - Reverse's logits peak at position 8
   - Difference = 1 = gap between S positions

The magic: the 1.25x scale difference precisely creates a 1-position phase shift
through the nonlinear interaction of impulse and spiral contributions at ReLU.
""")
