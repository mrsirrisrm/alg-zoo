"""
Analyze how the offset between forward and reverse trajectories is born.

Key question: In reverse (S before M), M arrives later and is LARGER (1.0 vs 0.8).
How does the network still output S_pos instead of M_pos?

The offset must be created at the moment of the 2nd impulse through differential
ReLU gating.
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
print("OFFSET BIRTH ANALYSIS: How does reverse still output S_pos?")
print("=" * 70)

# Test case: M@2, S@7 (forward) vs S@2, M@7 (reverse)
m_pos, s_pos = 2, 7
M_val, S_val = 1.0, 0.8

def run_sequence(positions_values, return_all=False):
    """Run a sequence and return hidden states at each timestep."""
    seq = th.zeros(1, 10)
    for pos, val in positions_values:
        seq[0, pos] = val

    h = th.zeros(1, 16)
    states = [h.numpy().copy()]

    for t in range(10):
        x_t = seq[0, t:t+1].unsqueeze(0)
        pre_relu = (th.tensor(W_ih) @ x_t.T).squeeze() + (th.tensor(W_hh) @ h.T).squeeze()
        h = th.relu(pre_relu).unsqueeze(0)
        states.append(h.numpy().copy())

    return states if return_all else states[-1]

def analyze_impulse_moment(h_before, impulse_val, label):
    """Analyze what happens when an impulse arrives into state h_before."""
    h_before = h_before.squeeze()

    # Pre-ReLU activation
    pre_relu = W_ih.squeeze() * impulse_val + W_hh @ h_before

    # Post-ReLU activation
    post_relu = np.maximum(0, pre_relu)

    # Which neurons are active?
    active = post_relu > 0
    clipped = (pre_relu < 0)

    print(f"\n{label}:")
    print(f"  Impulse magnitude: {impulse_val}")
    print(f"  ||h_before||: {np.linalg.norm(h_before):.2f}")
    print(f"  Active neurons: {np.sum(active)}/16 - {np.where(active)[0].tolist()}")
    print(f"  Clipped neurons: {np.sum(clipped)}/16 - {np.where(clipped)[0].tolist()}")

    return pre_relu, post_relu, active

# Run both directions
fwd_states = run_sequence([(m_pos, M_val), (s_pos, S_val)], return_all=True)
rev_states = run_sequence([(s_pos, S_val), (m_pos, M_val)], return_all=True)

print("\n" + "=" * 70)
print("FORWARD: M@2 (1.0), S@7 (0.8)")
print("=" * 70)

# In forward, 1st impulse is M@2, 2nd impulse is S@7
print("\n--- Before 1st impulse (t=2, M arrives) ---")
h_before_M_fwd = fwd_states[2]  # state just before t=2 update
pre_M_fwd, post_M_fwd, active_M_fwd = analyze_impulse_moment(h_before_M_fwd, M_val, "M impulse (fwd)")

print("\n--- Before 2nd impulse (t=7, S arrives) ---")
h_before_S_fwd = fwd_states[7]  # state just before t=7 update
pre_S_fwd, post_S_fwd, active_S_fwd = analyze_impulse_moment(h_before_S_fwd, S_val, "S impulse (fwd)")

print("\n" + "=" * 70)
print("REVERSE: S@2 (0.8), M@7 (1.0)")
print("=" * 70)

# In reverse, 1st impulse is S@2, 2nd impulse is M@7
print("\n--- Before 1st impulse (t=2, S arrives) ---")
h_before_S_rev = rev_states[2]
pre_S_rev, post_S_rev, active_S_rev = analyze_impulse_moment(h_before_S_rev, S_val, "S impulse (rev)")

print("\n--- Before 2nd impulse (t=7, M arrives) ---")
h_before_M_rev = rev_states[7]
pre_M_rev, post_M_rev, active_M_rev = analyze_impulse_moment(h_before_M_rev, M_val, "M impulse (rev)")

print("\n" + "=" * 70)
print("CRITICAL COMPARISON: State immediately after 2nd impulse (t=7 -> t=8)")
print("=" * 70)

h_fwd_after_2nd = fwd_states[8].squeeze()  # After S arrives at t=7
h_rev_after_2nd = rev_states[8].squeeze()  # After M arrives at t=7

offset_at_birth = h_fwd_after_2nd - h_rev_after_2nd

print(f"\n||h_fwd|| = {np.linalg.norm(h_fwd_after_2nd):.2f}")
print(f"||h_rev|| = {np.linalg.norm(h_rev_after_2nd):.2f}")
print(f"||offset|| = {np.linalg.norm(offset_at_birth):.2f}")

# Check the offset's projection onto discriminative direction
discrim = W_out[s_pos] - W_out[m_pos]
margin = discrim @ offset_at_birth
print(f"\nDiscriminative margin at birth: {margin:.2f}")
print(f"  (positive means fwd favors s_pos over m_pos, as desired)")

print("\n" + "=" * 70)
print("THE KEY INSIGHT: Differential ReLU gating at 2nd impulse")
print("=" * 70)

# What's different about the hidden state when 2nd impulse arrives?
print("\nState before 2nd impulse:")
print(f"  Forward (before S@7):  ||h|| = {np.linalg.norm(h_before_S_fwd):.2f}")
print(f"  Reverse (before M@7):  ||h|| = {np.linalg.norm(h_before_M_rev):.2f}")

# The pre-ReLU values determine what gets clipped
print("\nPre-ReLU activations at 2nd impulse:")

# Forward: S (0.8) arrives into state shaped by M's spiral
pre_fwd = W_ih.squeeze() * S_val + W_hh @ h_before_S_fwd.squeeze()
# Reverse: M (1.0) arrives into state shaped by S's spiral
pre_rev = W_ih.squeeze() * M_val + W_hh @ h_before_M_rev.squeeze()

print(f"\nForward (S=0.8 into M-spiral):")
for i in range(16):
    status = "ACTIVE" if pre_fwd[i] > 0 else "CLIPPED"
    print(f"  n{i:2d}: pre={pre_fwd[i]:7.2f} -> {status}")

print(f"\nReverse (M=1.0 into S-spiral):")
for i in range(16):
    status = "ACTIVE" if pre_rev[i] > 0 else "CLIPPED"
    print(f"  n{i:2d}: pre={pre_rev[i]:7.2f} -> {status}")

# Neurons that are clipped differently
fwd_clipped = pre_fwd < 0
rev_clipped = pre_rev < 0
diff_clipped = fwd_clipped != rev_clipped

print(f"\nNeurons clipped DIFFERENTLY between fwd and rev:")
for i in np.where(diff_clipped)[0]:
    print(f"  n{i}: fwd={'CLIP' if fwd_clipped[i] else 'ACTIVE':6s}, rev={'CLIP' if rev_clipped[i] else 'ACTIVE':6s}")
    print(f"        pre_fwd={pre_fwd[i]:.2f}, pre_rev={pre_rev[i]:.2f}")
    print(f"        W_out contribution to s_pos: {W_out[s_pos, i]:.2f}")
    print(f"        W_out contribution to m_pos: {W_out[m_pos, i]:.2f}")

print("\n" + "=" * 70)
print("DECOMPOSING THE OFFSET SOURCES")
print("=" * 70)

# The offset comes from three sources:
# 1. Different impulse magnitudes (M=1.0 vs S=0.8)
# 2. Different hidden states before impulse
# 3. Different ReLU clipping patterns

# Source 1: Impulse difference (if same h_before)
impulse_diff_contribution = W_ih.squeeze() * (S_val - M_val)  # -0.2 * W_ih
print(f"\n1. Impulse magnitude difference (S-M = {S_val - M_val}):")
print(f"   Contribution to offset: {np.linalg.norm(impulse_diff_contribution):.2f}")

# Source 2: Different h_before
h_diff = h_before_S_fwd.squeeze() - h_before_M_rev.squeeze()
h_evolution_contribution = W_hh @ h_diff
print(f"\n2. Different hidden states before 2nd impulse:")
print(f"   ||h_fwd_before - h_rev_before|| = {np.linalg.norm(h_diff):.2f}")
print(f"   After W_hh: {np.linalg.norm(h_evolution_contribution):.2f}")

# Source 3: ReLU differential
# If there were no ReLU, the offset would be:
linear_offset = impulse_diff_contribution + h_evolution_contribution
# Actual offset:
actual_offset = h_fwd_after_2nd - h_rev_after_2nd

print(f"\n3. ReLU differential clipping:")
print(f"   Linear prediction: {np.linalg.norm(linear_offset):.2f}")
print(f"   Actual offset:     {np.linalg.norm(actual_offset):.2f}")
print(f"   ReLU contribution: {np.linalg.norm(actual_offset - linear_offset):.2f}")

# How does each source contribute to the discriminative margin?
margin_impulse = discrim @ impulse_diff_contribution
margin_h_evol = discrim @ h_evolution_contribution
margin_relu = discrim @ (actual_offset - linear_offset)

print(f"\nDiscriminative margin breakdown:")
print(f"   From impulse diff:    {margin_impulse:+.2f}")
print(f"   From h evolution:     {margin_h_evol:+.2f}")
print(f"   From ReLU clipping:   {margin_relu:+.2f}")
print(f"   ----------------------------")
print(f"   Total margin:         {margin_impulse + margin_h_evol + margin_relu:+.2f}")

print("\n" + "=" * 70)
print("TRACING BACK: How did the pre-2nd-impulse states diverge?")
print("=" * 70)

# The h_before states are different because the 1st impulse was different
# Forward: M=1.0 at t=2, then evolves to t=7
# Reverse: S=0.8 at t=2, then evolves to t=7

# Let's trace the evolution
print("\nTrajectory comparison (||h|| at each timestep):")
print("t  | Forward | Reverse | Difference")
print("---|---------|---------|----------")
for t in range(11):
    h_f = np.linalg.norm(fwd_states[t])
    h_r = np.linalg.norm(rev_states[t])
    diff = np.linalg.norm(fwd_states[t].squeeze() - rev_states[t].squeeze())
    marker = " <-- 1st impulse" if t == 3 else (" <-- 2nd impulse" if t == 8 else "")
    print(f"{t:2d} | {h_f:7.2f} | {h_r:7.2f} | {diff:8.2f}{marker}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Pre-ReLU comparison at 2nd impulse
ax = axes[0, 0]
x = np.arange(16)
width = 0.35
ax.bar(x - width/2, pre_fwd, width, label='Fwd (S into M-spiral)', alpha=0.7)
ax.bar(x + width/2, pre_rev, width, label='Rev (M into S-spiral)', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Neuron')
ax.set_ylabel('Pre-ReLU activation')
ax.set_title('Pre-ReLU at 2nd impulse (t=7)')
ax.legend()

# 2. Post-ReLU comparison
ax = axes[0, 1]
post_fwd = np.maximum(0, pre_fwd)
post_rev = np.maximum(0, pre_rev)
ax.bar(x - width/2, post_fwd, width, label='Fwd', alpha=0.7)
ax.bar(x + width/2, post_rev, width, label='Rev', alpha=0.7)
ax.set_xlabel('Neuron')
ax.set_ylabel('Post-ReLU activation')
ax.set_title('Post-ReLU at 2nd impulse')
ax.legend()

# 3. The offset born at 2nd impulse
ax = axes[0, 2]
ax.bar(x, offset_at_birth, color='purple', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Neuron')
ax.set_ylabel('Offset (h_fwd - h_rev)')
ax.set_title(f'Offset at birth (||offset||={np.linalg.norm(offset_at_birth):.1f})')

# 4. Margin contribution by source
ax = axes[1, 0]
sources = ['Impulse\n(S-M)', 'h evolution\n(W_hh)', 'ReLU\nclipping']
margins = [margin_impulse, margin_h_evol, margin_relu]
colors = ['blue' if m > 0 else 'red' for m in margins]
ax.bar(sources, margins, color=colors, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Contribution to margin')
ax.set_title('Margin decomposition at birth')

# 5. Offset evolution after birth
ax = axes[1, 1]
offsets = []
margins_over_time = []
for t in range(11):
    off = fwd_states[t].squeeze() - rev_states[t].squeeze()
    offsets.append(np.linalg.norm(off))
    margins_over_time.append(discrim @ off)

ax.plot(range(11), offsets, 'o-', label='||offset||')
ax.plot(range(11), margins_over_time, 's-', label='margin')
ax.axvline(7, color='gray', linestyle='--', alpha=0.5, label='2nd impulse')
ax.set_xlabel('Timestep')
ax.set_ylabel('Value')
ax.set_title('Offset evolution over time')
ax.legend()

# 6. W_out contribution of differentially clipped neurons
ax = axes[1, 2]
diff_neurons = np.where(diff_clipped)[0]
if len(diff_neurons) > 0:
    contrib_s = W_out[s_pos, diff_neurons]
    contrib_m = W_out[m_pos, diff_neurons]
    x_diff = np.arange(len(diff_neurons))
    ax.bar(x_diff - width/2, contrib_s, width, label=f'W_out[s={s_pos}]', alpha=0.7)
    ax.bar(x_diff + width/2, contrib_m, width, label=f'W_out[m={m_pos}]', alpha=0.7)
    ax.set_xticks(x_diff)
    ax.set_xticklabels([f'n{i}' for i in diff_neurons])
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Differentially clipped neurons')
    ax.set_ylabel('W_out weight')
    ax.set_title('W_out weights for diff-clipped neurons')
    ax.legend()

plt.tight_layout()
plt.savefig('docs/offset_birth_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nVisualization saved to docs/offset_birth_analysis.png")

print("\n" + "=" * 70)
print("SUMMARY: How reverse still outputs S_pos")
print("=" * 70)
print("""
The puzzle: In reverse, M (1.0) arrives AFTER S (0.8) and is LARGER.
Why doesn't M "overwrite" S's position?

Answer: The offset is born through DIFFERENTIAL CLIPPING.

1. BEFORE the 2nd impulse arrives:
   - Forward: h has evolved from M@2 (larger impulse, specific spiral)
   - Reverse: h has evolved from S@2 (smaller impulse, different spiral)

2. WHEN the 2nd impulse arrives:
   - Forward: S (0.8) enters the M-shaped state
   - Reverse: M (1.0) enters the S-shaped state

3. The PRE-ReLU activations differ because:
   - Different impulse magnitudes (contributes {:.1f} to margin)
   - Different hidden states (contributes {:.1f} to margin)

4. ReLU CLIPS DIFFERENTLY, adding {:.1f} to the margin

5. The born offset has ||offset|| = {:.1f} with margin = {:.1f}

6. This offset then GROWS through W_hh's expanding eigenmodes,
   and ReLU continues to differentially clip, widening the gap.

The network doesn't need to "remember" S explicitly. The spiral
from S's impulse creates a hidden state that, when M arrives,
produces a specific clipping pattern that encodes "S was first."
""".format(margin_impulse, margin_h_evol, margin_relu,
           np.linalg.norm(offset_at_birth), margin))
