"""
Deep dive into how the REVERSE direction works.

The puzzle: S arrives first (smaller value, 0.8), M arrives second (larger, 1.0).
Yet the network outputs S's position, not M's.

Key insight from v2: The S-spiral "protects" S's position information.
When M arrives, it can't overwrite because the existing state shapes ReLU clipping.

This script examines EXACTLY how this protection works.
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax

# Load model
model = example_2nd_argmax()
model.eval()

W_ih = model.rnn.weight_ih_l0.detach().numpy().squeeze()  # (16,)
W_hh = model.rnn.weight_hh_l0.detach().numpy()  # (16, 16)
W_out = model.linear.weight.detach().numpy()     # (10, 16)

M_val, S_val = 1.0, 0.8

def run_to_state(impulses):
    """Run sequence with given impulses, return final state.
    impulses: list of (position, value) tuples
    """
    h = np.zeros(16)
    for t in range(10):
        x_t = 0.0
        for pos, val in impulses:
            if pos == t:
                x_t = val
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
    return h

def run_stepwise(impulses):
    """Return hidden state at each timestep."""
    h = np.zeros(16)
    states = [h.copy()]
    for t in range(10):
        x_t = 0.0
        for pos, val in impulses:
            if pos == t:
                x_t = val
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        states.append(h.copy())
    return states

print("=" * 70)
print("REVERSE MECHANISM: How S@2 survives M@7")
print("=" * 70)

# The reverse case: S@2, M@7
s_pos, m_pos = 2, 7
states_rev = run_stepwise([(s_pos, S_val), (m_pos, M_val)])

print("\n1. THE S-SPIRAL BEFORE M ARRIVES")
print("-" * 50)

h_before_M = states_rev[m_pos]  # State at t=7, just before M's impulse
print(f"||h|| at t={m_pos} (before M): {np.linalg.norm(h_before_M):.2f}")
print(f"Active neurons: {np.sum(h_before_M > 0)}/16")

# What position does this state encode?
logits_before_M = W_out @ h_before_M
pred_before_M = np.argmax(logits_before_M)
print(f"Prediction from h_before_M: {pred_before_M}")

# The S-spiral has been counting down!
print("\nCountdown from S@2:")
for t in range(3, 8):
    h_t = states_rev[t]
    pred_t = np.argmax(W_out @ h_t)
    expected = s_pos + (9 - t)  # Countdown formula
    print(f"  t={t}: pred={pred_t}, expected countdown={expected}")

print("\n2. WHAT HAPPENS WHEN M ARRIVES")
print("-" * 50)

# Pre-ReLU when M arrives
pre_relu_M = W_ih * M_val + W_hh @ h_before_M
post_relu_M = np.maximum(0, pre_relu_M)

# Compare to M arriving into empty state
pre_relu_M_alone = W_ih * M_val
post_relu_M_alone = np.maximum(0, pre_relu_M_alone)

print("Neuron-by-neuron comparison:")
print("n  | h_before | W_hh@h | W_ih*M | pre_total | post | M_alone | diff")
print("---|----------|--------|--------|-----------|------|---------|-----")

significant_diffs = []
for i in range(16):
    h_i = h_before_M[i]
    whh_contrib = (W_hh @ h_before_M)[i]
    wih_contrib = W_ih[i] * M_val
    pre = pre_relu_M[i]
    post = post_relu_M[i]
    alone = post_relu_M_alone[i]
    diff = post - alone

    if abs(diff) > 1:
        significant_diffs.append((i, diff, h_i, whh_contrib))

    print(f"{i:2d} | {h_i:8.2f} | {whh_contrib:6.2f} | {wih_contrib:6.2f} | {pre:9.2f} | {post:4.1f} | {alone:7.1f} | {diff:+5.1f}")

print(f"\nNeurons with significant difference (|diff| > 1):")
for i, diff, h_i, whh in significant_diffs:
    w_out_s = W_out[s_pos, i]
    w_out_m = W_out[m_pos, i]
    print(f"  n{i}: diff={diff:+.1f}, W_out[s={s_pos}]={w_out_s:.2f}, W_out[m={m_pos}]={w_out_m:.2f}")
    print(f"       -> contribution to s-m: {diff * (w_out_s - w_out_m):+.2f}")

print("\n3. THE OFFSET: (M into S-spiral) vs (M alone)")
print("-" * 50)

h_after_M_rev = states_rev[m_pos + 1]  # t=8
h_after_M_alone = run_stepwise([(m_pos, M_val)])[m_pos + 1]

offset = h_after_M_rev - h_after_M_alone
print(f"||offset|| = {np.linalg.norm(offset):.2f}")

# Project offset onto W_out
proj = W_out @ offset
print(f"\nOffset projection onto W_out:")
print(f"  Position {s_pos} (S): {proj[s_pos]:+.2f}")
print(f"  Position {m_pos} (M): {proj[m_pos]:+.2f}")
print(f"  Margin (s-m): {proj[s_pos] - proj[m_pos]:+.2f}")

print("\n4. DECOMPOSING THE S-SPIRAL'S CONTRIBUTION")
print("-" * 50)

# The S-spiral contributes through W_hh @ h_before_M
# This adds to the pre-ReLU activations when M arrives

whh_contribution = W_hh @ h_before_M
print(f"||W_hh @ h_before_M|| = {np.linalg.norm(whh_contribution):.2f}")

# How does this contribution affect the discriminative direction?
discrim = W_out[s_pos] - W_out[m_pos]

# If we could pass whh_contribution through to the output linearly...
linear_effect = discrim @ whh_contribution
print(f"Linear effect on margin: {linear_effect:.2f}")

# But ReLU modifies this!
# The actual effect is the difference in post-ReLU outputs
actual_diff = post_relu_M - post_relu_M_alone
actual_effect = discrim @ actual_diff
print(f"Actual effect on margin (after ReLU): {actual_effect:.2f}")

print("\n5. THE CRITICAL NEURONS")
print("-" * 50)

# Which neurons matter most for the s_pos vs m_pos discrimination?
neuron_importance = []
for i in range(16):
    importance = abs(discrim[i]) * abs(actual_diff[i])
    neuron_importance.append((i, importance, discrim[i], actual_diff[i]))

neuron_importance.sort(key=lambda x: -x[1])
print("Top neurons by |discriminative weight × activation diff|:")
for i, imp, d, a in neuron_importance[:5]:
    sign = "+" if d * a > 0 else "-"
    print(f"  n{i}: importance={imp:.2f}, discrim={d:.2f}, diff={a:.2f} -> {sign} margin")

print("\n6. FOLLOWING THE INFORMATION FLOW")
print("-" * 50)

print("""
The S-spiral encodes S's position through the PHASE of oscillation.
When M arrives at t=7:

1. S-spiral has been evolving for 5 timesteps (t=3 to t=7)
2. The phase has "counted down" from the start
3. h_before_M has a specific pattern that represents "S was at position 2"

4. M's impulse (W_ih * 1.0) tries to create a new pattern
5. But W_hh @ h_before_M ADDS to the pre-ReLU
6. This shifts some neurons across the ReLU threshold

7. The shifted neurons carry the "S@2" information forward
8. Even though M is larger, S's position is preserved in the offset
""")

# Let's verify the countdown continues correctly after M
print("\nCountdown verification after M arrives:")
for t in range(8, 11):
    h_t = states_rev[t]
    pred_t = np.argmax(W_out @ h_t)
    expected = s_pos + (9 - t)  # Should still be counting to s_pos!
    status = "✓" if pred_t == expected else "✗"
    print(f"  t={t}: pred={pred_t}, expected={expected} {status}")

print("\n7. THE PITHY EXPLANATION")
print("=" * 70)

print("""
╔════════════════════════════════════════════════════════════════════╗
║           HOW REVERSE WORKS: THE S-SPIRAL PROTECTION               ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  In FORWARD: S arrives 2nd → countdown naturally lands on S_pos    ║
║                                                                     ║
║  In REVERSE: M arrives 2nd, but S's spiral has already "claimed"   ║
║              the hidden state. When M arrives:                      ║
║                                                                     ║
║    pre_relu = W_ih × M  +  W_hh @ [S-spiral]                       ║
║                 ↑              ↑                                    ║
║            M wants to      S's phase info                          ║
║            reset phase     shifts thresholds                       ║
║                                                                     ║
║  The S-spiral's contribution (W_hh @ h) shifts ReLU boundaries:    ║
║  • Some neurons that M would activate get suppressed               ║
║  • Some neurons that M would suppress stay active                  ║
║                                                                     ║
║  Result: The post-ReLU state encodes BOTH positions, but the       ║
║          countdown continues toward S_pos, not M_pos               ║
║                                                                     ║
║  PITHY: "First impulse claims the spiral; second impulse can't     ║
║          evict it because ReLU boundaries have already shifted."   ║
║                                                                     ║
╚════════════════════════════════════════════════════════════════════╝
""")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. The S-spiral before M arrives
ax = axes[0, 0]
ax.bar(range(16), h_before_M, color='red', alpha=0.7)
ax.set_xlabel('Neuron')
ax.set_ylabel('Activation')
ax.set_title(f'S-spiral at t={m_pos} (before M arrives)')

# 2. W_hh contribution when M arrives
ax = axes[0, 1]
ax.bar(range(16), whh_contribution, color='purple', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Neuron')
ax.set_ylabel('W_hh @ h_before_M')
ax.set_title('S-spiral\'s contribution to M\'s reception')

# 3. The offset (M into S-spiral) vs (M alone)
ax = axes[0, 2]
ax.bar(range(16), actual_diff, color='green', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Neuron')
ax.set_ylabel('Activation difference')
ax.set_title('Offset: (M into S-spiral) - (M alone)')

# 4. Offset projection onto W_out
ax = axes[1, 0]
colors = ['red' if i == s_pos else ('blue' if i == m_pos else 'gray') for i in range(10)]
ax.bar(range(10), proj, color=colors, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('W_out @ offset')
ax.set_title(f'Offset favors s_pos={s_pos} over m_pos={m_pos}')

# 5. Countdown verification
ax = axes[1, 1]
preds = [np.argmax(W_out @ states_rev[t]) for t in range(1, 11)]
expected = [s_pos + (9 - t) if t >= 3 else None for t in range(1, 11)]
ax.plot(range(1, 11), preds, 'ro-', label='Actual prediction', markersize=8)
ax.plot(range(3, 11), [s_pos + (9 - t) for t in range(3, 11)], 'b--', label='Expected countdown', alpha=0.7)
ax.axhline(s_pos, color='red', linestyle=':', alpha=0.5, label=f's_pos={s_pos}')
ax.axvline(m_pos, color='gray', linestyle='--', alpha=0.5, label=f'M arrives at t={m_pos}')
ax.set_xlabel('Timestep')
ax.set_ylabel('Predicted position')
ax.set_title('Countdown survives M\'s arrival')
ax.legend(fontsize=8)

# 6. Summary diagram (text-based)
ax = axes[1, 2]
ax.axis('off')
summary_text = """
THE REVERSE MECHANISM

S@2 arrives first:
  → Kicks off spiral
  → Phase encodes "position 2"
  → Countdown begins

M@7 arrives second:
  → Larger impulse (1.0 > 0.8)
  → But S-spiral shifts ReLU thresholds
  → M can't reset the phase

Result at t=9:
  → Countdown lands on position 2 ✓
  → S's position preserved

Key insight:
  The first impulse "claims" the phase.
  ReLU makes this claim sticky.
"""
ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('docs/reverse_mechanism_deep_dive.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nVisualization saved to docs/reverse_mechanism_deep_dive.png")
