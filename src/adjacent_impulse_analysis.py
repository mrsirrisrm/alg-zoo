"""
What happens when S precedes M by only 1 timestep?
Does the S-spiral have time to build before M arrives?
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
    """Return hidden state at each timestep."""
    h = np.zeros(16)
    states = [h.copy()]
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        states.append(h.copy())
    return states

print("=" * 70)
print("ADJACENT IMPULSES: S precedes M by 1 timestep")
print("=" * 70)

# Test gaps from 1 to 5
for gap in [1, 2, 3, 5]:
    s_pos = 3
    m_pos = s_pos + gap

    if m_pos >= 10:
        continue

    print(f"\n{'='*70}")
    print(f"GAP = {gap}: S@{s_pos}, M@{m_pos}")
    print("=" * 70)

    states = run_stepwise([(s_pos, S_val), (m_pos, M_val)])

    # State just before M arrives
    h_before_M = states[m_pos]

    # What M would see if alone
    states_M_only = run_stepwise([(m_pos, M_val)])
    h_before_M_alone = states_M_only[m_pos]  # Should be zeros

    print(f"\nState before M arrives (t={m_pos}):")
    print(f"  ||h_S_spiral|| = {np.linalg.norm(h_before_M):.2f}")
    print(f"  ||h_M_alone||  = {np.linalg.norm(h_before_M_alone):.2f}")
    print(f"  Active neurons: {np.sum(h_before_M > 0)}/16")

    # W_hh contribution when M arrives
    whh_contrib = W_hh @ h_before_M
    wih_contrib = W_ih * M_val

    print(f"\nContributions to pre-ReLU when M arrives:")
    print(f"  ||W_hh @ h_S|| = {np.linalg.norm(whh_contrib):.2f}")
    print(f"  ||W_ih * M||   = {np.linalg.norm(wih_contrib):.2f}")
    print(f"  Ratio (S-spiral / M-input): {np.linalg.norm(whh_contrib) / np.linalg.norm(wih_contrib):.2f}")

    # Final prediction
    h_final = states[-1]
    pred = np.argmax(W_out @ h_final)
    print(f"\nFinal prediction: {pred} (expected: {s_pos})")
    print(f"  Correct: {'✓' if pred == s_pos else '✗'}")

    # Check margin
    logits = W_out @ h_final
    margin = logits[s_pos] - logits[m_pos]
    print(f"  Margin (s - m): {margin:.2f}")

print("\n" + "=" * 70)
print("DETAILED LOOK AT GAP=1")
print("=" * 70)

s_pos, m_pos = 3, 4
states = run_stepwise([(s_pos, S_val), (m_pos, M_val)])

print("\nTimestep-by-timestep:")
print("t  | ||h|| | pred | active neurons")
print("---|-------|------|---------------")
for t in range(11):
    h = states[t]
    if t == 0:
        pred = "-"
    else:
        pred = np.argmax(W_out @ h)
    active = np.where(h > 0)[0].tolist()
    marker = ""
    if t == s_pos + 1:
        marker = " <- after S"
    elif t == m_pos + 1:
        marker = " <- after M"
    print(f"{t:2d} | {np.linalg.norm(h):5.2f} | {pred:4} | {active}{marker}")

# The critical moment: what does M see?
print(f"\nAt t={m_pos} (M arrives):")
h_before = states[m_pos]
print(f"  h_before (S-spiral after 1 step) = ")
for i in range(16):
    if h_before[i] > 0.01:
        print(f"    n{i}: {h_before[i]:.2f}")

# Compare pre-ReLU
pre_with_S = W_ih * M_val + W_hh @ h_before
pre_without_S = W_ih * M_val  # M into empty state

print(f"\nPre-ReLU comparison (M into S-spiral vs M alone):")
print("n  | with S | alone | diff  | clipped?")
print("---|--------|-------|-------|----------")
for i in range(16):
    w = pre_with_S[i]
    a = pre_without_S[i]
    diff = w - a
    clip_w = "CLIP" if w < 0 else "    "
    clip_a = "CLIP" if a < 0 else "    "
    diff_clip = "DIFF" if (w < 0) != (a < 0) else "    "
    print(f"{i:2d} | {w:6.2f} | {a:5.2f} | {diff:+5.2f} | {clip_w} {clip_a} {diff_clip}")

# How many neurons are clipped differently?
diff_clipped = ((pre_with_S < 0) != (pre_without_S < 0))
print(f"\nNeurons clipped differently: {np.sum(diff_clipped)}")
print(f"  Which ones: {np.where(diff_clipped)[0].tolist()}")

print("\n" + "=" * 70)
print("ALL ADJACENT PAIRS (gap=1)")
print("=" * 70)

print("\nTesting all (s_pos, m_pos) pairs with gap=1:")
print("s_pos | m_pos | pred | correct | margin")
print("------|-------|------|---------|-------")

all_correct = True
for s_pos in range(9):
    m_pos = s_pos + 1
    states = run_stepwise([(s_pos, S_val), (m_pos, M_val)])
    h_final = states[-1]
    pred = np.argmax(W_out @ h_final)
    logits = W_out @ h_final
    margin = logits[s_pos] - logits[m_pos]
    correct = pred == s_pos
    all_correct = all_correct and correct
    print(f"  {s_pos}   |   {m_pos}   |  {pred}   |    {'✓' if correct else '✗'}    | {margin:+.1f}")

print(f"\nAll correct: {'✓' if all_correct else '✗'}")
