"""
Deep dive into n4: the critical hub that feeds dominant neurons but dies.

n4 has:
- Lowest tropical eigenvector (-242.21)
- Only 14% active at t=10
- But appears in ALL top cycles
- Highest W_hh outgoing weights to dominant neurons

This is the "hidden pump" of the network.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch as th
from alg_zoo.architectures import DistRNN


def load_local_model():
    path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
    state_dict = th.load(path, weights_only=True, map_location='cpu')
    seq_len, hidden_size = state_dict['linear.weight'].shape
    model = DistRNN(hidden_size=hidden_size, seq_len=seq_len, bias=False)
    model.load_state_dict(state_dict)
    return model


model = load_local_model()
W_ih = model.rnn.weight_ih_l0.detach().numpy().squeeze()
W_hh = model.rnn.weight_hh_l0.detach().numpy()
W_out = model.linear.weight.detach().numpy()

COMPS = [1, 6, 7, 8]
WAVES = [0, 10, 11, 12, 14]
BRIDGES = [3, 5, 13, 15]
OTHERS = [2, 4, 9]


def run_stepwise(impulses):
    h = np.zeros(16)
    states = [h.copy()]
    pre_states = [h.copy()]
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        states.append(h.copy())
        pre_states.append(pre.copy())
    return states, pre_states


def analyze_n4():
    print("=" * 70)
    print("N4: THE CRITICAL HUB ANALYSIS")
    print("=" * 70)

    # n4's connectivity
    print("\n1. N4's OUTGOING CONNECTIONS (W_hh[:, 4])")
    print("-" * 50)
    out_weights = W_hh[:, 4]  # Column 4 = weights from n4 to others
    sorted_out = sorted(enumerate(out_weights), key=lambda x: -x[1])

    for i, w in sorted_out:
        cat = 'comp' if i in COMPS else 'wave' if i in WAVES else 'bridge' if i in BRIDGES else 'other'
        print(f"  n4 -> n{i:2d} ({cat:6s}): {w:+.3f}")

    # n4's incoming connections
    print("\n2. N4's INCOMING CONNECTIONS (W_hh[4, :])")
    print("-" * 50)
    in_weights = W_hh[4, :]  # Row 4 = weights to n4 from others
    sorted_in = sorted(enumerate(in_weights), key=lambda x: -x[1])

    for i, w in sorted_in:
        cat = 'comp' if i in COMPS else 'wave' if i in WAVES else 'bridge' if i in BRIDGES else 'other'
        print(f"  n{i:2d} ({cat:6s}) -> n4: {w:+.3f}")

    # n4's input sensitivity
    print(f"\n3. N4's INPUT SENSITIVITY: W_ih[4] = {W_ih[4]:.3f}")
    print(f"   (Rank among all: {sorted(enumerate(W_ih), key=lambda x: -x[1]).index((4, W_ih[4])) + 1}/16)")

    # Trace n4 through dynamics
    print("\n4. N4's ACTIVATION PATTERN OVER TIME")
    print("-" * 50)

    # Test several cases
    test_cases = [
        ((0, 1.0), (5, 0.8), "Fwd M@0, S@5"),
        ((0, 0.8), (5, 1.0), "Rev S@0, M@5"),
        ((4, 1.0), (5, 0.8), "Fwd M@4, S@5"),
        ((4, 0.8), (5, 1.0), "Rev S@4, M@5"),
        ((0, 1.0), (9, 0.8), "Fwd M@0, S@9"),
    ]

    for impulses_desc in test_cases:
        impulses = [(impulses_desc[0][0], impulses_desc[0][1]), (impulses_desc[1][0], impulses_desc[1][1])]
        desc = impulses_desc[2]

        states, pre_states = run_stepwise(impulses)

        print(f"\n  {desc}:")
        print(f"    t   | h[4]     | pre[4]   | n4 active?")
        print(f"    " + "-" * 40)
        for t in range(1, 11):
            h4 = states[t][4]
            pre4 = pre_states[t][4]
            active = "YES" if h4 > 0 else "no"
            print(f"    {t:2d}  | {h4:8.3f} | {pre4:8.3f} | {active}")

    # Analyze what kills n4
    print("\n5. WHAT KILLS N4?")
    print("-" * 50)

    # n4's self-connection
    print(f"  n4 -> n4 weight: {W_hh[4, 4]:.3f}")

    # Check if negative inputs dominate
    neg_inputs = [i for i in range(16) if W_hh[4, i] < -0.5]
    print(f"  Strong negative inputs to n4: {neg_inputs}")
    for i in neg_inputs:
        print(f"    W_hh[4, {i}] = {W_hh[4, i]:.3f}")

    # The critical insight: n4 pumps other neurons but gets inhibited
    print("\n6. N4 AS A TRANSIENT PUMP")
    print("-" * 50)

    # Sum of outgoing positive weights vs incoming positive weights
    out_pos_sum = np.sum(np.maximum(0, W_hh[:, 4]))
    in_pos_sum = np.sum(np.maximum(0, W_hh[4, :]))

    print(f"  Sum of positive outgoing: {out_pos_sum:.3f}")
    print(f"  Sum of positive incoming: {in_pos_sum:.3f}")
    print(f"  Ratio (out/in): {out_pos_sum/in_pos_sum:.2f}")

    # Who does n4 pump the most?
    print("\n  Top neurons n4 pumps:")
    for i, w in sorted_out[:5]:
        if w > 0:
            cat = 'comp' if i in COMPS else 'wave' if i in WAVES else 'bridge' if i in BRIDGES else 'other'
            print(f"    n{i:2d} ({cat}): +{w:.3f}")

    # Create visualization
    create_n4_visualization()


def create_n4_visualization():
    """Visualize n4's role as the critical hub."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. n4's outgoing weights
    ax = axes[0, 0]
    out_weights = W_hh[:, 4]
    colors = ['red' if i in COMPS else 'blue' if i in WAVES else 'green' if i in BRIDGES else 'gray' for i in range(16)]
    ax.bar(range(16), out_weights, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Target neuron', fontsize=12)
    ax.set_ylabel('Weight W_hh[i, 4]', fontsize=12)
    ax.set_title('N4 OUTGOING: What n4 pumps\n(n4 feeds waves n10,n11 and other n2)', fontsize=12)
    ax.set_xticks(range(16))
    ax.grid(True, alpha=0.3, axis='y')

    # 2. n4's incoming weights
    ax = axes[0, 1]
    in_weights = W_hh[4, :]
    ax.bar(range(16), in_weights, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Source neuron', fontsize=12)
    ax.set_ylabel('Weight W_hh[4, j]', fontsize=12)
    ax.set_title('N4 INCOMING: What feeds n4\n(n4 fed by bridge n15, but inhibited by comps)', fontsize=12)
    ax.set_xticks(range(16))
    ax.grid(True, alpha=0.3, axis='y')

    # 3. n4's activation trace through time for several cases
    ax = axes[1, 0]

    cases = [
        ([(0, 1.0), (5, 0.8)], 'Fwd M@0,S@5', 'b-o'),
        ([(0, 0.8), (5, 1.0)], 'Rev S@0,M@5', 'r-s'),
        ([(4, 1.0), (5, 0.8)], 'Fwd M@4,S@5', 'g-^'),
        ([(0, 1.0), (9, 0.8)], 'Fwd M@0,S@9', 'm-d'),
    ]

    for impulses, label, style in cases:
        states, _ = run_stepwise(impulses)
        n4_trace = [states[t][4] for t in range(1, 11)]
        ax.plot(range(1, 11), n4_trace, style, label=label, markersize=8, linewidth=2)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('n4 activation', fontsize=12)
    ax.set_title('N4 activation over time\n(transient - dies by t=10 in most cases)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 11))

    # 4. Compare n4 pumping to n10, n11 activations
    ax = axes[1, 1]

    impulses = [(0, 1.0), (5, 0.8)]
    states, _ = run_stepwise(impulses)

    t = range(1, 11)
    ax.plot(t, [states[i][4] for i in range(1, 11)], 'gray', marker='o', linewidth=2, label='n4 (hub)', markersize=8)
    ax.plot(t, [states[i][10] for i in range(1, 11)], 'blue', marker='s', linewidth=2, label='n10 (wave)', markersize=8)
    ax.plot(t, [states[i][11] for i in range(1, 11)], 'cyan', marker='^', linewidth=2, label='n11 (wave)', markersize=8)
    ax.plot(t, [states[i][15] for i in range(1, 11)], 'green', marker='d', linewidth=2, label='n15 (bridge)', markersize=8)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(1, color='orange', linestyle='--', alpha=0.5, label='M@0 impulse')
    ax.axvline(6, color='red', linestyle='--', alpha=0.5, label='S@5 impulse')

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Activation', fontsize=12)
    ax.set_title('N4 pumps n10,n11,n15 then dies\n(Fwd M@0, S@5)', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 11))

    plt.tight_layout()
    plt.savefig('docs/n4_hub_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved to docs/n4_hub_analysis.png")


if __name__ == "__main__":
    analyze_n4()
