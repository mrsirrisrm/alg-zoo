"""
Tropical Eigenvector Analysis of W_hh

In tropical (max-plus) algebra:
- Addition is max: a ⊕ b = max(a, b)
- Multiplication is +: a ⊗ b = a + b

The tropical eigenvalue problem: A ⊗ v = λ ⊗ v
becomes: max_j(A_ij + v_j) = λ + v_i for all i

The tropical eigenvalue λ = maximum cycle mean in the graph of A.
The tropical eigenvector v shows which nodes "dominate" - higher v_i means
neuron i receives stronger recurrent input relative to others.

For the RNN, this reveals which neurons the dynamics "want" to amplify.
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

# Neuron categories
COMPS = [1, 6, 7, 8]
WAVES = [0, 10, 11, 12, 14]
BRIDGES = [3, 5, 13, 15]
OTHERS = [2, 4, 9]

NEURON_COLORS = {n: 'red' for n in COMPS}
NEURON_COLORS.update({n: 'blue' for n in WAVES})
NEURON_COLORS.update({n: 'green' for n in BRIDGES})
NEURON_COLORS.update({n: 'gray' for n in OTHERS})

NEURON_CATEGORY = {n: 'comp' for n in COMPS}
NEURON_CATEGORY.update({n: 'wave' for n in WAVES})
NEURON_CATEGORY.update({n: 'bridge' for n in BRIDGES})
NEURON_CATEGORY.update({n: 'other' for n in OTHERS})


def tropical_matrix_vector_mult(A, v):
    """Tropical matrix-vector product: (A ⊗ v)_i = max_j(A_ij + v_j)"""
    n = len(v)
    result = np.zeros(n)
    for i in range(n):
        result[i] = np.max(A[i, :] + v)
    return result


def compute_tropical_eigenvalue(A, max_iter=1000, tol=1e-10):
    """
    Compute tropical eigenvalue using power iteration.

    The tropical eigenvalue λ is the maximum cycle mean:
    λ = max over all cycles C of (sum of edge weights in C) / (length of C)

    Power iteration: v^(k+1) = A ⊗ v^(k) - λ^(k)
    where λ^(k) = max_i((A ⊗ v^(k))_i - v^(k)_i)
    """
    n = A.shape[0]
    v = np.zeros(n)  # Start with zero vector (tropical 1)

    eigenvalues = []
    for _ in range(max_iter):
        Av = tropical_matrix_vector_mult(A, v)

        # Tropical eigenvalue estimate
        lambda_k = np.max(Av - v)
        eigenvalues.append(lambda_k)

        # Normalize: subtract max to keep bounded
        v_new = Av - lambda_k

        if np.max(np.abs(v_new - v)) < tol:
            break
        v = v_new

    return lambda_k, v, eigenvalues


def find_critical_graph(A, eigenvalue, tol=1e-6):
    """
    Find the critical graph: edges (i,j) where A_ij + v_j = λ + v_i
    These are the "tight" edges that achieve the maximum.
    """
    n = A.shape[0]
    _, v, _ = compute_tropical_eigenvalue(A)

    critical_edges = []
    for i in range(n):
        for j in range(n):
            if abs(A[i, j] + v[j] - (eigenvalue + v[i])) < tol:
                critical_edges.append((j, i, A[i, j]))  # j -> i with weight A_ij

    return critical_edges, v


def compute_all_cycle_means(A):
    """Compute cycle means for all simple cycles up to length n."""
    n = A.shape[0]

    # Use Floyd-Warshall style to find max path means
    # D[k][i][j] = max mean of path from i to j using at most k edges

    cycles = []

    # Check self-loops (length 1 cycles)
    for i in range(n):
        if A[i, i] > -np.inf:
            cycles.append((A[i, i], 1, [i]))

    # Check length-2 cycles
    for i in range(n):
        for j in range(n):
            if i != j:
                weight = A[i, j] + A[j, i]
                cycles.append((weight / 2, 2, [i, j]))

    # Check length-3 cycles
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(n):
                    if k != i and k != j:
                        weight = A[i, j] + A[j, k] + A[k, i]
                        cycles.append((weight / 3, 3, [i, j, k]))

    # Sort by cycle mean
    cycles.sort(key=lambda x: -x[0])
    return cycles


def analyze_tropical_structure():
    """Main analysis of tropical eigenstructure."""

    print("=" * 70)
    print("TROPICAL EIGENVECTOR ANALYSIS OF W_hh")
    print("=" * 70)

    # Compute tropical eigenvalue and eigenvector
    eigenvalue, eigenvector, convergence = compute_tropical_eigenvalue(W_hh)

    print(f"\nTropical eigenvalue λ = {eigenvalue:.4f}")
    print("(This is the maximum cycle mean - the asymptotic growth rate)")

    # Find critical graph
    critical_edges, v = find_critical_graph(W_hh, eigenvalue)

    print(f"\nTropical eigenvector v (normalized so max = 0):")
    print("-" * 50)

    # Sort neurons by eigenvector value
    sorted_neurons = sorted(range(16), key=lambda i: -eigenvector[i])

    for rank, n in enumerate(sorted_neurons):
        cat = NEURON_CATEGORY[n]
        print(f"  n{n:2d} ({cat:6s}): v = {eigenvector[n]:7.3f}  (rank {rank+1})")

    # Analyze dominating neurons
    print("\n" + "=" * 70)
    print("DOMINATING NEURONS (highest tropical eigenvector components)")
    print("=" * 70)

    # Top neurons receive the most "flow" in the tropical sense
    top_5 = sorted_neurons[:5]
    bottom_5 = sorted_neurons[-5:]

    print("\nTop 5 (receive strongest recurrent amplification):")
    for n in top_5:
        # What feeds this neuron?
        feeders = [(j, W_hh[n, j]) for j in range(16)]
        feeders.sort(key=lambda x: -x[1])
        top_feeders = feeders[:3]
        feeder_str = ", ".join([f"n{j}({w:.2f})" for j, w in top_feeders])
        print(f"  n{n} ({NEURON_CATEGORY[n]}): v={eigenvector[n]:.3f}, fed by: {feeder_str}")

    print("\nBottom 5 (weakest recurrent support):")
    for n in bottom_5:
        print(f"  n{n} ({NEURON_CATEGORY[n]}): v={eigenvector[n]:.3f}")

    # Critical graph analysis
    print("\n" + "=" * 70)
    print("CRITICAL GRAPH (edges achieving the tropical eigenvalue)")
    print("=" * 70)

    print(f"\nNumber of critical edges: {len(critical_edges)}")
    print("\nCritical edges (j -> i means W_hh[i,j] is on a max-mean cycle):")

    critical_neurons = set()
    for j, i, w in critical_edges:
        critical_neurons.add(i)
        critical_neurons.add(j)
        print(f"  n{j} -> n{i}: weight {w:.3f}")

    print(f"\nNeurons in critical graph: {sorted(critical_neurons)}")
    print("Categories:", [NEURON_CATEGORY[n] for n in sorted(critical_neurons)])

    # Find maximum cycle mean cycles
    print("\n" + "=" * 70)
    print("TOP CYCLES BY MEAN WEIGHT")
    print("=" * 70)

    cycles = compute_all_cycle_means(W_hh)

    print("\nTop 10 cycles:")
    for mean, length, nodes in cycles[:10]:
        node_str = " -> ".join([f"n{n}" for n in nodes]) + f" -> n{nodes[0]}"
        cats = [NEURON_CATEGORY[n] for n in nodes]
        print(f"  Mean={mean:.3f}, Length={length}: {node_str}")
        print(f"    Categories: {cats}")

    # Analyze by category
    print("\n" + "=" * 70)
    print("TROPICAL EIGENVECTOR BY CATEGORY")
    print("=" * 70)

    for cat, neurons in [('comp', COMPS), ('wave', WAVES), ('bridge', BRIDGES), ('other', OTHERS)]:
        values = [eigenvector[n] for n in neurons]
        print(f"\n{cat.upper():8s}: mean={np.mean(values):.3f}, max={np.max(values):.3f}, min={np.min(values):.3f}")
        for n in neurons:
            print(f"    n{n}: {eigenvector[n]:.3f}")

    # Create visualization
    create_tropical_visualization(eigenvector, eigenvalue, critical_edges)

    return eigenvector, eigenvalue, critical_edges


def create_tropical_visualization(eigenvector, eigenvalue, critical_edges):
    """Create visualization of tropical eigenstructure."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Tropical eigenvector bar chart
    ax = axes[0, 0]
    colors = [NEURON_COLORS[n] for n in range(16)]
    ax.bar(range(16), eigenvector, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Neuron', fontsize=12)
    ax.set_ylabel('Tropical eigenvector v', fontsize=12)
    ax.set_title(f'Tropical Eigenvector (λ = {eigenvalue:.3f})\nHigher = more amplified by recurrence', fontsize=12)
    ax.set_xticks(range(16))
    ax.grid(True, alpha=0.3, axis='y')

    # 2. W_hh heatmap with critical edges highlighted
    ax = axes[0, 1]
    im = ax.imshow(W_hh, cmap='RdBu_r', aspect='equal', vmin=-3, vmax=3)
    plt.colorbar(im, ax=ax, label='Weight')

    # Mark critical edges
    for j, i, w in critical_edges:
        ax.scatter([j], [i], marker='o', s=100, facecolors='none',
                   edgecolors='lime', linewidths=2)

    ax.set_xlabel('From neuron j', fontsize=12)
    ax.set_ylabel('To neuron i', fontsize=12)
    ax.set_title('W_hh with critical edges (green circles)', fontsize=12)
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))

    # 3. Input weights W_ih vs tropical eigenvector
    ax = axes[0, 2]
    ax.scatter(W_ih, eigenvector, c=colors, s=150, edgecolors='black', alpha=0.8)
    for n in range(16):
        ax.annotate(f'{n}', (W_ih[n], eigenvector[n]), fontsize=8,
                   ha='center', va='bottom')
    ax.set_xlabel('W_ih (input sensitivity)', fontsize=12)
    ax.set_ylabel('Tropical eigenvector v', fontsize=12)
    ax.set_title('Input sensitivity vs tropical dominance', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 4. Output weights projection onto tropical eigenvector
    ax = axes[1, 0]
    # How much does each output class "see" the dominant neurons?
    output_tropical_proj = W_out @ eigenvector
    ax.bar(range(10), output_tropical_proj, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Output position', fontsize=12)
    ax.set_ylabel('W_out · v', fontsize=12)
    ax.set_title('Output projection onto tropical eigenvector\n(which outputs favor dominant neurons)', fontsize=12)
    ax.set_xticks(range(10))
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Rank eigenvector components vs W_out norms
    ax = axes[1, 1]
    w_out_norms = np.linalg.norm(W_out, axis=0)  # Norm for each hidden neuron
    ax.scatter(eigenvector, w_out_norms, c=colors, s=150, edgecolors='black', alpha=0.8)
    for n in range(16):
        ax.annotate(f'{n}', (eigenvector[n], w_out_norms[n]), fontsize=8,
                   ha='center', va='bottom')
    ax.set_xlabel('Tropical eigenvector v', fontsize=12)
    ax.set_ylabel('||W_out[:, n]|| (output influence)', fontsize=12)
    ax.set_title('Tropical dominance vs output influence', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 6. Legend and summary
    ax = axes[1, 2]
    ax.axis('off')

    # Compute summary stats
    comp_mean = np.mean([eigenvector[n] for n in COMPS])
    wave_mean = np.mean([eigenvector[n] for n in WAVES])
    bridge_mean = np.mean([eigenvector[n] for n in BRIDGES])
    other_mean = np.mean([eigenvector[n] for n in OTHERS])

    summary = f"""TROPICAL EIGENVECTOR SUMMARY

Tropical eigenvalue λ = {eigenvalue:.4f}
(Maximum cycle mean = asymptotic growth rate)

Mean eigenvector by category:
  COMPARATORS (red):  {comp_mean:.3f}
  WAVES (blue):       {wave_mean:.3f}
  BRIDGES (green):    {bridge_mean:.3f}
  OTHER (gray):       {other_mean:.3f}

INTERPRETATION:

Higher v_i means neuron i is more "tropically
dominant" - it receives stronger net recurrent
input relative to others.

The tropical eigenvalue λ determines how fast
the max activation grows per timestep (in the
tropical/max-plus sense).

Critical edges form cycles that achieve this
maximum growth rate - these are the "backbone"
of the recurrent dynamics.

Neurons NOT on critical cycles are "dominated" -
their activations grow slower than λ per step.
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            ha='left', va='top', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig('docs/tropical_eigenvector_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved visualization to docs/tropical_eigenvector_analysis.png")


if __name__ == "__main__":
    eigenvector, eigenvalue, critical_edges = analyze_tropical_structure()
