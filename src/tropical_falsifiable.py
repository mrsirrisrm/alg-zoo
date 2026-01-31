"""
Falsifiable tests for tropical genericity claim.

Claim: The model has found a minimum where W_out is tropically generic
(all relevant submatrices are non-singular).

Tests:
1. Perturbation robustness: does small noise create singularities?
2. Random baseline: are random matrices also tropically generic?
3. Gap to singularity: how close is W_out to having a tie?
"""

import torch as th
import numpy as np
from itertools import permutations, combinations
from alg_zoo.architectures import DistRNN
from scipy.optimize import linear_sum_assignment


def load_local_model():
    import os
    path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
    state_dict = th.load(path, weights_only=True, map_location='cpu')
    seq_len, hidden_size = state_dict['linear.weight'].shape
    model = DistRNN(hidden_size=hidden_size, seq_len=seq_len, bias=False)
    model.load_state_dict(state_dict)
    return model


def tropical_det_gap(A):
    """Return the gap between best and second-best tropical determinant.
    Larger gap = more robustly non-singular."""
    n = A.shape[0]
    assert A.shape[1] == n

    # Enumerate all permutations and find top 2 values
    vals = []
    for perm in permutations(range(n)):
        val = sum(A[i, perm[i]] for i in range(n))
        vals.append(val)

    vals.sort(reverse=True)
    return vals[0] - vals[1]  # gap between best and second-best


def section(title):
    print(f"\n{'='*80}")
    print(title)
    print('='*80)


def main():
    model = load_local_model()
    W_out = model.linear.weight.data.clone().numpy()  # [10, 16]

    # =========================================================================
    # 1. Gap to singularity for full W_out
    # =========================================================================
    section("1. Gap to tropical singularity for W_out submatrices")

    print("\nFor each 10×10 submatrix, compute the gap between best and")
    print("second-best assignment. Gap=0 means singular (tie).\n")

    gaps = []
    for cols in combinations(range(16), 10):
        submat = W_out[:, cols]
        gap = tropical_det_gap(submat)
        gaps.append((gap, cols))

    gaps.sort()  # smallest gap first
    print(f"Smallest gaps (closest to singularity):")
    for gap, cols in gaps[:5]:
        print(f"  gap={gap:.6f}, cols={cols}")

    print(f"\nLargest gaps (most robust):")
    for gap, cols in gaps[-5:]:
        print(f"  gap={gap:.6f}, cols={cols}")

    print(f"\nSummary:")
    print(f"  Min gap: {gaps[0][0]:.6f}")
    print(f"  Max gap: {gaps[-1][0]:.6f}")
    print(f"  Mean gap: {np.mean([g for g, _ in gaps]):.6f}")

    # =========================================================================
    # 2. Perturbation test: add noise, check for singularities
    # =========================================================================
    section("2. Perturbation robustness")

    print("\nAdd Gaussian noise to W_out, check if singularities appear.")

    noise_levels = [0.001, 0.01, 0.1, 0.5, 1.0]
    n_trials = 20

    print(f"\n{'noise':>8} {'n_singular':>12} {'min_gap':>10} {'mean_gap':>10}")
    print("-" * 45)

    for noise in noise_levels:
        n_singular = 0
        min_gaps = []

        for trial in range(n_trials):
            W_noisy = W_out + np.random.randn(*W_out.shape) * noise

            # Check a subset of submatrices
            trial_min_gap = float('inf')
            for cols in combinations(range(16), 10):
                submat = W_noisy[:, cols]
                gap = tropical_det_gap(submat)
                if gap < 1e-9:
                    n_singular += 1
                trial_min_gap = min(trial_min_gap, gap)
            min_gaps.append(trial_min_gap)

        print(f"{noise:>8.3f} {n_singular:>12} {np.min(min_gaps):>10.4f} {np.mean(min_gaps):>10.4f}")

    # =========================================================================
    # 3. Random baseline: are random matrices also non-singular?
    # =========================================================================
    section("3. Random baseline comparison")

    print("\nGenerate random 10×16 matrices, check tropical genericity.")

    # Match W_out statistics
    w_mean = W_out.mean()
    w_std = W_out.std()
    print(f"W_out stats: mean={w_mean:.3f}, std={w_std:.3f}")

    n_random = 50
    n_singular_random = 0
    random_min_gaps = []

    for _ in range(n_random):
        W_rand = np.random.randn(10, 16) * w_std + w_mean

        trial_min_gap = float('inf')
        for cols in combinations(range(16), 10):
            submat = W_rand[:, cols]
            gap = tropical_det_gap(submat)
            if gap < 1e-9:
                n_singular_random += 1
            trial_min_gap = min(trial_min_gap, gap)
        random_min_gaps.append(trial_min_gap)

    print(f"\nRandom matrices ({n_random} trials):")
    print(f"  Singular submatrices found: {n_singular_random}")
    print(f"  Min gap across all: {np.min(random_min_gaps):.6f}")
    print(f"  Mean min gap: {np.mean(random_min_gaps):.6f}")
    print(f"\nLearned W_out:")
    print(f"  Min gap: {gaps[0][0]:.6f}")

    if np.mean(random_min_gaps) > gaps[0][0]:
        print(f"\n→ Random matrices have LARGER gaps on average!")
        print(f"  This suggests W_out is closer to singularity than random.")
    else:
        print(f"\n→ W_out has gaps comparable to or larger than random.")
        print(f"  Tropical genericity is the expected baseline.")

    # =========================================================================
    # 4. Which neurons, if removed, would most reduce discriminability?
    # =========================================================================
    section("4. Critical neurons for tropical genericity")

    print("\nFor each neuron, compute the min gap when that column is excluded.")
    print("(Using remaining 15 columns to form 10×10 submatrices)\n")

    print(f"{'exclude':>8} {'min_gap':>10} {'mean_gap':>10}")
    print("-" * 32)

    for exclude in range(16):
        cols_remaining = [c for c in range(16) if c != exclude]
        gaps_excl = []
        for cols in combinations(cols_remaining, 10):
            submat = W_out[:, cols]
            gap = tropical_det_gap(submat)
            gaps_excl.append(gap)

        print(f"  n{exclude:<6} {np.min(gaps_excl):>10.4f} {np.mean(gaps_excl):>10.4f}")


if __name__ == "__main__":
    main()
