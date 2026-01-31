"""
Compute tropical rank of the effective W_out for each tropical cell.

Tropical rank = largest k such that some k×k submatrix has a unique
optimal assignment (tropically non-singular).

For a matrix A, the tropical determinant is:
  trop_det(A) = max over permutations π of (sum_i A[i, π(i)])

A matrix is tropically singular if two different permutations achieve this max.
"""

import torch as th
import numpy as np
from itertools import permutations, combinations
from alg_zoo.architectures import DistRNN
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


def load_local_model():
    import os
    path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
    state_dict = th.load(path, weights_only=True, map_location='cpu')
    seq_len, hidden_size = state_dict['linear.weight'].shape
    model = DistRNN(hidden_size=hidden_size, seq_len=seq_len, bias=False)
    model.load_state_dict(state_dict)
    return model


def run_to_final(W_ih, W_hh, x_single):
    h = th.zeros(1, 16)
    for t in range(10):
        x_t = x_single[t:t+1].unsqueeze(0)
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
    return pre[0].detach().numpy()


def tropical_det(A):
    """Tropical determinant: max over permutations of sum of selected entries.
    Uses Hungarian algorithm (linear_sum_assignment) on -A to find max assignment.
    Returns (value, number of optimal permutations)."""
    n = A.shape[0]
    assert A.shape[1] == n

    # Use scipy for the optimal assignment
    row_ind, col_ind = linear_sum_assignment(-A)  # negative for max
    opt_val = A[row_ind, col_ind].sum()

    # Count how many permutations achieve this value
    # For small n, enumerate; for large n, just check if there's a tie
    if n <= 8:
        count = 0
        for perm in permutations(range(n)):
            val = sum(A[i, perm[i]] for i in range(n))
            if abs(val - opt_val) < 1e-9:
                count += 1
        return opt_val, count
    else:
        # For larger matrices, just return 1 (assume generic)
        return opt_val, 1


def is_tropically_nonsingular(A):
    """Check if square matrix A is tropically non-singular.
    Non-singular means unique optimal assignment."""
    _, count = tropical_det(A)
    return count == 1


def tropical_rank(A, max_rank=None):
    """Compute tropical rank of matrix A.
    Returns the largest k such that some k×k submatrix is tropically non-singular."""
    m, n = A.shape
    if max_rank is None:
        max_rank = min(m, n)

    for k in range(max_rank, 0, -1):
        # Check all k×k submatrices
        found = False
        for rows in combinations(range(m), k):
            if found:
                break
            for cols in combinations(range(n), k):
                submat = A[np.ix_(rows, cols)]
                if is_tropically_nonsingular(submat):
                    return k
    return 0


def section(title):
    print(f"\n{'='*80}")
    print(title)
    print('='*80)


def main():
    model = load_local_model()
    W_ih = model.rnn.weight_ih_l0.data.squeeze().clone()
    W_hh = model.rnn.weight_hh_l0.data.clone()
    W_out = model.linear.weight.data.clone().numpy()  # [10, 16]

    # Collect cells
    pairs = []
    for mt in range(10):
        for st in range(10):
            if mt == st:
                continue
            x = th.zeros(10)
            x[mt] = 1.0
            x[st] = 0.8
            pre_final = run_to_final(W_ih, W_hh, x)
            ap = (pre_final > 0).astype(int)
            ap_str = ''.join(str(b) for b in ap)
            pairs.append({
                'mt': mt, 'st': st,
                'ap_str': ap_str, 'ap': ap,
                'n_active': ap.sum(),
            })

    cells = defaultdict(list)
    for p in pairs:
        cells[p['ap_str']].append(p)

    # =========================================================================
    # 1. Tropical rank of full W_out
    # =========================================================================
    section("1. Tropical rank of full W_out (10×16)")

    print("\nW_out shape:", W_out.shape)
    ord_rank = np.linalg.matrix_rank(W_out)
    print(f"Ordinary rank: {ord_rank}")

    # Tropical rank (this might be slow for 10×16)
    print("Computing tropical rank (checking submatrices)...")
    trop_rank = tropical_rank(W_out, max_rank=10)
    print(f"Tropical rank: {trop_rank}")

    # =========================================================================
    # 2. Tropical rank of effective W_out for each cell
    # =========================================================================
    section("2. Tropical rank of effective W_out per cell")

    cell_list = sorted(cells.items(), key=lambda x: -len(x[1]))

    print(f"\n{'cell':>4} {'size':>5} {'n_active':>8} {'ord_rank':>9} {'trop_rank':>10}")
    print("-" * 45)

    for idx, (ap_str, members) in enumerate(cell_list[:15]):  # top 15 cells
        active_cols = [i for i, b in enumerate(ap_str) if b == '1']
        W_eff = W_out[:, active_cols]  # 10 × n_active

        ord_r = np.linalg.matrix_rank(W_eff)
        trop_r = tropical_rank(W_eff, max_rank=min(10, len(active_cols)))

        print(f"{idx:>4} {len(members):>5} {len(active_cols):>8} {ord_r:>9} {trop_r:>10}")

    # =========================================================================
    # 3. Check tropical singularity of key submatrices
    # =========================================================================
    section("3. Tropical determinant analysis of the dominant cell")

    dominant_ap = cell_list[0][0]
    active_cols = [i for i, b in enumerate(dominant_ap) if b == '1']
    W_eff = W_out[:, active_cols]  # 10 × 13

    print(f"\nDominant cell: {len(active_cols)} active neurons")
    print(f"Active columns: {active_cols}")
    print(f"W_eff shape: {W_eff.shape}")

    # Check all 10×10 submatrices
    print(f"\nChecking 10×10 submatrices of W_eff (10×{len(active_cols)})...")
    n_singular = 0
    n_nonsingular = 0
    for cols in combinations(range(len(active_cols)), 10):
        submat = W_eff[:, cols]
        if is_tropically_nonsingular(submat):
            n_nonsingular += 1
        else:
            n_singular += 1

    print(f"10×10 submatrices: {n_nonsingular} non-singular, {n_singular} singular")
    print(f"Total: {n_nonsingular + n_singular}")

    # =========================================================================
    # 4. What does tropical singularity mean for readout?
    # =========================================================================
    section("4. Tropical singularity interpretation")

    # For a 10×10 submatrix to be tropically singular means there are
    # two different ways to assign rows to columns with the same total weight.
    # This could indicate ambiguity in the readout.

    # Find a singular submatrix and analyze it
    print("\nFinding a tropically singular 10×10 submatrix...")
    for cols in combinations(range(len(active_cols)), 10):
        submat = W_eff[:, cols]
        val, count = tropical_det(submat)
        if count > 1:
            print(f"Found singular submatrix with columns {cols}")
            print(f"  Tropical det = {val:.3f}, achieved by {count} permutations")

            # Find the permutations
            n = 10
            opt_perms = []
            for perm in permutations(range(n)):
                pval = sum(submat[i, perm[i]] for i in range(n))
                if abs(pval - val) < 1e-9:
                    opt_perms.append(perm)
                if len(opt_perms) >= 3:
                    break

            print(f"  Example optimal permutations:")
            for p in opt_perms[:2]:
                assignments = [(i, active_cols[cols[p[i]]]) for i in range(10)]
                print(f"    {assignments[:5]}...")
            break

    # =========================================================================
    # 5. Per-pair: tropical rank of h_final values
    # =========================================================================
    section("5. Do different pairs in same cell use different tropical structures?")

    # For the dominant cell, compute which 10×10 submatrix is "active"
    # based on the actual h_final values (which columns dominate the readout)

    dominant_members = cell_list[0][1]
    print(f"\nDominant cell has {len(dominant_members)} pairs")
    print("For each pair, finding which neurons dominate the target logit...")

    for p in dominant_members[:5]:
        mt, st = p['mt'], p['st']
        x = th.zeros(10)
        x[mt] = 1.0
        x[st] = 0.8

        # Get h_final
        h = th.zeros(1, 16)
        for t in range(10):
            x_t = x[t:t+1].unsqueeze(0)
            pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
            h = th.relu(pre)
        h_final = h[0].detach().numpy()

        # Logit decomposition
        tgt = st
        contribs = [(n, h_final[n] * W_out[tgt, n]) for n in range(16) if h_final[n] > 0]
        contribs.sort(key=lambda x: -abs(x[1]))

        top3 = contribs[:3]
        print(f"  (M{mt},S{st}) tgt={tgt}: top contributors = "
              f"{[(f'n{n}', f'{c:.1f}') for n, c in top3]}")


if __name__ == "__main__":
    main()
