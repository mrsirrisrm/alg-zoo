"""
Investigate D[t] after the second impulse.

Current model iterates h after the second impulse to get D[t] for remaining
timesteps. If D[t] after the second impulse depends only on gap (and not on
the specific positions), we can precompute a canonical D for each gap.

This would eliminate the Phase 4 iteration entirely.
"""

import numpy as np
from alg_zoo import example_2nd_argmax

model = example_2nd_argmax()
model.eval()

W_ih = model.rnn.weight_ih_l0.detach().numpy().squeeze()
W_hh = model.rnn.weight_hh_l0.detach().numpy()
W_out = model.linear.weight.detach().numpy()

M_val = 1.0
S_val = 0.8


def get_full_D_sequence(m, s, M_val, S_val):
    """Run full model, return D[t] at each timestep."""
    h = np.zeros(16)
    D_seq = []

    for t in range(10):
        x = M_val if t == m else (S_val if t == s else 0.0)
        pre = W_ih * x + W_hh @ h
        D = frozenset(np.where(pre > 0)[0])
        D_seq.append(D)
        h = np.maximum(0, pre)

    return D_seq


print("=" * 70)
print("D[t] AFTER SECOND IMPULSE: DEPENDENCE ON GAP")
print("=" * 70)

# =============================================================================
# Collect D sequences for all pairs, organized by gap
# =============================================================================

# gap = second_pos - first_pos (always positive)
# steps_after = number of timesteps after second impulse = 9 - second_pos

# For each gap, collect the D[t] values for timesteps AFTER the second impulse
# Key question: are these the same for all pairs with the same gap?

print("\nCollecting D sequences for all 90 pairs...")

# Organize by: gap, direction (fwd = M first, rev = S first), steps_after_second
# D_after[gap][direction] = list of (pair, D_after_list) tuples

D_after_by_gap = {}  # gap -> {(first_pos, second_pos, direction): D_after_sequence}

for m in range(10):
    for s in range(10):
        if m == s:
            continue

        D_seq = get_full_D_sequence(m, s, M_val, S_val)

        first_pos = min(m, s)
        second_pos = max(m, s)
        gap = second_pos - first_pos
        direction = 'fwd' if m < s else 'rev'
        first_val = M_val if m < s else S_val
        second_val = S_val if m < s else M_val

        # D values at and after second impulse
        D_after = D_seq[second_pos:]  # includes D at second impulse

        key = (first_pos, second_pos, direction)

        if gap not in D_after_by_gap:
            D_after_by_gap[gap] = {}
        D_after_by_gap[gap][key] = D_after

# =============================================================================
# Check: for each gap and direction, are D_after sequences identical?
# =============================================================================

print("\n" + "=" * 70)
print("CHECKING D INVARIANCE BY GAP")
print("=" * 70)

for gap in sorted(D_after_by_gap.keys()):
    entries = D_after_by_gap[gap]

    # Separate by direction
    fwd_entries = {k: v for k, v in entries.items() if k[2] == 'fwd'}
    rev_entries = {k: v for k, v in entries.items() if k[2] == 'rev'}

    print(f"\n### Gap = {gap}")

    for direction, dir_entries in [('fwd (M first)', fwd_entries), ('rev (S first)', rev_entries)]:
        if not dir_entries:
            continue

        print(f"\n  {direction}: {len(dir_entries)} pairs")

        # Check if all D_after sequences are the same
        sequences = list(dir_entries.values())
        reference = sequences[0]

        all_same = True
        diffs = []
        for key, seq in dir_entries.items():
            first_pos, second_pos, d = key
            for step_idx, (d_ref, d_actual) in enumerate(zip(reference, seq)):
                if d_ref != d_actual:
                    all_same = False
                    t_actual = second_pos + step_idx
                    diffs.append((first_pos, second_pos, step_idx, t_actual,
                                  d_ref - d_actual, d_actual - d_ref))

        if all_same:
            print(f"    ALL IDENTICAL across positions")
            print(f"    D sequence after 2nd impulse ({len(reference)} steps):")
            for step_idx, D in enumerate(reference):
                print(f"      +{step_idx}: {len(D):2d} neurons: {sorted(D)}")
        else:
            print(f"    DIFFERENCES FOUND:")
            # Group diffs by step
            for step_idx in sorted(set(d[2] for d in diffs)):
                step_diffs = [d for d in diffs if d[2] == step_idx]
                print(f"      +{step_idx}: {len(step_diffs)} pairs differ")
                for first_pos, second_pos, _, t_actual, missing, extra in step_diffs[:3]:
                    print(f"        pair ({first_pos},{second_pos}) at t={t_actual}: "
                          f"missing {sorted(missing)}, extra {sorted(extra)}")

# =============================================================================
# Detailed analysis: what varies and what doesn't
# =============================================================================

print("\n" + "=" * 70)
print("DETAILED D ANALYSIS: WHICH NEURONS VARY?")
print("=" * 70)

for gap in sorted(D_after_by_gap.keys()):
    entries = D_after_by_gap[gap]

    fwd_entries = {k: v for k, v in entries.items() if k[2] == 'fwd'}
    rev_entries = {k: v for k, v in entries.items() if k[2] == 'rev'}

    for direction, dir_entries in [('fwd', fwd_entries), ('rev', rev_entries)]:
        if not dir_entries:
            continue

        sequences = list(dir_entries.values())
        max_steps = max(len(seq) for seq in sequences)

        for step_idx in range(max_steps):
            # Collect all D sets at this step
            D_sets = [seq[step_idx] for seq in sequences if step_idx < len(seq)]
            if not D_sets:
                continue

            # Per-neuron: how often is each neuron active?
            active_counts = np.zeros(16)
            for D in D_sets:
                for n in D:
                    active_counts[n] += 1

            n_samples = len(D_sets)
            always_active = [n for n in range(16) if active_counts[n] == n_samples]
            never_active = [n for n in range(16) if active_counts[n] == 0]
            variable = [n for n in range(16) if 0 < active_counts[n] < n_samples]

            if variable:
                print(f"\n  Gap={gap}, {direction}, +{step_idx}: "
                      f"{len(always_active)} always, {len(never_active)} never, "
                      f"{len(variable)} VARIABLE")
                for n in variable:
                    pct = 100 * active_counts[n] / n_samples
                    print(f"    n{n}: active {pct:.0f}% ({int(active_counts[n])}/{n_samples})")

# =============================================================================
# Can we define a canonical D for each (gap, direction)?
# =============================================================================

print("\n" + "=" * 70)
print("CANONICAL D BY GAP")
print("=" * 70)

# For the canonical D, use "majority vote" for variable neurons
# Then test if using this canonical D gives correct predictions

canonical_D_by_gap = {}  # (gap, direction) -> list of D sets

for gap in sorted(D_after_by_gap.keys()):
    entries = D_after_by_gap[gap]

    for direction in ['fwd', 'rev']:
        dir_entries = {k: v for k, v in entries.items() if k[2] == direction}
        if not dir_entries:
            continue

        sequences = list(dir_entries.values())
        max_steps = max(len(seq) for seq in sequences)

        canonical = []
        for step_idx in range(max_steps):
            D_sets = [seq[step_idx] for seq in sequences if step_idx < len(seq)]
            n_samples = len(D_sets)

            # Majority vote
            active_counts = np.zeros(16)
            for D in D_sets:
                for n in D:
                    active_counts[n] += 1

            canonical_D = frozenset(n for n in range(16) if active_counts[n] > n_samples / 2)
            canonical.append(canonical_D)

        canonical_D_by_gap[(gap, direction)] = canonical

print("\nCanonical D sequences by (gap, direction):")
for key in sorted(canonical_D_by_gap.keys()):
    gap, direction = key
    seq = canonical_D_by_gap[key]
    print(f"\n  Gap={gap}, {direction}:")
    for step_idx, D in enumerate(seq):
        print(f"    +{step_idx}: {len(D):2d} neurons: {sorted(D)}")

# =============================================================================
# Test: use canonical D (by gap) for the full prediction
# =============================================================================

print("\n" + "=" * 70)
print("TESTING CANONICAL D BY GAP")
print("=" * 70)


def predict_with_canonical_gap_D(m, s, M_val, S_val, canonical_D_by_gap, canonical_D_pre):
    """
    Predict using:
    - canonical_D_pre for steps before second impulse
    - canonical_D_by_gap for steps at and after second impulse
    No iteration needed.
    """
    first_pos = min(m, s)
    second_pos = max(m, s)
    gap = second_pos - first_pos
    direction = 'fwd' if m < s else 'rev'

    # Build full D sequence
    D_seq = []

    # Phase 1: before first impulse
    for t in range(first_pos):
        D_seq.append(frozenset())

    # Phase 2: after first, before second (canonical single-impulse)
    for t in range(first_pos, second_pos):
        k = t - first_pos
        D_seq.append(canonical_D_pre[k])

    # Phase 3+4: at and after second impulse (canonical by gap)
    gap_D = canonical_D_by_gap[(gap, direction)]
    for step_idx in range(len(gap_D)):
        D_seq.append(gap_D[step_idx])

    # Now run the dynamics with these D masks (no ReLU needed, D is known)
    h = np.zeros(16)
    for t in range(10):
        x = M_val if t == m else (S_val if t == s else 0.0)
        pre = W_ih * x + W_hh @ h
        D_mask = np.array([1.0 if i in D_seq[t] else 0.0 for i in range(16)])
        h = D_mask * np.maximum(0, pre)

    logits = W_out @ h
    pred = np.argmax(logits)
    return pred


# Compute canonical D for single impulse (pre-second)
canonical_D_pre = []
h = np.maximum(0, W_ih)
canonical_D_pre.append(frozenset(np.where(W_ih > 0)[0]))
for k in range(1, 10):
    pre = W_hh @ h
    canonical_D_pre.append(frozenset(np.where(pre > 0)[0]))
    h = np.maximum(0, pre)

# Test all 90 pairs
correct_canonical = 0
correct_actual = 0
mismatches = []

for m in range(10):
    for s in range(10):
        if m == s:
            continue

        # Canonical prediction
        pred_can = predict_with_canonical_gap_D(m, s, M_val, S_val,
                                                 canonical_D_by_gap, canonical_D_pre)
        # Actual model
        h = np.zeros(16)
        for t in range(10):
            x = M_val if t == m else (S_val if t == s else 0.0)
            pre = W_ih * x + W_hh @ h
            h = np.maximum(0, pre)
        logits_actual = W_out @ h
        pred_actual = np.argmax(logits_actual)

        if pred_can == s:
            correct_canonical += 1
        if pred_actual == s:
            correct_actual += 1
        if pred_can != pred_actual:
            mismatches.append((m, s, pred_can, pred_actual))

print(f"\nCanonical-by-gap accuracy: {correct_canonical}/90 = {100*correct_canonical/90:.1f}%")
print(f"Actual model accuracy:     {correct_actual}/90 = {100*correct_actual/90:.1f}%")
print(f"Mismatches: {len(mismatches)}")

if mismatches:
    print("\nMismatch details:")
    for m, s, pred_can, pred_actual in mismatches:
        gap = abs(m - s)
        direction = 'fwd' if m < s else 'rev'
        print(f"  (m={m}, s={s}) gap={gap} {direction}: "
              f"canonical predicts {pred_can}, actual predicts {pred_actual}")

# =============================================================================
# Further: does D depend on direction at all, or just gap?
# =============================================================================

print("\n" + "=" * 70)
print("D DEPENDENCE ON DIRECTION (FWD vs REV)")
print("=" * 70)

for gap in sorted(set(g for g, d in canonical_D_by_gap.keys())):
    fwd_key = (gap, 'fwd')
    rev_key = (gap, 'rev')

    if fwd_key in canonical_D_by_gap and rev_key in canonical_D_by_gap:
        fwd_seq = canonical_D_by_gap[fwd_key]
        rev_seq = canonical_D_by_gap[rev_key]

        same = all(f == r for f, r in zip(fwd_seq, rev_seq))
        print(f"\n  Gap={gap}: fwd vs rev D identical? {same}")
        if not same:
            for step_idx, (f, r) in enumerate(zip(fwd_seq, rev_seq)):
                if f != r:
                    print(f"    +{step_idx}: fwd={sorted(f)}")
                    print(f"           rev={sorted(r)}")
                    print(f"           fwd_only={sorted(f - r)}, rev_only={sorted(r - f)}")
