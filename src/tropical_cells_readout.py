"""
Does W_out work uniformly across tropical cells, or does it rely on
cell-specific structure?

Within a cell at t=9, some neurons are dead (h=0). The effective readout
is W_out[:, active_neurons] @ h_active. Different cells use different
subsets of W_out columns.

Questions:
1. For the dominant cell (23 pairs), how does W_out discriminate 23 different targets?
2. If we force all neurons active (use pre-activation instead of post-ReLU),
   does W_out still work? This tests whether the zeroing matters.
3. How much logit contribution comes from neurons that differ between cells?
4. If we swap a pair into a different cell's activation pattern, does it break?
"""

import torch as th
import numpy as np
from alg_zoo.architectures import DistRNN
from collections import defaultdict


def load_local_model():
    import os
    path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
    state_dict = th.load(path, weights_only=True, map_location='cpu')
    seq_len, hidden_size = state_dict['linear.weight'].shape
    model = DistRNN(hidden_size=hidden_size, seq_len=seq_len, bias=False)
    model.load_state_dict(state_dict)
    return model


def run_trace_full(W_ih, W_hh, x_single):
    h = th.zeros(1, 16)
    h_trace = []
    pre_trace = []
    for t in range(10):
        x_t = x_single[t:t+1].unsqueeze(0)
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
        pre_trace.append(pre[0].detach().numpy().copy())
        h_trace.append(h[0].detach().numpy().copy())
    return np.array(h_trace), np.array(pre_trace)


def section(title):
    print(f"\n{'='*80}")
    print(title)
    print('='*80)


def main():
    model = load_local_model()
    W_ih = model.rnn.weight_ih_l0.data.squeeze().clone()
    W_hh = model.rnn.weight_hh_l0.data.clone()
    W_out = model.linear.weight.data.clone().numpy()  # [10, 16]

    # Collect all data
    pairs = []
    for mt in range(10):
        for st in range(10):
            if mt == st:
                continue
            x = th.zeros(10)
            x[mt] = 1.0
            x[st] = 0.8
            h_trace, pre_trace = run_trace_full(W_ih, W_hh, x)
            h_final = h_trace[9]
            pre_final = pre_trace[9]
            ap_final = (pre_final > 0).astype(int)
            ap_str = ''.join(str(b) for b in ap_final)
            ordering = "fwd" if st > mt else "rev"
            pairs.append({
                'mt': mt, 'st': st, 'target': st,
                'h_final': h_final, 'pre_final': pre_final,
                'ap_str': ap_str, 'ap': ap_final,
                'ordering': ordering,
                'gap': abs(st - mt),
                'last_clip': max(mt, st),
            })

    # Group by cell
    cells = defaultdict(list)
    for p in pairs:
        cells[p['ap_str']].append(p)

    # =========================================================================
    # 1. Overview: cells, sizes, and accuracy within each
    # =========================================================================
    section("1. Tropical cells at t=9: size and readout accuracy")

    print(f"\n{'cell':>4} {'size':>5} {'active':>7} {'dead_neurons':>30} {'targets':>30}")
    print("-" * 85)

    cell_list = sorted(cells.items(), key=lambda x: -len(x[1]))
    for idx, (ap_str, members) in enumerate(cell_list):
        dead = [f'n{i}' for i, b in enumerate(ap_str) if b == '0']
        targets = sorted(set(p['target'] for p in members))
        n_active = sum(int(b) for b in ap_str)
        print(f"{idx:>4} {len(members):>5} {n_active:>4}/16 "
              f"{','.join(dead):>30} {str(targets):>30}")

    # =========================================================================
    # 2. The dominant cell: how does W_out separate 23 pairs?
    # =========================================================================
    dominant_ap, dominant_members = cell_list[0]
    section(f"2. Dominant cell ({len(dominant_members)} pairs): readout analysis")

    dead_neurons = [i for i, b in enumerate(dominant_ap) if b == '0']
    active_neurons = [i for i, b in enumerate(dominant_ap) if b == '1']
    print(f"\nDead neurons: {[f'n{n}' for n in dead_neurons]}")
    print(f"Active neurons: {[f'n{n}' for n in active_neurons]}")

    # Within this cell, the effective readout is W_out[:, active] @ h[active]
    # Show that it correctly separates all targets
    print(f"\n{'mt':>3} {'st':>3} {'ord':>4} {'gap':>4} {'tgt':>4} {'pred':>5} {'margin':>8} {'ok':>3}")
    print("-" * 40)

    for p in sorted(dominant_members, key=lambda p: (p['ordering'], p['target'])):
        logits = p['h_final'] @ W_out.T
        pred = logits.argmax()
        tgt = p['target']
        margin = logits[tgt] - max(logits[j] for j in range(10) if j != tgt)
        ok = "Y" if pred == tgt else "N"
        print(f"{p['mt']:>3} {p['st']:>3} {p['ordering']:>4} {p['gap']:>4} {tgt:>4} {pred:>5} {margin:>8.2f} {ok:>3}")

    # =========================================================================
    # 3. Test: use pre-activation (bypass ReLU at t=9) instead of h_final
    #    This removes the cell structure entirely — all neurons contribute
    # =========================================================================
    section("3. Bypass ReLU at t=9: use pre-activation for readout")

    print(f"\nIf we use pre_final instead of h_final = relu(pre_final),")
    print(f"dead neurons now contribute their NEGATIVE pre-activations.\n")

    correct_h = 0
    correct_pre = 0
    margin_diffs = []

    for p in pairs:
        logits_h = p['h_final'] @ W_out.T
        logits_pre = p['pre_final'] @ W_out.T
        tgt = p['target']

        pred_h = logits_h.argmax()
        pred_pre = logits_pre.argmax()

        if pred_h == tgt:
            correct_h += 1
        if pred_pre == tgt:
            correct_pre += 1

        margin_h = logits_h[tgt] - max(logits_h[j] for j in range(10) if j != tgt)
        margin_pre = logits_pre[tgt] - max(logits_pre[j] for j in range(10) if j != tgt)
        margin_diffs.append(margin_pre - margin_h)

    print(f"Accuracy with h_final (relu):    {correct_h}/90 = {correct_h/90*100:.1f}%")
    print(f"Accuracy with pre_final (no relu): {correct_pre}/90 = {correct_pre/90*100:.1f}%")
    print(f"Mean margin change (pre - h):     {np.mean(margin_diffs):.2f}")
    print(f"Min margin change:                {np.min(margin_diffs):.2f}")
    print(f"Pairs where margin IMPROVED:      {sum(1 for d in margin_diffs if d > 0)}/90")

    # =========================================================================
    # 4. Contribution of dead neurons: how much logit would they add?
    # =========================================================================
    section("4. Contribution of dead neurons (suppressed by ReLU)")

    print(f"\nFor each pair, compute what the dead neurons WOULD contribute")
    print(f"if ReLU didn't zero them: dead_contrib = pre[dead] @ W_out[:, dead].T\n")

    print(f"{'mt':>3} {'st':>3} {'ord':>4} {'tgt':>4} {'dead_to_tgt':>12} {'dead_to_best_wrong':>19} "
          f"{'dead_margin_effect':>19} {'n_dead':>7}")
    print("-" * 75)

    # Show a sample
    for p in sorted(pairs, key=lambda p: (p['ordering'], p['gap'], p['mt']))[:20]:
        dead = [i for i, b in enumerate(p['ap_str']) if b == '0']
        if not dead:
            continue
        tgt = p['target']
        h = p['h_final']
        pre = p['pre_final']

        # Current logits
        logits = h @ W_out.T
        best_wrong = max(logits[j] for j in range(10) if j != tgt)
        bw_idx = [j for j in range(10) if j != tgt and logits[j] == best_wrong][0]

        # Dead neuron contributions
        dead_to_tgt = sum(pre[n] * W_out[tgt, n] for n in dead)
        dead_to_bw = sum(pre[n] * W_out[bw_idx, n] for n in dead)
        dead_margin = dead_to_tgt - dead_to_bw

        print(f"{p['mt']:>3} {p['st']:>3} {p['ordering']:>4} {tgt:>4} {dead_to_tgt:>12.2f} "
              f"{dead_to_bw:>19.2f} {dead_margin:>19.2f} {len(dead):>7}")

    # Summary stats
    all_dead_margins = []
    all_dead_tgt = []
    for p in pairs:
        dead = [i for i, b in enumerate(p['ap_str']) if b == '0']
        if not dead:
            continue
        tgt = p['target']
        logits = p['h_final'] @ W_out.T
        bw_idx = sorted(range(10), key=lambda j: -logits[j])
        bw_idx = [j for j in bw_idx if j != tgt][0]
        dead_to_tgt = sum(p['pre_final'][n] * W_out[tgt, n] for n in dead)
        dead_to_bw = sum(p['pre_final'][n] * W_out[bw_idx, n] for n in dead)
        all_dead_margins.append(dead_to_tgt - dead_to_bw)
        all_dead_tgt.append(dead_to_tgt)

    print(f"\n--- Summary ---")
    print(f"Mean dead-neuron margin effect: {np.mean(all_dead_margins):.2f}")
    print(f"Std: {np.std(all_dead_margins):.2f}")
    print(f"Mean |dead → target|: {np.mean(np.abs(all_dead_tgt)):.2f}")
    print(f"Cases where dead neurons would HELP margin: {sum(1 for d in all_dead_margins if d > 0)}/90")
    print(f"Cases where dead neurons would HURT margin: {sum(1 for d in all_dead_margins if d < 0)}/90")

    # =========================================================================
    # 5. Cross-cell test: take h_final from cell A, apply cell B's mask
    # =========================================================================
    section("5. Cross-cell test: apply wrong cell's activation mask")

    print(f"\nFor pairs in the dominant cell, zero out neurons that are dead")
    print(f"in OTHER cells. Does the prediction survive?\n")

    # Get all distinct cell patterns
    all_cell_patterns = list(set(p['ap_str'] for p in pairs))
    all_cell_patterns.sort()

    # For each pair in the dominant cell, try every other cell's mask
    n_survive = 0
    n_break = 0
    n_tests = 0

    break_examples = []

    for p in dominant_members[:5]:  # first 5 pairs for detail
        tgt = p['target']
        h = p['h_final']
        logits_orig = h @ W_out.T
        pred_orig = logits_orig.argmax()

        for other_ap in all_cell_patterns:
            if other_ap == p['ap_str']:
                continue

            # Apply other cell's mask to this pair's h_final
            mask = np.array([int(b) for b in other_ap])
            h_masked = h * mask
            logits_masked = h_masked @ W_out.T
            pred_masked = logits_masked.argmax()

            n_tests += 1
            if pred_masked == tgt:
                n_survive += 1
            else:
                n_break += 1
                if len(break_examples) < 5:
                    diff_neurons = [i for i in range(16) if mask[i] != p['ap'][i]]
                    break_examples.append((p['mt'], p['st'], tgt, pred_masked,
                                          [f'n{n}' for n in diff_neurons]))

    print(f"Tests: {n_tests}, survive: {n_survive} ({n_survive/n_tests*100:.1f}%), "
          f"break: {n_break} ({n_break/n_tests*100:.1f}%)")

    if break_examples:
        print(f"\nBreak examples:")
        for mt, st, tgt, pred, diffs in break_examples:
            print(f"  (M{mt},S{st}) tgt={tgt} → pred={pred}, changed neurons: {diffs}")

    # Now do it systematically for ALL pairs
    print(f"\n--- Systematic: all pairs × all other cells ---")
    n_survive_all = 0
    n_break_all = 0
    n_tests_all = 0

    for p in pairs:
        tgt = p['target']
        h = p['h_final']

        for other_ap in all_cell_patterns:
            if other_ap == p['ap_str']:
                continue
            mask = np.array([int(b) for b in other_ap])
            h_masked = h * mask
            logits_masked = h_masked @ W_out.T
            pred_masked = logits_masked.argmax()

            n_tests_all += 1
            if pred_masked == tgt:
                n_survive_all += 1
            else:
                n_break_all += 1

    print(f"Tests: {n_tests_all}, survive: {n_survive_all} ({n_survive_all/n_tests_all*100:.1f}%), "
          f"break: {n_break_all} ({n_break_all/n_tests_all*100:.1f}%)")

    # =========================================================================
    # 6. The affine map difference between cells: effective W_out
    # =========================================================================
    section("6. Effective W_out similarity across cells")

    print(f"\nFor each cell, the effective readout is W_out[:, active_neurons].")
    print(f"Compare cells by: what fraction of W_out columns are shared?\n")

    # Pairwise comparison of the top cells
    top_cells = cell_list[:8]
    print(f"{'cell_i':>7} {'cell_j':>7} {'shared':>7} {'only_i':>7} {'only_j':>7} "
          f"{'jaccard':>8}")
    print("-" * 50)

    for i in range(len(top_cells)):
        for j in range(i+1, len(top_cells)):
            ap_i = set(k for k, b in enumerate(top_cells[i][0]) if b == '1')
            ap_j = set(k for k, b in enumerate(top_cells[j][0]) if b == '1')
            shared = ap_i & ap_j
            only_i = ap_i - ap_j
            only_j = ap_j - ap_i
            jaccard = len(shared) / len(ap_i | ap_j)
            print(f"{i:>7} {j:>7} {len(shared):>7} {len(only_i):>7} {len(only_j):>7} "
                  f"{jaccard:>8.3f}")

    # =========================================================================
    # 7. Per-cell: what is the contribution of cell-specific neurons?
    # =========================================================================
    section("7. Logit contribution from cell-differentiating neurons")

    # The dominant cell's active set
    dominant_active = set(i for i, b in enumerate(dominant_ap) if b == '1')

    print(f"\nDominant cell active: {sorted(dominant_active)}")
    print(f"\nFor each other cell, the neurons that differ, and how much")
    print(f"those neurons contribute to logits for pairs IN that cell:\n")

    for idx, (ap_str, members) in enumerate(cell_list[:8]):
        active = set(i for i, b in enumerate(ap_str) if b == '1')
        diff = active.symmetric_difference(dominant_active)
        if not diff:
            continue

        # For pairs in this cell, contribution of the differing neurons
        diff_contribs = []
        total_contribs = []
        for p in members:
            tgt = p['target']
            h = p['h_final']
            diff_c = sum(h[n] * W_out[tgt, n] for n in diff if h[n] > 0)
            total_c = h @ W_out[tgt]
            diff_contribs.append(diff_c)
            total_contribs.append(total_c)

        only_here = active - dominant_active
        only_dominant = dominant_active - active

        print(f"  Cell {idx} ({len(members)} pairs): "
              f"+{[f'n{n}' for n in sorted(only_here)]}, "
              f"-{[f'n{n}' for n in sorted(only_dominant)]}")
        print(f"    Diff neuron contrib to target: {np.mean(diff_contribs):.2f} "
              f"/ total {np.mean(total_contribs):.2f} "
              f"({np.mean(diff_contribs)/np.mean(total_contribs)*100:.1f}%)")


if __name__ == "__main__":
    main()
