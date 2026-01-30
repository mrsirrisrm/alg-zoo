"""
Round 2 Claims 9-15.
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


def make_clean_dataset():
    samples, labels, max_positions, sec_positions = [], [], [], []
    for max_pos in range(10):
        for sec_pos in range(10):
            if max_pos == sec_pos:
                continue
            x = th.zeros(10)
            x[max_pos] = 1.0
            x[sec_pos] = 0.8
            samples.append(x)
            labels.append(sec_pos)
            max_positions.append(max_pos)
            sec_positions.append(sec_pos)
    return th.stack(samples), th.tensor(labels), max_positions, sec_positions


def run_model_hidden(W_ih, W_hh, X):
    """Return h_final for all samples."""
    batch_size = X.shape[0]
    h = th.zeros(batch_size, 16)
    for t in range(10):
        x_t = X[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
    return h.detach().numpy()


def run_model(W_ih, W_hh, W_out, X):
    h = th.zeros(X.shape[0], 16)
    for t in range(10):
        x_t = X[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
    return h @ W_out.T


def _acc(preds, y, mask=None):
    if mask is not None:
        if mask.sum() == 0:
            return float('nan')
        return (preds[mask] == y[mask]).float().mean().item() * 100
    return (preds == y).float().mean().item() * 100


def section(title):
    print(f"\n{'='*80}")
    print(title)
    print('='*80)


COMPS = [1, 6, 7, 8]
WAVES = [0, 10, 11, 12, 14]
BRIDGES = [3, 5, 13, 15]


# =============================================================================
# CLAIM 9: n7 is a pure M-position encoder
# =============================================================================
def test_claim_9(W_ih, W_hh, X, max_pos_list, sec_pos_list):
    section("CLAIM 9: n7 as pure M-position encoder")
    print("Does n7's h_final depend only on Mt, not St?\n")

    h_final = run_model_hidden(W_ih, W_hh, X)

    # Group n7 values by Mt
    by_mt = defaultdict(list)
    by_st = defaultdict(list)
    for i, (mp, sp) in enumerate(zip(max_pos_list, sec_pos_list)):
        by_mt[mp].append(h_final[i, 7])
        by_st[sp].append(h_final[i, 7])

    print("n7 h_final grouped by Mt:")
    print(f"{'Mt':>4} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'range':>8}")
    print("-" * 50)
    for mp in range(10):
        vals = by_mt[mp]
        print(f"{mp:>4} {np.mean(vals):>8.2f} {np.std(vals):>8.3f} "
              f"{np.min(vals):>8.2f} {np.max(vals):>8.2f} {np.max(vals)-np.min(vals):>8.3f}")

    print("\nn7 h_final grouped by St:")
    print(f"{'St':>4} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'range':>8}")
    print("-" * 50)
    for sp in range(10):
        vals = by_st[sp]
        print(f"{sp:>4} {np.mean(vals):>8.2f} {np.std(vals):>8.3f} "
              f"{np.min(vals):>8.2f} {np.max(vals):>8.2f} {np.max(vals)-np.min(vals):>8.3f}")

    # R² for Mt-only model
    all_n7 = np.array([h_final[i, 7] for i in range(len(max_pos_list))])
    mt_means = np.array([np.mean(by_mt[mp]) for mp in max_pos_list])
    ss_res = np.sum((all_n7 - mt_means)**2)
    ss_tot = np.sum((all_n7 - all_n7.mean())**2)
    r2_mt = 1 - ss_res / ss_tot
    print(f"\nR²(Mt only) = {r2_mt:.4f}")

    # Compare all comps
    print("\nR²(Mt only) for all comps:")
    for c in COMPS:
        all_c = np.array([h_final[i, c] for i in range(len(max_pos_list))])
        by_mt_c = defaultdict(list)
        for i, mp in enumerate(max_pos_list):
            by_mt_c[mp].append(h_final[i, c])
        mt_means_c = np.array([np.mean(by_mt_c[mp]) for mp in max_pos_list])
        ss_res = np.sum((all_c - mt_means_c)**2)
        ss_tot = np.sum((all_c - all_c.mean())**2)
        r2 = 1 - ss_res / ss_tot
        print(f"  n{c}: R²(Mt) = {r2:.4f}")


# =============================================================================
# CLAIM 10: n2 encodes first-impulse timing
# =============================================================================
def test_claim_10(W_ih, W_hh, X, max_pos_list, sec_pos_list):
    section("CLAIM 10: n2 encodes first-impulse timing")
    print("n2's h_final should correlate with first-impulse position.\n")

    h_final = run_model_hidden(W_ih, W_hh, X)

    first_pos = []
    for mp, sp in zip(max_pos_list, sec_pos_list):
        first_pos.append(min(mp, sp))

    all_n2 = np.array([h_final[i, 2] for i in range(len(max_pos_list))])
    first_arr = np.array(first_pos)

    # Group by first impulse position
    by_first = defaultdict(list)
    for i, fp in enumerate(first_pos):
        by_first[fp].append(h_final[i, 2])

    print("n2 h_final grouped by first-impulse position:")
    print(f"{'first':>6} {'mean':>8} {'std':>8} {'n':>4}")
    print("-" * 30)
    for fp in range(10):
        if by_first[fp]:
            vals = by_first[fp]
            print(f"{fp:>6} {np.mean(vals):>8.2f} {np.std(vals):>8.3f} {len(vals):>4}")

    # R² with first position
    first_means = np.array([np.mean(by_first[fp]) for fp in first_pos])
    ss_res = np.sum((all_n2 - first_means)**2)
    ss_tot = np.sum((all_n2 - all_n2.mean())**2)
    r2_first = 1 - ss_res / ss_tot
    print(f"\nR²(first_impulse_pos) = {r2_first:.4f}")

    # Also check: is it monotonic?
    means_by_pos = [np.mean(by_first[fp]) for fp in range(10) if by_first[fp]]
    diffs = [means_by_pos[i+1] - means_by_pos[i] for i in range(len(means_by_pos)-1)]
    all_decreasing = all(d < 0 for d in diffs)
    all_increasing = all(d > 0 for d in diffs)
    print(f"Monotonic? {'decreasing' if all_decreasing else 'increasing' if all_increasing else 'NO'}")
    print(f"Consecutive diffs: {['%.2f' % d for d in diffs]}")

    # Separate fwd/rev
    print("\nBy ordering:")
    for label, cond in [("Forward (Mt first)", lambda mp, sp: mp < sp),
                         ("Reversed (St first)", lambda mp, sp: sp < mp)]:
        vals = [(first_pos[i], all_n2[i]) for i, (mp, sp) in enumerate(zip(max_pos_list, sec_pos_list)) if cond(mp, sp)]
        if vals:
            fps, n2s = zip(*vals)
            corr = np.corrcoef(fps, n2s)[0, 1]
            print(f"  {label}: correlation(first_pos, n2) = {corr:.4f}")


# =============================================================================
# CLAIM 11: (last_clip, gap, ordering) lookup predicts accuracy
# =============================================================================
def test_claim_11(W_ih, W_hh, W_out, X, y, max_pos_list, sec_pos_list):
    section("CLAIM 11: (last_clip, gap, ordering) lookup table")
    print("Build lookup from actual model outputs. Check if every cell is correct.\n")

    logits = run_model(W_ih, W_hh, W_out, X)
    preds = logits.argmax(dim=-1).numpy()
    y_np = y.numpy()

    # Build lookup: (last_clip, gap, ordering) → prediction
    lookup = {}
    for i, (mp, sp) in enumerate(zip(max_pos_list, sec_pos_list)):
        last_clip = max(mp, sp)
        gap = abs(sp - mp)
        ordering = "fwd" if sp > mp else "rev"
        key = (last_clip, gap, ordering)

        if key not in lookup:
            lookup[key] = {'pred': preds[i], 'target': y_np[i], 'pairs': []}
        lookup[key]['pairs'].append((mp, sp, y_np[i], preds[i]))

    # Check consistency — all pairs in same cell should get same prediction
    inconsistent = 0
    for key, val in sorted(lookup.items()):
        preds_in_cell = set(p[3] for p in val['pairs'])
        if len(preds_in_cell) > 1:
            inconsistent += 1
            print(f"  INCONSISTENT: {key} → preds {preds_in_cell}")

    print(f"Total cells: {len(lookup)}")
    print(f"Inconsistent cells: {inconsistent}")

    # Check correctness
    correct_cells = sum(1 for v in lookup.values() if all(p[2] == p[3] for p in v['pairs']))
    total_cells = len(lookup)
    print(f"Correct cells: {correct_cells}/{total_cells}")

    # Show the lookup table
    print(f"\nLookup table (last_clip, gap, ordering → target, pred, correct?):")
    print(f"{'lc':>3} {'gap':>4} {'ord':>4} {'tgt':>4} {'pred':>5} {'ok':>3}")
    print("-" * 28)
    for key in sorted(lookup.keys()):
        val = lookup[key]
        # All pairs in cell have same target? (should, since target = St)
        targets = set(p[2] for p in val['pairs'])
        preds_set = set(p[3] for p in val['pairs'])
        tgt_str = str(targets.pop()) if len(targets) == 1 else str(targets)
        pred_str = str(preds_set.pop()) if len(preds_set) == 1 else str(preds_set)
        ok = "Y" if all(p[2] == p[3] for p in val['pairs']) else "N"
        print(f"{key[0]:>3} {key[1]:>4} {key[2]:>4} {tgt_str:>4} {pred_str:>5} {ok:>3}")


# =============================================================================
# CLAIM 12: n9 compensates for insufficient cascade time
# =============================================================================
def test_claim_12(W_ih, W_hh, X, max_pos_list, sec_pos_list):
    section("CLAIM 12: n9 activity vs gap in reversed pairs")
    print("n9 should be more active for small-gap reversed pairs.\n")

    h_final = run_model_hidden(W_ih, W_hh, X)

    # Group n9 h_final by (gap, ordering)
    by_gap_ord = defaultdict(list)
    for i, (mp, sp) in enumerate(zip(max_pos_list, sec_pos_list)):
        gap = abs(sp - mp)
        ordering = "fwd" if sp > mp else "rev"
        by_gap_ord[(gap, ordering)].append(h_final[i, 9])

    print(f"{'gap':>4} {'fwd_mean':>10} {'rev_mean':>10} {'fwd_n':>6} {'rev_n':>6}")
    print("-" * 40)
    for g in range(1, 10):
        fwd = by_gap_ord.get((g, 'fwd'), [])
        rev = by_gap_ord.get((g, 'rev'), [])
        fwd_mean = np.mean(fwd) if fwd else float('nan')
        rev_mean = np.mean(rev) if rev else float('nan')
        print(f"{g:>4} {fwd_mean:>10.3f} {rev_mean:>10.3f} {len(fwd):>6} {len(rev):>6}")

    # Correlation: gap vs n9 in reversed pairs
    rev_gaps, rev_n9 = [], []
    for i, (mp, sp) in enumerate(zip(max_pos_list, sec_pos_list)):
        if sp < mp:  # reversed
            rev_gaps.append(abs(sp - mp))
            rev_n9.append(h_final[i, 9])

    if rev_gaps:
        corr = np.corrcoef(rev_gaps, rev_n9)[0, 1]
        print(f"\nCorrelation(gap, n9) in reversed pairs: {corr:.4f}")


# =============================================================================
# CLAIM 13: n4 second firing is approximately constant (gain control)
# =============================================================================
def test_claim_13(W_ih, W_hh, X, max_pos_list, sec_pos_list):
    section("CLAIM 13: n4 second-impulse firing — gain control check")
    print("n4 at second impulse should be ~constant across all pairs.\n")

    second_fires = []
    n2_at_second = []
    for i, (mp, sp) in enumerate(zip(max_pos_list, sec_pos_list)):
        x = X[i:i+1]
        second_pos = max(mp, sp)  # second in time

        h = th.zeros(1, 16)
        for t in range(10):
            x_t = x[:, t:t+1]
            pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
            h = th.relu(pre)
            if t == second_pos:
                second_fires.append(h[0, 4].item())
                # n2 value just before this step
                # Actually h is already updated, let's get pre-step n2
        # Redo to get n2 before second impulse
        h = th.zeros(1, 16)
        for t in range(10):
            if t == second_pos:
                n2_at_second.append(h[0, 2].item())
            x_t = x[:, t:t+1]
            pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
            h = th.relu(pre)

    fires = np.array(second_fires)
    n2s = np.array(n2_at_second)

    print(f"n4 at second impulse:")
    print(f"  mean={fires.mean():.2f}, std={fires.std():.2f}, "
          f"min={fires.min():.2f}, max={fires.max():.2f}")
    print(f"  CV (std/mean) = {fires.std()/fires.mean():.3f}")

    print(f"\nn2 just before second impulse:")
    print(f"  mean={n2s.mean():.2f}, std={n2s.std():.2f}, "
          f"min={n2s.min():.2f}, max={n2s.max():.2f}")

    # By ordering
    fwd_fires, rev_fires = [], []
    for i, (mp, sp) in enumerate(zip(max_pos_list, sec_pos_list)):
        if sp > mp:
            fwd_fires.append(second_fires[i])
        else:
            rev_fires.append(second_fires[i])

    fwd_arr = np.array(fwd_fires)
    rev_arr = np.array(rev_fires)
    print(f"\nBy ordering:")
    print(f"  Forward:  mean={fwd_arr.mean():.2f}, std={fwd_arr.std():.2f}, range=[{fwd_arr.min():.2f}, {fwd_arr.max():.2f}]")
    print(f"  Reversed: mean={rev_arr.mean():.2f}, std={rev_arr.std():.2f}, range=[{rev_arr.min():.2f}, {rev_arr.max():.2f}]")

    # Expected: n4_fire ≈ 10.16*mag + n2*(-0.49) + other recurrent
    # Check if n4_fire + n2*0.49 is more constant
    compensated = fires + n2s * 0.49
    print(f"\nn4 + n2*0.49 (should be ~constant if pure gain control):")
    print(f"  mean={compensated.mean():.2f}, std={compensated.std():.2f}, "
          f"CV={compensated.std()/compensated.mean():.3f}")


# =============================================================================
# CLAIM 14: Comp→wave direct edges damage specific positions
# =============================================================================
def test_claim_14(model):
    section("CLAIM 14: Comp→wave damage by position")
    print("Zeroing direct comp→wave edges — which positions are hurt?\n")

    W_ih = model.rnn.weight_ih_l0.data.squeeze().clone()
    W_hh_orig = model.rnn.weight_hh_l0.data.clone()
    W_out = model.linear.weight.data.clone()
    X, y, max_pos_list, sec_pos_list = make_clean_dataset()
    sec_arr = np.array(sec_pos_list)

    # Zero all direct comp→wave
    W_hh = W_hh_orig.clone()
    for w in WAVES:
        for c in COMPS:
            W_hh[w, c] = 0.0
            W_hh[c, w] = 0.0

    logits = run_model(W_ih, W_hh, W_out, X)
    preds = logits.argmax(dim=-1).numpy()
    correct = (preds == y.numpy())

    print(f"By St (target) position:")
    print(f"{'St':>4} {'acc':>6}")
    print("-" * 12)
    for sp in range(10):
        mask = sec_arr == sp
        if mask.sum() > 0:
            print(f"{sp:>4} {correct[mask].mean()*100:>5.1f}%")

    # Also by Mt
    max_arr = np.array(max_pos_list)
    print(f"\nBy Mt position:")
    print(f"{'Mt':>4} {'acc':>6}")
    print("-" * 12)
    for mp in range(10):
        mask = max_arr == mp
        if mask.sum() > 0:
            print(f"{mp:>4} {correct[mask].mean()*100:>5.1f}%")


# =============================================================================
# CLAIM 15: W_out uses comps and waves as two channels
# =============================================================================
def test_claim_15(model):
    section("CLAIM 15: W_out comp vs wave channel ablation")
    print("Zero comp or wave columns in W_out — different damage patterns?\n")

    W_ih = model.rnn.weight_ih_l0.data.squeeze().clone()
    W_hh = model.rnn.weight_hh_l0.data.clone()
    W_out_orig = model.linear.weight.data.clone()
    X, y, max_pos_list, sec_pos_list = make_clean_dataset()

    forward_mask = th.tensor([s > m for m, s in zip(max_pos_list, sec_pos_list)])
    reversed_mask = ~forward_mask

    max_arr = np.array(max_pos_list)
    sec_arr = np.array(sec_pos_list)
    gaps = np.abs(sec_arr - max_arr)

    # Baseline
    logits = run_model(W_ih, W_hh, W_out_orig, X)
    preds = logits.argmax(dim=-1)
    print(f"Baseline: fwd={_acc(preds, y, forward_mask):.1f}%, "
          f"rev={_acc(preds, y, reversed_mask):.1f}%, all={_acc(preds, y):.1f}%")

    # Zero comp columns in W_out
    W_out = W_out_orig.clone()
    for c in COMPS:
        W_out[:, c] = 0.0
    logits = run_model(W_ih, W_hh, W_out, X)
    preds = logits.argmax(dim=-1)
    print(f"No comps in readout: fwd={_acc(preds, y, forward_mask):.1f}%, "
          f"rev={_acc(preds, y, reversed_mask):.1f}%, all={_acc(preds, y):.1f}%")

    # Zero wave columns in W_out
    W_out = W_out_orig.clone()
    for w in WAVES:
        W_out[:, w] = 0.0
    logits = run_model(W_ih, W_hh, W_out, X)
    preds = logits.argmax(dim=-1)
    print(f"No waves in readout: fwd={_acc(preds, y, forward_mask):.1f}%, "
          f"rev={_acc(preds, y, reversed_mask):.1f}%, all={_acc(preds, y):.1f}%")

    # Zero both
    W_out = W_out_orig.clone()
    for c in COMPS:
        W_out[:, c] = 0.0
    for w in WAVES:
        W_out[:, w] = 0.0
    logits = run_model(W_ih, W_hh, W_out, X)
    preds = logits.argmax(dim=-1)
    print(f"No comps+waves: fwd={_acc(preds, y, forward_mask):.1f}%, "
          f"rev={_acc(preds, y, reversed_mask):.1f}%, all={_acc(preds, y):.1f}%")

    # Gap breakdown for comp vs wave ablation
    print("\nGap breakdown:")
    print(f"{'gap':>4} {'no_comp_fwd':>12} {'no_comp_rev':>12} {'no_wave_fwd':>12} {'no_wave_rev':>12}")
    print("-" * 56)

    # No comps
    W_out = W_out_orig.clone()
    for c in COMPS:
        W_out[:, c] = 0.0
    logits_nc = run_model(W_ih, W_hh, W_out, X)
    preds_nc = logits_nc.argmax(dim=-1).numpy()
    correct_nc = (preds_nc == y.numpy())

    # No waves
    W_out = W_out_orig.clone()
    for w in WAVES:
        W_out[:, w] = 0.0
    logits_nw = run_model(W_ih, W_hh, W_out, X)
    preds_nw = logits_nw.argmax(dim=-1).numpy()
    correct_nw = (preds_nw == y.numpy())

    is_fwd = sec_arr > max_arr
    for g in range(1, 10):
        fwd_mask_g = (gaps == g) & is_fwd
        rev_mask_g = (gaps == g) & ~is_fwd

        nc_fwd = correct_nc[fwd_mask_g].mean()*100 if fwd_mask_g.sum() > 0 else float('nan')
        nc_rev = correct_nc[rev_mask_g].mean()*100 if rev_mask_g.sum() > 0 else float('nan')
        nw_fwd = correct_nw[fwd_mask_g].mean()*100 if fwd_mask_g.sum() > 0 else float('nan')
        nw_rev = correct_nw[rev_mask_g].mean()*100 if rev_mask_g.sum() > 0 else float('nan')

        print(f"{g:>4} {nc_fwd:>11.0f}% {nc_rev:>11.0f}% {nw_fwd:>11.0f}% {nw_rev:>11.0f}%")


def main():
    model = load_local_model()
    W_ih = model.rnn.weight_ih_l0.data.squeeze().clone()
    W_hh = model.rnn.weight_hh_l0.data.clone()
    W_out = model.linear.weight.data.clone()
    X, y, max_pos_list, sec_pos_list = make_clean_dataset()

    test_claim_9(W_ih, W_hh, X, max_pos_list, sec_pos_list)
    test_claim_10(W_ih, W_hh, X, max_pos_list, sec_pos_list)
    test_claim_11(W_ih, W_hh, W_out, X, y, max_pos_list, sec_pos_list)
    test_claim_12(W_ih, W_hh, X, max_pos_list, sec_pos_list)
    test_claim_13(W_ih, W_hh, X, max_pos_list, sec_pos_list)
    test_claim_14(model)
    test_claim_15(model)


if __name__ == "__main__":
    main()
