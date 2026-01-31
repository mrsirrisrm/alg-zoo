"""
Investigating the gap readout mechanism.

Forward: target = last_clip (gap ignored)
Reversed: target = last_clip - gap (gap subtracted)

Questions:
1. How does W_out suppress the gap for forward and include it for reversed?
2. How is the subtraction actually implemented in the linear readout?
3. What role does n9 (only fires for reversed) play in the gap shift?
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
    samples, labels, mps, sps = [], [], [], []
    for mp in range(10):
        for sp in range(10):
            if mp == sp:
                continue
            x = th.zeros(10)
            x[mp] = 1.0
            x[sp] = 0.8
            samples.append(x)
            labels.append(sp)
            mps.append(mp)
            sps.append(sp)
    return th.stack(samples), th.tensor(labels), np.array(mps), np.array(sps)


def run_model_hidden(W_ih, W_hh, X):
    h = th.zeros(X.shape[0], 16)
    for t in range(10):
        x_t = X[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
    return h.detach().numpy()


COMPS = [1, 6, 7, 8]
WAVES = [0, 10, 11, 12, 14]
BRIDGES = [3, 5, 13, 15]
N9 = [9]
N4 = [4]
N2 = [2]
GROUPS = {
    'comps': COMPS,
    'waves': WAVES,
    'bridges': BRIDGES,
    'n9': N9,
    'n4': N4,
    'n2': N2,
}


def section(title):
    print(f"\n{'='*80}")
    print(title)
    print('='*80)


def main():
    model = load_local_model()
    W_ih = model.rnn.weight_ih_l0.data.squeeze().clone()
    W_hh = model.rnn.weight_hh_l0.data.clone()
    W_out = model.linear.weight.data.clone().numpy()  # [10, 16]

    X, Y, mps, sps = make_clean_dataset()
    h_final = run_model_hidden(W_ih, W_hh, X)  # [90, 16]
    y_np = Y.numpy()

    fwd_mask = sps > mps
    rev_mask = sps < mps
    last_clips = np.maximum(mps, sps)
    gaps = np.abs(sps - mps)

    # =========================================================================
    # 1. Per-neuron logit contribution for target position
    #    logit[pos] = sum_n h_final[n] * W_out[pos, n]
    #    For each pair, look at contributions to the TARGET logit and to
    #    the LAST_CLIP logit (these differ only for reversed pairs)
    # =========================================================================
    section("1. Per-neuron-group contribution to target vs last_clip logit")

    # For each pair, compute contribution of each group to target logit and last_clip logit
    for label, mask in [("FORWARD", fwd_mask), ("REVERSED", rev_mask)]:
        print(f"\n--- {label} pairs ---")
        print(f"(For forward, target == last_clip, so these should be identical)")
        print(f"{'group':>10} {'→target':>10} {'→last_clip':>12} {'diff':>10}")
        print("-" * 45)

        for gname, gneurons in GROUPS.items():
            tgt_contribs = []
            lc_contribs = []
            for i in np.where(mask)[0]:
                tgt = y_np[i]
                lc = last_clips[i]
                h = h_final[i]
                # contribution of this group to target logit
                c_tgt = sum(h[n] * W_out[tgt, n] for n in gneurons)
                c_lc = sum(h[n] * W_out[lc, n] for n in gneurons)
                tgt_contribs.append(c_tgt)
                lc_contribs.append(c_lc)
            print(f"{gname:>10} {np.mean(tgt_contribs):>10.2f} {np.mean(lc_contribs):>12.2f} "
                  f"{np.mean(tgt_contribs) - np.mean(lc_contribs):>10.2f}")

    # =========================================================================
    # 2. For reversed pairs: decompose the "shift" from last_clip to target
    #    The readout needs to suppress last_clip and boost target = last_clip - gap
    #    For each group, compute: contrib[target] - contrib[last_clip]
    #    A positive value means this group helps shift toward the correct answer
    # =========================================================================
    section("2. Reversed: per-group shift (contrib[target] - contrib[last_clip]) by gap")

    print(f"\n{'gap':>4}", end="")
    for gname in GROUPS:
        print(f" {gname:>10}", end="")
    print(f" {'total':>10}")
    print("-" * 80)

    for g in range(1, 10):
        gmask = rev_mask & (gaps == g)
        if gmask.sum() == 0:
            continue
        shifts = {gname: [] for gname in GROUPS}
        total_shifts = []
        for i in np.where(gmask)[0]:
            tgt = y_np[i]
            lc = last_clips[i]
            h = h_final[i]
            total_shift = 0
            for gname, gneurons in GROUPS.items():
                s = sum(h[n] * (W_out[tgt, n] - W_out[lc, n]) for n in gneurons)
                shifts[gname].append(s)
                total_shift += s
            total_shifts.append(total_shift)
        print(f"{g:>4}", end="")
        for gname in GROUPS:
            print(f" {np.mean(shifts[gname]):>10.2f}", end="")
        print(f" {np.mean(total_shifts):>10.2f}")

    # =========================================================================
    # 3. n9's role: it only fires for reversed. What does its W_out column look like?
    #    And what happens to reversed predictions if we zero n9 at readout?
    # =========================================================================
    section("3. n9's W_out column and readout-only ablation")

    print("\nn9's W_out weights (W_out[:, 9]):")
    for pos in range(10):
        print(f"  pos {pos}: {W_out[pos, 9]:>8.4f}")

    # Readout-only ablation: zero specific neurons in h_final before W_out
    print("\nReadout-only ablation (zero neurons in h_final, keep dynamics intact):")
    print(f"{'zeroed':>15} {'fwd%':>6} {'rev%':>6} {'all%':>6}  rev_pred_dist")
    print("-" * 70)

    for label, zero_neurons in [
        ("none", []),
        ("n9", [9]),
        ("n4", [4]),
        ("n2", [2]),
        ("n9+n4", [9, 4]),
        ("n9+n4+n2", [9, 4, 2]),
        ("waves", WAVES),
        ("bridges", BRIDGES),
        ("comps", COMPS),
    ]:
        h_mod = h_final.copy()
        for n in zero_neurons:
            h_mod[:, n] = 0.0
        logits = h_mod @ W_out.T
        preds = logits.argmax(axis=1)

        fwd_acc = (preds[fwd_mask] == y_np[fwd_mask]).mean() * 100
        rev_acc = (preds[rev_mask] == y_np[rev_mask]).mean() * 100
        all_acc = (preds == y_np).mean() * 100

        # For reversed failures, what do they predict?
        rev_wrong = rev_mask & (preds != y_np)
        pred_lc = 0
        pred_other = 0
        for i in np.where(rev_wrong)[0]:
            if preds[i] == last_clips[i]:
                pred_lc += 1
            else:
                pred_other += 1

        rev_info = ""
        if rev_wrong.sum() > 0:
            rev_info = f"pred_last_clip={pred_lc}, pred_other={pred_other}"

        print(f"{label:>15} {fwd_acc:>5.1f}% {rev_acc:>5.1f}% {all_acc:>5.1f}%  {rev_info}")

    # =========================================================================
    # 4. The key question: for reversed pairs, what IS different in h_final
    #    compared to the forward pair with the same last_clip?
    #    e.g., forward (Mt=2, St=7) and reversed (Mt=7, St=2) both have last_clip=7
    #    but target=7 vs target=2. What changes in h_final?
    # =========================================================================
    section("4. h_final difference: forward vs reversed at same (last_clip, gap)")

    print("\nFor each (last_clip, gap) where both forward and reversed exist,")
    print("show the per-neuron difference (rev - fwd) in h_final.\n")

    # Build lookup: (mp, sp) -> index
    pair_idx = {}
    for i in range(len(mps)):
        pair_idx[(mps[i], sps[i])] = i

    # For a few example gaps
    for gap in [1, 2, 3, 5]:
        print(f"\n--- gap={gap} ---")
        print(f"{'lc':>3} {'fwd_tgt':>8} {'rev_tgt':>8}  ", end="")
        for gname in ['comps', 'waves', 'bridges', 'n9', 'n4', 'n2']:
            print(f" {gname:>8}", end="")
        print(f" {'logit_shift':>11}")
        print("-" * 100)

        for lc in range(gap, 10):
            # forward: Mt = lc - gap, St = lc (St > Mt)
            mt_fwd = lc - gap
            st_fwd = lc
            # reversed: Mt = lc, St = lc - gap (St < Mt)
            mt_rev = lc
            st_rev = lc - gap

            if (mt_fwd, st_fwd) not in pair_idx or (mt_rev, st_rev) not in pair_idx:
                continue

            i_fwd = pair_idx[(mt_fwd, st_fwd)]
            i_rev = pair_idx[(mt_rev, st_rev)]

            h_diff = h_final[i_rev] - h_final[i_fwd]

            # Show per-group total difference
            print(f"{lc:>3} {st_fwd:>8} {st_rev:>8}  ", end="")
            for gname in ['comps', 'waves', 'bridges', 'n9', 'n4', 'n2']:
                gneurons = GROUPS[gname]
                d = sum(abs(h_diff[n]) for n in gneurons)
                print(f" {d:>8.2f}", end="")

            # The logit shift: what does this h_diff contribute via W_out?
            # For correct readout, h_diff @ W_out.T should boost target_rev and suppress target_fwd
            logit_diff = h_diff @ W_out.T
            # target_fwd = lc, target_rev = lc - gap
            shift = logit_diff[st_rev] - logit_diff[st_fwd]
            print(f" {shift:>11.2f}")

    # =========================================================================
    # 5. Per-neuron contribution to the "gap shift" for reversed
    #    For each neuron n: h_rev[n] * W_out[target_rev, n] - h_fwd[n] * W_out[target_fwd, n]
    #    This shows which neurons are responsible for the readout shifting
    #    from last_clip to last_clip - gap
    # =========================================================================
    section("5. Per-neuron contribution to gap shift (averaged over all reversed pairs)")

    # For each reversed pair, the "shift" each neuron provides is:
    #   h[n] * W_out[target, n] - h[n] * W_out[last_clip, n]
    #   = h[n] * (W_out[target, n] - W_out[last_clip, n])
    # This uses the SAME h_final — the shift comes from W_out reading
    # different rows for target vs last_clip

    neuron_shift = np.zeros(16)
    neuron_shift_abs = np.zeros(16)
    n_rev = rev_mask.sum()

    for i in np.where(rev_mask)[0]:
        tgt = y_np[i]
        lc = last_clips[i]
        for n in range(16):
            s = h_final[i, n] * (W_out[tgt, n] - W_out[lc, n])
            neuron_shift[n] += s
            neuron_shift_abs[n] += abs(s)

    neuron_shift /= n_rev
    neuron_shift_abs /= n_rev

    print(f"\n{'neuron':>7} {'group':>8} {'mean_shift':>11} {'mean_|shift|':>13} {'h_rev_mean':>11} {'W_out_range':>12}")
    print("-" * 70)

    # Also compute mean h_final for reversed pairs
    h_rev_mean = h_final[rev_mask].mean(axis=0)

    for n in range(16):
        # Find group
        grp = "?"
        for gname, gneurons in GROUPS.items():
            if n in gneurons:
                grp = gname
                break
        w_range = W_out[:, n].max() - W_out[:, n].min()
        print(f"  n{n:<5} {grp:>8} {neuron_shift[n]:>11.3f} {neuron_shift_abs[n]:>13.3f} "
              f"{h_rev_mean[n]:>11.3f} {w_range:>12.3f}")

    # =========================================================================
    # 6. W_out structure: does it have a "position gradient" that enables subtraction?
    #    If W_out[pos, n] varies linearly with pos for some neuron n, then
    #    h[n] * W_out[target, n] naturally creates a position-proportional signal
    # =========================================================================
    section("6. W_out position structure per neuron")

    print("\nFor each neuron, fit W_out[:, n] = a*pos + b and report the slope.")
    print("A strong linear slope means the neuron's contribution scales with position.\n")

    positions = np.arange(10)
    print(f"{'neuron':>7} {'group':>8} {'slope':>8} {'R²':>6} {'W_out values (pos 0-9)':>50}")
    print("-" * 90)

    for n in range(16):
        grp = "?"
        for gname, gneurons in GROUPS.items():
            if n in gneurons:
                grp = gname
                break
        w = W_out[:, n]
        # Linear fit
        slope, intercept = np.polyfit(positions, w, 1)
        pred = slope * positions + intercept
        ss_res = ((w - pred) ** 2).sum()
        ss_tot = ((w - w.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        vals = " ".join(f"{v:>5.2f}" for v in w)
        print(f"  n{n:<5} {grp:>8} {slope:>8.3f} {r2:>6.3f}  {vals}")

    # =========================================================================
    # 7. The critical test: can we decompose W_out into a "last_clip reader"
    #    and a "gap shifter"? Look at it from the W_out perspective.
    #    For reversed pair with target = lc - gap:
    #    logit[lc - gap] needs to be max.
    #    What row of W_out does this correspond to?
    # =========================================================================
    section("7. W_out row similarity: do nearby positions have similar readout vectors?")

    print("\nCosine similarity between W_out rows (readout vectors for each position):")
    print(f"{'':>4}", end="")
    for j in range(10):
        print(f" {j:>5}", end="")
    print()

    for i in range(10):
        print(f"{i:>3} ", end="")
        for j in range(10):
            dot = np.dot(W_out[i], W_out[j])
            norm_i = np.linalg.norm(W_out[i])
            norm_j = np.linalg.norm(W_out[j])
            cos = dot / (norm_i * norm_j) if norm_i > 0 and norm_j > 0 else 0
            print(f" {cos:>5.2f}", end="")
        print()

    # =========================================================================
    # 8. Direct test: for reversed pairs, manually shift h_final by the
    #    difference between a forward and reversed pair at same (lc, gap).
    #    Does the "ordering signal" in h_final act as a gate on the gap?
    # =========================================================================
    section("8. What reversed h_final neurons distinguish from forward (same lc, gap)?")

    # Average h_final for forward and reversed, grouped by gap
    print("\nMean h_final for n9, n4, n2 by gap:")
    print(f"{'gap':>4} {'n9_fwd':>8} {'n9_rev':>8} {'n4_fwd':>8} {'n4_rev':>8} {'n2_fwd':>8} {'n2_rev':>8}")
    print("-" * 55)

    for g in range(1, 10):
        fg = fwd_mask & (gaps == g)
        rg = rev_mask & (gaps == g)
        if fg.sum() == 0 or rg.sum() == 0:
            continue
        h_f = h_final[fg].mean(axis=0)
        h_r = h_final[rg].mean(axis=0)
        print(f"{g:>4} {h_f[9]:>8.3f} {h_r[9]:>8.3f} {h_f[4]:>8.3f} {h_r[4]:>8.3f} "
              f"{h_f[2]:>8.3f} {h_r[2]:>8.3f}")

    # =========================================================================
    # 9. The subtraction mechanism: for reversed pairs, decompose the winning
    #    logit into "base" (what you'd get from last_clip alone) and "shift"
    #    (what moves it to target). Track how each neuron group contributes.
    # =========================================================================
    section("9. Reversed logit decomposition: base (last_clip) + shift → target")

    print("\nFor each reversed pair, logit[target] = sum_n h[n] * W_out[target, n]")
    print("Split into: h[n] * W_out[lc, n] (base) + h[n] * (W_out[target, n] - W_out[lc, n]) (shift)")
    print("\nAveraged by gap:\n")

    print(f"{'gap':>4} {'tgt_logit':>10} ", end="")
    for gname in ['comps', 'waves', 'bridges', 'n9', 'n4', 'n2']:
        print(f" {gname+'_b':>8} {gname+'_s':>8}", end="")
    print()
    print("-" * 130)

    for g in range(1, 10):
        gmask = rev_mask & (gaps == g)
        if gmask.sum() == 0:
            continue
        tgt_logits = []
        bases = {gn: [] for gn in GROUPS}
        shifts = {gn: [] for gn in GROUPS}

        for i in np.where(gmask)[0]:
            tgt = y_np[i]
            lc = last_clips[i]
            h = h_final[i]

            tgt_logit = h @ W_out[tgt]
            tgt_logits.append(tgt_logit)

            for gname, gneurons in GROUPS.items():
                base = sum(h[n] * W_out[lc, n] for n in gneurons)
                shift = sum(h[n] * (W_out[tgt, n] - W_out[lc, n]) for n in gneurons)
                bases[gname].append(base)
                shifts[gname].append(shift)

        print(f"{g:>4} {np.mean(tgt_logits):>10.2f} ", end="")
        for gname in ['comps', 'waves', 'bridges', 'n9', 'n4', 'n2']:
            print(f" {np.mean(bases[gname]):>8.2f} {np.mean(shifts[gname]):>8.2f}", end="")
        print()

    # =========================================================================
    # 10. Same decomposition for forward pairs to confirm gap-shift is ~0
    # =========================================================================
    section("10. Forward logit decomposition: base + shift (should be ~0 shift)")

    print(f"\n{'gap':>4} {'tgt_logit':>10} ", end="")
    for gname in ['comps', 'waves', 'bridges', 'n9', 'n4', 'n2']:
        print(f" {gname+'_b':>8} {gname+'_s':>8}", end="")
    print()
    print("-" * 130)

    for g in range(1, 10):
        gmask = fwd_mask & (gaps == g)
        if gmask.sum() == 0:
            continue
        tgt_logits = []
        bases = {gn: [] for gn in GROUPS}
        shifts = {gn: [] for gn in GROUPS}

        for i in np.where(gmask)[0]:
            tgt = y_np[i]
            lc = last_clips[i]
            h = h_final[i]
            # For forward, target == last_clip, so shift should be 0
            tgt_logit = h @ W_out[tgt]
            tgt_logits.append(tgt_logit)

            for gname, gneurons in GROUPS.items():
                base = sum(h[n] * W_out[lc, n] for n in gneurons)
                shift = sum(h[n] * (W_out[tgt, n] - W_out[lc, n]) for n in gneurons)
                bases[gname].append(base)
                shifts[gname].append(shift)

        print(f"{g:>4} {np.mean(tgt_logits):>10.2f} ", end="")
        for gname in ['comps', 'waves', 'bridges', 'n9', 'n4', 'n2']:
            print(f" {np.mean(bases[gname]):>8.2f} {np.mean(shifts[gname]):>8.2f}", end="")
        print()


if __name__ == "__main__":
    main()
