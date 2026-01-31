"""
Follow-up investigation of the gap readout mechanism.

Key hypothesis from part 1: n2 acts as a "last_clip anchor" that opposes the gap
shift for reversed pairs. This should mean:
1. Reducing n2 at readout should HELP reversed (remove the opposing force)
2. The margin for reversed pairs should correlate with how much n2 opposes

Also: how does the comp rebuild trajectory actually differ between fwd and rev
to produce the gap shift?
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


def run_trace(W_ih, W_hh, x_single):
    h = th.zeros(1, 16)
    trace = []
    for t in range(10):
        x_t = x_single[t:t+1].unsqueeze(0)
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
        trace.append(h[0].detach().numpy().copy())
    return np.array(trace)


COMPS = [1, 6, 7, 8]
WAVES = [0, 10, 11, 12, 14]
BRIDGES = [3, 5, 13, 15]


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
    h_final = run_model_hidden(W_ih, W_hh, X)
    y_np = Y.numpy()

    fwd_mask = sps > mps
    rev_mask = sps < mps
    last_clips = np.maximum(mps, sps)
    gaps = np.abs(sps - mps)

    # =========================================================================
    # 1. Scale n2 at readout: does reducing n2 help reversed?
    # =========================================================================
    section("1. Scale n2 at readout only — does reducing n2 help reversed?")

    print(f"{'n2_scale':>9} {'fwd%':>6} {'rev%':>6} {'all%':>6}  {'fwd_margin':>11} {'rev_margin':>11}")
    print("-" * 60)

    for scale in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        h_mod = h_final.copy()
        h_mod[:, 2] *= scale
        logits = h_mod @ W_out.T
        preds = logits.argmax(axis=1)

        fwd_acc = (preds[fwd_mask] == y_np[fwd_mask]).mean() * 100
        rev_acc = (preds[rev_mask] == y_np[rev_mask]).mean() * 100
        all_acc = (preds == y_np).mean() * 100

        # Margins
        fwd_margins = []
        rev_margins = []
        for i in range(len(y_np)):
            tgt = y_np[i]
            l = logits[i]
            margin = l[tgt] - max(l[j] for j in range(10) if j != tgt)
            if fwd_mask[i]:
                fwd_margins.append(margin)
            else:
                rev_margins.append(margin)

        print(f"{scale:>9.2f} {fwd_acc:>5.1f}% {rev_acc:>5.1f}% {all_acc:>5.1f}%  "
              f"{np.mean(fwd_margins):>11.2f} {np.mean(rev_margins):>11.2f}")

    # =========================================================================
    # 2. For a specific pair, trace the comp rebuild trajectory step by step
    #    Compare forward (M2, S7) vs reversed (M7, S2) — same lc=7, gap=5
    # =========================================================================
    section("2. Comp rebuild trajectory: forward (M2,S7) vs reversed (M7,S2)")

    for label, mt, st in [("Forward M2,S7", 2, 7), ("Reversed M7,S2", 7, 2)]:
        x = th.zeros(10)
        x[mt] = 1.0
        x[st] = 0.8
        trace = run_trace(W_ih, W_hh, x)

        tgt = st
        lc = max(mt, st)
        logits = trace[9] @ W_out.T

        print(f"\n{label} → target={tgt}, last_clip={lc}, pred={logits.argmax()}")
        print(f"{'t':>3}", end="")
        for n in COMPS:
            print(f" {'n'+str(n):>8}", end="")
        print(f" {'n2':>8} {'n4':>8} {'n9':>8}")
        print("-" * 75)
        for t in range(10):
            print(f"{t:>3}", end="")
            for n in COMPS:
                print(f" {trace[t, n]:>8.3f}", end="")
            print(f" {trace[t, 2]:>8.3f} {trace[t, 4]:>8.3f} {trace[t, 9]:>8.3f}")

        # Show the logit decomposition for this specific pair
        print(f"\nLogit decomposition for target={tgt}:")
        h = trace[9]
        for gname, gneurons in [('comps', COMPS), ('waves', WAVES), ('bridges', BRIDGES),
                                  ('n9', [9]), ('n4', [4]), ('n2', [2])]:
            c = sum(h[n] * W_out[tgt, n] for n in gneurons)
            print(f"  {gname:>10}: {c:>8.2f}")
        print(f"  {'TOTAL':>10}: {sum(h[n] * W_out[tgt, n] for n in range(16)):>8.2f}")

    # =========================================================================
    # 3. The n4 gap encoding: verify n4_rev grows because n2 decays with gap
    # =========================================================================
    section("3. n4 second firing vs n2 at M arrival for reversed pairs")

    print("\nFor reversed pairs, S arrives first, then M. n2 decays between S and M.")
    print(f"{'gap':>4} {'n2_at_M':>8} {'n4_at_M':>8} {'n2_at_9':>8} {'n4_at_9':>8}")
    print("-" * 40)

    pair_idx = {}
    for i in range(len(mps)):
        pair_idx[(mps[i], sps[i])] = i

    for gap in range(1, 10):
        n2_at_M_vals = []
        n4_at_M_vals = []
        n2_at_9_vals = []
        n4_at_9_vals = []

        for mt in range(gap, 10):
            st = mt - gap
            if (mt, st) not in pair_idx:
                continue
            x = th.zeros(10)
            x[mt] = 1.0
            x[st] = 0.8
            trace = run_trace(W_ih, W_hh, x)
            n2_at_M_vals.append(trace[mt, 2])
            n4_at_M_vals.append(trace[mt, 4])
            n2_at_9_vals.append(trace[9, 2])
            n4_at_9_vals.append(trace[9, 4])

        if n2_at_M_vals:
            print(f"{gap:>4} {np.mean(n2_at_M_vals):>8.3f} {np.mean(n4_at_M_vals):>8.3f} "
                  f"{np.mean(n2_at_9_vals):>8.3f} {np.mean(n4_at_9_vals):>8.3f}")

    # =========================================================================
    # 4. Key question: what's the actual mechanism of the comp shift?
    #    For forward at last_clip=7, comps rebuild from t=7 to t=9 (2 steps)
    #    For reversed at last_clip=7, comps also rebuild from t=7 to t=9 (2 steps)
    #    But the starting conditions differ because of different first-impulse residues
    #    Show the comp state at last_clip and at t=9 for matched pairs
    # =========================================================================
    section("4. Comp state at last_clip vs h_final for matched forward/reversed pairs")

    print("\nPairs with last_clip=7, varying gap:")
    print(f"{'pair':>12} {'gap':>4}", end="")
    for n in COMPS:
        print(f" {'n'+str(n)+'_lc':>8} {'n'+str(n)+'_9':>8}", end="")
    print()
    print("-" * 80)

    for gap in range(1, 8):
        # Forward: Mt = 7 - gap, St = 7
        mt_f = 7 - gap
        st_f = 7
        # Reversed: Mt = 7, St = 7 - gap
        mt_r = 7
        st_r = 7 - gap

        for label, mt, st in [(f"fwd M{mt_f},S{st_f}", mt_f, st_f),
                               (f"rev M{mt_r},S{st_r}", mt_r, st_r)]:
            x = th.zeros(10)
            x[mt] = 1.0
            x[st] = 0.8
            trace = run_trace(W_ih, W_hh, x)
            print(f"{label:>12} {gap:>4}", end="")
            for n in COMPS:
                print(f" {trace[7, n]:>8.3f} {trace[9, n]:>8.3f}", end="")
            print()
        print()

    # =========================================================================
    # 5. The forward/reversed logit margin per pair — is n2's opposition
    #    proportional to the margin difference?
    # =========================================================================
    section("5. n2's opposition vs margin difference (forward - reversed)")

    print("\nFor matched (lc, gap) pairs, compare margins and n2 contributions:")
    print(f"{'lc':>3} {'gap':>4} {'fwd_margin':>11} {'rev_margin':>11} {'n2_fwd→tgt':>11} {'n2_rev→tgt':>11} {'n2_rev→lc':>10}")
    print("-" * 70)

    for gap in [1, 3, 5, 7]:
        for lc in range(gap, 10):
            mt_f = lc - gap
            st_f = lc
            mt_r = lc
            st_r = lc - gap

            if (mt_f, st_f) not in pair_idx or (mt_r, st_r) not in pair_idx:
                continue

            i_f = pair_idx[(mt_f, st_f)]
            i_r = pair_idx[(mt_r, st_r)]

            # Forward
            h_f = h_final[i_f]
            logits_f = h_f @ W_out.T
            margin_f = logits_f[st_f] - max(logits_f[j] for j in range(10) if j != st_f)

            # Reversed
            h_r = h_final[i_r]
            logits_r = h_r @ W_out.T
            margin_r = logits_r[st_r] - max(logits_r[j] for j in range(10) if j != st_r)

            # n2 contributions
            n2_fwd_tgt = h_f[2] * W_out[st_f, 2]
            n2_rev_tgt = h_r[2] * W_out[st_r, 2]
            n2_rev_lc = h_r[2] * W_out[lc, 2]

            print(f"{lc:>3} {gap:>4} {margin_f:>11.2f} {margin_r:>11.2f} "
                  f"{n2_fwd_tgt:>11.2f} {n2_rev_tgt:>11.2f} {n2_rev_lc:>10.2f}")
        print()

    # =========================================================================
    # 6. What if we swap the h_final of a fwd pair into a rev pair at readout?
    #    Specifically: take h_final from forward (M2,S7) and read it out
    #    as if it were reversed — does W_out give position 7 or 2?
    # =========================================================================
    section("6. Cross-read: forward h_final through W_out, reversed h_final through W_out")

    print("\nFor each (lc, gap) pair, what does W_out predict from each h_final?")
    print(f"{'lc':>3} {'gap':>4} {'fwd_h→pred':>11} {'rev_h→pred':>11} {'fwd_tgt':>8} {'rev_tgt':>8}")
    print("-" * 55)

    for gap in [1, 2, 3, 5]:
        for lc in range(gap, 10):
            mt_f = lc - gap
            st_f = lc
            mt_r = lc
            st_r = lc - gap

            if (mt_f, st_f) not in pair_idx or (mt_r, st_r) not in pair_idx:
                continue

            i_f = pair_idx[(mt_f, st_f)]
            i_r = pair_idx[(mt_r, st_r)]

            pred_f = (h_final[i_f] @ W_out.T).argmax()
            pred_r = (h_final[i_r] @ W_out.T).argmax()

            print(f"{lc:>3} {gap:>4} {pred_f:>11} {pred_r:>11} {st_f:>8} {st_r:>8}")
        print()


if __name__ == "__main__":
    main()
