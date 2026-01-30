"""
Deep dive into odd/even gap asymmetry in reversed pairs at low S magnitude.

Questions:
1. Does odd/even hold across s_mag values?
2. Is M9 a special case? (gap=9 is always M9,S0)
3. What's happening at gap=3 specifically?
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


def run_model(W_ih, W_hh, W_out, X):
    h = th.zeros(X.shape[0], 16)
    for t in range(10):
        x_t = X[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
    return h @ W_out.T


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


def section(title):
    print(f"\n{'='*80}")
    print(title)
    print('='*80)


def make_reversed_dataset(s_mag):
    """All reversed pairs: St < Mt."""
    samples, labels, mps, sps = [], [], [], []
    for mt in range(10):
        for st in range(mt):
            x = th.zeros(10)
            x[mt] = 1.0
            x[st] = s_mag
            samples.append(x)
            labels.append(st)
            mps.append(mt)
            sps.append(st)
    return th.stack(samples), th.tensor(labels), np.array(mps), np.array(sps)


def main():
    model = load_local_model()
    W_ih = model.rnn.weight_ih_l0.data.squeeze().clone()
    W_hh = model.rnn.weight_hh_l0.data.clone()
    W_out = model.linear.weight.data.clone()

    # =========================================================================
    # 1. Full heatmap: accuracy by (gap, s_mag) — reversed only
    # =========================================================================
    section("1. Accuracy heatmap: gap x s_mag (reversed pairs only)")

    s_mags = [0.80, 0.50, 0.30, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04]

    print(f"\n{'s_mag':>6}", end="")
    for g in range(1, 10):
        parity = "o" if g % 2 == 1 else "e"
        print(f"  g{g}{parity:>1}", end="")
    print("   all")
    print("-" * 68)

    for s_mag in s_mags:
        X, y, mps, sps = make_reversed_dataset(s_mag)
        gaps = mps - sps
        logits = run_model(W_ih, W_hh, W_out, X)
        preds = logits.argmax(dim=-1).numpy()
        correct = (preds == y.numpy())

        print(f"{s_mag:>6.2f}", end="")
        for g in range(1, 10):
            mask = gaps == g
            if mask.sum() > 0:
                acc = correct[mask].mean() * 100
                print(f" {acc:>4.0f}%", end="")
            else:
                print(f"   --", end="")
        all_acc = correct.mean() * 100
        print(f"  {all_acc:>4.0f}%")

    # =========================================================================
    # 2. Same but EXCLUDING M9 pairs
    # =========================================================================
    section("2. Same heatmap, EXCLUDING Mt=9 pairs")

    print(f"\n{'s_mag':>6}", end="")
    for g in range(1, 10):
        parity = "o" if g % 2 == 1 else "e"
        print(f"  g{g}{parity:>1}", end="")
    print("   all")
    print("-" * 68)

    for s_mag in s_mags:
        X, y, mps, sps = make_reversed_dataset(s_mag)
        gaps = mps - sps
        not_m9 = mps != 9
        logits = run_model(W_ih, W_hh, W_out, X)
        preds = logits.argmax(dim=-1).numpy()
        correct = (preds == y.numpy())

        print(f"{s_mag:>6.2f}", end="")
        for g in range(1, 10):
            mask = (gaps == g) & not_m9
            if mask.sum() > 0:
                acc = correct[mask].mean() * 100
                print(f" {acc:>4.0f}%", end="")
            else:
                print(f"   --", end="")
        all_acc = correct[not_m9].mean() * 100
        print(f"  {all_acc:>4.0f}%")

    # =========================================================================
    # 3. M9 pairs specifically
    # =========================================================================
    section("3. Mt=9 pairs accuracy by gap and s_mag")

    print(f"\nMt=9 reversed pairs: (M9, S8)=gap1, (M9,S7)=gap2, ... (M9,S0)=gap9")
    print(f"\n{'s_mag':>6}", end="")
    for g in range(1, 10):
        print(f"  g{g}", end="")
    print()
    print("-" * 50)

    for s_mag in s_mags:
        X, y, mps, sps = make_reversed_dataset(s_mag)
        gaps = mps - sps
        is_m9 = mps == 9
        logits = run_model(W_ih, W_hh, W_out, X)
        preds = logits.argmax(dim=-1).numpy()
        correct = (preds == y.numpy())

        print(f"{s_mag:>6.2f}", end="")
        for g in range(1, 10):
            mask = (gaps == g) & is_m9
            if mask.sum() > 0:
                acc = correct[mask].mean() * 100
                print(f" {acc:>4.0f}%", end="")
            else:
                print(f"   --", end="")
        print()

    # =========================================================================
    # 4. Individual pair accuracy at s_mag=0.10 — every reversed pair
    # =========================================================================
    section("4. Every reversed pair at s_mag=0.10")

    s_mag = 0.10
    X, y, mps, sps = make_reversed_dataset(s_mag)
    gaps = mps - sps
    logits = run_model(W_ih, W_hh, W_out, X)
    preds = logits.argmax(dim=-1).numpy()
    y_np = y.numpy()
    logits_np = logits.detach().numpy()

    # Grid: Mt on rows, St on columns
    print(f"\nPred grid (rows=Mt, cols=St, '.' = correct, 'X'=wrong, '-'=n/a):")
    print(f"    ", end="")
    for st in range(10):
        print(f" S{st}", end="")
    print()
    for mt in range(10):
        print(f" M{mt} ", end="")
        for st in range(10):
            if st >= mt:
                print(f"  -", end="")
            else:
                idx = np.where((mps == mt) & (sps == st))[0]
                if len(idx) > 0:
                    i = idx[0]
                    if preds[i] == y_np[i]:
                        print(f"  .", end="")
                    else:
                        print(f" X{preds[i]}", end="")
        print(f"  | gap parity: ", end="")
        for st in range(mt):
            g = mt - st
            print("o" if g % 2 == 1 else "e", end="")
        print()

    # =========================================================================
    # 5. Logit margin grid at s_mag=0.10
    # =========================================================================
    section("5. Logit margins at s_mag=0.10 — every reversed pair")

    print(f"\nMargin grid (correct_logit - max_wrong_logit):")
    print(f"      ", end="")
    for st in range(9):
        print(f"   S{st}", end="")
    print()
    for mt in range(1, 10):
        print(f" M{mt}  ", end="")
        for st in range(mt):
            idx = np.where((mps == mt) & (sps == st))[0]
            if len(idx) > 0:
                i = idx[0]
                correct_l = logits_np[i, y_np[i]]
                wrong_l = np.delete(logits_np[i], y_np[i]).max()
                margin = correct_l - wrong_l
                if margin >= 0:
                    print(f" {margin:>4.1f}", end="")
                else:
                    print(f" {margin:>4.1f}", end="")
            else:
                print(f"    -", end="")
        print()

    # =========================================================================
    # 6. Gap=3 deep dive: what are these pairs and why do they fail?
    # =========================================================================
    section("6. Gap=3 deep dive at s_mag=0.10")

    print("\nAll gap=3 reversed pairs:")
    print(f"{'Mt':>3} {'St':>3} {'pred':>5} {'tgt':>4} {'ok':>3} {'margin':>8} "
          f"{'n2_preM':>8} {'n4_atM':>8} {'n7_final':>9}")
    print("-" * 62)

    for mt in range(3, 10):
        st = mt - 3
        x = th.zeros(10)
        x[mt] = 1.0
        x[st] = s_mag
        trace = run_trace(W_ih, W_hh, x)
        logits_single = (th.tensor(trace[9]) @ W_out.T).numpy()
        pred = logits_single.argmax()
        correct_l = logits_single[st]
        wrong_l = np.delete(logits_single, st).max()
        margin = correct_l - wrong_l

        n2_pre = trace[mt-1, 2] if mt > 0 else 0.0
        n4_at = trace[mt, 4]
        n7_final = trace[9, 7]

        ok = "Y" if pred == st else "N"
        print(f"{mt:>3} {st:>3} {pred:>5} {st:>4} {ok:>3} {margin:>8.3f} "
              f"{n2_pre:>8.3f} {n4_at:>8.3f} {n7_final:>9.3f}")

    # Compare with gap=2 and gap=4
    for gap_label, gap_val in [("Gap=2", 2), ("Gap=4", 4)]:
        print(f"\n{gap_label} reversed pairs:")
        print(f"{'Mt':>3} {'St':>3} {'pred':>5} {'tgt':>4} {'ok':>3} {'margin':>8}")
        print("-" * 32)
        for mt in range(gap_val, 10):
            st = mt - gap_val
            x = th.zeros(10)
            x[mt] = 1.0
            x[st] = s_mag
            trace = run_trace(W_ih, W_hh, x)
            logits_single = (th.tensor(trace[9]) @ W_out.T).numpy()
            pred = logits_single.argmax()
            margin = logits_single[st] - np.delete(logits_single, st).max()
            ok = "Y" if pred == st else "N"
            print(f"{mt:>3} {st:>3} {pred:>5} {st:>4} {ok:>3} {margin:>8.3f}")

    # =========================================================================
    # 7. Gap=3: trace all 4 comps through time for a failing vs passing pair
    # =========================================================================
    section("7. Full comp traces: gap=3 failing (M5,S2) vs passing (M7,S4)")

    for mt, st, label in [(5, 2, "FAILS"), (7, 4, "PASSES")]:
        x = th.zeros(10)
        x[mt] = 1.0
        x[st] = s_mag
        trace = run_trace(W_ih, W_hh, x)
        logits_single = (th.tensor(trace[9]) @ W_out.T).numpy()
        pred = logits_single.argmax()

        print(f"\n(M{mt}, S{st}) gap=3 — {label}, pred={pred}, target={st}")
        print(f"{'t':>3} {'n1':>8} {'n6':>8} {'n7':>8} {'n8':>8}  "
              f"{'n2':>8} {'n4':>8} {'n9':>8}")
        print("-" * 72)
        for t in range(10):
            mark = ""
            if t == st:
                mark = " ← S"
            elif t == mt:
                mark = " ← M"
            print(f"{t:>3} {trace[t,1]:>8.3f} {trace[t,6]:>8.3f} "
                  f"{trace[t,7]:>8.3f} {trace[t,8]:>8.3f}  "
                  f"{trace[t,2]:>8.3f} {trace[t,4]:>8.3f} {trace[t,9]:>8.3f}{mark}")
        print(f"Top 3: pos {np.argsort(logits_single)[-3:][::-1]} = "
              f"{logits_single[np.argsort(logits_single)[-3:][::-1]]}")

    # =========================================================================
    # 8. Same gap=3 at s_mag=0.8 (clean) to see what changes
    # =========================================================================
    section("8. Gap=3 at s_mag=0.8 (clean) for reference")

    s_mag_clean = 0.8
    for mt, st in [(5, 2), (3, 0), (6, 3)]:
        x = th.zeros(10)
        x[mt] = 1.0
        x[st] = s_mag_clean
        trace = run_trace(W_ih, W_hh, x)
        logits_single = (th.tensor(trace[9]) @ W_out.T).numpy()
        pred = logits_single.argmax()
        margin = logits_single[st] - np.delete(logits_single, st).max()

        print(f"\n(M{mt}, S{st}) s=0.8, pred={pred}, margin={margin:.3f}")
        print(f"{'t':>3} {'n1':>8} {'n6':>8} {'n7':>8} {'n8':>8}  "
              f"{'n2':>8} {'n4':>8}")
        print("-" * 60)
        for t in range(10):
            mark = ""
            if t == st:
                mark = " ← S"
            elif t == mt:
                mark = " ← M"
            print(f"{t:>3} {trace[t,1]:>8.3f} {trace[t,6]:>8.3f} "
                  f"{trace[t,7]:>8.3f} {trace[t,8]:>8.3f}  "
                  f"{trace[t,2]:>8.3f} {trace[t,4]:>8.3f}{mark}")

    # =========================================================================
    # 9. Check: is the gap=3 problem about n2 level or cascade state?
    # Clamp n2 to its clean-dataset value and rerun gap=3 at s_mag=0.1
    # =========================================================================
    section("9. Gap=3 rescue: what if n2 had the right value?")
    print("Inject n2's clean-dataset value at the M arrival step.\n")

    for mt, st in [(3, 0), (5, 2), (6, 3)]:
        # Get clean n2 value at M-1
        x_clean = th.zeros(10)
        x_clean[mt] = 1.0
        x_clean[st] = 0.8
        trace_clean = run_trace(W_ih, W_hh, x_clean)
        n2_clean_pre = trace_clean[mt-1, 2] if mt > 0 else 0.0

        # Run weak S with n2 clamped at M arrival
        x = th.zeros(10)
        x[mt] = 1.0
        x[st] = 0.1
        h = th.zeros(1, 16)
        for t in range(10):
            if t == mt:
                # Inject clean n2 just before M processes
                h[0, 2] = n2_clean_pre
            x_t = x[t:t+1].unsqueeze(0)
            pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
            h = th.relu(pre)

        logits_rescue = (h @ W_out.T).detach().numpy()[0]
        pred_rescue = logits_rescue.argmax()

        # Also get original weak prediction
        trace_weak = run_trace(W_ih, W_hh, x)
        logits_weak = (th.tensor(trace_weak[9]) @ W_out.T).numpy()
        pred_weak = logits_weak.argmax()

        print(f"(M{mt}, S{st}) gap=3:")
        print(f"  Clean n2 pre-M = {n2_clean_pre:.3f}, weak n2 pre-M = "
              f"{trace_weak[mt-1,2] if mt>0 else 0:.3f}")
        print(f"  Weak pred = {pred_weak} (target {st}), "
              f"rescued pred = {pred_rescue} (target {st})")
        margin_weak = logits_weak[st] - np.delete(logits_weak, st).max()
        margin_rescue = logits_rescue[st] - np.delete(logits_rescue, st).max()
        print(f"  Weak margin = {margin_weak:.3f}, rescued margin = {margin_rescue:.3f}")
        print()


if __name__ == "__main__":
    main()
