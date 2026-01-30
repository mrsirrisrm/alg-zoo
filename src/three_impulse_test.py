"""
Failure mode test: forward pairs where a 3rd-largest value (Tt) follows St
and is close in magnitude.

Setup: M at some early position (mag 1.0), S at a middle position (mag 0.8),
T at a later position (mag close to 0.8).

If T comes after S and is close in magnitude, the model might:
- Clip comps again at T, making T's position the last_clip
- Confuse T for S and output T's position
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


def section(title):
    print(f"\n{'='*80}")
    print(title)
    print('='*80)


def main():
    model = load_local_model()
    W_ih = model.rnn.weight_ih_l0.data.squeeze().clone()
    W_hh = model.rnn.weight_hh_l0.data.clone()
    W_out = model.linear.weight.data.clone()

    # =========================================================================
    # 1. Sweep T magnitude for fixed M, S, T positions (forward: Mt < St < Tt)
    # =========================================================================
    section("1. Forward: M=1.0 at pos 1, S=0.8 at pos 4, T at pos 7, sweep T magnitude")

    t_mags = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.79, 0.8]

    print(f"{'t_mag':>6} {'pred':>5} {'tgt':>4} {'ok':>3} {'logit_S':>8} {'logit_T':>8} {'margin':>8}")
    print("-" * 48)
    for t_mag in t_mags:
        x = th.zeros(10)
        x[1] = 1.0   # M
        x[4] = 0.8   # S (target = 4)
        x[7] = t_mag  # T (3rd largest)

        logits = run_model(W_ih, W_hh, W_out, x.unsqueeze(0))
        logits_np = logits[0].detach().numpy()
        pred = logits_np.argmax()
        ok = "Y" if pred == 4 else "N"

        print(f"{t_mag:>6.2f} {pred:>5} {4:>4} {ok:>3} {logits_np[4]:>8.3f} "
              f"{logits_np[7]:>8.3f} {logits_np[4]-logits_np[7]:>8.3f}")

    # =========================================================================
    # 2. Systematic: all valid (Mt, St, Tt) forward triples, T close to S
    # =========================================================================
    section("2. Systematic: all forward triples (Mt<St<Tt), T_mag=0.79")

    t_mag = 0.79
    results = []

    for mt in range(8):
        for st in range(mt+1, 9):
            for tt in range(st+1, 10):
                x = th.zeros(10)
                x[mt] = 1.0
                x[st] = 0.8
                x[tt] = t_mag
                logits = run_model(W_ih, W_hh, W_out, x.unsqueeze(0))
                pred = logits[0].argmax().item()
                results.append((mt, st, tt, pred, st, pred == st))

    correct = sum(r[5] for r in results)
    total = len(results)
    print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"Predict Tt instead of St: {sum(1 for r in results if r[3]==r[2])}/{total}")

    # Show failures
    failures = [r for r in results if not r[5]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        print(f"{'Mt':>3} {'St':>3} {'Tt':>3} {'pred':>5} {'pred==Tt':>9}")
        print("-" * 28)
        for mt, st, tt, pred, tgt, ok in failures:
            pred_tt = "YES" if pred == tt else "no"
            print(f"{mt:>3} {st:>3} {tt:>3} {pred:>5} {pred_tt:>9}")

    # =========================================================================
    # 3. Sweep T_mag more finely to find the breaking point
    # =========================================================================
    section("3. Accuracy vs T_mag for all forward triples")

    t_mags = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.78, 0.79, 0.795, 0.8]

    print(f"{'t_mag':>6} {'acc%':>6} {'n':>4} {'pred_Tt%':>9} {'pred_other%':>12}")
    print("-" * 42)

    for t_mag in t_mags:
        correct = 0
        total = 0
        pred_tt_count = 0
        pred_other = 0

        for mt in range(8):
            for st in range(mt+1, 9):
                for tt in range(st+1, 10):
                    x = th.zeros(10)
                    x[mt] = 1.0
                    x[st] = 0.8
                    x[tt] = t_mag
                    logits = run_model(W_ih, W_hh, W_out, x.unsqueeze(0))
                    pred = logits[0].argmax().item()
                    total += 1
                    if pred == st:
                        correct += 1
                    elif pred == tt:
                        pred_tt_count += 1
                    else:
                        pred_other += 1

        print(f"{t_mag:>6.3f} {correct/total*100:>5.1f}% {total:>4} "
              f"{pred_tt_count/total*100:>8.1f}% {pred_other/total*100:>11.1f}%")

    # =========================================================================
    # 4. T_mag = 0.79: breakdown by gap(Mt,St) and gap(St,Tt)
    # =========================================================================
    section("4. T_mag=0.79: accuracy by gap(Mt,St) and gap(St,Tt)")

    t_mag = 0.79
    by_gap_ms = defaultdict(lambda: [0, 0])
    by_gap_st = defaultdict(lambda: [0, 0])

    for mt in range(8):
        for st in range(mt+1, 9):
            for tt in range(st+1, 10):
                x = th.zeros(10)
                x[mt] = 1.0
                x[st] = 0.8
                x[tt] = t_mag
                logits = run_model(W_ih, W_hh, W_out, x.unsqueeze(0))
                pred = logits[0].argmax().item()

                gap_ms = st - mt
                gap_st = tt - st

                by_gap_ms[gap_ms][1] += 1
                by_gap_st[gap_st][1] += 1
                if pred == st:
                    by_gap_ms[gap_ms][0] += 1
                    by_gap_st[gap_st][0] += 1

    print(f"\nBy gap(Mt, St):")
    print(f"{'gap':>4} {'acc':>8} {'n':>4}")
    print("-" * 18)
    for g in sorted(by_gap_ms.keys()):
        c, t = by_gap_ms[g]
        print(f"{g:>4} {c/t*100:>7.1f}% {t:>4}")

    print(f"\nBy gap(St, Tt):")
    print(f"{'gap':>4} {'acc':>8} {'n':>4}")
    print("-" * 18)
    for g in sorted(by_gap_st.keys()):
        c, t = by_gap_st[g]
        print(f"{g:>4} {c/t*100:>7.1f}% {t:>4}")

    # =========================================================================
    # 5. Trace comparison: with and without T
    # =========================================================================
    section("5. Trace: (M1, S4) without T vs with T=0.79 at pos 7")

    for label, t_mag_val, t_pos in [("No T", 0.0, 7), ("T=0.79 at 7", 0.79, 7)]:
        x = th.zeros(10)
        x[1] = 1.0
        x[4] = 0.8
        if t_mag_val > 0:
            x[t_pos] = t_mag_val

        trace = run_trace(W_ih, W_hh, x)
        logits = (th.tensor(trace[9]) @ W_out.T).numpy()
        pred = logits.argmax()

        print(f"\n{label}, pred={pred}, target=4")
        print(f"{'t':>3} {'n2':>8} {'n4':>8} {'n7':>8} {'n9':>8}  {'n1':>8} {'n6':>8} {'n8':>8}")
        print("-" * 68)
        for t in range(10):
            print(f"{t:>3} {trace[t,2]:>8.3f} {trace[t,4]:>8.3f} "
                  f"{trace[t,7]:>8.3f} {trace[t,9]:>8.3f}  "
                  f"{trace[t,1]:>8.3f} {trace[t,6]:>8.3f} {trace[t,8]:>8.3f}")
        print(f"Top 3 logits: pos {np.argsort(logits)[-3:][::-1]} = "
              f"{logits[np.argsort(logits)[-3:][::-1]]}")

    # =========================================================================
    # 6. What if T comes BETWEEN M and S? (Mt < Tt < St)
    # =========================================================================
    section("6. T between M and S: (Mt < Tt < St), T_mag=0.79")

    t_mag = 0.79
    results = []

    for mt in range(7):
        for tt in range(mt+1, 8):
            for st in range(tt+1, 10):
                x = th.zeros(10)
                x[mt] = 1.0
                x[tt] = t_mag
                x[st] = 0.8
                logits = run_model(W_ih, W_hh, W_out, x.unsqueeze(0))
                pred = logits[0].argmax().item()
                results.append((mt, tt, st, pred, st, pred == st))

    correct = sum(r[5] for r in results)
    total = len(results)
    print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"Predict Tt: {sum(1 for r in results if r[3]==r[1])}/{total}")

    failures = [r for r in results if not r[5]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        print(f"{'Mt':>3} {'Tt':>3} {'St':>3} {'pred':>5} {'pred==Tt':>9}")
        print("-" * 28)
        for mt, tt, st, pred, tgt, ok in failures:
            pred_tt = "YES" if pred == tt else "no"
            print(f"{mt:>3} {tt:>3} {st:>3} {pred:>5} {pred_tt:>9}")


if __name__ == "__main__":
    main()
