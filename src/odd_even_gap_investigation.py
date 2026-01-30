"""
Investigate the odd/even gap asymmetry in reversed pairs at low S magnitude.

At s_mag=0.1, reversed pairs fail at gaps 1,3,7,9 but survive at 2,4,6,8.
Why?
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
    # 1. Confirm the pattern across multiple S magnitudes
    # =========================================================================
    section("1. Odd/even gap pattern across S magnitudes (reversed only)")

    s_mags = [0.8, 0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05]

    print(f"\n{'s_mag':>6}", end="")
    for g in range(1, 10):
        print(f" {'g'+str(g):>6}", end="")
    print()
    print("-" * 62)

    for s_mag in s_mags:
        samples, labels = [], []
        max_positions, sec_positions = [], []

        for max_pos in range(10):
            for sec_pos in range(10):
                if max_pos == sec_pos or sec_pos >= max_pos:
                    continue  # reversed only: St < Mt
                x = th.zeros(10)
                x[max_pos] = 1.0
                x[sec_pos] = s_mag
                samples.append(x)
                labels.append(sec_pos)
                max_positions.append(max_pos)
                sec_positions.append(sec_pos)

        X = th.stack(samples)
        y = th.tensor(labels)
        logits = run_model(W_ih, W_hh, W_out, X)
        preds = logits.argmax(dim=-1).numpy()
        y_np = y.numpy()
        mp_arr = np.array(max_positions)
        sp_arr = np.array(sec_positions)
        gaps = mp_arr - sp_arr  # Mt - St for reversed

        print(f"{s_mag:>6.2f}", end="")
        for g in range(1, 10):
            mask = gaps == g
            if mask.sum() > 0:
                acc = (preds[mask] == y_np[mask]).mean() * 100
                print(f" {acc:>5.0f}%", end="")
            else:
                print(f"    --", end="")
        print()

    # =========================================================================
    # 2. At s_mag=0.1, trace specific odd vs even gap pairs
    # =========================================================================
    section("2. Comp h_final for reversed pairs at s_mag=0.1: odd vs even gaps")

    s_mag = 0.1
    print("\nFor each gap, show all reversed pairs' comp h_final and prediction.")
    print(f"{'Mt':>3} {'St':>3} {'gap':>4} {'pred':>5} {'tgt':>4} {'ok':>3}  "
          f"{'n1':>7} {'n6':>7} {'n7':>7} {'n8':>7}")
    print("-" * 60)

    for g in [1, 2, 3, 4]:  # Compare first few odd/even
        for max_pos in range(10):
            sec_pos = max_pos - g
            if sec_pos < 0:
                continue
            x = th.zeros(10)
            x[max_pos] = 1.0
            x[sec_pos] = s_mag
            X_single = x.unsqueeze(0)

            h_final = run_model_hidden(W_ih, W_hh, X_single)[0]
            logits = run_model(W_ih, W_hh, W_out, X_single)
            pred = logits.argmax(dim=-1).item()
            ok = "Y" if pred == sec_pos else "N"

            print(f"{max_pos:>3} {sec_pos:>3} {g:>4} {pred:>5} {sec_pos:>4} {ok:>3}  "
                  f"{h_final[1]:>7.2f} {h_final[6]:>7.2f} {h_final[7]:>7.2f} {h_final[8]:>7.2f}")
        print()

    # =========================================================================
    # 3. Compare n2 latch at time of M arrival for odd vs even gaps
    # =========================================================================
    section("3. n2 and n4 at M arrival time: odd vs even gaps, s_mag=0.1")

    s_mag = 0.1
    print(f"\n{'Mt':>3} {'St':>3} {'gap':>4} {'n2_preM':>8} {'n4_atM':>8} {'n2_atM':>8} {'pred':>5} {'ok':>3}")
    print("-" * 52)

    for g in range(1, 10):
        for max_pos in range(g, 10):
            sec_pos = max_pos - g
            x = th.zeros(10)
            x[max_pos] = 1.0
            x[sec_pos] = s_mag
            trace = run_trace(W_ih, W_hh, x)

            n2_pre = trace[max_pos - 1, 2] if max_pos > 0 else 0.0
            n4_at = trace[max_pos, 4]
            n2_at = trace[max_pos, 2]

            logits = (th.tensor(trace[9]) @ W_out.T).numpy()
            pred = logits.argmax()
            ok = "Y" if pred == sec_pos else "N"

            print(f"{max_pos:>3} {sec_pos:>3} {g:>4} {n2_pre:>8.3f} {n4_at:>8.3f} "
                  f"{n2_at:>8.3f} {pred:>5} {ok:>3}")
        print()

    # =========================================================================
    # 4. Check if it's a parity thing in the W_hh eigenstructure
    # =========================================================================
    section("4. W_hh^k patterns for comps — odd vs even step counts")

    # The rebuild trajectory after M clip depends on W_hh^k.
    # Check if there's a parity structure in the eigenvalues.
    eigenvalues = np.linalg.eigvals(W_hh.numpy())
    print("\nW_hh eigenvalues:")
    for i, ev in enumerate(sorted(eigenvalues, key=lambda x: -abs(x))):
        print(f"  λ{i:>2}: {ev.real:>8.4f} + {ev.imag:>8.4f}j  |λ|={abs(ev):.4f}")

    # Count negative real eigenvalues (these flip sign each step → parity)
    neg_real = sum(1 for ev in eigenvalues if abs(ev.imag) < 0.01 and ev.real < 0)
    complex_pairs = sum(1 for ev in eigenvalues if abs(ev.imag) > 0.01) // 2
    print(f"\nNegative real eigenvalues: {neg_real}")
    print(f"Complex conjugate pairs: {complex_pairs}")
    print("(Negative real eigenvalues create odd/even parity in trajectories)")

    # =========================================================================
    # 5. Single-impulse rebuild: check for parity in comp trajectories
    # =========================================================================
    section("5. Single-impulse comp rebuild trajectories — parity check")

    print("\nSingle impulse at t=0, magnitude 1.0:")
    x = th.zeros(10)
    x[0] = 1.0
    trace = run_trace(W_ih, W_hh, x)

    print(f"{'t':>3} {'steps':>6} {'n1':>8} {'n6':>8} {'n7':>8} {'n8':>8}")
    print("-" * 40)
    for t in range(10):
        print(f"{t:>3} {t:>6} {trace[t,1]:>8.3f} {trace[t,6]:>8.3f} "
              f"{trace[t,7]:>8.3f} {trace[t,8]:>8.3f}")

    # Check: do comps alternate in some way?
    print("\nStep-to-step differences (shows oscillation):")
    print(f"{'t':>3} {'Δn1':>8} {'Δn6':>8} {'Δn7':>8} {'Δn8':>8}")
    print("-" * 36)
    for t in range(1, 10):
        print(f"{t:>3} {trace[t,1]-trace[t-1,1]:>8.3f} {trace[t,6]-trace[t-1,6]:>8.3f} "
              f"{trace[t,7]-trace[t-1,7]:>8.3f} {trace[t,8]-trace[t-1,8]:>8.3f}")

    # =========================================================================
    # 6. At s_mag=0.1 reversed: logit margins for odd vs even gaps
    # =========================================================================
    section("6. Logit margins: correct_logit - max_wrong_logit, s_mag=0.1 reversed")

    s_mag = 0.1
    print(f"\n{'Mt':>3} {'St':>3} {'gap':>4} {'margin':>8} {'correct':>8} {'top_wrong':>10} {'pred':>5}")
    print("-" * 52)

    for g in range(1, 10):
        for max_pos in range(g, 10):
            sec_pos = max_pos - g
            x = th.zeros(10)
            x[max_pos] = 1.0
            x[sec_pos] = s_mag

            trace = run_trace(W_ih, W_hh, x)
            logits = (th.tensor(trace[9]) @ W_out.T).numpy()
            correct_logit = logits[sec_pos]
            wrong_logits = np.delete(logits, sec_pos)
            max_wrong = wrong_logits.max()
            margin = correct_logit - max_wrong
            pred = logits.argmax()

            print(f"{max_pos:>3} {sec_pos:>3} {g:>4} {margin:>8.3f} {correct_logit:>8.3f} "
                  f"{max_wrong:>10.3f} {pred:>5}")
        print()

    # Summary: mean margin by gap
    print("Mean margin by gap:")
    print(f"{'gap':>4} {'mean_margin':>12} {'min_margin':>12} {'parity':>7}")
    print("-" * 40)
    for g in range(1, 10):
        margins = []
        for max_pos in range(g, 10):
            sec_pos = max_pos - g
            x = th.zeros(10)
            x[max_pos] = 1.0
            x[sec_pos] = s_mag

            trace = run_trace(W_ih, W_hh, x)
            logits = (th.tensor(trace[9]) @ W_out.T).numpy()
            correct_logit = logits[sec_pos]
            max_wrong = np.delete(logits, sec_pos).max()
            margins.append(correct_logit - max_wrong)

        margins = np.array(margins)
        parity = "ODD" if g % 2 == 1 else "EVEN"
        print(f"{g:>4} {margins.mean():>12.3f} {margins.min():>12.3f} {parity:>7}")


if __name__ == "__main__":
    main()
