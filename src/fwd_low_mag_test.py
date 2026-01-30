"""
Failure mode test: forward pairs where mag(St) << mag(Mt).

Hypothesis: when forward and S magnitude is very low relative to M,
the model may mistake Mt for St (output Mt instead of St).
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


def run_model_trace(W_ih, W_hh, x_single):
    """Return full hidden trace for a single input."""
    h = th.zeros(1, 16)
    trace = []
    for t in range(10):
        x_t = x_single[t:t+1].unsqueeze(0)
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
        trace.append(h[0].detach().numpy().copy())
    return np.array(trace)  # (10, 16)


def main():
    model = load_local_model()
    W_ih = model.rnn.weight_ih_l0.data.squeeze().clone()
    W_hh = model.rnn.weight_hh_l0.data.clone()
    W_out = model.linear.weight.data.clone()

    print("="*80)
    print("FAILURE MODE: Forward pairs with low S magnitude")
    print("="*80)
    print()

    # Sweep S magnitudes
    s_mags = [0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01]

    print(f"{'s_mag':>6} {'fwd%':>6} {'rev%':>6} {'all%':>6}  "
          f"{'fwd_pred_Mt':>12} {'rev_pred_Mt':>12}")
    print("-" * 60)

    for s_mag in s_mags:
        samples, labels = [], []
        max_positions, sec_positions = [], []
        is_forward = []

        for max_pos in range(10):
            for sec_pos in range(10):
                if max_pos == sec_pos:
                    continue
                x = th.zeros(10)
                x[max_pos] = 1.0
                x[sec_pos] = s_mag
                samples.append(x)
                labels.append(sec_pos)
                max_positions.append(max_pos)
                sec_positions.append(sec_pos)
                is_forward.append(sec_pos > max_pos)

        X = th.stack(samples)
        y = th.tensor(labels)
        fwd = th.tensor(is_forward)
        rev = ~fwd

        logits = run_model(W_ih, W_hh, W_out, X)
        preds = logits.argmax(dim=-1)

        fwd_acc = (preds[fwd] == y[fwd]).float().mean().item() * 100
        rev_acc = (preds[rev] == y[rev]).float().mean().item() * 100
        all_acc = (preds == y).float().mean().item() * 100

        mp_arr = th.tensor(max_positions)
        fwd_pred_mt = (preds[fwd] == mp_arr[fwd]).float().mean().item() * 100
        rev_pred_mt = (preds[rev] == mp_arr[rev]).float().mean().item() * 100

        print(f"{s_mag:>6.2f} {fwd_acc:>5.1f}% {rev_acc:>5.1f}% {all_acc:>5.1f}%  "
              f"{fwd_pred_mt:>11.1f}% {rev_pred_mt:>11.1f}%")

    # Detailed reversed failures at s_mag=0.1
    print("\n" + "="*80)
    print("DETAILED: s_mag=0.1 — reversed pair failures by gap")
    print("="*80)

    s_mag = 0.1
    samples, labels = [], []
    max_positions, sec_positions = [], []

    for max_pos in range(10):
        for sec_pos in range(10):
            if max_pos == sec_pos:
                continue
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
    gaps = np.abs(sp_arr - mp_arr)
    is_fwd = sp_arr > mp_arr

    print(f"\n{'gap':>4} {'fwd_acc':>8} {'rev_acc':>8} {'fwd_n':>6} {'rev_n':>6}")
    print("-" * 36)
    for g in range(1, 10):
        fwd_mask = (gaps == g) & is_fwd
        rev_mask = (gaps == g) & ~is_fwd
        fwd_acc = (preds[fwd_mask] == y_np[fwd_mask]).mean() * 100 if fwd_mask.sum() > 0 else float('nan')
        rev_acc = (preds[rev_mask] == y_np[rev_mask]).mean() * 100 if rev_mask.sum() > 0 else float('nan')
        print(f"{g:>4} {fwd_acc:>7.1f}% {rev_acc:>7.1f}% {int(fwd_mask.sum()):>6} {int(rev_mask.sum()):>6}")

    # Every wrong reversed pair at s_mag=0.1
    print(f"\nWrong reversed pairs at s_mag=0.1:")
    print(f"{'Mt':>4} {'St':>4} {'gap':>4} {'pred':>5} {'tgt':>4}")
    print("-" * 25)
    for i in range(len(preds)):
        if not is_fwd[i] and preds[i] != y_np[i]:
            print(f"{mp_arr[i]:>4} {sp_arr[i]:>4} {gaps[i]:>4} {preds[i]:>5} {y_np[i]:>4}")

    # What does failure look like internally?
    # Compare n2/n4 trace for a clean vs weak S reversed pair
    print("\n" + "="*80)
    print("TRACE COMPARISON: reversed pair (M5, S2) at s_mag=0.8 vs 0.1")
    print("="*80)

    for s_mag_trace in [0.8, 0.1]:
        x = th.zeros(10)
        x[5] = 1.0
        x[2] = s_mag_trace
        trace = run_model_trace(W_ih, W_hh, x)
        logit = (th.tensor(trace[9]) @ W_out.T).numpy()
        pred = logit.argmax()

        print(f"\ns_mag={s_mag_trace}, pred={pred}, target=2")
        print(f"{'t':>3} {'n2':>8} {'n4':>8} {'n7':>8} {'n9':>8}")
        print("-" * 38)
        for t in range(10):
            print(f"{t:>3} {trace[t,2]:>8.3f} {trace[t,4]:>8.3f} "
                  f"{trace[t,7]:>8.3f} {trace[t,9]:>8.3f}")
        print(f"Top 3 logits: {np.argsort(logit)[-3:][::-1]} = "
              f"{logit[np.argsort(logit)[-3:][::-1]]}")

    # Also trace a forward pair to see why forward survives
    print("\n" + "="*80)
    print("TRACE COMPARISON: forward pair (M2, S5) at s_mag=0.8 vs 0.1")
    print("="*80)

    for s_mag_trace in [0.8, 0.1]:
        x = th.zeros(10)
        x[2] = 1.0
        x[5] = s_mag_trace
        trace = run_model_trace(W_ih, W_hh, x)
        logit = (th.tensor(trace[9]) @ W_out.T).numpy()
        pred = logit.argmax()

        print(f"\ns_mag={s_mag_trace}, pred={pred}, target=5")
        print(f"{'t':>3} {'n2':>8} {'n4':>8} {'n7':>8} {'n9':>8}")
        print("-" * 38)
        for t in range(10):
            print(f"{t:>3} {trace[t,2]:>8.3f} {trace[t,4]:>8.3f} "
                  f"{trace[t,7]:>8.3f} {trace[t,9]:>8.3f}")
        print(f"Top 3 logits: {np.argsort(logit)[-3:][::-1]} = "
              f"{logit[np.argsort(logit)[-3:][::-1]]}")


if __name__ == "__main__":
    main()
