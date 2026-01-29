"""
Impulse input analysis and n0/n14 connectivity deep dive.

1. Feed impulse inputs [1,0,0,...], [0,1,0,...], etc. and record full trajectories
2. Examine W_hh connectivity of n0 and n14 in detail
"""

import torch as th
import numpy as np
import sys
sys.path.insert(0, '/Users/martin/code/alg-zoo')
from alg_zoo.architectures import DistRNN

# Load model from local file
state_dict = th.load('/private/tmp/claude/-Users-martin-code-alg-zoo/0015c501-bf0c-4060-b9c3-623d5c02e5f1/scratchpad/model.pth', weights_only=True, map_location='cpu')
seq_len_m, hidden_size_m = state_dict['linear.weight'].shape
model = DistRNN(hidden_size=hidden_size_m, seq_len=seq_len_m, bias=False)
model.load_state_dict(state_dict)
W_ih = model.rnn.weight_ih_l0.data.squeeze()  # (16,)
W_hh = model.rnn.weight_hh_l0.data            # (16, 16)
W_out = model.linear.weight.data               # (10, 16)

n_neurons = 16
seq_len = 10

# ============================================================
# PART 1: IMPULSE INPUTS
# ============================================================
print("=" * 80)
print("PART 1: IMPULSE INPUT ANALYSIS")
print("=" * 80)

# Create 10 impulse sequences: [1,0,...], [0,1,0,...], etc.
impulses = th.zeros(seq_len, seq_len)
for i in range(seq_len):
    impulses[i, i] = 1.0

# Run each impulse through the model, recording full trajectory
for imp_pos in range(seq_len):
    x = impulses[imp_pos].unsqueeze(0).unsqueeze(-1)  # (1, 10, 1)

    h = th.zeros(1, n_neurons)
    hidden_states = []
    pre_acts = []
    clipped = []

    for t in range(seq_len):
        x_t = x[:, t, :]  # (1, 1)
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        clip_mask = (pre < 0)
        h = th.relu(pre)
        hidden_states.append(h.squeeze(0).clone())
        pre_acts.append(pre.squeeze(0).clone())
        clipped.append(clip_mask.squeeze(0).clone())

    hidden_states = th.stack(hidden_states)  # (10, 16)
    pre_acts = th.stack(pre_acts)
    clipped = th.stack(clipped)

    # Output logits
    logits = hidden_states[-1] @ W_out.T  # (10,)
    pred = logits.argmax().item()

    print(f"\n--- Impulse at position {imp_pos} ---")
    print(f"  Prediction: position {pred} (logits argmax)")
    print(f"  Top-3 logits: ", end="")
    vals, idxs = logits.topk(3)
    for v, i in zip(vals, idxs):
        print(f"pos{i.item()}={v.item():.2f}  ", end="")
    print()

    # Show which neurons are active (non-zero) at each timestep
    active_counts = (hidden_states > 0).sum(dim=1)
    print(f"  Active neurons per step: {[c.item() for c in active_counts]}")

    # Show key neuron trajectories
    for n in [0, 2, 4, 7, 10, 11, 12, 14]:
        traj = [f"{hidden_states[t, n].item():6.2f}" for t in range(seq_len)]
        clip_str = ["C" if clipped[t, n] else "." for t in range(seq_len)]
        print(f"  n{n:2d}: {' '.join(traj)}  clip: {''.join(clip_str)}")

# ============================================================
# Impulse response summary: h_final for each impulse position
# ============================================================
print("\n\n" + "=" * 80)
print("IMPULSE RESPONSE SUMMARY: h_final[neuron] for impulse at each position")
print("=" * 80)

h_finals = []
for imp_pos in range(seq_len):
    x = impulses[imp_pos].unsqueeze(0).unsqueeze(-1)
    h = th.zeros(1, n_neurons)
    for t in range(seq_len):
        x_t = x[:, t, :]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
    h_finals.append(h.squeeze(0).clone())

h_finals = th.stack(h_finals)  # (10, 16)

# Print as table
header = "imp_pos " + " ".join([f"  n{n:2d}" for n in range(n_neurons)])
print(header)
for imp_pos in range(seq_len):
    row = f"  t={imp_pos}   "
    for n in range(n_neurons):
        val = h_finals[imp_pos, n].item()
        row += f"{val:6.2f}"
    print(row)

# Output logits for each impulse
print("\n\nOUTPUT LOGITS for each impulse position:")
print("imp_pos " + " ".join([f" p{p}" for p in range(seq_len)]))
for imp_pos in range(seq_len):
    logits = h_finals[imp_pos] @ W_out.T
    row = f"  t={imp_pos}   "
    for p in range(seq_len):
        row += f"{logits[p].item():6.1f}"
    print(row)

# ============================================================
# PART 2: N0 AND N14 CONNECTIVITY
# ============================================================
print("\n\n" + "=" * 80)
print("PART 2: N0 AND N14 CONNECTIVITY ANALYSIS")
print("=" * 80)

for target_n in [0, 14]:
    print(f"\n{'='*60}")
    print(f"NEURON n{target_n}")
    print(f"{'='*60}")

    print(f"\n  W_ih[{target_n}] = {W_ih[target_n].item():.4f}")
    print(f"  W_hh[{target_n},{target_n}] (self-recurrence) = {W_hh[target_n, target_n].item():.4f}")

    # Incoming connections (who feeds into this neuron)
    print(f"\n  INCOMING connections (W_hh[{target_n}, :]):")
    incoming = W_hh[target_n, :]
    sorted_idx = incoming.abs().argsort(descending=True)
    for i in sorted_idx:
        val = incoming[i].item()
        if abs(val) > 0.1:
            print(f"    n{i.item():2d} → n{target_n}: {val:+.4f}")

    # Outgoing connections (who this neuron feeds)
    print(f"\n  OUTGOING connections (W_hh[:, {target_n}]):")
    outgoing = W_hh[:, target_n]
    sorted_idx = outgoing.abs().argsort(descending=True)
    for i in sorted_idx:
        val = outgoing[i].item()
        if abs(val) > 0.1:
            print(f"    n{target_n} → n{i.item():2d}: {val:+.4f}")

    # W_out contribution
    print(f"\n  W_out[:, {target_n}] (output weights per position):")
    wout = W_out[:, target_n]
    for p in range(seq_len):
        bar = "+" * int(abs(wout[p].item()) * 5) if wout[p].item() > 0 else "-" * int(abs(wout[p].item()) * 5)
        print(f"    pos {p}: {wout[p].item():+.4f}  {bar}")

    # DFT of W_out column
    wout_np = wout.numpy()
    dft = np.fft.rfft(wout_np)
    dft_mag = np.abs(dft)
    total_energy = np.sum(dft_mag[1:]**2)  # exclude DC
    print(f"\n  DFT of W_out[:, {target_n}]:")
    for k in range(len(dft)):
        energy_pct = (dft_mag[k]**2 / total_energy * 100) if total_energy > 0 and k > 0 else 0
        print(f"    k={k}: magnitude={dft_mag[k]:.3f}  phase={np.angle(dft[k]):.2f} rad  energy={energy_pct:.1f}%")

# ============================================================
# N0 and N14 behavior on random data
# ============================================================
print("\n\n" + "=" * 80)
print("N0 AND N14 BEHAVIOR ON RANDOM DATA (100k samples)")
print("=" * 80)

th.manual_seed(42)
N = 100000
x = th.rand(N, seq_len)

# Manual forward pass
h = th.zeros(N, n_neurons)
all_hidden = th.zeros(N, n_neurons, seq_len)
all_clipped = th.zeros(N, n_neurons, seq_len, dtype=th.bool)

for t in range(seq_len):
    x_t = x[:, t:t+1]
    pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
    all_clipped[:, :, t] = pre < 0
    h = th.relu(pre)
    all_hidden[:, :, t] = h

h_final = all_hidden[:, :, -1]  # (N, 16)

# Task ground truth
vals_sorted, _ = x.sort(dim=1, descending=True)
max_vals = vals_sorted[:, 0]
second_vals = vals_sorted[:, 1]
argmax_pos = x.argmax(dim=1)

# 2nd argmax
x_masked = x.clone()
x_masked[th.arange(N), argmax_pos] = -1
second_argmax_pos = x_masked.argmax(dim=1)

for target_n in [0, 14]:
    print(f"\n--- Neuron n{target_n} ---")

    h_n = h_final[:, target_n]

    # Clipping statistics
    clip_rates = all_clipped[:, target_n, :].float().mean(dim=0)
    print(f"  Clip rate per timestep: {[f'{r:.2f}' for r in clip_rates.tolist()]}")
    print(f"  Mean h_final: {h_n.mean():.3f}, std: {h_n.std():.3f}")

    # Correlations with key variables
    def corr(a, b):
        a, b = a.float(), b.float()
        a = a - a.mean()
        b = b - b.mean()
        return (a * b).mean() / (a.std() * b.std() + 1e-10)

    print(f"  Correlations:")
    print(f"    r(h_final, max_val)       = {corr(h_n, max_vals):.4f}")
    print(f"    r(h_final, 2nd_val)       = {corr(h_n, second_vals):.4f}")
    print(f"    r(h_final, argmax_pos)    = {corr(h_n, argmax_pos.float()):.4f}")
    print(f"    r(h_final, 2nd_argmax_pos)= {corr(h_n, second_argmax_pos.float()):.4f}")
    print(f"    r(h_final, max-2nd gap)   = {corr(h_n, max_vals - second_vals):.4f}")
    print(f"    r(h_final, sum of inputs) = {corr(h_n, x.sum(dim=1)):.4f}")

    # Correlation with each individual input
    print(f"  Correlation with each x[t]:")
    for t in range(seq_len):
        r = corr(h_n, x[:, t])
        bar = "+" * int(abs(r.item()) * 50)
        sign = "+" if r > 0 else "-"
        print(f"    x[{t}]: r={r.item():+.4f}  {sign}{bar}")

    # Correlation with other neurons at h_final
    print(f"  Correlation with other neurons at h_final:")
    for other_n in range(n_neurons):
        if other_n == target_n:
            continue
        r = corr(h_n, h_final[:, other_n])
        if abs(r.item()) > 0.15:
            print(f"    n{other_n:2d}: r={r.item():+.4f}")

    # Mean h_final by argmax position
    print(f"  Mean h_final by argmax position:")
    for p in range(seq_len):
        mask = argmax_pos == p
        if mask.sum() > 0:
            m = h_n[mask].mean().item()
            print(f"    argmax={p}: mean={m:.3f} (n={mask.sum().item()})")

    # Mean h_final by 2nd_argmax position
    print(f"  Mean h_final by 2nd_argmax position:")
    for p in range(seq_len):
        mask = second_argmax_pos == p
        if mask.sum() > 0:
            m = h_n[mask].mean().item()
            print(f"    2nd_argmax={p}: mean={m:.3f} (n={mask.sum().item()})")

    # Contribution to output (h_final * W_out)
    contrib = h_n.unsqueeze(1) * W_out[:, target_n].unsqueeze(0)  # (N, 10)
    mean_contrib = contrib.mean(dim=0)
    print(f"  Mean output contribution (h * W_out) per position:")
    for p in range(seq_len):
        print(f"    pos {p}: {mean_contrib[p].item():+.3f}")


# ============================================================
# N0 and N14: conditional analysis
# ============================================================
print("\n\n" + "=" * 80)
print("N0 AND N14: CONDITIONAL ON CORRECTNESS")
print("=" * 80)

with th.no_grad():
    logits = model(x)
    preds = logits.argmax(dim=-1)
correct = preds == second_argmax_pos

for target_n in [0, 14]:
    h_n = h_final[:, target_n]
    print(f"\n--- n{target_n} ---")
    print(f"  Mean h_final (correct):   {h_n[correct].mean():.3f}")
    print(f"  Mean h_final (incorrect): {h_n[~correct].mean():.3f}")

    # Contribution to correct logit when correct vs incorrect
    contrib_correct_logit = th.zeros(N)
    for i in range(N):
        target_pos = second_argmax_pos[i].item()
        contrib_correct_logit[i] = h_n[i] * W_out[target_pos, target_n]

    print(f"  Mean contrib to correct logit (correct):   {contrib_correct_logit[correct].mean():.3f}")
    print(f"  Mean contrib to correct logit (incorrect): {contrib_correct_logit[~correct].mean():.3f}")
