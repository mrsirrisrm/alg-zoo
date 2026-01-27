"""
Clipping Analysis for the 2nd Argmax Model

Analyzes how ReLU clipping patterns encode information about large values
in the M_16,10 model (432 parameters).
"""

import torch as th
import numpy as np
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def get_clipping_pattern(model, x):
    """
    Compute binary clipping pattern for each neuron at each timestep.

    Returns:
        clipped: [batch, 16, 10] bool tensor where True means pre-activation < 0
    """
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data

    batch_size = x.shape[0]
    clipped = th.zeros(batch_size, 16, 10, dtype=th.bool)
    h = th.zeros(batch_size, 16)

    for t in range(10):
        x_t = x[:, t:t+1]
        pre_act = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        clipped[:, :, t] = pre_act < 0
        h = th.relu(pre_act)

    return clipped


def get_full_trajectory(model, x):
    """
    Compute full trajectory including pre-activations and hidden states.

    Returns:
        clipped: [batch, 16, 10] bool
        preact: [batch, 16, 10] float
        hidden: [batch, 16, 10] float
    """
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data

    batch_size = x.shape[0]
    clipped = th.zeros(batch_size, 16, 10, dtype=th.bool)
    preact = th.zeros(batch_size, 16, 10)
    hidden = th.zeros(batch_size, 16, 10)
    h = th.zeros(batch_size, 16)

    for t in range(10):
        x_t = x[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        preact[:, :, t] = pre
        clipped[:, :, t] = pre < 0
        h = th.relu(pre)
        hidden[:, :, t] = h

    return clipped, preact, hidden


def compute_clipping_threshold(model, hidden_prev, neuron_idx):
    """
    Compute the input threshold at which a neuron clips.

    For neuron n with negative W_ih:
        threshold = -W_hh[n,:] @ h_prev / W_ih[n]

    If x[t] > threshold, the neuron clips.
    """
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data

    recurrent_contrib = hidden_prev @ W_hh[neuron_idx, :]
    threshold = -recurrent_contrib / W_ih[neuron_idx]
    return threshold


def analyze_clipping_vs_correctness(model, n_samples=100000):
    """Analyze correlation between clipping patterns and prediction correctness."""
    x = th.rand(n_samples, 10)
    clipped = get_clipping_pattern(model, x)

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
        targets = task_2nd_argmax(x)
        correct = preds == targets

    print("Correlation between clipping at t=9 and correctness:")
    print("(Positive = clipped correlates with CORRECT)")

    for n in range(16):
        clip_n_9 = clipped[:, n, 9].float().numpy()
        correct_np = correct.float().numpy()

        if clip_n_9.std() > 0.01:
            corr = np.corrcoef(clip_n_9, correct_np)[0, 1]
            print(f"  Neuron {n:2d}: corr = {corr:+.4f}")


def analyze_adaptive_threshold(model, n_samples=50000):
    """Verify that clipping threshold tracks the running maximum."""
    x = th.rand(n_samples, 10)
    _, _, hidden = get_full_trajectory(model, x)

    print("Correlation between neuron 7 threshold and max(x[0:t-1]):")

    for t in range(1, 10):
        h_prev = hidden[:, :, t-1]
        threshold = compute_clipping_threshold(model, h_prev, 7)
        running_max = x[:, :t].max(dim=-1).values

        corr = np.corrcoef(threshold.numpy(), running_max.numpy())[0, 1]
        print(f"  t={t}: corr = {corr:.4f}")


def analyze_clipping_as_max_detector(model, n_samples=50000):
    """Check how well clipping predicts whether current position is the max."""
    x = th.rand(n_samples, 10)
    clipped = get_clipping_pattern(model, x)
    argmax_pos = x.argmax(dim=-1)

    print("P(neuron 7 clips | position is max) vs P(clips | not max):")

    for t in range(10):
        is_max_at_t = argmax_pos == t
        n7_clipped = clipped[:, 7, t]

        if is_max_at_t.sum() > 100:
            p_clip_given_max = n7_clipped[is_max_at_t].float().mean().item()
            p_clip_given_not = n7_clipped[~is_max_at_t].float().mean().item()
            print(f"  t={t}: P(clip|max)={p_clip_given_max:.1%}, P(clip|¬max)={p_clip_given_not:.1%}")


def main():
    model = example_2nd_argmax()

    print("=" * 70)
    print("CLIPPING ANALYSIS FOR 2ND ARGMAX MODEL")
    print("=" * 70)

    # Basic info
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    print("\nInput weights (W_ih):")
    for i in range(16):
        if abs(W_ih[i].item()) > 5:
            print(f"  Neuron {i:2d}: {W_ih[i].item():+.2f}")

    print("\n" + "-" * 70)
    analyze_adaptive_threshold(model)

    print("\n" + "-" * 70)
    analyze_clipping_as_max_detector(model)

    print("\n" + "-" * 70)
    analyze_clipping_vs_correctness(model)


if __name__ == "__main__":
    main()
