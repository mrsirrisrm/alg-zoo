"""
ADC-like Circuit Analysis for the 2nd Argmax Model

Investigates how the model combines:
- Pure analog neurons (never clip) for continuous value tracking
- Binary-like neurons (1, 6, 7, 8) that act as comparators with different thresholds
"""

import torch as th
import numpy as np
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def get_full_trajectory(model, x):
    """Compute full trajectory including pre-activations and hidden states."""
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
    threshold = -W_hh[n,:] @ h_prev / W_ih[n]
    """
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data

    recurrent_contrib = hidden_prev @ W_hh[neuron_idx, :]
    threshold = -recurrent_contrib / W_ih[neuron_idx]
    return threshold


def analyze_neuron_behaviors(model, n_samples=50000):
    """Categorize neurons by their clipping behavior."""
    x = th.rand(n_samples, 10)
    clipped, preact, hidden = get_full_trajectory(model, x)

    W_ih = model.rnn.weight_ih_l0.data.squeeze()

    print("=" * 70)
    print("NEURON BEHAVIOR CLASSIFICATION")
    print("=" * 70)

    print("\n1. INPUT WEIGHTS (W_ih)")
    print("-" * 50)
    print("Neuron   W_ih      Type")

    analog_neurons = []
    binary_neurons = []

    for n in range(16):
        w = W_ih[n].item()
        clip_rate = clipped[:, n, :].float().mean().item()

        # Classify based on W_ih magnitude and clip rate
        if abs(w) < 1.0:
            neuron_type = "analog (weak input)"
            analog_neurons.append(n)
        elif clip_rate < 0.05 or clip_rate > 0.95:
            neuron_type = "saturated"
        elif w < -5:
            neuron_type = "BINARY (large-value detector)"
            binary_neurons.append(n)
        elif w > 5:
            neuron_type = "BINARY (small-value detector)"
            binary_neurons.append(n)
        else:
            neuron_type = "mixed"

        print(f"  {n:2d}:    {w:+7.2f}   {neuron_type}")

    return analog_neurons, binary_neurons


def analyze_threshold_diversity(model, n_samples=50000):
    """Check if different binary neurons have different thresholds."""
    x = th.rand(n_samples, 10)
    _, _, hidden = get_full_trajectory(model, x)

    W_ih = model.rnn.weight_ih_l0.data.squeeze()

    # Focus on the large-value detectors
    detectors = [n for n in range(16) if W_ih[n].item() < -5]

    print("\n2. THRESHOLD DIVERSITY (ADC-like behavior)")
    print("-" * 50)
    print("Do different neurons have different clipping thresholds?")
    print("\nThreshold statistics at t=5 (mid-sequence):")
    print("Neuron   W_ih      Mean_thresh   Std_thresh    Correlation")

    thresholds = {}
    for n in detectors:
        h_prev = hidden[:, :, 4]  # State before t=5
        thresh = compute_clipping_threshold(model, h_prev, n)
        thresholds[n] = thresh.numpy()

        running_max = x[:, :5].max(dim=-1).values.numpy()
        corr = np.corrcoef(thresh.numpy(), running_max)[0, 1]

        print(f"  {n:2d}:    {W_ih[n].item():+7.2f}   {thresh.mean().item():.3f}         "
              f"{thresh.std().item():.3f}         {corr:.3f}")

    # Check if thresholds are different from each other
    print("\nInter-neuron threshold correlations:")
    detector_list = list(detectors)
    for i, n1 in enumerate(detector_list):
        for n2 in detector_list[i+1:]:
            corr = np.corrcoef(thresholds[n1], thresholds[n2])[0, 1]
            diff = np.mean(np.abs(thresholds[n1] - thresholds[n2]))
            print(f"  n{n1} vs n{n2}: corr={corr:.3f}, mean_diff={diff:.3f}")


def analyze_quantization_levels(model, n_samples=50000):
    """
    Analyze whether the binary neurons create quantization levels.

    In a true ADC, different comparators fire at different input levels,
    creating a binary encoding of the input magnitude.
    """
    x = th.rand(n_samples, 10)
    clipped, _, hidden = get_full_trajectory(model, x)

    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    detectors = [n for n in range(16) if W_ih[n].item() < -5]

    print("\n3. QUANTIZATION ANALYSIS")
    print("-" * 50)
    print("At each timestep, which neurons clip given the input value?")

    # For t=5, bin inputs by value and see which neurons clip
    t = 5
    x_val = x[:, t].numpy()

    print(f"\nAt t={t}, clipping probability by input value:")
    print("x_range       " + "  ".join(f"n{n}" for n in detectors))

    for low, high in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (x_val >= low) & (x_val < high)
        if mask.sum() > 100:
            rates = []
            for n in detectors:
                rate = clipped[mask, n, t].float().mean().item()
                rates.append(f"{rate:.0%}")
            print(f"[{low:.1f}, {high:.1f})    " + "   ".join(rates))


def analyze_running_max_encoding(model, n_samples=50000):
    """
    Check if the clipping pattern encodes the running maximum.

    Key insight: the threshold adapts to track the running max,
    so the pattern of which neurons clip encodes information about
    how the current input compares to the history.
    """
    x = th.rand(n_samples, 10)
    clipped, _, _ = get_full_trajectory(model, x)

    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    detectors = [n for n in range(16) if W_ih[n].item() < -5]

    print("\n4. RUNNING MAX ENCODING")
    print("-" * 50)
    print("How well does the clipping pattern predict if current x > running_max?")

    for t in range(2, 10):
        running_max = x[:, :t].max(dim=-1).values
        is_new_max = x[:, t] > running_max

        # Count how many detectors clip
        n_clipped = clipped[:, detectors, t].sum(dim=-1)

        # When x[t] > running_max, how many clip?
        n_clip_when_max = n_clipped[is_new_max].float().mean().item()
        n_clip_when_not = n_clipped[~is_new_max].float().mean().item()

        print(f"t={t}: clip_count when new_max={n_clip_when_max:.1f}/{len(detectors)}, "
              f"when not={n_clip_when_not:.1f}/{len(detectors)}")


def analyze_analog_neuron(model, n_samples=50000):
    """
    Analyze the pure analog neuron (n2) that never clips.
    What information does it preserve?
    """
    x = th.rand(n_samples, 10)
    _, _, hidden = get_full_trajectory(model, x)

    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data

    print("\n5. ANALOG NEURON ANALYSIS (n2)")
    print("-" * 50)

    # Find neurons with small W_ih (pure analog)
    analog_neurons = [n for n in range(16) if abs(W_ih[n].item()) < 1.0]
    print(f"Analog neurons (|W_ih| < 1): {analog_neurons}")

    for n in analog_neurons:
        print(f"\nNeuron {n}:")
        print(f"  W_ih[{n}] = {W_ih[n].item():.4f}")

        # What does the hidden state correlate with?
        h_final = hidden[:, n, 9].numpy()

        # Test correlations
        max_val = x.max(dim=-1).values.numpy()
        second_max = th.topk(x, 2, dim=-1).values[:, 1].numpy()
        sum_val = x.sum(dim=-1).numpy()
        mean_val = x.mean(dim=-1).numpy()

        corr_max = np.corrcoef(h_final, max_val)[0, 1]
        corr_2nd = np.corrcoef(h_final, second_max)[0, 1]
        corr_sum = np.corrcoef(h_final, sum_val)[0, 1]
        corr_mean = np.corrcoef(h_final, mean_val)[0, 1]

        print(f"  Correlation of h[{n}] at t=9 with:")
        print(f"    max(x):      {corr_max:.3f}")
        print(f"    2nd_max(x):  {corr_2nd:.3f}")
        print(f"    sum(x):      {corr_sum:.3f}")
        print(f"    mean(x):     {corr_mean:.3f}")


def analyze_combined_representation(model, n_samples=50000):
    """
    Analyze how the analog and digital components combine.
    """
    x = th.rand(n_samples, 10)
    clipped, _, hidden = get_full_trajectory(model, x)

    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    detectors = [n for n in range(16) if W_ih[n].item() < -5]

    print("\n6. COMBINED ANALOG-DIGITAL REPRESENTATION")
    print("-" * 50)

    # The final hidden state combines:
    # - Analog values from non-clipping neurons
    # - Binary-ish values from frequently-clipping neurons

    h_final = hidden[:, :, 9]

    # Look at the distribution of values for each neuron
    print("Final hidden state statistics:")
    print("Neuron   Mean      Std       Min       Max       Type")

    for n in range(16):
        h_n = h_final[:, n]
        w = W_ih[n].item()

        if n in detectors:
            ntype = "BINARY"
        elif abs(w) < 1.0:
            ntype = "analog"
        else:
            ntype = "mixed"

        print(f"  {n:2d}:   {h_n.mean().item():7.3f}   {h_n.std().item():7.3f}   "
              f"{h_n.min().item():7.3f}   {h_n.max().item():7.3f}   {ntype}")


def main():
    model = example_2nd_argmax()

    print("=" * 70)
    print("ADC-LIKE CIRCUIT ANALYSIS")
    print("=" * 70)
    print("\nHypothesis: The model combines pure analog neurons with")
    print("binary-like comparators to create an ADC-like encoding.\n")

    analog, binary = analyze_neuron_behaviors(model)
    analyze_threshold_diversity(model)
    analyze_quantization_levels(model)
    analyze_running_max_encoding(model)
    analyze_analog_neuron(model)
    analyze_combined_representation(model)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nAnalog neurons (weak input coupling): {analog}")
    print(f"Binary neurons (large-value detectors): {binary}")
    print("\nThe model appears to use a hybrid analog-digital encoding where:")
    print("1. Binary neurons detect whether x[t] > threshold (where threshold ≈ running_max)")
    print("2. Different binary neurons may have slightly different thresholds")
    print("3. Analog neurons preserve continuous value information")
    print("4. The combination allows distinguishing between similar large values")


if __name__ == "__main__":
    main()
