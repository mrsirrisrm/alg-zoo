"""
Padding Experiment for the 2nd Argmax Model

Tests whether adding a small trailing value improves accuracy by preventing
information loss from ReLU clipping at the final timestep.
"""

import torch as th
from alg_zoo import example_2nd_argmax, task_2nd_argmax

from clipping_analysis import get_clipping_pattern


def evaluate_with_padding(model, n_samples=100000, pad_value=0.01):
    """
    Evaluate model accuracy with padded sequences.

    Args:
        model: The 2nd argmax model
        n_samples: Number of test samples
        pad_value: Value to use for padding at position 9

    Returns:
        dict with accuracy and other statistics
    """
    # Generate 9 random values + padding
    x_9 = th.rand(n_samples, 9)
    x_padded = th.cat([x_9, th.full((n_samples, 1), pad_value)], dim=1)

    targets = task_2nd_argmax(x_padded)

    with th.no_grad():
        preds = model(x_padded).argmax(dim=-1)
        correct = preds == targets

    # Check if padding ever in top 2
    is_top2 = (x_padded.argsort(dim=-1)[:, -2:] == 9).any(dim=1)

    return {
        'accuracy': correct.float().mean().item(),
        'pad_in_top2': is_top2.float().mean().item(),
        'n_samples': n_samples,
    }


def evaluate_baseline(model, n_samples=100000):
    """Evaluate model on standard random sequences."""
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
        correct = preds == targets

    argmax_pos = x.argmax(dim=-1)

    return {
        'accuracy': correct.float().mean().item(),
        'accuracy_max_at_9': correct[argmax_pos == 9].float().mean().item(),
        'accuracy_max_not_9': correct[argmax_pos != 9].float().mean().item(),
        'n_samples': n_samples,
    }


def compare_clipping_rates(model, n_samples=100000):
    """Compare clipping rates at t=9 between original and padded sequences."""
    # Original
    x_orig = th.rand(n_samples, 10)
    clip_orig = get_clipping_pattern(model, x_orig)

    # Padded
    x_9 = th.rand(n_samples, 9)
    x_pad = th.cat([x_9, th.full((n_samples, 1), 0.01)], dim=1)
    clip_pad = get_clipping_pattern(model, x_pad)

    print("Clipping rate at t=9:")
    print("Neuron   Original   Padded")
    for n in [1, 4, 6, 7, 8]:
        rate_orig = clip_orig[:, n, 9].float().mean().item()
        rate_pad = clip_pad[:, n, 9].float().mean().item()
        print(f"  {n}:      {rate_orig:.1%}      {rate_pad:.1%}")


def analyze_x9_effect(model, n_samples=100000):
    """Analyze how the value of x[9] affects accuracy even when it's not the max."""
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
        correct = preds == targets

    argmax_pos = x.argmax(dim=-1)
    mask_not_max = argmax_pos != 9

    print("Accuracy by x[9] value (when x[9] is NOT the max):")
    for thresh in [0.3, 0.5, 0.7]:
        mask = mask_not_max & (x[:, 9] > thresh)
        if mask.sum() > 100:
            acc = correct[mask].float().mean().item()
            print(f"  x[9] > {thresh}: {acc:.2%} ({mask.sum().item()} samples)")


def accuracy_by_position(model, x, pad_label=""):
    """Compute accuracy broken down by 2nd argmax position."""
    targets = task_2nd_argmax(x)

    with th.no_grad():
        preds = model(x).argmax(dim=-1)
        correct = preds == targets

    print(f"Accuracy by 2nd argmax position {pad_label}:")
    for pos in range(10):
        mask = targets == pos
        if mask.sum() > 100:
            acc = correct[mask].float().mean().item()
            print(f"  Position {pos}: {acc:.2%}")


def main():
    model = example_2nd_argmax()

    print("=" * 70)
    print("PADDING EXPERIMENT")
    print("=" * 70)

    # Baseline
    print("\n1. BASELINE (10 random values)")
    print("-" * 50)
    baseline = evaluate_baseline(model)
    print(f"Overall accuracy: {baseline['accuracy']:.2%}")
    print(f"Accuracy when max at pos 9: {baseline['accuracy_max_at_9']:.2%}")
    print(f"Accuracy when max NOT at pos 9: {baseline['accuracy_max_not_9']:.2%}")

    # Padded
    print("\n2. PADDED (9 random + small constant)")
    print("-" * 50)
    for pad_val in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]:
        result = evaluate_with_padding(model, pad_value=pad_val)
        print(f"pad={pad_val:.2f}: accuracy={result['accuracy']:.2%}, "
              f"pad in top-2: {result['pad_in_top2']:.1%}")

    # Clipping comparison
    print("\n3. CLIPPING RATE COMPARISON")
    print("-" * 50)
    compare_clipping_rates(model)

    # Effect of x[9] value
    print("\n4. EFFECT OF x[9] VALUE")
    print("-" * 50)
    analyze_x9_effect(model)

    # Summary
    print("\n5. SUMMARY")
    print("-" * 50)
    padded = evaluate_with_padding(model, pad_value=0.01)
    improvement = padded['accuracy'] - baseline['accuracy']
    print(f"Original accuracy:  {baseline['accuracy']:.2%}")
    print(f"Padded accuracy:    {padded['accuracy']:.2%}")
    print(f"Improvement:        {improvement:+.2%}")


if __name__ == "__main__":
    main()
