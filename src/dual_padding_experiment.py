"""
Dual Padding Experiment for the 2nd Argmax Model

Tests whether padding at BOTH ends (initial + trailing) provides
additive accuracy improvements.
"""

import torch as th
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def evaluate_accuracy(model, x):
    """Evaluate model accuracy on given inputs."""
    targets = task_2nd_argmax(x)
    with th.no_grad():
        preds = model(x).argmax(dim=-1)
    return (preds == targets).float().mean().item()


def run_experiment(n_samples=100000):
    """Run the dual padding experiment."""
    model = example_2nd_argmax()

    print("=" * 70)
    print("DUAL PADDING EXPERIMENT")
    print("=" * 70)

    # Generate base random values
    x_10 = th.rand(n_samples, 10)  # For original
    x_9 = th.rand(n_samples, 9)    # For single padding
    x_8 = th.rand(n_samples, 8)    # For dual padding

    pad_val = 0.01

    # Original (10 random)
    acc_orig = evaluate_accuracy(model, x_10)

    # Trailing padding (9 random + small)
    x_trail = th.cat([x_9, th.full((n_samples, 1), pad_val)], dim=1)
    acc_trail = evaluate_accuracy(model, x_trail)

    # Initial padding (small + 9 random)
    x_init = th.cat([th.full((n_samples, 1), pad_val), x_9], dim=1)
    acc_init = evaluate_accuracy(model, x_init)

    # Both padding (small + 8 random + small)
    x_both = th.cat([
        th.full((n_samples, 1), pad_val),
        x_8,
        th.full((n_samples, 1), pad_val)
    ], dim=1)
    acc_both = evaluate_accuracy(model, x_both)

    print(f"\n1. ACCURACY SUMMARY")
    print("-" * 50)
    print(f"Original (10 random):      {acc_orig:.2%}")
    print(f"Trailing only:             {acc_trail:.2%} (+{acc_trail-acc_orig:.2%})")
    print(f"Initial only:              {acc_init:.2%} (+{acc_init-acc_orig:.2%})")
    print(f"Both:                      {acc_both:.2%} (+{acc_both-acc_orig:.2%})")

    # Check additivity
    expected_both = acc_orig + (acc_trail - acc_orig) + (acc_init - acc_orig)
    print(f"\nExpected if additive:      {expected_both:.2%}")
    print(f"Actual both:               {acc_both:.2%}")
    print(f"Synergy:                   {acc_both - expected_both:+.2%}")

    # Accuracy by position for each configuration
    print(f"\n2. ACCURACY BY 2ND ARGMAX POSITION")
    print("-" * 50)

    configs = [
        ("Original", x_10),
        ("Trail", x_trail),
        ("Init", x_init),
        ("Both", x_both),
    ]

    print("Position  " + "  ".join(f"{name:>8}" for name, _ in configs))

    for pos in range(10):
        accs = []
        for name, x in configs:
            targets = task_2nd_argmax(x)
            with th.no_grad():
                preds = model(x).argmax(dim=-1)
            correct = preds == targets
            mask = targets == pos

            if mask.sum() > 100:
                acc = correct[mask].float().mean().item()
                accs.append(f"{acc:>7.1%}")
            else:
                accs.append("      -")

        print(f"    {pos}:    {'  '.join(accs)}")

    # Analyze why initial padding helps
    print(f"\n3. WHY INITIAL PADDING HELPS")
    print("-" * 50)

    argmax_orig = x_10.argmax(dim=-1)
    targets_orig = task_2nd_argmax(x_10)
    with th.no_grad():
        preds_orig = model(x_10).argmax(dim=-1)
    correct_orig = preds_orig == targets_orig

    acc_max_at_0 = correct_orig[argmax_orig == 0].float().mean().item()
    acc_max_not_0 = correct_orig[argmax_orig != 0].float().mean().item()

    print(f"Original accuracy when max at pos 0:     {acc_max_at_0:.2%}")
    print(f"Original accuracy when max NOT at pos 0: {acc_max_not_0:.2%}")
    print(f"\nWith initial padding, max is never at position 0,")
    print(f"eliminating the lower-accuracy edge case.")

    return {
        'original': acc_orig,
        'trailing': acc_trail,
        'initial': acc_init,
        'both': acc_both,
    }


if __name__ == "__main__":
    run_experiment()
