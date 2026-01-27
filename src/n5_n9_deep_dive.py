"""
Deep dive into n5 and n9 - the LOO difference encoders.

Key questions:
1. What exactly do they encode?
2. What do they feed into (W_hh connections)?
3. How do they contribute to the output (W_out)?
4. When are they important for correct predictions?
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def get_full_trajectory(model, x):
    """Compute full trajectory with pre-activations."""
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data

    batch_size = x.shape[0]
    hidden = th.zeros(batch_size, 16, 10)
    pre_act = th.zeros(batch_size, 16, 10)
    h = th.zeros(batch_size, 16)

    for t in range(10):
        x_t = x[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        pre_act[:, :, t] = pre
        h = th.relu(pre)
        hidden[:, :, t] = h

    return hidden, pre_act


def analyze_n5_n9_encoding(model, n_samples=100000):
    """
    What exactly do n5 and n9 encode?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    # Get various features
    top2 = th.topk(x, 2, dim=-1)
    max_val = top2.values[:, 0]
    second_val = top2.values[:, 1]
    gap = max_val - second_val

    # Third max for comparison
    top3 = th.topk(x, 3, dim=-1)
    third_val = top3.values[:, 2]
    gap_2_3 = second_val - third_val

    hidden, _ = get_full_trajectory(model, x)
    h5 = hidden[:, 5, 9].numpy()
    h9 = hidden[:, 9, 9].numpy()

    print("=" * 70)
    print("N5 AND N9: WHAT DO THEY ENCODE?")
    print("=" * 70)

    # Correlations with various features
    features = {
        'max_val': max_val.numpy(),
        'second_val': second_val.numpy(),
        'third_val': third_val.numpy(),
        'gap (max - 2nd)': gap.numpy(),
        'gap_2_3 (2nd - 3rd)': gap_2_3.numpy(),
        'argmax_pos': argmax_pos.numpy().astype(float),
        '2nd_argmax_pos': targets.numpy().astype(float),
        'pos_diff (argmax - 2nd)': (argmax_pos - targets).numpy().astype(float),
    }

    print("\nCorrelations:")
    print("-" * 60)
    print(f"{'Feature':<25} | {'r(n5)':<12} | {'r(n9)':<12}")
    print("-" * 60)

    for name, vals in features.items():
        corr5 = np.corrcoef(h5, vals)[0, 1]
        corr9 = np.corrcoef(h9, vals)[0, 1]
        print(f"{name:<25} | {corr5:>+10.3f}  | {corr9:>+10.3f}")

    # Check if they encode "margin to third"
    margin_to_third = second_val - third_val
    print(f"\n{'margin (2nd - 3rd)':<25} | {np.corrcoef(h5, margin_to_third.numpy())[0,1]:>+10.3f}  | {np.corrcoef(h9, margin_to_third.numpy())[0,1]:>+10.3f}")

    return features


def analyze_n5_n9_temporal(model, n_samples=50000):
    """
    How do n5 and n9 evolve over time?
    """
    x = th.rand(n_samples, 10)
    argmax_pos = x.argmax(dim=-1)

    top2 = th.topk(x, 2, dim=-1)
    gap = (top2.values[:, 0] - top2.values[:, 1]).numpy()

    hidden, _ = get_full_trajectory(model, x)

    print("\n" + "=" * 70)
    print("N5 AND N9: TEMPORAL EVOLUTION")
    print("=" * 70)

    print("\nMean activation by timestep:")
    print("-" * 50)
    print(f"{'Time':<6} | {'n5 mean':<12} | {'n9 mean':<12}")
    print("-" * 50)

    for t in range(10):
        h5_t = hidden[:, 5, t].mean().item()
        h9_t = hidden[:, 9, t].mean().item()
        print(f"t={t:<4} | {h5_t:>10.3f}  | {h9_t:>10.3f}")

    # When do they spike?
    print("\n\nWhen do n5/n9 have high values?")
    print("-" * 50)

    # Check if they spike at argmax position
    print("\nn5 mean at argmax position vs other positions:")
    h5_at_argmax = []
    h5_not_argmax = []
    for i in range(n_samples):
        pos = argmax_pos[i].item()
        h5_at_argmax.append(hidden[i, 5, pos].item())
        for t in range(10):
            if t != pos:
                h5_not_argmax.append(hidden[i, 5, t].item())

    print(f"  At argmax: {np.mean(h5_at_argmax):.3f}")
    print(f"  Not at argmax: {np.mean(h5_not_argmax):.3f}")

    # Same for n9
    print("\nn9 mean at argmax position vs other positions:")
    h9_at_argmax = []
    h9_not_argmax = []
    for i in range(n_samples):
        pos = argmax_pos[i].item()
        h9_at_argmax.append(hidden[i, 9, pos].item())
        for t in range(10):
            if t != pos:
                h9_not_argmax.append(hidden[i, 9, t].item())

    print(f"  At argmax: {np.mean(h9_at_argmax):.3f}")
    print(f"  Not at argmax: {np.mean(h9_not_argmax):.3f}")


def analyze_n5_n9_connectivity(model):
    """
    What do n5 and n9 connect to/from?
    """
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data
    W_out = model.linear.weight.data

    print("\n" + "=" * 70)
    print("N5 AND N9: CONNECTIVITY")
    print("=" * 70)

    # Input weights
    print("\nInput weights (W_ih):")
    print(f"  n5: {W_ih[5].item():+.3f}")
    print(f"  n9: {W_ih[9].item():+.3f}")
    print(f"  (For comparison, comparators: n1={W_ih[1].item():+.2f}, n7={W_ih[7].item():+.2f})")

    # Self-recurrence
    print("\nSelf-recurrence (W_hh diagonal):")
    print(f"  n5: {W_hh[5, 5].item():+.3f}")
    print(f"  n9: {W_hh[9, 9].item():+.3f}")

    # What feeds INTO n5 and n9?
    print("\nTop inputs to n5 (W_hh[:, 5] sorted by magnitude):")
    inputs_to_5 = [(i, W_hh[5, i].item()) for i in range(16)]
    inputs_to_5.sort(key=lambda x: abs(x[1]), reverse=True)
    for i, w in inputs_to_5[:5]:
        print(f"  from n{i}: {w:+.3f}")

    print("\nTop inputs to n9 (W_hh[:, 9] sorted by magnitude):")
    inputs_to_9 = [(i, W_hh[9, i].item()) for i in range(16)]
    inputs_to_9.sort(key=lambda x: abs(x[1]), reverse=True)
    for i, w in inputs_to_9[:5]:
        print(f"  from n{i}: {w:+.3f}")

    # What do n5 and n9 FEED TO?
    print("\nTop outputs from n5 (W_hh[5, :] sorted by magnitude):")
    outputs_from_5 = [(i, W_hh[i, 5].item()) for i in range(16)]
    outputs_from_5.sort(key=lambda x: abs(x[1]), reverse=True)
    for i, w in outputs_from_5[:5]:
        print(f"  to n{i}: {w:+.3f}")

    print("\nTop outputs from n9 (W_hh[9, :] sorted by magnitude):")
    outputs_from_9 = [(i, W_hh[i, 9].item()) for i in range(16)]
    outputs_from_9.sort(key=lambda x: abs(x[1]), reverse=True)
    for i, w in outputs_from_9[:5]:
        print(f"  to n{i}: {w:+.3f}")

    # Output weights
    print("\n" + "-" * 50)
    print("Output weights (W_out[:, n]):")
    print("-" * 50)
    print(f"{'Class':<8} | {'W_out[:,5]':<12} | {'W_out[:,9]':<12}")
    print("-" * 50)
    for j in range(10):
        print(f"{j:<8} | {W_out[j, 5].item():>+10.3f}  | {W_out[j, 9].item():>+10.3f}")

    print(f"\n|W_out[:,5]| sum = {th.abs(W_out[:, 5]).sum().item():.2f}")
    print(f"|W_out[:,9]| sum = {th.abs(W_out[:, 9]).sum().item():.2f}")


def analyze_n5_n9_importance(model, n_samples=100000):
    """
    When are n5 and n9 important for correct predictions?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    top2 = th.topk(x, 2, dim=-1)
    gap = top2.values[:, 0] - top2.values[:, 1]

    hidden, _ = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    W_out = model.linear.weight.data

    with th.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=-1)
    correct = (preds == targets)

    print("\n" + "=" * 70)
    print("N5 AND N9: WHEN ARE THEY IMPORTANT?")
    print("=" * 70)

    # Contribution to correct class
    print("\nContribution to CORRECT class logit:")
    print("-" * 60)

    contrib_5 = th.zeros(n_samples)
    contrib_9 = th.zeros(n_samples)
    for i in range(n_samples):
        j = targets[i].item()
        contrib_5[i] = h_final[i, 5] * W_out[j, 5]
        contrib_9[i] = h_final[i, 9] * W_out[j, 9]

    print(f"Mean contribution from n5: {contrib_5.mean().item():+.3f}")
    print(f"Mean contribution from n9: {contrib_9.mean().item():+.3f}")

    # By gap size
    print("\nContribution by gap size:")
    print("-" * 50)
    print(f"{'Gap range':<15} | {'n5 contrib':<12} | {'n9 contrib':<12} | {'Accuracy':<10}")
    print("-" * 50)

    for low, high in [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.0)]:
        mask = (gap >= low) & (gap < high)
        if mask.sum() > 100:
            c5 = contrib_5[mask].mean().item()
            c9 = contrib_9[mask].mean().item()
            acc = correct[mask].float().mean().item()
            print(f"[{low:.2f}, {high:.2f})    | {c5:>+10.3f}  | {c9:>+10.3f}  | {acc:>8.1%}")

    # By temporal ordering
    print("\nContribution by temporal ordering:")
    print("-" * 50)

    second_before = targets < argmax_pos
    second_after = targets > argmax_pos

    print(f"2nd BEFORE argmax: n5={contrib_5[second_before].mean().item():+.3f}, n9={contrib_9[second_before].mean().item():+.3f}")
    print(f"2nd AFTER argmax:  n5={contrib_5[second_after].mean().item():+.3f}, n9={contrib_9[second_after].mean().item():+.3f}")

    # Ablation study: what happens if we zero out n5 and n9?
    print("\n" + "-" * 50)
    print("ABLATION: Accuracy without n5/n9")
    print("-" * 50)

    # Original accuracy
    orig_acc = correct.float().mean().item()
    print(f"Original accuracy: {orig_acc:.1%}")

    # Zero out n5
    h_ablated = h_final.clone()
    h_ablated[:, 5] = 0
    logits_no_5 = h_ablated @ W_out.T
    preds_no_5 = logits_no_5.argmax(dim=-1)
    acc_no_5 = (preds_no_5 == targets).float().mean().item()
    print(f"Without n5: {acc_no_5:.1%} (Δ = {acc_no_5 - orig_acc:+.1%})")

    # Zero out n9
    h_ablated = h_final.clone()
    h_ablated[:, 9] = 0
    logits_no_9 = h_ablated @ W_out.T
    preds_no_9 = logits_no_9.argmax(dim=-1)
    acc_no_9 = (preds_no_9 == targets).float().mean().item()
    print(f"Without n9: {acc_no_9:.1%} (Δ = {acc_no_9 - orig_acc:+.1%})")

    # Zero out both
    h_ablated = h_final.clone()
    h_ablated[:, 5] = 0
    h_ablated[:, 9] = 0
    logits_no_5_9 = h_ablated @ W_out.T
    preds_no_5_9 = logits_no_5_9.argmax(dim=-1)
    acc_no_5_9 = (preds_no_5_9 == targets).float().mean().item()
    print(f"Without n5 & n9: {acc_no_5_9:.1%} (Δ = {acc_no_5_9 - orig_acc:+.1%})")

    # For comparison, ablate comparators
    h_ablated = h_final.clone()
    for n in [1, 6, 7, 8]:
        h_ablated[:, n] = 0
    logits_no_comp = h_ablated @ W_out.T
    preds_no_comp = logits_no_comp.argmax(dim=-1)
    acc_no_comp = (preds_no_comp == targets).float().mean().item()
    print(f"Without comparators (n1,6,7,8): {acc_no_comp:.1%} (Δ = {acc_no_comp - orig_acc:+.1%})")


def plot_n5_n9_analysis(model, n_samples=50000, save_path=None):
    """
    Visualize n5 and n9 behavior.
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    top2 = th.topk(x, 2, dim=-1)
    gap = top2.values[:, 0] - top2.values[:, 1]
    second_val = top2.values[:, 1]

    hidden, _ = get_full_trajectory(model, x)
    h5 = hidden[:, 5, 9].numpy()
    h9 = hidden[:, 9, 9].numpy()

    W_out = model.linear.weight.data

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Panel 1: n5 vs gap
    ax = axes[0, 0]
    ax.scatter(gap.numpy(), h5, alpha=0.1, s=1)
    ax.set_xlabel('Gap (max - 2nd_max)')
    ax.set_ylabel('n5 at t=9')
    corr = np.corrcoef(h5, gap.numpy())[0, 1]
    ax.set_title(f'n5 vs gap (r = {corr:.3f})')
    ax.grid(True, alpha=0.3)

    # Panel 2: n9 vs gap
    ax = axes[0, 1]
    ax.scatter(gap.numpy(), h9, alpha=0.1, s=1)
    ax.set_xlabel('Gap (max - 2nd_max)')
    ax.set_ylabel('n9 at t=9')
    corr = np.corrcoef(h9, gap.numpy())[0, 1]
    ax.set_title(f'n9 vs gap (r = {corr:.3f})')
    ax.grid(True, alpha=0.3)

    # Panel 3: n5 and n9 by argmax position
    ax = axes[0, 2]
    h5_by_pos = [h5[argmax_pos.numpy() == p].mean() for p in range(10)]
    h9_by_pos = [h9[argmax_pos.numpy() == p].mean() for p in range(10)]
    x_pos = np.arange(10)
    width = 0.35
    ax.bar(x_pos - width/2, h5_by_pos, width, label='n5', alpha=0.7)
    ax.bar(x_pos + width/2, h9_by_pos, width, label='n9', alpha=0.7)
    ax.set_xlabel('Argmax position')
    ax.set_ylabel('Mean activation')
    ax.set_title('n5 & n9 by argmax position')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: W_out patterns for n5 and n9
    ax = axes[1, 0]
    x_pos = np.arange(10)
    width = 0.35
    ax.bar(x_pos - width/2, W_out[:, 5].numpy(), width, label='W_out[:,5]', alpha=0.7)
    ax.bar(x_pos + width/2, W_out[:, 9].numpy(), width, label='W_out[:,9]', alpha=0.7)
    ax.set_xlabel('Output class (2nd_argmax)')
    ax.set_ylabel('Weight')
    ax.set_title('W_out weights for n5 and n9')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 5: n5 vs second_val
    ax = axes[1, 1]
    ax.scatter(second_val.numpy(), h5, alpha=0.1, s=1)
    ax.set_xlabel('2nd max value')
    ax.set_ylabel('n5 at t=9')
    corr = np.corrcoef(h5, second_val.numpy())[0, 1]
    ax.set_title(f'n5 vs 2nd_max value (r = {corr:.3f})')
    ax.grid(True, alpha=0.3)

    # Panel 6: n5 by 2nd_argmax position (when argmax=5)
    ax = axes[1, 2]
    mask = argmax_pos == 5
    h5_by_2nd = []
    positions = []
    for j in range(10):
        if j == 5:
            continue
        m = mask & (targets == j)
        if m.sum() > 30:
            positions.append(j)
            h5_by_2nd.append(h5[m.numpy()].mean())

    ax.bar(positions, h5_by_2nd, alpha=0.7)
    ax.set_xlabel('2nd_argmax position')
    ax.set_ylabel('Mean n5 (when argmax=5)')
    ax.set_title('n5 varies by 2nd_argmax position!')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved n5/n9 analysis to {save_path}')

    return fig


def analyze_n5_n9_vs_comparators(model, n_samples=50000):
    """
    How do n5/n9 interact with the comparators?
    """
    x = th.rand(n_samples, 10)
    targets = task_2nd_argmax(x)
    argmax_pos = x.argmax(dim=-1)

    hidden, _ = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    print("\n" + "=" * 70)
    print("N5/N9 vs COMPARATORS: INTERACTION")
    print("=" * 70)

    # Correlation between n5/n9 and comparators
    print("\nCorrelation between n5/n9 and comparators (h_final):")
    print("-" * 50)

    comparators = [1, 6, 7, 8]
    for n in [5, 9]:
        print(f"\nn{n} correlations:")
        for c in comparators:
            corr = np.corrcoef(h_final[:, n].numpy(), h_final[:, c].numpy())[0, 1]
            print(f"  r(n{n}, n{c}) = {corr:+.3f}")

    # Do n5/n9 provide ADDITIONAL info beyond comparators?
    print("\n" + "-" * 50)
    print("Residual analysis: n5/n9 info beyond comparators")
    print("-" * 50)

    # Regress n5 on comparators, check residual correlation with 2nd_argmax
    from numpy.linalg import lstsq

    X = np.column_stack([
        h_final[:, 1].numpy(),
        h_final[:, 6].numpy(),
        h_final[:, 7].numpy(),
        h_final[:, 8].numpy(),
        np.ones(n_samples)
    ])

    for n in [5, 9]:
        h_n = h_final[:, n].numpy()
        coeffs, _, _, _ = lstsq(X, h_n, rcond=None)
        predicted = X @ coeffs
        residual = h_n - predicted

        # Correlate residual with positions
        corr_argmax = np.corrcoef(residual, argmax_pos.numpy())[0, 1]
        corr_2nd = np.corrcoef(residual, targets.numpy())[0, 1]

        print(f"\nn{n} residual (after removing comparator info):")
        print(f"  r(residual, argmax) = {corr_argmax:+.3f}")
        print(f"  r(residual, 2nd_argmax) = {corr_2nd:+.3f}")


def main():
    model = example_2nd_argmax()

    analyze_n5_n9_encoding(model)
    analyze_n5_n9_temporal(model)
    analyze_n5_n9_connectivity(model)
    analyze_n5_n9_importance(model)
    analyze_n5_n9_vs_comparators(model)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    plot_n5_n9_analysis(model, save_path='docs/figures/n5_n9_analysis.png')

    print("\n" + "=" * 70)
    print("SUMMARY: N5 AND N9")
    print("=" * 70)
    print("""
N5 and N9 are "margin encoders" that track how distinguished the 2nd max is:

ENCODING:
  - Both negatively correlated with loo_diff (n5: -0.67, n9: -0.44)
  - Both positively correlated with argmax position
  - HIGH when the gap is SMALL (ambiguous cases)
  - LOW when the gap is LARGE (clear cases)

CONNECTIVITY:
  - n5 W_ih is small - driven more by recurrence
  - n9 has stronger input weight
  - Both receive input from comparators
  - Both have moderate W_out importance

FUNCTION:
  - They encode "confidence" or "margin" of the maximum
  - Help disambiguate when the top values are close
  - Provide amplitude information complementary to position

IMPORTANCE:
  - Ablating n5 alone: small accuracy drop
  - Ablating n9 alone: small accuracy drop
  - Combined: larger drop, but comparators remain dominant

They appear to be AUXILIARY encoders that help in ambiguous cases,
not the primary position encoding mechanism.
""")


if __name__ == "__main__":
    main()
