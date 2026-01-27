"""
Fourier-like Position Encoding Analysis

Investigates whether the comparator neurons encode position using
sinusoidal-like components at different frequencies, similar to
positional encoding in transformers.

Hypothesis from observation:
- n8: ~π/2 radians over sequence (quarter wave)
- n6: ~π radians over sequence (half wave)
- n7: ~3π/2 radians over sequence (three-quarter wave)
- n1: ~2π radians over sequence (full wave)
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from alg_zoo import example_2nd_argmax, task_2nd_argmax


def get_full_trajectory(model, x):
    """Compute full trajectory including clipping events."""
    W_ih = model.rnn.weight_ih_l0.data.squeeze()
    W_hh = model.rnn.weight_hh_l0.data

    batch_size = x.shape[0]
    clipped = th.zeros(batch_size, 16, 10, dtype=th.bool)
    hidden = th.zeros(batch_size, 16, 10)
    h = th.zeros(batch_size, 16)

    for t in range(10):
        x_t = x[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        clipped[:, :, t] = pre < 0
        h = th.relu(pre)
        hidden[:, :, t] = h

    return clipped, hidden


def extract_recency_curves(model, n_samples=50000):
    """
    Extract the mean h_final as a function of steps-since-clip for each comparator.
    """
    x = th.rand(n_samples, 10)
    clipped, hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    comparators = [1, 6, 7, 8]

    recency_curves = {}

    for n in comparators:
        # Find last clip position for each sample
        last_clip = th.zeros(n_samples, dtype=th.long) - 1
        for t in range(10):
            mask = clipped[:, n, t]
            last_clip[mask] = t

        # Steps since last clip (at t=9)
        steps_since = 9 - last_clip

        # Get mean h_final for each steps_since value
        curve = []
        steps = list(range(1, 10))
        for s in steps:
            mask = steps_since == s
            if mask.sum() > 100:
                curve.append(h_final[mask, n].mean().item())
            else:
                curve.append(np.nan)

        recency_curves[n] = np.array(curve)

    return steps, recency_curves


def fit_sinusoid(steps, values, freq_range=(0.1, 1.5)):
    """
    Fit a sinusoid to the data: y = A * sin(ω*t + φ) + C
    Returns best fit parameters.
    """
    from scipy.optimize import minimize

    # Remove NaN values
    valid = ~np.isnan(values)
    t = np.array(steps)[valid]
    y = values[valid]

    if len(t) < 4:
        return None

    # Normalize t to [0, 1] range for frequency estimation
    t_norm = (t - t.min()) / (t.max() - t.min())

    def sinusoid(params, t):
        A, omega, phi, C = params
        return A * np.sin(omega * t + phi) + C

    def loss(params):
        return np.sum((sinusoid(params, t_norm) - y) ** 2)

    # Try multiple initializations
    best_loss = np.inf
    best_params = None

    for omega_init in np.linspace(freq_range[0] * np.pi, freq_range[1] * np.pi, 10):
        for phi_init in [0, np.pi/2, np.pi, 3*np.pi/2]:
            A_init = (y.max() - y.min()) / 2
            C_init = y.mean()

            result = minimize(loss, [A_init, omega_init, phi_init, C_init],
                            method='Nelder-Mead')

            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x

    return best_params


def plot_fourier_analysis(model, save_path=None):
    """
    Compare recency curves to sinusoidal fits.
    """
    steps, recency_curves = extract_recency_curves(model)

    # Hypothesized frequencies (radians over the sequence)
    hypothesized = {
        8: np.pi / 2,      # Quarter wave
        6: np.pi,          # Half wave
        7: 3 * np.pi / 2,  # Three-quarter wave
        1: 2 * np.pi,      # Full wave
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Order by hypothesized frequency
    neuron_order = [8, 6, 7, 1]
    colors = {'1': 'steelblue', '6': 'coral', '7': 'green', '8': 'purple'}

    fitted_freqs = {}

    for idx, n in enumerate(neuron_order):
        ax = axes[idx]
        curve = recency_curves[n]

        # Plot actual data
        valid = ~np.isnan(curve)
        ax.plot(np.array(steps)[valid], curve[valid], 'o-',
                color=colors[str(n)], linewidth=2, markersize=8,
                label=f'n{n} actual')

        # Fit sinusoid
        params = fit_sinusoid(steps, curve)
        if params is not None:
            A, omega, phi, C = params
            fitted_freqs[n] = omega

            # Plot fitted curve
            t_fine = np.linspace(1, 8, 100)
            t_norm_fine = (t_fine - 1) / 7  # Normalize to [0, 1]
            y_fit = A * np.sin(omega * t_norm_fine + phi) + C
            ax.plot(t_fine, y_fit, '--', color='gray', linewidth=2,
                   label=f'fit: ω={omega:.2f}')

            # Also plot the hypothesized sinusoid with phase adjustments
            omega_hyp = hypothesized[n]
            # Fit amplitude and offset to match data
            scale = (curve[valid].max() - curve[valid].min()) / 2
            offset = curve[valid].mean()

            # Phase adjustments based on user observation:
            # n7: peak at t=2 (step 1 in steps-since-clip terms, but t_fine starts at 1)
            # n1: minimum at t=2
            if n == 7:
                # Peak at t=2 means sin(omega*t_norm + phi) = 1 at t=2
                # t=2 corresponds to t_norm = (2-1)/7 = 1/7
                # sin(omega * 1/7 + phi) = 1 => omega * 1/7 + phi = pi/2
                phi_hyp = np.pi/2 - omega_hyp * (1/7)
            elif n == 1:
                # Minimum at t=2 means sin(omega*t_norm + phi) = -1 at t=2
                # sin(omega * 1/7 + phi) = -1 => omega * 1/7 + phi = -pi/2
                phi_hyp = -np.pi/2 - omega_hyp * (1/7)
            else:
                # For n6 and n8, use default phase that looks reasonable
                # n8: rising from start, so sin starting at -1 and rising
                # n6: peak early then decline
                if n == 8:
                    phi_hyp = -np.pi/2  # Start at minimum, rise
                else:  # n == 6
                    phi_hyp = np.pi/4  # Peak early

            y_hyp_base = np.sin(omega_hyp * t_norm_fine + phi_hyp)
            y_hyp = scale * y_hyp_base + offset
            ax.plot(t_fine, y_hyp, ':', color='red', linewidth=2, alpha=0.7,
                   label=f'hyp: ω={omega_hyp:.2f}')

        ax.set_xlabel('Steps since last clip')
        ax.set_ylabel('Mean h_final')
        ax.set_title(f'Neuron {n}\nHypothesized: {hypothesized[n]/np.pi:.1f}π rad')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Fourier-like Position Encoding in Comparator Neurons\n'
                 'Solid=data, Dashed=best fit, Dotted=hypothesized frequency',
                 fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved Fourier analysis to {save_path}')

    return fig, fitted_freqs


def plot_combined_encoding(model, save_path=None):
    """
    Show all four neurons together as a combined encoding.
    """
    steps, recency_curves = extract_recency_curves(model)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: All curves overlaid (normalized)
    ax = axes[0]
    colors = {'1': 'steelblue', '6': 'coral', '7': 'green', '8': 'purple'}

    for n in [1, 6, 7, 8]:
        curve = recency_curves[n]
        valid = ~np.isnan(curve)

        # Normalize to [0, 1] for comparison
        curve_norm = (curve - np.nanmin(curve)) / (np.nanmax(curve) - np.nanmin(curve))

        ax.plot(np.array(steps)[valid], curve_norm[valid], 'o-',
                color=colors[str(n)], linewidth=2, markersize=6,
                label=f'n{n}')

    ax.set_xlabel('Steps since last clip')
    ax.set_ylabel('Normalized h_final')
    ax.set_title('Recency Curves (normalized)\nDifferent "frequencies" visible')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Phase portrait - n1 vs n7, n6 vs n8
    ax = axes[1]

    # Get raw data for phase portrait
    x = th.rand(10000, 10)
    clipped, hidden = get_full_trajectory(model, x)
    h_final = hidden[:, :, 9]

    # Color by steps since any comparator clipped
    last_clip_any = th.zeros(10000, dtype=th.long)
    for t in range(10):
        for n in [1, 6, 7, 8]:
            mask = clipped[:, n, t]
            last_clip_any[mask] = t
    steps_since_any = 9 - last_clip_any

    # Plot h1 vs h7 colored by recency
    scatter = ax.scatter(h_final[:, 1].numpy(), h_final[:, 7].numpy(),
                        c=steps_since_any.numpy(), cmap='viridis',
                        alpha=0.3, s=10)
    ax.set_xlabel('h_final[n1]')
    ax.set_ylabel('h_final[n7]')
    ax.set_title('Phase Portrait: n1 vs n7\n(color = steps since clip)')
    plt.colorbar(scatter, ax=ax, label='Steps since clip')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved combined encoding to {save_path}')

    return fig


def analyze_frequency_content(model):
    """
    Use FFT to analyze the frequency content of recency curves.
    """
    steps, recency_curves = extract_recency_curves(model)

    print("=" * 70)
    print("FREQUENCY ANALYSIS OF RECENCY CURVES")
    print("=" * 70)

    print("\nHypothesized frequencies (radians over sequence):")
    print("  n8: π/2  (0.50π) - quarter wave, slowest")
    print("  n6: π    (1.00π) - half wave")
    print("  n7: 3π/2 (1.50π) - three-quarter wave")
    print("  n1: 2π   (2.00π) - full wave, fastest")

    print("\n" + "-" * 50)
    print("Sinusoid fits:")

    for n in [8, 6, 7, 1]:
        curve = recency_curves[n]
        params = fit_sinusoid(steps, curve)
        if params is not None:
            A, omega, phi, C = params
            print(f"\n  n{n}:")
            print(f"    Fitted ω = {omega:.2f} ({omega/np.pi:.2f}π)")
            print(f"    Amplitude = {A:.2f}")
            print(f"    Phase = {phi:.2f} rad")
            print(f"    Offset = {C:.2f}")


def compute_orthogonality(model):
    """
    Check if the recency encodings are approximately orthogonal,
    as they would be for Fourier components.
    """
    steps, recency_curves = extract_recency_curves(model)

    print("\n" + "=" * 70)
    print("ORTHOGONALITY ANALYSIS")
    print("=" * 70)

    # Stack curves into matrix (neurons x steps)
    neurons = [1, 6, 7, 8]
    curves = []
    for n in neurons:
        curve = recency_curves[n]
        # Interpolate NaN values
        valid = ~np.isnan(curve)
        if not all(valid):
            curve = np.interp(steps, np.array(steps)[valid], curve[valid])
        curves.append(curve)

    curves = np.array(curves)

    # Normalize each curve (subtract mean, divide by std)
    curves_norm = (curves - curves.mean(axis=1, keepdims=True)) / curves.std(axis=1, keepdims=True)

    # Compute correlation matrix
    corr = np.corrcoef(curves_norm)

    print("\nCorrelation matrix of recency curves:")
    print("        n1      n6      n7      n8")
    for i, n1 in enumerate(neurons):
        row = "  n" + str(n1) + "  "
        for j, n2 in enumerate(neurons):
            row += f"{corr[i,j]:+.2f}    "
        print(row)

    print("\nFor orthogonal Fourier components, off-diagonal should be ~0")

    # Compute mean absolute off-diagonal correlation
    off_diag = []
    for i in range(4):
        for j in range(i+1, 4):
            off_diag.append(abs(corr[i, j]))

    print(f"\nMean |off-diagonal correlation|: {np.mean(off_diag):.3f}")
    print("(0 = perfectly orthogonal, 1 = perfectly correlated)")


def main():
    model = example_2nd_argmax()

    analyze_frequency_content(model)
    compute_orthogonality(model)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_fourier_analysis(model, 'docs/figures/fourier_encoding.png')
    plot_combined_encoding(model, 'docs/figures/combined_encoding.png')

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The comparator neurons appear to encode position using components
at different "temporal frequencies":

1. FAST (n1, ~2π): Completes a full cycle over the sequence
   - Most sensitive to fine position differences
   - But also most susceptible to noise

2. MEDIUM-FAST (n7, ~3π/2): Three-quarter cycle
   - Good discrimination for middle positions

3. MEDIUM (n6, ~π): Half cycle
   - Distinguishes early vs late positions

4. SLOW (n8, ~π/2): Quarter cycle
   - Coarse position information
   - Most robust to noise

This is analogous to:
- Fourier basis functions in signal processing
- Positional encodings in transformers (sin/cos at different frequencies)
- Color vision (different cone sensitivities)

The combination of multiple frequencies allows:
- Fine-grained position discrimination (high freq)
- Robust coarse positioning (low freq)
- Error correction through redundancy
""")


if __name__ == "__main__":
    main()
