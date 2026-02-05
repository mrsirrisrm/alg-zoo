"""
KOOPMAN K^9 DISCRIMINATION TEST

Test whether the approximate Koopman operator K, raised to the power of
remaining timesteps, correctly predicts the discrimination margin for
all 90 (m, s) position pairs.

If this works, the Koopman lifting provides a LINEAR predictor of
the nonlinear ReLU dynamics that preserves the discrimination signal.
"""

import numpy as np
from alg_zoo import example_2nd_argmax

model = example_2nd_argmax()
model.eval()

W_ih = model.rnn.weight_ih_l0.detach().numpy().squeeze()
W_hh = model.rnn.weight_hh_l0.detach().numpy()
W_out = model.linear.weight.detach().numpy()

M_val = 1.0
S_val = 0.8

def run_rnn(impulses):
    """Run RNN and return trajectory."""
    h = np.zeros(16)
    trajectory = [h.copy()]
    for t in range(10):
        x_t = sum(val for pos, val in impulses if pos == t)
        pre = W_ih * x_t + W_hh @ h
        h = np.maximum(0, pre)
        trajectory.append(h.copy())
    return trajectory

def lift_state(h):
    """Lift h to (h, h²) observable space."""
    return np.concatenate([h, h**2])

def unlift_state(g):
    """Extract h from lifted state."""
    return g[:16]

print("=" * 70)
print("KOOPMAN K^9 DISCRIMINATION TEST")
print("=" * 70)

# =============================================================================
# STEP 1: Fit Koopman operator from all trajectory data
# =============================================================================

print("\n[1] Fitting Koopman operator K from trajectory data...")

lifted_in = []
lifted_out = []

for m in range(10):
    for s in range(10):
        if m == s:
            continue

        # Forward trajectory
        traj_fwd = run_rnn([(m, M_val), (s, S_val)])
        for t in range(1, 10):
            lifted_in.append(lift_state(traj_fwd[t]))
            lifted_out.append(lift_state(traj_fwd[t+1]))

        # Reverse trajectory
        traj_rev = run_rnn([(m, S_val), (s, M_val)])
        for t in range(1, 10):
            lifted_in.append(lift_state(traj_rev[t]))
            lifted_out.append(lift_state(traj_rev[t+1]))

lifted_in = np.array(lifted_in)
lifted_out = np.array(lifted_out)

# Fit K via least squares: lifted_out ≈ lifted_in @ K
K, residuals, rank, s = np.linalg.lstsq(lifted_in, lifted_out, rcond=None)

# Quality check
predicted = lifted_in @ K
ss_res = np.sum((lifted_out - predicted) ** 2)
ss_tot = np.sum((lifted_out - np.mean(lifted_out, axis=0)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"   Lifted dimension: {K.shape[0]}")
print(f"   Training samples: {len(lifted_in)}")
print(f"   Koopman R²: {r2:.4f}")

# Koopman eigenvalues
K_eig = np.linalg.eigvals(K)
K_eig_sorted = sorted(K_eig, key=lambda x: -abs(x))
print(f"   Top eigenvalues: {', '.join(f'{abs(e):.3f}' for e in K_eig_sorted[:5])}")

# =============================================================================
# STEP 2: Test K^n prediction for varying n
# =============================================================================

print("\n[2] Testing K^n prediction accuracy...")

def test_koopman_power(n_steps, start_t):
    """
    Test if K^n_steps correctly predicts h[start_t + n_steps] from h[start_t].
    Returns R² and per-component errors.
    """
    actual_h = []
    predicted_h = []

    K_power = np.linalg.matrix_power(K, n_steps)

    for m in range(10):
        for s in range(10):
            if m == s:
                continue

            traj = run_rnn([(m, M_val), (s, S_val)])

            if start_t + n_steps <= 10:
                g_start = lift_state(traj[start_t])
                g_pred = g_start @ K_power
                h_pred = unlift_state(g_pred)
                h_actual = traj[start_t + n_steps]

                actual_h.append(h_actual)
                predicted_h.append(h_pred)

    actual_h = np.array(actual_h)
    predicted_h = np.array(predicted_h)

    ss_res = np.sum((actual_h - predicted_h) ** 2)
    ss_tot = np.sum((actual_h - np.mean(actual_h, axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return r2, actual_h, predicted_h

print("\n   K^n prediction R² (starting from t=2, after both impulses typically arrived):")
print("   n_steps | R²     | Notes")
print("   --------|--------|------")
for n in [1, 2, 3, 5, 7]:
    r2, _, _ = test_koopman_power(n, start_t=2)
    note = ""
    if n == 7:
        note = " ← predicting t=9 from t=2"
    print(f"      {n}    | {r2:.4f} | {note}")

# =============================================================================
# STEP 3: K^9 discrimination test - the main event
# =============================================================================

print("\n" + "=" * 70)
print("[3] K^9 DISCRIMINATION TEST")
print("=" * 70)

# For each (m, s) pair, we want to test:
# 1. Start from h after the 2nd impulse (t = max(m,s) + 1)
# 2. Apply K^(9 - start_t) to get predicted h[9]
# 3. Check if W_out @ h_pred gives correct discrimination

correct_actual = 0
correct_koopman = 0
margin_comparison = []

print("\nPer-pair analysis (showing first 20):")
print("(m,s) | start_t | actual_margin | koopman_margin | correct?")
print("------|---------|---------------|----------------|----------")

for idx, (m, s) in enumerate([(m, s) for m in range(10) for s in range(10) if m != s]):
    # Forward: M at m, S at s → should output s
    traj_fwd = run_rnn([(m, M_val), (s, S_val)])
    # Reverse: S at m, M at s → should output m
    traj_rev = run_rnn([(m, S_val), (s, M_val)])

    # Start after both impulses have arrived
    start_t = max(m, s) + 1
    if start_t > 9:
        start_t = 9  # Edge case

    n_steps = 9 - start_t

    # Actual final states
    h_fwd_actual = traj_fwd[10]  # t=9 is index 10 in trajectory
    h_rev_actual = traj_rev[10]

    # Koopman predictions
    if n_steps > 0:
        K_power = np.linalg.matrix_power(K, n_steps)
        g_fwd = lift_state(traj_fwd[start_t])
        g_rev = lift_state(traj_rev[start_t])
        h_fwd_pred = unlift_state(g_fwd @ K_power)
        h_rev_pred = unlift_state(g_rev @ K_power)
    else:
        h_fwd_pred = traj_fwd[start_t]
        h_rev_pred = traj_rev[start_t]

    # Actual margins
    # Forward should have logit[s] > logit[m]
    logits_fwd_actual = W_out @ h_fwd_actual
    margin_fwd_actual = logits_fwd_actual[s] - logits_fwd_actual[m]

    # Reverse should have logit[m] > logit[s]
    logits_rev_actual = W_out @ h_rev_actual
    margin_rev_actual = logits_rev_actual[m] - logits_rev_actual[s]

    # Koopman margins
    logits_fwd_pred = W_out @ h_fwd_pred
    margin_fwd_koopman = logits_fwd_pred[s] - logits_fwd_pred[m]

    logits_rev_pred = W_out @ h_rev_pred
    margin_rev_koopman = logits_rev_pred[m] - logits_rev_pred[s]

    # Check correctness
    actual_correct = (margin_fwd_actual > 0) and (margin_rev_actual > 0)
    koopman_correct = (margin_fwd_koopman > 0) and (margin_rev_koopman > 0)

    if actual_correct:
        correct_actual += 1
    if koopman_correct:
        correct_koopman += 1

    margin_comparison.append({
        'm': m, 's': s,
        'margin_fwd_actual': margin_fwd_actual,
        'margin_fwd_koopman': margin_fwd_koopman,
        'margin_rev_actual': margin_rev_actual,
        'margin_rev_koopman': margin_rev_koopman,
        'n_steps': n_steps
    })

    if idx < 20:
        status = "✓" if koopman_correct else "✗"
        print(f"({m},{s}) |    {start_t}    |    {margin_fwd_actual:6.1f}    |     {margin_fwd_koopman:6.1f}     |    {status}")

print("...")

# =============================================================================
# STEP 4: Summary statistics
# =============================================================================

print("\n" + "=" * 70)
print("[4] SUMMARY")
print("=" * 70)

print(f"\nActual model accuracy: {correct_actual}/90 = {100*correct_actual/90:.1f}%")
print(f"Koopman K^n accuracy:  {correct_koopman}/90 = {100*correct_koopman/90:.1f}%")

# Margin correlation
margins_fwd_actual = [d['margin_fwd_actual'] for d in margin_comparison]
margins_fwd_koopman = [d['margin_fwd_koopman'] for d in margin_comparison]
margins_rev_actual = [d['margin_rev_actual'] for d in margin_comparison]
margins_rev_koopman = [d['margin_rev_koopman'] for d in margin_comparison]

corr_fwd = np.corrcoef(margins_fwd_actual, margins_fwd_koopman)[0, 1]
corr_rev = np.corrcoef(margins_rev_actual, margins_rev_koopman)[0, 1]

print(f"\nMargin correlation (actual vs Koopman):")
print(f"   Forward: r = {corr_fwd:.4f}")
print(f"   Reverse: r = {corr_rev:.4f}")

# Margin error statistics
errors_fwd = np.array(margins_fwd_koopman) - np.array(margins_fwd_actual)
errors_rev = np.array(margins_rev_koopman) - np.array(margins_rev_actual)

print(f"\nMargin prediction error:")
print(f"   Forward: mean = {np.mean(errors_fwd):.2f}, std = {np.std(errors_fwd):.2f}")
print(f"   Reverse: mean = {np.mean(errors_rev):.2f}, std = {np.std(errors_rev):.2f}")

# Find the cases where Koopman fails (if any)
if correct_koopman < 90:
    print("\n" + "-" * 50)
    print("FAILED CASES (Koopman gives wrong discrimination):")
    print("-" * 50)
    for d in margin_comparison:
        fwd_wrong = d['margin_fwd_koopman'] <= 0 and d['margin_fwd_actual'] > 0
        rev_wrong = d['margin_rev_koopman'] <= 0 and d['margin_rev_actual'] > 0
        if fwd_wrong or rev_wrong:
            print(f"   ({d['m']},{d['s']}): n_steps={d['n_steps']}")
            print(f"      Fwd: actual={d['margin_fwd_actual']:.2f}, koopman={d['margin_fwd_koopman']:.2f}")
            print(f"      Rev: actual={d['margin_rev_actual']:.2f}, koopman={d['margin_rev_koopman']:.2f}")

# =============================================================================
# STEP 5: Analyze where Koopman error concentrates
# =============================================================================

print("\n" + "=" * 70)
print("[5] ERROR ANALYSIS: Does Koopman error correlate with margin?")
print("=" * 70)

# Is the error larger for small-margin cases?
all_margins_actual = margins_fwd_actual + margins_rev_actual
all_errors = list(errors_fwd) + list(errors_rev)

# Correlation between |error| and margin
corr_error_margin = np.corrcoef(np.abs(all_errors), all_margins_actual)[0, 1]
print(f"\nCorrelation(|error|, margin): r = {corr_error_margin:.4f}")

# Binned analysis
small_margin_idx = [i for i, m in enumerate(all_margins_actual) if m < np.median(all_margins_actual)]
large_margin_idx = [i for i, m in enumerate(all_margins_actual) if m >= np.median(all_margins_actual)]

small_margin_errors = [all_errors[i] for i in small_margin_idx]
large_margin_errors = [all_errors[i] for i in large_margin_idx]

print(f"\nError by margin bin:")
print(f"   Small margin (< median): mean |error| = {np.mean(np.abs(small_margin_errors)):.2f}")
print(f"   Large margin (≥ median): mean |error| = {np.mean(np.abs(large_margin_errors)):.2f}")

# Minimum margin with Koopman prediction
min_margin_koopman = min(min(margins_fwd_koopman), min(margins_rev_koopman))
min_margin_actual = min(min(margins_fwd_actual), min(margins_rev_actual))

print(f"\nMinimum discrimination margin:")
print(f"   Actual model: {min_margin_actual:.2f}")
print(f"   Koopman pred: {min_margin_koopman:.2f}")

# =============================================================================
# STEP 6: What observables matter most?
# =============================================================================

print("\n" + "=" * 70)
print("[6] OBSERVABLE IMPORTANCE: Which components of K matter?")
print("=" * 70)

# The discrimination margin is W_out @ h = W_out @ g[:16]
# So we only care about how K maps to the first 16 components

K_to_h = K[:, :16]  # Columns that affect h (not h²)

# SVD to find important directions
U, S_k, Vt = np.linalg.svd(K_to_h)

print("\nSingular values of K[:, :16] (maps lifted state to h):")
print(f"   Top 5: {S_k[:5]}")
print(f"   Condition number: {S_k[0]/S_k[-1]:.1f}")

# How much does h² contribute vs h?
K_h_to_h = K[:16, :16]   # h → h
K_h2_to_h = K[16:, :16]  # h² → h

norm_h_to_h = np.linalg.norm(K_h_to_h, 'fro')
norm_h2_to_h = np.linalg.norm(K_h2_to_h, 'fro')

print(f"\nContribution to next h:")
print(f"   ||K[h→h]||_F  = {norm_h_to_h:.2f}")
print(f"   ||K[h²→h]||_F = {norm_h2_to_h:.2f}")
print(f"   Ratio (h²/h):   {norm_h2_to_h/norm_h_to_h:.3f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if correct_koopman == 90:
    print("""
✓ KOOPMAN K^n PERFECTLY PREDICTS DISCRIMINATION

The approximate Koopman operator K, fitted on (h, h²) observables,
correctly predicts the sign of the discrimination margin for all
90 position pairs when propagated K^n steps.

This means the ReLU nonlinearity has been successfully "absorbed"
into the lifted linear dynamics for discrimination purposes.

NEXT STEPS FOR MECHANISTIC ESTIMATE:
1. Derive K analytically from (W_ih, W_hh) + activation statistics
2. Show K^9 @ g_init has positive margin for all inputs
3. This would be a weight-based proof of 100% accuracy
""")
else:
    pct = 100 * correct_koopman / 90
    print(f"""
⚠ KOOPMAN K^n ACHIEVES {pct:.1f}% DISCRIMINATION ACCURACY

The quadratic lifting (h, h²) captures most but not all of the
discrimination signal. The failures may be due to:
- Boundary crossings not captured by quadratic terms
- Higher-order nonlinearities in specific trajectories
- Trajectory-dependent effects not in local observables

NEXT STEPS:
1. Analyze failed cases - are they near cell boundaries?
2. Try richer observables: cross-terms h_i * h_j, ReLU features
3. Consider trajectory-specific Koopman operators
""")
