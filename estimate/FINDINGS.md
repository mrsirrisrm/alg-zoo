# Mechanistic accuracy estimate for M(16,10) — running log

Target (ARC, AlgZoo post): a mechanistic estimate of accuracy that matches random
sampling in MSE vs compute, measured on average over noisy copies of the weights.

## Facts from the weights (no sampling)
- No biases ⇒ positively homogeneous: output depends only on x/‖x‖.
- Only n1, n4, n6, n7, n8 have large input weights (|W_ih| ≈ 10–13); n4 positive,
  the others negative. n14 −1.3; the rest |W_ih| < 0.3.
- n2: self-weight 0.97 (memory), fed by n4 (+1.73), feeds back on n4 (−0.49);
  n4 self-weight −0.99 ⇒ comparator structure.

## Ground truth (Monte Carlo, validation only)
- acc = 0.9531186 ± 1.1e-6 (N = 2^35, RTX 3090, 6.5 min). `gt_2e35.npz`.
- Failures: 63% predict the 3rd largest, 20% predict the max, 16% other.
- Accuracy falls when S − 3rd is small (0.58 below 0.01) AND when M − S is large
  (0.92 at 1.5, 0.83 at 2.5) — homogeneity: a large M makes S look like background.
- Worst cells (M pos, S pos): (0,7) 0.849, (7,6) 0.900, (9,8) 0.900, (9,0) 0.904.

## Measured structure
- n4 active ⇔ x_t is a new running max: 98.5% agreement for t ≥ 3.
  No neuron tracks "new 2nd max" directly.
- "Lucky seed" claim is wrong: all 5 zoo h16/T10 seeds reach 88–95%
  (example model 95.3%, best). h8 fails (35–43%), h32 ≈ 99%.

## Mechanism (derived from weights, then checked)
- Activation-weighted contributions E|W_ij h_j| (not raw weights — "small" weights
  from large periphery neurons matter; cutting periphery→core drops acc to 45%).
- n2 = memory: n2' = 0.97 n2 + 1.73 n4; on a record of value a with n4 = 10.16a − 0.49 n2,
  n2' = 17.58 a + 0.12 n2. Measured median n2/runmax = 17.6 (IQR 16.6–18.4). ✔
- n4 = soft comparator vs n2. One-term threshold 0.49·17.58/10.16 = 0.848·runmax is
  WRONG (prediction of a break at vS/vM ≈ 0.85 falsified: accuracy is smooth in the
  ratio). Full recurrent drive averages −9.53/runmax ⇒ mean threshold ≈ 0.94, but
  context-dependent: P(n4 on) = 0.25 at x = 0.90·max, 0.90 at 0.98, 0.994 at 1.00.
  Main threshold movers after n2: n4 self (−1.0), n6 (+0.93), n8 (−0.76).
- n6, n7 (and n1, n8 with mixing) = relu(−11..13·x + 0.5..0.7·n2): gap-below-max signals.
- Fragility lives in the W_hh rows of n1, n2, n6, n7, n8 (row noise ×4 → 0.75–0.86),
  not n4 or the periphery (≈0.94).
- Readout splits into two mechanisms:
  * S first (reverse, acc 0.9557): S is a record (n4 fires 98.9%), M is the last n4
    event (99.8%), S is the previous n4 event (95.6%); net outputs the previous event
    92.8%. Readout is amplitude-weighted, not a hard rule: when a false record
    intervenes the net still gets 71% right. Error budget: 0.013 from intervening
    false records, 0.031 with the event chain correct (decode errors).
  * M first (forward, acc 0.9508): S is not a record (n4 at S only 12%, and those are
    worse: 0.90). Errors: 65% pick R, 23% pick M, 29% pick something before M.
    R after M is harder (0.947) than R before M (0.959).
- **Unified decomposition** (`probes/ab_candidates.py`): S = argmax(A, B) with
  A = max before M (the penultimate record), B = max after M. Error budget, as a
  fraction of all inputs (total 0.0468):
  * within S's side, picks the wrong element: 0.0238 (51%) — equal on both sides
    (0.0121 before M, 0.0117 after M);
  * A/B swap: 0.0118 (25%) — a clean noisy comparator, P(swap) ≈ Φ(−d/σ),
    d = |vA − vB|/vM, σ ≈ 0.02 near 0, heavier tail beyond;
  * predicts M (M/S near-tie, vS/vM > 0.975): 0.0095 (20%);
  * other side, not its best: 0.0017 (4%).
  With only one candidate side (M at 0 or 9) accuracy is lower (0.93 vs 0.958), so
  within-side selection, not the A/B comparison, is the weak step.
- Weight-level origin of the A trace: after M's record, n2 = 17.58 vM + 0.122·n2_old,
  where 0.122 = 0.97 − 1.73·0.49 — n2 keeps a ~12% trace of the previous record.
- Offset rule argmax_{j≠M}(x_j − β(j,m)·x_M) fitted to the net's outputs explains
  nothing (96.39% vs 96.23% agreement) ⇒ errors are context-dependent measurement
  noise, not position biases.

## Evaluation harness
- σ = 0.005 (relative RMS, per matrix) noisy copies: accuracy 0.86–0.93.
  `noisy_sigma0.005.npz`, 2^30 samples each.

## Dead ends (with reasons)
1. **Exact polyhedral enumeration.** 1.65M distinct 160-bit activation patterns in
   4M samples, ~30% unseen mass; 99% of mass needs ~all of them. Even conditioned on
   (sign pattern, record set) the dominant pattern holds only 6% of mass.
   Activation patterns are not the right explanatory variable on Gaussian inputs
   (impulse-on-zero-background stability does not transfer).
2. **Gaussian closure of h_t (E0, `moment_prop.py`).** Fails at step 1: h_0 lies on
   two rays (sign of x_0). A moment-matched Gaussian lets the ray parameter go
   negative; e.g. n9's next pre-activation is ≈ −80|x_0| (never positive) but the
   Gaussian makes it fire 9% of the time. Splitting on sign(x_t) (Gaussian-sum, 2^8
   components) helps forward cases but reverse cases stay badly wrong.
   Lesson: the state lives on a positive cone; closures must respect it.

## Numerics
- `gauss.py`: Genz BVN (needed: ray geometry gives correlations ≈ ±1, where naive
  quadrature for Φ₂ breaks and second-moment matrices go non-PSD).
