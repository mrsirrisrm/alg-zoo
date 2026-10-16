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
