# Note on the noisy-copy protocol (draft for ARC / Jacob; PLAN A5)

**Claim.** At the noise level that keeps the mechanism intact (σ = 0.005 relative rms per
matrix, accuracy 0.83–0.94 from 0.953), the perturbed copies of M(16,10) are in the
perturbative regime of the trained model. Their accuracy is predicted from the perturbation
and the trained model's sensitivity alone, with no evaluation of the copy. So the copy
harness measures sensitivity around a known point, not estimation; it is not an
independent test of a mechanistic estimator.

**Evidence** (`estimate/probes/weight_perturbation.py`, `wp_calibrate.py`, 32 copies with
2^30-sample truth each):

| predictor of the copy's accuracy loss | corr with truth | rms error | cost per copy |
|---|---|---|---|
| constant (mean loss over copies) | – | 0.026 | 0 |
| c · δᵀHδ, H = Gauss-Newton Fisher of the trained model, c one scalar fitted on 16 fresh copies (leave-one-out within the harness: 0.009) | 0.95 | 0.013 | 3.7e5 FLOPs (65 MC samples) |
| first-order logit shift J·δ on a fixed 65k pool, count flips | 0.996 | 0.0024 | 5.7e8 FLOPs |

Monte Carlo needs ~560 samples (3 MFLOP) to reach the quadratic form's 0.013, and the
quadratic form costs the equivalent of 65 samples. H has participation ratio 4.9: which
copies are bad is set by how much of the noise lands on about five stiff directions, all
in the recurrent weights of five neurons.

**What this does and does not say.**
- It does not say the estimator problem is solved: the copies' accuracy is deduced from
  the trained model's *measured* accuracy plus perturbation theory. The trained accuracy
  itself still has to come from somewhere, and that is the hard part the protocol was
  meant to test.
- It does say that any estimator allowed to use the trained model as advice can beat
  sampling on the copies at low budget without understanding the mechanism, so the copy
  harness cannot separate mechanistic from advice-based estimators.
  (Common random numbers against the trained model add nothing above that budget: the
  copies' flips are one-directional, so the paired difference is as variable as the
  copy's own outcome; `probes/wp_crn.py`.)
- The independently trained seeds (0–4, accuracy 0.88–0.95) are not perturbations of the
  trained model, and on them the picture is unchanged: the best unbiased mechanism-guided
  sampling gains 2–4× over MC, the best deduction is 13–15% off.

**Suggestion.** Report results on independently trained models as the primary test, and
treat noisy copies as a sensitivity check. If noisy copies are kept, choose σ large enough
that the quadratic prediction fails (for this model the mean loss departs from quadratic
scaling between σ = 0.005 and 0.01, and the mechanism is destroyed by 0.02), and report
the perturbation-theory baseline alongside MC.
