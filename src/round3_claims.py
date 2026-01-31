"""
Round 3 Claims 16-21.
"""

import torch as th
import numpy as np
from alg_zoo.architectures import DistRNN
from collections import defaultdict


def load_local_model():
    import os
    path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
    state_dict = th.load(path, weights_only=True, map_location='cpu')
    seq_len, hidden_size = state_dict['linear.weight'].shape
    model = DistRNN(hidden_size=hidden_size, seq_len=seq_len, bias=False)
    model.load_state_dict(state_dict)
    return model


def make_clean_dataset():
    samples, labels, mps, sps = [], [], [], []
    for mp in range(10):
        for sp in range(10):
            if mp == sp:
                continue
            x = th.zeros(10)
            x[mp] = 1.0
            x[sp] = 0.8
            samples.append(x)
            labels.append(sp)
            mps.append(mp)
            sps.append(sp)
    return th.stack(samples), th.tensor(labels), np.array(mps), np.array(sps)


def run_model(W_ih, W_hh, W_out, X):
    h = th.zeros(X.shape[0], 16)
    for t in range(10):
        x_t = X[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
    return h @ W_out.T


def run_model_hidden(W_ih, W_hh, X):
    h = th.zeros(X.shape[0], 16)
    for t in range(10):
        x_t = X[:, t:t+1]
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
    return h.detach().numpy()


COMPS = [1, 6, 7, 8]
WAVES = [0, 10, 11, 12, 14]
BRIDGES = [3, 5, 13, 15]


def section(title):
    print(f"\n{'='*80}")
    print(title)
    print('='*80)


# =============================================================================
# CLAIM 16: Single impulse → fixed default output
# =============================================================================
def test_claim_16(W_ih, W_hh, W_out):
    section("CLAIM 16: Single impulse → default output")

    print(f"{'pos':>4} {'pred':>5} {'top3_pos':>20} {'top3_logit':>30}")
    print("-" * 64)

    preds = []
    for pos in range(10):
        x = th.zeros(10)
        x[pos] = 1.0
        logits = run_model(W_ih, W_hh, W_out, x.unsqueeze(0))[0].detach().numpy()
        pred = logits.argmax()
        preds.append(pred)
        top3 = np.argsort(logits)[-3:][::-1]
        print(f"{pos:>4} {pred:>5}   {str(top3):>20} {logits[top3]}")

    unique = set(preds)
    print(f"\nDistinct outputs: {len(unique)} → {sorted(unique)}")
    print(f"Most common: {max(set(preds), key=preds.count)} "
          f"(appears {preds.count(max(set(preds), key=preds.count))}/10)")

    # Also test with magnitude 0.8 (like S)
    print(f"\nSame with magnitude 0.8:")
    preds08 = []
    for pos in range(10):
        x = th.zeros(10)
        x[pos] = 0.8
        logits = run_model(W_ih, W_hh, W_out, x.unsqueeze(0))[0].detach().numpy()
        pred = logits.argmax()
        preds08.append(pred)
        print(f"  pos={pos} → pred={pred}")
    print(f"  Distinct: {len(set(preds08))} → {sorted(set(preds08))}")


# =============================================================================
# CLAIM 17: Waves encode gap for reversed pairs
# =============================================================================
def test_claim_17(W_ih, W_hh, X, mps, sps):
    section("CLAIM 17: Waves encode gap for reversed pairs")

    h_final = run_model_hidden(W_ih, W_hh, X)
    is_rev = sps < mps
    gaps = np.abs(sps - mps)
    last_clips = np.maximum(mps, sps)

    rev_idx = np.where(is_rev)[0]
    rev_gaps = gaps[rev_idx]
    rev_lc = last_clips[rev_idx]

    # R² for waves grouped by gap vs by last_clip
    for label, neurons, name in [("Waves", WAVES, "wave"),
                                  ("Comps", COMPS, "comp")]:
        for group_var, group_name in [(rev_gaps, "gap"), (rev_lc, "last_clip")]:
            # Compute R² as fraction of variance explained by group means
            features = h_final[rev_idx][:, neurons]
            total_var = 0
            explained_var = 0
            for col in range(len(neurons)):
                vals = features[:, col]
                overall_mean = vals.mean()
                ss_tot = np.sum((vals - overall_mean)**2)

                group_means = {}
                for g in np.unique(group_var):
                    mask = group_var == g
                    group_means[g] = vals[mask].mean()

                predicted = np.array([group_means[g] for g in group_var])
                ss_res = np.sum((vals - predicted)**2)

                total_var += ss_tot
                explained_var += (ss_tot - ss_res)

            r2 = explained_var / total_var if total_var > 0 else 0
            print(f"{name} R²({group_name}) for reversed: {r2:.4f}")

    # Also: within-gap variance for waves
    print(f"\nWave h_final within-gap std (reversed):")
    print(f"{'gap':>4} {'mean_std':>10} {'n':>4}")
    print("-" * 20)
    for g in range(1, 10):
        mask = (is_rev) & (gaps == g)
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        stds = []
        for w in WAVES:
            stds.append(np.std(h_final[idx, w]))
        print(f"{g:>4} {np.mean(stds):>10.4f} {len(idx):>4}")


# =============================================================================
# CLAIM 18: Forward from comps + n2 alone
# =============================================================================
def test_claim_18(W_ih, W_hh, W_out, X, y, mps, sps):
    section("CLAIM 18: Forward from comps + n2 linear readout")

    h_final = run_model_hidden(W_ih, W_hh, X)
    is_fwd = sps > mps
    is_rev = ~is_fwd

    # Use only comp + n2 features
    comp_n2_idx = COMPS + [2]  # n1, n6, n7, n8, n2
    features = h_final[:, comp_n2_idx]

    # Fit linear readout via least squares (one-hot targets)
    y_np = y.numpy()
    Y_onehot = np.zeros((len(y_np), 10))
    for i, yi in enumerate(y_np):
        Y_onehot[i, yi] = 1.0

    # Fit on all data, test on forward/reversed splits
    # W_fit = (X^T X)^{-1} X^T Y
    XtX = features.T @ features
    XtY = features.T @ Y_onehot
    W_fit = np.linalg.solve(XtX + 1e-8 * np.eye(len(comp_n2_idx)), XtY)

    logits_fit = features @ W_fit
    preds_fit = logits_fit.argmax(axis=1)

    fwd_acc = (preds_fit[is_fwd] == y_np[is_fwd]).mean() * 100
    rev_acc = (preds_fit[is_rev] == y_np[is_rev]).mean() * 100
    all_acc = (preds_fit == y_np).mean() * 100

    print(f"Comps + n2 only (5 neurons):")
    print(f"  Forward:  {fwd_acc:.1f}%")
    print(f"  Reversed: {rev_acc:.1f}%")
    print(f"  All:      {all_acc:.1f}%")

    # Compare: waves + n2
    wave_n2_idx = WAVES + [2]
    features_w = h_final[:, wave_n2_idx]
    XtX_w = features_w.T @ features_w
    XtY_w = features_w.T @ Y_onehot
    W_fit_w = np.linalg.solve(XtX_w + 1e-8 * np.eye(len(wave_n2_idx)), XtY_w)
    preds_w = (features_w @ W_fit_w).argmax(axis=1)
    print(f"\nWaves + n2 only (6 neurons):")
    print(f"  Forward:  {(preds_w[is_fwd] == y_np[is_fwd]).mean()*100:.1f}%")
    print(f"  Reversed: {(preds_w[is_rev] == y_np[is_rev]).mean()*100:.1f}%")
    print(f"  All:      {(preds_w == y_np).mean()*100:.1f}%")

    # Compare: comps only (no n2)
    features_c = h_final[:, COMPS]
    XtX_c = features_c.T @ features_c
    XtY_c = features_c.T @ Y_onehot
    W_fit_c = np.linalg.solve(XtX_c + 1e-8 * np.eye(len(COMPS)), XtY_c)
    preds_c = (features_c @ W_fit_c).argmax(axis=1)
    print(f"\nComps only (4 neurons, no n2):")
    print(f"  Forward:  {(preds_c[is_fwd] == y_np[is_fwd]).mean()*100:.1f}%")
    print(f"  Reversed: {(preds_c[is_rev] == y_np[is_rev]).mean()*100:.1f}%")
    print(f"  All:      {(preds_c == y_np).mean()*100:.1f}%")

    # Original W_out for reference
    logits_orig = run_model(W_ih, W_hh, W_out, X)
    preds_orig = logits_orig.argmax(dim=-1).numpy()
    print(f"\nOriginal W_out (16 neurons): {(preds_orig == y_np).mean()*100:.1f}%")


# =============================================================================
# CLAIM 19: Parity pattern is method-independent
# =============================================================================
def test_claim_19(W_ih, W_hh, W_out):
    section("CLAIM 19: Parity under cascade attenuation (not magnitude reduction)")
    print("Scale hidden state at t=S+1 by factor alpha.\n")

    alphas = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]

    print(f"{'alpha':>6}", end="")
    for g in range(1, 10):
        p = "o" if g % 2 == 1 else "e"
        print(f"  g{g}{p}", end="")
    print("   all")
    print("-" * 62)

    for alpha in alphas:
        correct_by_gap = defaultdict(lambda: [0, 0])
        total_correct = 0
        total = 0

        for mt in range(10):
            for st in range(mt):  # reversed only
                x = th.zeros(10)
                x[mt] = 1.0
                x[st] = 0.8  # full magnitude!

                h = th.zeros(1, 16)
                for t in range(10):
                    x_t = x[t:t+1].unsqueeze(0)
                    pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
                    h = th.relu(pre)
                    # Attenuate cascade right after S impulse
                    if t == st:
                        # Scale everything except n2 (to isolate cascade effect)
                        for n in list(range(16)):
                            if n != 2:
                                h[0, n] *= alpha

                logits = (h @ W_out.T).detach().numpy()[0]
                pred = logits.argmax()
                gap = mt - st
                correct_by_gap[gap][1] += 1
                total += 1
                if pred == st:
                    correct_by_gap[gap][0] += 1
                    total_correct += 1

        print(f"{alpha:>6.2f}", end="")
        for g in range(1, 10):
            c, t = correct_by_gap[g]
            acc = c / t * 100 if t > 0 else float('nan')
            print(f" {acc:>4.0f}%", end="")
        print(f"  {total_correct/total*100:>4.0f}%")

    # Also try: attenuate ONLY waves after S
    print(f"\nAttenuate ONLY waves at t=S (alpha on waves only):")
    print(f"{'alpha':>6}", end="")
    for g in range(1, 10):
        p = "o" if g % 2 == 1 else "e"
        print(f"  g{g}{p}", end="")
    print("   all")
    print("-" * 62)

    for alpha in alphas:
        correct_by_gap = defaultdict(lambda: [0, 0])
        total_correct = 0
        total = 0

        for mt in range(10):
            for st in range(mt):
                x = th.zeros(10)
                x[mt] = 1.0
                x[st] = 0.8

                h = th.zeros(1, 16)
                for t in range(10):
                    x_t = x[t:t+1].unsqueeze(0)
                    pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
                    h = th.relu(pre)
                    if t == st:
                        for w in WAVES + BRIDGES:
                            h[0, w] *= alpha

                logits = (h @ W_out.T).detach().numpy()[0]
                pred = logits.argmax()
                gap = mt - st
                correct_by_gap[gap][1] += 1
                total += 1
                if pred == st:
                    correct_by_gap[gap][0] += 1
                    total_correct += 1

        print(f"{alpha:>6.2f}", end="")
        for g in range(1, 10):
            c, t = correct_by_gap[g]
            acc = c / t * 100 if t > 0 else float('nan')
            print(f" {acc:>4.0f}%", end="")
        print(f"  {total_correct/total*100:>4.0f}%")


# =============================================================================
# CLAIM 20: Clean margins predict failure order
# =============================================================================
def test_claim_20(W_ih, W_hh, W_out, X, y, mps, sps):
    section("CLAIM 20: Clean margins predict failure order")

    # Get clean margins for all 90 pairs
    logits_clean = run_model(W_ih, W_hh, W_out, X).detach().numpy()
    y_np = y.numpy()

    clean_margins = []
    for i in range(len(y_np)):
        correct = logits_clean[i, y_np[i]]
        wrong = np.delete(logits_clean[i], y_np[i]).max()
        clean_margins.append(correct - wrong)
    clean_margins = np.array(clean_margins)

    # Find failure threshold for each pair: lowest s_mag where it's still correct
    s_mags = np.arange(0.80, 0.0, -0.02)
    failure_smag = np.full(90, 0.0)  # s_mag at which pair first fails

    for s_mag in s_mags:
        samples = []
        for i, (mp, sp) in enumerate(zip(mps, sps)):
            x = th.zeros(10)
            x[mp] = 1.0
            x[sp] = s_mag
            samples.append(x)
        X_test = th.stack(samples)
        logits = run_model(W_ih, W_hh, W_out, X_test).detach().numpy()
        preds = logits.argmax(axis=1)
        for i in range(90):
            if preds[i] == y_np[i]:
                failure_smag[i] = s_mag  # last s_mag where correct

    # Correlation between clean margin and failure threshold
    # Higher margin → should survive to lower s_mag → failure_smag should be lower
    # So we expect negative correlation between margin and failure_smag
    # Or positive correlation between margin and "robustness" (survival s_mag)
    # Actually: failure_smag = last s_mag where correct. Lower = more robust.
    # Wait, that's backwards. Let me think...
    # If pair fails at s_mag=0.1, failure_smag=0.12 (last correct)
    # If pair never fails, failure_smag=0.02 (last tested)
    # More robust pairs have lower failure_smag
    # Higher margin should → lower failure_smag (more robust)
    # So correlation(margin, failure_smag) should be NEGATIVE

    from scipy.stats import spearmanr
    rho, p = spearmanr(clean_margins, failure_smag)

    print(f"Clean margin stats: mean={clean_margins.mean():.2f}, "
          f"min={clean_margins.min():.2f}, max={clean_margins.max():.2f}")
    print(f"Failure s_mag stats: mean={failure_smag.mean():.3f}, "
          f"min={failure_smag.min():.3f}, max={failure_smag.max():.3f}")
    print(f"\nSpearman correlation(clean_margin, failure_smag) = {rho:.4f} (p={p:.2e})")
    print(f"(Negative = higher margin survives longer, as expected)")

    # Show the 10 smallest-margin pairs and their failure points
    sorted_idx = np.argsort(clean_margins)
    print(f"\n10 smallest-margin pairs:")
    print(f"{'Mt':>3} {'St':>3} {'gap':>4} {'ord':>4} {'margin':>8} {'fail_at':>8}")
    print("-" * 34)
    for i in sorted_idx[:10]:
        gap = abs(int(mps[i]) - int(sps[i]))
        ordering = "fwd" if sps[i] > mps[i] else "rev"
        print(f"{mps[i]:>3} {sps[i]:>3} {gap:>4} {ordering:>4} "
              f"{clean_margins[i]:>8.3f} {failure_smag[i]:>8.2f}")

    print(f"\n10 largest-margin pairs:")
    for i in sorted_idx[-10:]:
        gap = abs(int(mps[i]) - int(sps[i]))
        ordering = "fwd" if sps[i] > mps[i] else "rev"
        print(f"{mps[i]:>3} {sps[i]:>3} {gap:>4} {ordering:>4} "
              f"{clean_margins[i]:>8.3f} {failure_smag[i]:>8.2f}")

    # By ordering
    fwd_mask = sps > mps
    rho_fwd, _ = spearmanr(clean_margins[fwd_mask], failure_smag[fwd_mask])
    rho_rev, _ = spearmanr(clean_margins[~fwd_mask], failure_smag[~fwd_mask])
    print(f"\nBy ordering: fwd rho={rho_fwd:.4f}, rev rho={rho_rev:.4f}")


# =============================================================================
# CLAIM 21: Predict random 2-impulse accuracy
# =============================================================================
def test_claim_21(W_ih, W_hh, W_out):
    section("CLAIM 21: Predict random 2-impulse accuracy")

    np.random.seed(42)
    N = 5000

    # Generate random 2-impulse inputs
    m_mags = np.random.uniform(0.5, 1.0, N)
    s_mags = np.array([np.random.uniform(0.1, mm) for mm in m_mags])
    m_pos = np.random.randint(0, 10, N)
    s_pos = np.random.randint(0, 9, N)
    # Shift s_pos to avoid collision with m_pos
    s_pos = np.where(s_pos >= m_pos, s_pos + 1, s_pos)

    samples = []
    labels = []
    for i in range(N):
        x = th.zeros(10)
        x[m_pos[i]] = m_mags[i]
        x[s_pos[i]] = s_mags[i]
        samples.append(x)
        labels.append(s_pos[i])

    X = th.stack(samples)
    y = th.tensor(labels)

    logits = run_model(W_ih, W_hh, W_out, X)
    preds = logits.argmax(dim=-1).numpy()
    y_np = y.numpy()
    actual_acc = (preds == y_np).mean() * 100

    print(f"Random 2-impulse: N={N}, actual accuracy = {actual_acc:.1f}%")

    # Breakdown by ordering and gap
    is_fwd = s_pos > m_pos
    gaps = np.abs(s_pos - m_pos)
    ratios = s_mags / m_mags

    fwd_acc = (preds[is_fwd] == y_np[is_fwd]).mean() * 100
    rev_acc = (preds[~is_fwd] == y_np[~is_fwd]).mean() * 100
    print(f"Forward: {fwd_acc:.1f}%, Reversed: {rev_acc:.1f}%")

    # By ratio bins
    print(f"\nBy S/M ratio:")
    print(f"{'ratio_bin':>10} {'acc':>6} {'n':>5}")
    print("-" * 24)
    bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    for lo, hi in bins:
        mask = (ratios >= lo) & (ratios < hi)
        if mask.sum() > 0:
            acc = (preds[mask] == y_np[mask]).mean() * 100
            print(f"[{lo:.1f},{hi:.1f}) {acc:>5.1f}% {mask.sum():>5}")

    # By gap
    print(f"\nBy gap:")
    print(f"{'gap':>4} {'fwd_acc':>8} {'rev_acc':>8} {'fwd_n':>6} {'rev_n':>6}")
    print("-" * 36)
    for g in range(1, 10):
        fwd_mask = (gaps == g) & is_fwd
        rev_mask = (gaps == g) & ~is_fwd
        fa = (preds[fwd_mask] == y_np[fwd_mask]).mean() * 100 if fwd_mask.sum() > 0 else float('nan')
        ra = (preds[rev_mask] == y_np[rev_mask]).mean() * 100 if rev_mask.sum() > 0 else float('nan')
        print(f"{g:>4} {fa:>7.1f}% {ra:>7.1f}% {fwd_mask.sum():>6} {rev_mask.sum():>6}")

    # Simple predictor: use clean margins + ratio scaling
    # For each pair, look up the (last_clip, gap, ordering) clean margin,
    # then predict correct if margin * (s_mag/0.8) > threshold
    # First get clean margins
    clean_margins = {}
    for mp in range(10):
        for sp in range(10):
            if mp == sp:
                continue
            x_clean = th.zeros(10)
            x_clean[mp] = 1.0
            x_clean[sp] = 0.8
            logits_clean = run_model(W_ih, W_hh, W_out, x_clean.unsqueeze(0))[0].detach().numpy()
            correct = logits_clean[sp]
            wrong = np.delete(logits_clean, sp).max()
            clean_margins[(mp, sp)] = correct - wrong

    # For each random pair, find the matching clean pair and scale margin
    # The matching clean pair has the same (Mt, St) positions
    predicted_correct = np.zeros(N, dtype=bool)
    for i in range(N):
        key = (int(m_pos[i]), int(s_pos[i]))
        if key in clean_margins:
            # Scale margin by ratio (heuristic: margin degrades with lower S/M ratio)
            scaled = clean_margins[key] * (ratios[i] / 0.8)
            predicted_correct[i] = scaled > 0
        else:
            predicted_correct[i] = False

    pred_acc = predicted_correct.mean() * 100
    actual_correct = (preds == y_np)
    agreement = (predicted_correct == actual_correct).mean() * 100

    print(f"\nSimple predictor (margin * ratio/0.8 > 0):")
    print(f"  Predicted accuracy: {pred_acc:.1f}%")
    print(f"  Actual accuracy:    {actual_acc:.1f}%")
    print(f"  Difference:         {abs(pred_acc - actual_acc):.1f}pp")
    print(f"  Per-pair agreement: {agreement:.1f}%")

    # Better predictor: threshold sweep
    print(f"\nThreshold sweep for margin * (ratio/0.8)^k predictor:")
    print(f"{'k':>4} {'threshold':>10} {'pred_acc':>9} {'actual':>7} {'diff':>6} {'agree':>7}")
    print("-" * 48)
    best_agree = 0
    best_params = (0, 0)
    for k in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for thresh in [0, 1, 2, 3, 5]:
            pc = np.zeros(N, dtype=bool)
            for i in range(N):
                key = (int(m_pos[i]), int(s_pos[i]))
                if key in clean_margins:
                    scaled = clean_margins[key] * (ratios[i] / 0.8)**k
                    pc[i] = scaled > thresh
            pa = pc.mean() * 100
            ag = (pc == actual_correct).mean() * 100
            diff = abs(pa - actual_acc)
            if ag > best_agree:
                best_agree = ag
                best_params = (k, thresh)
            if diff < 8:
                print(f"{k:>4.1f} {thresh:>10} {pa:>8.1f}% {actual_acc:>6.1f}% {diff:>5.1f} {ag:>6.1f}%")

    print(f"\nBest agreement: {best_agree:.1f}% at k={best_params[0]}, thresh={best_params[1]}")


def main():
    model = load_local_model()
    W_ih = model.rnn.weight_ih_l0.data.squeeze().clone()
    W_hh = model.rnn.weight_hh_l0.data.clone()
    W_out = model.linear.weight.data.clone()
    X, y, mps, sps = make_clean_dataset()

    test_claim_16(W_ih, W_hh, W_out)
    test_claim_17(W_ih, W_hh, X, mps, sps)
    test_claim_18(W_ih, W_hh, W_out, X, y, mps, sps)
    test_claim_19(W_ih, W_hh, W_out)
    test_claim_20(W_ih, W_hh, W_out, X, y, mps, sps)
    test_claim_21(W_ih, W_hh, W_out)


if __name__ == "__main__":
    main()
