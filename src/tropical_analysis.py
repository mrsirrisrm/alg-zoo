"""
Tropical geometry analysis of the RNN.

Each ReLU is a tropical max: relu(x) = max(x, 0).
The activation pattern (which neurons are active at each timestep) defines
a cell of the tropical hyperplane arrangement. Within each cell, the
input→output map is affine.

Questions:
1. How many distinct activation pattern sequences exist across 90 clean pairs?
2. Where do forward/reversed pairs diverge in activation pattern?
3. Which tropical hyperplane crossings correspond to the parity structure?
4. How do activation patterns change under perturbation (low S magnitude)?
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


COMPS = [1, 6, 7, 8]
WAVES = [0, 10, 11, 12, 14]
BRIDGES = [3, 5, 13, 15]


def run_trace_full(W_ih, W_hh, x_single):
    """Run model and return both h (post-relu) and pre (pre-relu) at each step."""
    h = th.zeros(1, 16)
    h_trace = []
    pre_trace = []
    for t in range(10):
        x_t = x_single[t:t+1].unsqueeze(0)
        pre = h @ W_hh.T + x_t * W_ih.unsqueeze(0)
        h = th.relu(pre)
        pre_trace.append(pre[0].detach().numpy().copy())
        h_trace.append(h[0].detach().numpy().copy())
    return np.array(h_trace), np.array(pre_trace)


def activation_pattern(pre_trace):
    """Convert pre-activation trace to binary activation pattern.
    1 = neuron active (pre > 0), 0 = dead (pre <= 0)."""
    return (pre_trace > 0).astype(int)


def pattern_to_str(pattern_2d):
    """Convert [10, 16] activation pattern to a hashable string."""
    return ''.join(str(b) for row in pattern_2d for b in row)


def section(title):
    print(f"\n{'='*80}")
    print(title)
    print('='*80)


def main():
    model = load_local_model()
    W_ih = model.rnn.weight_ih_l0.data.squeeze().clone()
    W_hh = model.rnn.weight_hh_l0.data.clone()
    W_out = model.linear.weight.data.clone().numpy()

    # =========================================================================
    # 1. Enumerate activation patterns for all 90 clean pairs
    # =========================================================================
    section("1. Activation patterns for all 90 clean pairs")

    patterns = {}  # (mt, st) -> activation pattern [10, 16]
    pattern_strs = {}  # (mt, st) -> string
    h_traces = {}
    pre_traces = {}

    for mt in range(10):
        for st in range(10):
            if mt == st:
                continue
            x = th.zeros(10)
            x[mt] = 1.0
            x[st] = 0.8
            h_trace, pre_trace = run_trace_full(W_ih, W_hh, x)
            ap = activation_pattern(pre_trace)
            patterns[(mt, st)] = ap
            pattern_strs[(mt, st)] = pattern_to_str(ap)
            h_traces[(mt, st)] = h_trace
            pre_traces[(mt, st)] = pre_trace

    unique_patterns = set(pattern_strs.values())
    print(f"\nTotal pairs: {len(patterns)}")
    print(f"Distinct activation pattern sequences: {len(unique_patterns)}")

    # Check if any pairs share patterns
    str_to_pairs = defaultdict(list)
    for (mt, st), s in pattern_strs.items():
        str_to_pairs[s].append((mt, st))

    shared = {s: pairs for s, pairs in str_to_pairs.items() if len(pairs) > 1}
    if shared:
        print(f"\nPairs sharing activation patterns ({len(shared)} groups):")
        for s, pairs in shared.items():
            print(f"  {pairs}")
    else:
        print("\nNo pairs share activation patterns — all 90 are in distinct tropical cells.")

    # =========================================================================
    # 2. Per-timestep activation pattern: how many neurons are active?
    # =========================================================================
    section("2. Neurons active per timestep (averaged)")

    fwd_counts = np.zeros((10, 16))
    rev_counts = np.zeros((10, 16))
    n_fwd = 0
    n_rev = 0

    for (mt, st), ap in patterns.items():
        if st > mt:
            fwd_counts += ap
            n_fwd += 1
        else:
            rev_counts += ap
            n_rev += 1

    print(f"\nFraction of pairs where neuron is active, by timestep:")
    print(f"\nFORWARD ({n_fwd} pairs):")
    print(f"{'t':>3}", end="")
    for n in range(16):
        print(f" {'n'+str(n):>5}", end="")
    print(f" {'total':>6}")
    for t in range(10):
        print(f"{t:>3}", end="")
        for n in range(16):
            print(f" {fwd_counts[t, n]/n_fwd:>5.2f}", end="")
        print(f" {fwd_counts[t].sum()/n_fwd:>6.1f}")

    print(f"\nREVERSED ({n_rev} pairs):")
    print(f"{'t':>3}", end="")
    for n in range(16):
        print(f" {'n'+str(n):>5}", end="")
    print(f" {'total':>6}")
    for t in range(10):
        print(f"{t:>3}", end="")
        for n in range(16):
            print(f" {rev_counts[t, n]/n_rev:>5.2f}", end="")
        print(f" {rev_counts[t].sum()/n_rev:>6.1f}")

    # =========================================================================
    # 3. Where do forward/reversed pairs diverge for same (last_clip, gap)?
    # =========================================================================
    section("3. Activation pattern divergence: forward vs reversed at same (lc, gap)")

    print(f"\nFor each (lc, gap), the timestep where activation patterns first differ:")
    print(f"{'lc':>3} {'gap':>4} {'diverge_t':>10} {'neurons_diff_at_div':>20} {'total_diffs':>12}")
    print("-" * 55)

    for gap in range(1, 10):
        for lc in range(gap, 10):
            mt_f = lc - gap
            st_f = lc
            mt_r = lc
            st_r = lc - gap

            ap_f = patterns[(mt_f, st_f)]
            ap_r = patterns[(mt_r, st_r)]

            # Find first divergence timestep
            div_t = None
            div_neurons = []
            for t in range(10):
                diff = np.where(ap_f[t] != ap_r[t])[0]
                if len(diff) > 0 and div_t is None:
                    div_t = t
                    div_neurons = diff.tolist()

            total_diffs = (ap_f != ap_r).sum()

            neurons_str = ','.join(f'n{n}' for n in div_neurons) if div_neurons else '-'
            div_str = str(div_t) if div_t is not None else "SAME"
            print(f"{lc:>3} {gap:>4} {div_str:>10} {neurons_str:>20} {total_diffs:>12}")

    # =========================================================================
    # 4. Activation pattern at the rebuild phase (after last_clip)
    #    Compare odd vs even gaps for reversed pairs
    # =========================================================================
    section("4. Rebuild-phase activation patterns: odd vs even gap (reversed)")

    print(f"\nActivation pattern at each rebuild step after last_clip (reversed pairs):")
    print(f"Showing which comps are active at each step after last_clip.\n")

    for gap in range(1, 8):
        lc = 7  # fix last_clip = 7 for comparison
        mt_r = 7
        st_r = 7 - gap
        if st_r < 0:
            continue

        ap = patterns[(mt_r, st_r)]
        parity = "odd" if gap % 2 == 1 else "even"
        print(f"  gap={gap} ({parity:>4}), rebuild t=8,9:")
        for t in [7, 8, 9]:
            active = [f'n{n}' for n in range(16) if ap[t, n] == 1]
            dead = [f'n{n}' for n in range(16) if ap[t, n] == 0]
            print(f"    t={t}: active={active}")
        print()

    # =========================================================================
    # 5. Which neurons CHANGE activation during rebuild (tropical hyperplane crossings)
    # =========================================================================
    section("5. Tropical hyperplane crossings during rebuild (reversed, lc=7)")

    print(f"Neurons that change activation state between consecutive rebuild steps:\n")
    print(f"{'gap':>4} {'parity':>6} {'t7→t8 crossings':>30} {'t8→t9 crossings':>30}")
    print("-" * 75)

    for gap in range(1, 8):
        mt_r = 7
        st_r = 7 - gap
        if st_r < 0:
            continue

        ap = patterns[(mt_r, st_r)]
        parity = "odd" if gap % 2 == 1 else "even"

        crossings_78 = []
        crossings_89 = []
        for n in range(16):
            if ap[7, n] != ap[8, n]:
                direction = "↑" if ap[8, n] == 1 else "↓"
                crossings_78.append(f"n{n}{direction}")
            if ap[8, n] != ap[9, n]:
                direction = "↑" if ap[9, n] == 1 else "↓"
                crossings_89.append(f"n{n}{direction}")

        c78 = ', '.join(crossings_78) if crossings_78 else "none"
        c89 = ', '.join(crossings_89) if crossings_89 else "none"
        print(f"{gap:>4} {parity:>6} {c78:>30} {c89:>30}")

    # =========================================================================
    # 6. How do activation patterns change under perturbation?
    #    Compare clean (s_mag=0.8) vs weak (s_mag=0.1) for reversed pairs
    # =========================================================================
    section("6. Activation pattern changes under weak S (reversed pairs)")

    print(f"\nComparing clean (s=0.8) vs weak (s=0.1) activation patterns.")
    print(f"Showing timesteps and neurons where patterns differ.\n")

    for gap in [1, 2, 3, 5]:
        for lc in [5, 7]:
            mt_r = lc
            st_r = lc - gap
            if st_r < 0:
                continue

            # Clean
            x_clean = th.zeros(10)
            x_clean[mt_r] = 1.0
            x_clean[st_r] = 0.8
            _, pre_clean = run_trace_full(W_ih, W_hh, x_clean)
            ap_clean = activation_pattern(pre_clean)

            # Weak
            x_weak = th.zeros(10)
            x_weak[mt_r] = 1.0
            x_weak[st_r] = 0.1
            h_weak, pre_weak = run_trace_full(W_ih, W_hh, x_weak)
            ap_weak = activation_pattern(pre_weak)

            # Check prediction
            logits_weak = h_weak[9] @ W_out.T
            pred_weak = logits_weak.argmax()
            correct = pred_weak == st_r

            n_diffs = (ap_clean != ap_weak).sum()

            print(f"  (M{mt_r},S{st_r}) gap={gap}: {n_diffs} activation diffs, "
                  f"weak pred={pred_weak} {'OK' if correct else 'WRONG'}")

            # Show where they differ
            for t in range(10):
                diff_neurons = np.where(ap_clean[t] != ap_weak[t])[0]
                if len(diff_neurons) > 0:
                    changes = []
                    for n in diff_neurons:
                        changes.append(f"n{n}:{ap_clean[t,n]}→{ap_weak[t,n]}")
                    print(f"    t={t}: {', '.join(changes)}")
            print()

    # =========================================================================
    # 7. Pre-activation margins: how close are neurons to crossing?
    #    This measures distance to the nearest tropical hyperplane.
    # =========================================================================
    section("7. Pre-activation margins at t=9 (distance to tropical hyperplane)")

    print(f"How close each neuron's pre-activation is to 0 at the final timestep.")
    print(f"Small |pre| = close to tropical hyperplane = fragile.\n")

    # Compute for all reversed pairs
    print(f"REVERSED pairs — mean |pre[9]| by neuron:")
    print(f"{'neuron':>7} {'mean|pre|':>10} {'min|pre|':>10} {'frac_active':>12}")
    print("-" * 45)

    rev_pres = []
    for mt in range(10):
        for st in range(mt):
            rev_pres.append(pre_traces[(mt, st)][9])
    rev_pres = np.array(rev_pres)

    for n in range(16):
        vals = rev_pres[:, n]
        active_frac = (vals > 0).mean()
        print(f"  n{n:<5} {np.mean(np.abs(vals)):>10.3f} {np.min(np.abs(vals)):>10.3f} {active_frac:>12.2f}")

    # =========================================================================
    # 8. Count distinct cells at each phase of computation
    # =========================================================================
    section("8. Distinct activation patterns at each timestep")

    print(f"\nHow many distinct activation vectors exist at each timestep?")
    print(f"{'t':>3} {'all':>5} {'fwd':>5} {'rev':>5}")
    print("-" * 22)

    for t in range(10):
        all_pats = set()
        fwd_pats = set()
        rev_pats = set()
        for (mt, st), ap in patterns.items():
            pat_str = ''.join(str(b) for b in ap[t])
            all_pats.add(pat_str)
            if st > mt:
                fwd_pats.add(pat_str)
            else:
                rev_pats.add(pat_str)
        print(f"{t:>3} {len(all_pats):>5} {len(fwd_pats):>5} {len(rev_pats):>5}")

    # Do fwd and rev share any patterns at t=9?
    fwd_final = set()
    rev_final = set()
    for (mt, st), ap in patterns.items():
        pat_str = ''.join(str(b) for b in ap[9])
        if st > mt:
            fwd_final.add(pat_str)
        else:
            rev_final.add(pat_str)
    shared_final = fwd_final & rev_final
    print(f"\nt=9: fwd patterns shared with rev: {len(shared_final)}")
    if shared_final:
        for pat in shared_final:
            fwd_pairs = [(mt, st) for (mt, st) in patterns
                         if st > mt and ''.join(str(b) for b in patterns[(mt, st)][9]) == pat]
            rev_pairs = [(mt, st) for (mt, st) in patterns
                         if st < mt and ''.join(str(b) for b in patterns[(mt, st)][9]) == pat]
            active = [f'n{i}' for i, b in enumerate(pat) if b == '1']
            print(f"  Pattern {active}: fwd={fwd_pairs}, rev={rev_pairs}")

    # =========================================================================
    # 9. The full tropical cell identity: cumulative pattern through time
    # =========================================================================
    section("9. Cumulative cell identity: when do pairs become distinguishable?")

    print(f"\nNumber of distinct cumulative patterns (t=0..T) as T grows:")
    print(f"{'T':>3} {'all':>5} {'fwd':>5} {'rev':>5}")
    print("-" * 22)

    for T in range(10):
        all_pats = set()
        fwd_pats = set()
        rev_pats = set()
        for (mt, st), ap in patterns.items():
            # Cumulative: pattern from t=0 to T
            pat_str = ''.join(str(b) for row in ap[:T+1] for b in row)
            all_pats.add(pat_str)
            if st > mt:
                fwd_pats.add(pat_str)
            else:
                rev_pats.add(pat_str)
        print(f"{T:>3} {len(all_pats):>5} {len(fwd_pats):>5} {len(rev_pats):>5}")


if __name__ == "__main__":
    main()
