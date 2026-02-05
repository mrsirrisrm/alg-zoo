"""
Mechanistic Discrimination for M₁₆,₁₀ RNN

This module implements a complete weight-based explanation of how the 2nd argmax
RNN discriminates between input positions. Instead of treating the RNN as a black
box, we decompose its computation into interpretable components:

1. Canonical D sequence: The pattern of active neurons after a single impulse
2. W_eff: The effective linear map (product of masked W_hh matrices)
3. Offset mechanism: Why the network favors S position over M position

Key insight: The ReLU activation pattern D[t] is almost entirely determined by
a single "canonical" sequence that can be precomputed from weights. Only the
timestep when the second impulse arrives requires fresh computation.

This reduces "simulation" to table lookups + ~50 arithmetic operations.
"""

import numpy as np


class MechanisticDiscriminator:
    """
    Weight-based discriminator for 2nd argmax RNN.

    Precomputes the canonical D sequence from weights, then uses lookups
    and minimal computation to predict discrimination for any input.
    """

    def __init__(self, W_ih, W_hh, W_out):
        """
        Initialize with RNN weights.

        Args:
            W_ih: Input-to-hidden weights, shape (hidden_size,)
            W_hh: Hidden-to-hidden weights, shape (hidden_size, hidden_size)
            W_out: Hidden-to-output weights, shape (output_size, hidden_size)
        """
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.W_out = W_out
        self.hidden_size = len(W_ih)
        self.seq_len = W_out.shape[0]

        # Precompute canonical sequences
        self._precompute_canonical()

    def _precompute_canonical(self):
        """
        Precompute the canonical D sequence and pre-activations.

        This is done once and applies to all inputs. The canonical sequence
        represents the evolution of active neurons after a single positive
        impulse, which is independent of position and magnitude.
        """
        self.canonical_D = []      # Active neuron sets
        self.canonical_pre = []    # Pre-activations (before ReLU)
        self.canonical_h = []      # Hidden states (after ReLU)

        # At first impulse: h = ReLU(W_ih * val) = val * ReLU(W_ih)
        # D[0] = {i : W_ih[i] > 0} (same for any positive val)
        h = np.maximum(0, self.W_ih)
        self.canonical_D.append(frozenset(np.where(self.W_ih > 0)[0]))
        self.canonical_pre.append(self.W_ih.copy())
        self.canonical_h.append(h.copy())

        # Iterate: h[k+1] = ReLU(W_hh @ h[k])
        for k in range(1, self.seq_len):
            pre = self.W_hh @ h
            self.canonical_D.append(frozenset(np.where(pre > 0)[0]))
            self.canonical_pre.append(pre.copy())
            h = np.maximum(0, pre)
            self.canonical_h.append(h.copy())

    def compute_D_sequence(self, m, s, M_val, S_val):
        """
        Compute the D[t] sequence for a specific input.

        Uses precomputed canonical sequences for most timesteps.
        Only the second impulse and subsequent steps need fresh computation.

        Args:
            m: Position of M (max value)
            s: Position of S (second max value)
            M_val: Magnitude of M
            S_val: Magnitude of S

        Returns:
            List of frozensets, D[t] for t = 0, ..., seq_len-1
        """
        D_seq = []

        # Determine order
        first_pos = min(m, s)
        second_pos = max(m, s)
        first_val = M_val if m < s else S_val
        second_val = S_val if m < s else M_val

        # Phase 1: Before first impulse - empty
        for t in range(first_pos):
            D_seq.append(frozenset())

        # Phase 2: At and after first impulse, before second - canonical lookup
        for t in range(first_pos, second_pos):
            k = t - first_pos
            D_seq.append(self.canonical_D[k])

        # Phase 3: At second impulse - linear combination
        gap = second_pos - first_pos
        pre_at_second = first_val * self.canonical_pre[gap] + second_val * self.W_ih
        D_at_second = frozenset(np.where(pre_at_second > 0)[0])
        D_seq.append(D_at_second)

        # Phase 4: After second impulse - iterate (typically 0-2 steps)
        h = np.maximum(0, pre_at_second)
        for t in range(second_pos + 1, self.seq_len):
            pre = self.W_hh @ h
            D_seq.append(frozenset(np.where(pre > 0)[0]))
            h = np.maximum(0, pre)

        return D_seq

    def compute_final_state(self, m, s, M_val, S_val, D_seq=None):
        """
        Compute the final hidden state h[seq_len-1].

        Args:
            m, s, M_val, S_val: Input specification
            D_seq: Optional precomputed D sequence

        Returns:
            Final hidden state, shape (hidden_size,)
        """
        if D_seq is None:
            D_seq = self.compute_D_sequence(m, s, M_val, S_val)

        h = np.zeros(self.hidden_size)
        for t in range(self.seq_len):
            # Input at this timestep
            x = M_val if t == m else (S_val if t == s else 0.0)

            # Pre-activation
            pre = self.W_hh @ h + self.W_ih * x

            # Apply D[t] mask (equivalent to ReLU given correct D)
            D_mask = np.array([1.0 if i in D_seq[t] else 0.0
                              for i in range(self.hidden_size)])
            h = D_mask * np.maximum(0, pre)

        return h

    def predict(self, m, s, M_val=1.0, S_val=0.8):
        """
        Predict the network output for given input.

        Args:
            m: Position of M (max value)
            s: Position of S (second max value)
            M_val: Magnitude of M (default 1.0)
            S_val: Magnitude of S (default 0.8)

        Returns:
            Tuple of (predicted_position, margin, D_sequence)
        """
        D_seq = self.compute_D_sequence(m, s, M_val, S_val)
        h_final = self.compute_final_state(m, s, M_val, S_val, D_seq)

        logits = self.W_out @ h_final
        pred = np.argmax(logits)
        target = s  # 2nd argmax position
        margin = logits[target] - np.max([logits[i] for i in range(self.seq_len) if i != target])

        return pred, margin, D_seq

    def explain(self, m, s, M_val=1.0, S_val=0.8):
        """
        Provide a detailed explanation of the discrimination.

        Args:
            m, s, M_val, S_val: Input specification

        Returns:
            Dictionary with explanation components
        """
        D_seq = self.compute_D_sequence(m, s, M_val, S_val)
        h_final = self.compute_final_state(m, s, M_val, S_val, D_seq)
        logits = self.W_out @ h_final

        # Compute forward and reverse for offset analysis
        h_fwd = self.compute_final_state(m, s, M_val, S_val)
        h_rev = self.compute_final_state(s, m, M_val, S_val)
        h_main = (h_fwd + h_rev) / 2
        offset = h_fwd - h_rev

        # Margin decomposition
        discrim = self.W_out[s] - self.W_out[m]
        h_main_contrib = discrim @ h_main
        offset_contrib = discrim @ (offset / 2)

        return {
            'prediction': np.argmax(logits),
            'target': s,
            'correct': np.argmax(logits) == s,
            'margin': logits[s] - np.max([logits[i] for i in range(self.seq_len) if i != s]),
            'D_sequence': D_seq,
            'h_final': h_final,
            'h_main': h_main,
            'offset': offset,
            'h_main_contribution': h_main_contrib,
            'offset_contribution': offset_contrib,
            'logits': logits,
        }

    def predict_sequence(self, x):
        """
        Predict the 2nd argmax for a real input sequence.

        This extracts M (max) and S (2nd max) positions and values from
        the input, then uses the mechanistic predictor. Note that this
        ignores other input values (simplified model).

        Args:
            x: Input sequence, shape (seq_len,)

        Returns:
            Dictionary with prediction details
        """
        x = np.asarray(x)
        sorted_idx = np.argsort(x)
        m = sorted_idx[-1]   # Position of max
        s = sorted_idx[-2]   # Position of 2nd max (target)
        M_val = x[m]
        S_val = x[s]

        pred, margin, D_seq = self.predict(m, s, M_val, S_val)

        return {
            'input': x,
            'target': s,
            'prediction': pred,
            'correct': pred == s,
            'margin': margin,
            'm': m,
            's': s,
            'M_val': M_val,
            'S_val': S_val,
            'gap_MS': M_val - S_val,
            'gap_S3': S_val - x[sorted_idx[-3]] if len(x) > 2 else None,
            'D_sequence': D_seq,
        }


def load_from_weights_file(path):
    """Load discriminator from weights file."""
    w = np.load(path)
    return MechanisticDiscriminator(w['W_ih'], w['W_hh'], w['W_out'])

if __name__ == '__main__':
    # Load weights
    w = np.load('/tmp/weights.npz')
    disc = MechanisticDiscriminator(w['W_ih'], w['W_hh'], w['W_out'])

    print("=" * 70)
    print("MECHANISTIC DISCRIMINATOR DEMONSTRATION")
    print("=" * 70)

    # Show canonical D sequence
    print("\n### Canonical D sequence (precomputed from weights):")
    for k in range(10):
        print(f"  D[{k}]: {len(disc.canonical_D[k]):2d} neurons active")

    # Test all 90 pairs
    print("\n### Testing all 90 (m, s) pairs:")
    correct = 0
    margins = []
    for m in range(10):
        for s in range(10):
            if m != s:
                pred, margin, _ = disc.predict(m, s)
                if pred == s:
                    correct += 1
                margins.append(margin)

    print(f"  Accuracy: {correct}/90 = {100*correct/90:.1f}%")
    print(f"  Min margin: {min(margins):.2f}")
    print(f"  Mean margin: {np.mean(margins):.2f}")

    # Detailed explanation for one pair
    print("\n### Detailed explanation for (m=2, s=7):")
    exp = disc.explain(2, 7)
    print(f"  Prediction: {exp['prediction']} (target: {exp['target']})")
    print(f"  Correct: {exp['correct']}")
    print(f"  Margin: {exp['margin']:.2f}")
    print(f"  h_main contribution: {exp['h_main_contribution']:.2f}")
    print(f"  offset contribution: {exp['offset_contribution']:.2f}")

    print("\n### D sequence for (m=2, s=7):")
    for t, D in enumerate(exp['D_sequence']):
        label = ""
        if t == 2:
            label = " <- M arrives"
        elif t == 7:
            label = " <- S arrives"
        print(f"  t={t}: {len(D):2d} neurons{label}")

    # Test on real data sequences
    print("\n" + "=" * 70)
    print("REAL DATA SEQUENCES (randn)")
    print("=" * 70)

    np.random.seed(42)
    for i in range(5):
        x = np.random.randn(10)
        result = disc.predict_sequence(x)

        status = "OK" if result['correct'] else "MISS"
        print(f"\n### Sample {i+1}: [{status}]")
        print(f"  Input: [{', '.join(f'{v:+.2f}' for v in x)}]")
        print(f"  M at pos {result['m']} = {result['M_val']:+.2f}")
        print(f"  S at pos {result['s']} = {result['S_val']:+.2f} (target)")
        print(f"  Prediction: {result['prediction']}, Margin: {result['margin']:.2f}")
        print(f"  Gap M-S: {result['gap_MS']:.3f}, Gap S-3rd: {result['gap_S3']:.3f}")

    # Accuracy on larger sample
    print("\n### Accuracy on 1000 real sequences:")
    correct = 0
    for _ in range(1000):
        x = np.random.randn(10)
        result = disc.predict_sequence(x)
        if result['correct']:
            correct += 1
    print(f"  {correct}/1000 = {100*correct/1000:.1f}%")
