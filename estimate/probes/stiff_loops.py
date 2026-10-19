"""Where the Gauss-Newton stiffness sits (precision.py, relative coords): energy of the top
eigenvectors on self-loops vs the W_hh block among the stiff neurons {1,2,6,7,8}, and that
block's weights and eigenvalues. Run from estimate/."""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'probes')
import numpy as np, torch as th
from ground_truth import load_weights
from precision import gauss_newton
W = [th.tensor(a, dtype=th.float32) for a in load_weights('weights.npz')]
Wih, Whh, Wout = [w.numpy() for w in W]
H = gauss_newton(W).numpy().astype(np.float64)
lam, U = np.linalg.eigh(H); lam, U = lam[::-1], U[:, ::-1]
def name(a):
    if a < 16: return f"in->n{a}"
    if a < 272: i, j = divmod(a - 16, 16); return f"n{j}->n{i}" + (" (self)" if i == j else "")
    i, j = divmod(a - 272, 16); return f"n{j}->out{i}"
S = {2, 7, 8, 1, 6}
diag = [16 + 17 * i for i in range(16)]
within = [16 + 16 * i + j for i in S for j in S]
print("stiff set", sorted(S))
for k in range(6):
    u = U[:, k] ** 2
    top = np.argsort(u)[::-1][:6]
    print(f"\neigvec {k} ({lam[k] / lam.sum():.3f} of trace): self-loop energy {u[diag].sum():.2f}, "
          f"within-stiff-set W_hh {u[within].sum():.2f}, all W_hh {u[16:272].sum():.2f}")
    for a in top:
        w = Wih[a] if a < 16 else (Whh.ravel()[a - 16] if a < 272 else Wout.ravel()[a - 272])
        print(f"   {name(a):18s} energy {u[a]:.3f}  weight {w:+.3f}")
# whole-H view: stiffness (diag of H, relative coords) by category
d = np.diag(H); tot = d.sum()
print(f"\ndiag(H) share: self-loops {d[diag].sum() / tot:.3f}, within-stiff-set W_hh {d[within].sum() / tot:.3f} "
      f"(of which self {sum(d[16 + 17 * i] for i in S) / tot:.3f}), other W_hh {(d[16:272].sum() - d[within].sum()) / tot:.3f}, "
      f"W_ih {d[:16].sum() / tot:.3f}, W_out {d[272:].sum() / tot:.3f}")
idx = sorted(S)
print("\nW_hh restricted to stiff set (row = receiving, col = sending):")
print("      " + " ".join(f"n{j:<6d}" for j in idx))
for i in idx: print(f"n{i:<4d} " + " ".join(f"{Whh[i, j]:+.3f} " for j in idx))
print("eigenvalues of that 5x5 block:", np.round(np.linalg.eigvals(Whh[np.ix_(idx, idx)]), 3))
print("self weights all neurons:", {i: round(float(Whh[i, i]), 3) for i in range(16)})
