"""Hypothesis: S = argmax(A, B), A = max before M, B = max after M.
The network compares A vs B; errors = picking the other candidate."""
import sys; sys.path.insert(0, '.')
import torch as th, numpy as np
from ground_truth import forward, load_weights
W = [th.tensor(a) for a in load_weights('weights.npz')]
th.manual_seed(4); N = 1 << 22; x = th.randn(N, 10)
pred = forward(x, *W).argmax(-1)
M = x.argmax(-1); vM = x.max(-1).values; S = x.topk(2, -1).indices[:, 1]
tt = th.arange(10).expand(N, 10)
xb = th.where(tt < M[:, None], x, -th.inf); xa = th.where(tt > M[:, None], x, -th.inf)
A = xb.argmax(1); vA = xb.max(1).values; B = xa.argmax(1); vB = xa.max(1).values
hasA = M > 0; hasB = M < 9; both = hasA & hasB
ok = pred == S
other = th.where(S == A, B, A)
err = ~ok
print(f'errors: pick other candidate {(pred == other)[err & both].float().mean():.3f} (of errors with both candidates); pick M {(pred == M)[err].float().mean():.3f}')
print(f'acc: M at 0 (no A) {ok[~hasA].float().mean():.4f}, M at 9 (no B) {ok[~hasB].float().mean():.4f}, both {ok[both].float().mean():.4f}')
d = ((vA - vB).abs() / vM)[both]
okb = ok[both]; po = (pred == other)[both]
print(' |vA-vB|/vM   acc     P(pick other)  frac')
for lo, hi in [(0, .01), (.01, .02), (.02, .04), (.04, .08), (.08, .16), (.16, .32), (.32, .64), (.64, 10)]:
    q = (d >= lo) & (d < hi)
    print(f' [{lo:.2f},{hi:.2f})  {okb[q].float().mean():.4f}   {po[q].float().mean():.4f}        {q.float().mean():.3f}')

print('\n--- error budget over all inputs (rates are fractions of ALL inputs)')
sideS = th.where(S < M, 0, 1); sidep = th.where(pred < M, 0, th.where(pred > M, 1, -1))
cats = {
    'pred == M (M/S confusion)': err & (pred == M),
    'A/B swap (other side\'s best)': err & both & (pred == other),
    'within S\'s side, not S': err & (sidep == sideS) & (pred != M),
    'other side, not its best': err & (sidep != sideS) & (sidep >= 0) & (pred != other),
}
tot = err.float().mean().item()
for k, v in cats.items():
    r = v.float().mean().item(); print(f'  {k:34s} {r:.4f}  ({r / tot:.2f} of errors)')
print(f'  total error rate {tot:.4f}')
w = err & (sidep == sideS) & (pred != M)
print('within-side errors by side: S before M {:.4f}, S after M {:.4f}'.format(
    (w & (S < M)).float().mean().item(), (w & (S > M)).float().mean().item()))
