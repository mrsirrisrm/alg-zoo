"""Reverse branch (S before M) with a correct event chain (prev n4 event == S):
what predicts the remaining errors?"""
import sys; sys.path.insert(0, '.')
import torch as th, numpy as np
from ground_truth import load_weights
np.set_printoptions(precision=3, suppress=True, linewidth=200)
Wi, Wh, Wo = [th.tensor(a) for a in load_weights('weights.npz')]
th.manual_seed(3); N = 1 << 22; x = th.randn(N, 10); h = th.zeros(N, 16); on4 = []
for t in range(10):
    pre = x[:, t:t+1] * Wi + h @ Wh.T; on4.append(pre[:, 4] > 0); h = th.relu(pre)
on4 = th.stack(on4, 1); pred = (h @ Wo.T).argmax(-1)
top = x.topk(3, -1); M, S, R = top.indices.unbind(-1); vM, vS, vR = top.values.unbind(-1)
tt = th.arange(10).expand(N, 10)
last = th.where(on4, tt, -1).max(1).values
prev = th.where(on4 & (tt < last[:, None]), tt, -1).max(1).values
sel = (S < M) & (prev == S) & (last == M)
ok = (pred == S)[sel]; Ms, Ss, Rs = M[sel], S[sel], R[sel]
print(f'subset frac {sel.float().mean():.3f}, acc {ok.float().mean():.4f}')
wrong = ~ok; pw = pred[sel][wrong]
print(f'errors pick: R {(pw == Rs[wrong]).float().mean():.3f}, M {(pw == Ms[wrong]).float().mean():.3f}, other {((pw != Rs[wrong]) & (pw != Ms[wrong])).float().mean():.3f}')
print(f'errors pick a position before S {(pw < Ss[wrong]).float().mean():.3f}, between S,M {((pw > Ss[wrong]) & (pw < Ms[wrong])).float().mean():.3f}, after M {(pw > Ms[wrong]).float().mean():.3f}')
print(f'|pred - S| for errors: ', np.bincount((pw - Ss[wrong]).abs().numpy(), minlength=10)[:10] / wrong.sum().item())
A = np.full((10, 10), np.nan)
for s_ in range(10):
    for m_ in range(s_ + 1, 10):
        q = (Ss == s_) & (Ms == m_)
        if q.sum() > 500: A[s_, m_] = ok[q].float().mean()
print('acc by (row S pos, col M pos):'); print(A)
d = (Ms - Ss)
print('acc by M-S distance:', [round(ok[d == k].float().mean().item(), 4) for k in range(1, 10)])
print('acc by #steps after M:', [round(ok[(9 - Ms) == k].float().mean().item(), 4) for k in range(0, 9)])
