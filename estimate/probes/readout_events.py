"""Does the readout follow n4 firing events? Split by M-first / S-first."""
import sys; sys.path.insert(0, '.')
import torch as th, numpy as np
from ground_truth import load_weights
Wi, Wh, Wo = [th.tensor(a) for a in load_weights('weights.npz')]
th.manual_seed(0); N = 1 << 20; x = th.randn(N, 10); h = th.zeros(N, 16); on4 = []
for t in range(10):
    pre = x[:, t:t+1] * Wi + h @ Wh.T; on4.append(pre[:, 4] > 0); h = th.relu(pre)
on4 = th.stack(on4, 1)
pred = (h @ Wo.T).argmax(-1); top = x.topk(3, -1); M, S, R = top.indices.unbind(-1)
ok = pred == S; ar = th.arange(N)
# last n4 event, and last n4 event strictly before it
tt = th.arange(10).expand(N, 10)
ev = th.where(on4, tt, th.full_like(tt, -1))
last = ev.max(1).values
ev2 = th.where(on4 & (tt < last[:, None]), tt, th.full_like(tt, -1))
prev = ev2.max(1).values
for name, sel in [('M first', M < S), ('S first', S < M)]:
    print(name, f'acc {ok[sel].float().mean():.4f}')
    print(f'  P(n4 fires at S) {on4[ar, S][sel].float().mean():.3f}   at M {on4[ar, M][sel].float().mean():.3f}   at R {on4[ar, R][sel].float().mean():.3f}')
    print(f'  last n4 event == M: {(last == M)[sel].float().mean():.3f};  previous event == S: {(prev == S)[sel].float().mean():.3f}')
    print(f'  pred == previous event: {(pred == prev)[sel].float().mean():.3f};  pred == last event: {(pred == last)[sel].float().mean():.3f}')
    fs = sel & on4[ar, S]; nf = sel & ~on4[ar, S]
    print(f'  acc when n4 fired at S {ok[fs].float().mean():.4f} (frac {fs.float().sum()/sel.float().sum():.3f}); when not {ok[nf].float().mean():.4f}')

print('\n--- error budget')
rev = S < M; fwd = M < S
p = prev == S
print(f'reverse: P(prev==S) {p[rev].float().mean():.4f}; acc|prev==S {ok[rev & p].float().mean():.4f}; acc|prev!=S {ok[rev & ~p].float().mean():.4f}')
bad = rev & ~p
print(f'  when prev!=S: prev==R {(prev == R)[bad].float().mean():.3f}; prev between S and M {((prev > S) & (prev < M))[bad].float().mean():.3f}; pred==prev {(pred == prev)[bad].float().mean():.3f}')
rb = R < M
print(f'forward: R before M frac {rb[fwd].float().mean():.3f}: acc {ok[fwd & rb].float().mean():.4f};  R after M: acc {ok[fwd & ~rb].float().mean():.4f}')
wr = fwd & ~ok
print(f'  forward errors: pred==R {(pred == R)[wr].float().mean():.3f}, pred==M {(pred == M)[wr].float().mean():.3f}, pred before M {(pred < M)[wr].float().mean():.3f}')
# forward: is the network's choice the argmax of post-M values?
post = th.where(tt > M[:, None], x, th.full_like(x, -1e9)); pm = post.argmax(1)
print(f'  forward: S == argmax post-M (sanity) {(pm == S)[fwd].float().mean():.3f}')
