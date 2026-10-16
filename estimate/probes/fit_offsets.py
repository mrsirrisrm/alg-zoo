import sys; sys.path.insert(0,'.')
import torch as th, numpy as np
from ground_truth import forward, load_weights
np.set_printoptions(precision=3, suppress=True, linewidth=220)
W=[th.tensor(a) for a in load_weights('weights.npz')]
th.manual_seed(0); N=1<<20; x=th.randn(N,10)
pred=forward(x,*W).argmax(-1)
m=x.argmax(-1); s=x.topk(2,-1).indices[:,1]
print('net predicts M: %.4f, predicts S: %.4f'%((pred==m).float().mean(),(pred==s).float().mean()))
keep=pred!=m
xk,mk,pk=x[keep],m[keep],pred[keep]; xM=xk.gather(1,mk[:,None])
beta=th.zeros(10,10,requires_grad=True); logtau=th.tensor(-3.0,requires_grad=True)
opt=th.optim.Adam([beta,logtau],lr=0.05)
mask=th.nn.functional.one_hot(mk,10).bool()
for it in range(400):
    sc=(xk-beta[mk]*xM)/logtau.exp()
    sc=sc.masked_fill(mask,-1e9)
    loss=th.nn.functional.cross_entropy(sc,pk)
    opt.zero_grad(); loss.backward(); opt.step()
with th.no_grad():
    sc=(xk-beta[mk]*xM).masked_fill(mask,-1e9); rule=sc.argmax(-1)
    print('agreement with network (excluding net->M cases): %.4f'%((rule==pk).float().mean()))
    sc0=xk.masked_fill(mask,-1e9).argmax(-1)
    print('baseline agreement of the true rule (beta=0): %.4f'%((sc0==pk).float().mean()))
    print('beta[m, j] (row = M position, col = candidate j):'); b=beta.detach().numpy().copy(); np.fill_diagonal(b,np.nan); print(b)
