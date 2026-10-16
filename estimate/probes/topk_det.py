import sys; sys.path.insert(0,'.')
import torch as th, numpy as np
from gauss import Phi, Phi_inv
from ground_truth import load_weights
Wi,Wh,Wo=(th.tensor(a) for a in load_weights('weights.npz'))
def fwd(x):
    h=th.zeros(x.shape[0],16)
    for t in range(10): h=th.relu(x[:,t:t+1]*Wi+h@Wh.T)
    return (h@Wo.T).argmax(-1)
th.manual_seed(1)
C=4000; R=256  # configs, resamples of the rest
for k in [2,3,4,5]:
    base=th.randn(C,10); top=base.topk(k,1)
    vk=top.values[:,-1]  # k-th largest value: rest must be below
    s=top.indices[:,1]
    x=base[:,None,:].repeat(1,R,1)
    u=Phi(vk)[:,None,None]
    fresh=Phi_inv(th.rand(C,R,10)*u)
    mask=th.zeros(C,10,dtype=th.bool); mask.scatter_(1,top.indices,True)
    x=th.where(mask[:,None,:],x,fresh)
    ok=(fwd(x.reshape(-1,10)).reshape(C,R)==s[:,None]).float().mean(1)
    det=((ok<0.02)|(ok>0.98)).float().mean().item()
    # variance explained: Var(E[ok|top-k]) / Var(ok)
    p=ok.mean().item(); between=ok.var().item(); total=p*(1-p)
    print(f'condition on top-{k}: acc {p:.4f}  frac configs determined(>98% one way) {det:.3f}  var explained by top-{k}: {between/total:.3f}')
