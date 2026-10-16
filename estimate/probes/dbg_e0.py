import sys; sys.path.insert(0,'.')
import torch as th, numpy as np
from gauss import *
from moment_prop import bg_nodes
from ground_truth import load_weights
dt=th.float64
Wi,Wh,Wo=(th.tensor(a,dtype=dt) for a in load_weights('weights.npz'))
m,s,vM,vS=3,6,1.5,1.0
u=Phi(th.tensor([vS],dtype=dt))
# MC
N=400000; x=Phi_inv(th.rand(N,10,dtype=dt)*u); x[:,m]=vM; x[:,s]=vS
h=th.zeros(N,16,dtype=dt); hs=[]
for t in range(10):
    h=th.relu(x[:,t:t+1]*Wi+h@Wh.T); hs.append(h)
ok=((h@Wo.T).argmax(-1)==s).double().mean().item()
# moment prop
mu=th.zeros(1,16,dtype=dt); S=th.zeros(1,16,16,dtype=dt)
xb,wb=bg_nodes(u,12,dt,'cpu')
for t in range(10):
    if t in (m,s):
        xv=th.tensor([[vM if t==m else vS]],dtype=dt); w=th.ones(1,dtype=dt)
    else: xv=xb; w=wb
    pm=xv[...,None]*Wi+(mu@Wh.T)[:,None,:]; pS=(Wh@S@Wh.T)[:,None]
    m1,m2=relu_moments(pm,pS)
    mu=(w[None,:,None]*m1).sum(1); E2=(w[None,:,None,None]*m2).sum(1); S=E2-mu[...,:,None]*mu[...,None,:]
    mc_mu=hs[t].mean(0); mc_S=th.cov(hs[t].T)
    print(t, 'mean err %.3f (scale %.2f)  cov err %.3f (scale %.2f)  minEig %.2e'%((mu[0]-mc_mu).abs().max(), mc_mu.abs().max(), (S[0]-mc_S).abs().max(), mc_S.abs().max(), th.linalg.eigvalsh(S[0]).min()))
l=hs[-1]@Wo.T
print('MC P(correct)',ok)
print('logit mean MC', (l.mean(0)).numpy().round(2)); print('logit mean MP', (mu@Wo.T)[0].numpy().round(2))
print('logit std MC', l.std(0).numpy().round(2)); print('logit std MP', th.diagonal(Wo@S[0]@Wo.T).sqrt().numpy().round(2))
