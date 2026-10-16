import sys; sys.path.insert(0,'.')
import torch as th, numpy as np
from gauss import *
from ground_truth import load_weights
import numpy.polynomial.legendre as L
dt=th.float64
Wi,Wh,Wo=(th.tensor(a,dtype=dt) for a in load_weights('weights.npz'))
def run(m,s,vM,vS,K=8):
    u=Phi(th.tensor(vS,dtype=dt)).item(); u0=0.5
    g,gw=L.leggauss(K); v=(g+1)/2; gw=gw/2
    # branches: x<0 : x=Phi_inv(v*u0), mass u0/u ; 0<x<vS : x=Phi_inv(u0+v*(u-u0)), mass (u-u0)/u
    br=[(Phi_inv(th.tensor(v*u0,dtype=dt)), th.tensor(gw,dtype=dt), u0/u)]
    if vS>0: br.append((Phi_inv(th.tensor(u0+v*(u-u0),dtype=dt)), th.tensor(gw,dtype=dt), (u-u0)/u))
    mu=th.zeros(1,16,dtype=dt); S=th.zeros(1,16,16,dtype=dt); p=th.ones(1,dtype=dt)
    for t in range(10):
        if t in (m,s):
            xv=th.tensor([vM if t==m else vS],dtype=dt)
            pm=xv*Wi+mu@Wh.T; m1,m2=relu_moments(pm,Wh@S@Wh.T)
            mu=m1; S=m2-mu[...,:,None]*mu[...,None,:]; continue
        new=[]
        for xs,w,pb in br:
            pm=xs[None,:,None]*Wi+(mu@Wh.T)[:,None,:]; m1,m2=relu_moments(pm,(Wh@S@Wh.T)[:,None])
            mn=(w[None,:,None]*m1).sum(1); E2=(w[None,:,None,None]*m2).sum(1)
            new.append((mn,E2-mn[...,:,None]*mn[...,None,:],p*pb))
        mu=th.cat([a for a,_,_ in new]); S=th.cat([b for _,b,_ in new]); p=th.cat([c for _,_,c in new])
    lm=mu@Wo.T; lS=Wo@S@Wo.T
    idx=[j for j in range(10) if j!=s]
    D=th.zeros(9,10,dtype=dt); D[:,s]=1
    for r,j in enumerate(idx): D[r,j]-=1
    pc=orthant_prob(lm@D.T, D@lS@D.T, n=1024)
    return (p*pc).sum().item(), len(p)
def mc(m,s,vM,vS,N=400000):
    u=Phi(th.tensor(vS,dtype=dt)); x=Phi_inv(th.rand(N,10,dtype=dt)*u); x[:,m]=vM; x[:,s]=vS
    h=th.zeros(N,16,dtype=dt)
    for t in range(10): h=th.relu(x[:,t:t+1]*Wi+h@Wh.T)
    return ((h@Wo.T).argmax(-1)==s).double().mean().item()
for (m,s,vM,vS) in [(3,6,1.5,1.0),(6,3,1.5,1.0),(0,7,1.6,1.1),(2,8,2.5,0.6),(8,1,1.2,1.1)]:
    print((m,s,vM,vS),'E1 %.4f comps %d'%run(m,s,vM,vS),' MC %.4f'%mc(m,s,vM,vS))

print('--- diagnose (8,1,1.2,1.1)')
m,s,vM,vS=8,1,1.2,1.1
N=400000; u=Phi(th.tensor(vS,dtype=dt)); x=Phi_inv(th.rand(N,10,dtype=dt)*u); x[:,m]=vM; x[:,s]=vS
h=th.zeros(N,16,dtype=dt); hs=[]
for t in range(10): h=th.relu(x[:,t:t+1]*Wi+h@Wh.T); hs.append(h)
l=h@Wo.T; marg=l[:,s:s+1]-th.cat([l[:,:s],l[:,s+1:]],1)
mm=marg.min(1).values
print('min-margin quantiles', np.quantile(mm.numpy(),[0.001,0.01,0.05,0.25,0.5,0.75]).round(2))
print('margin mean', marg.mean(0).numpy().round(1)); print('margin std', marg.std(0).numpy().round(1))
for t in range(10):
    print(t,'h mean',hs[t].mean(0).numpy().round(1))
