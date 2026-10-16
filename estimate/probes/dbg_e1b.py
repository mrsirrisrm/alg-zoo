import sys; sys.path.insert(0,'.'); sys.argv=['x']
exec(open('probes/dbg_e1.py').read().split("for (m,s,vM,vS) in")[0])
import numpy.polynomial.legendre as L
m,s,vM,vS=8,1,1.2,1.1
N=400000; u=Phi(th.tensor(vS,dtype=dt)); x=Phi_inv(th.rand(N,10,dtype=dt)*u); x[:,m]=vM; x[:,s]=vS
h=th.zeros(N,16,dtype=dt); hs=[]
for t in range(10): h=th.relu(x[:,t:t+1]*Wi+h@Wh.T); hs.append(h)
uu=u.item(); u0=0.5; K=8
g,gw=L.leggauss(K); v=(g+1)/2; gw=gw/2
br=[(Phi_inv(th.tensor(v*u0,dtype=dt)), th.tensor(gw,dtype=dt), u0/uu),(Phi_inv(th.tensor(u0+v*(uu-u0),dtype=dt)), th.tensor(gw,dtype=dt), (uu-u0)/uu)]
mu=th.zeros(1,16,dtype=dt); S=th.zeros(1,16,16,dtype=dt); p=th.ones(1,dtype=dt)
for t in range(10):
    if t in (m,s):
        xv=th.tensor([vM if t==m else vS],dtype=dt)
        m1,m2=relu_moments(xv*Wi+mu@Wh.T,Wh@S@Wh.T); mu=m1; S=m2-mu[...,:,None]*mu[...,None,:]
    else:
        new=[]
        for xs,w,pb in br:
            m1,m2=relu_moments(xs[None,:,None]*Wi+(mu@Wh.T)[:,None,:],(Wh@S@Wh.T)[:,None])
            mn=(w[None,:,None]*m1).sum(1); E2=(w[None,:,None,None]*m2).sum(1)
            new.append((mn,E2-mn[...,:,None]*mn[...,None,:],p*pb))
        mu=th.cat([a for a,_,_ in new]); S=th.cat([b for _,b,_ in new]); p=th.cat([c for _,_,c in new])
    mix=(p[:,None]*mu).sum(0)
    E2=(p[:,None,None]*(S+mu[:,:,None]*mu[:,None,:])).sum(0); C=E2-mix[:,None]*mix[None]
    print(t,'mean err %.2f  sd err %.2f  (mc mean scale %.1f) min comp var %.2e'%((mix-hs[t].mean(0)).abs().max(), (C.diagonal().clamp_min(0).sqrt()-hs[t].std(0)).abs().max(), hs[t].mean(0).abs().max(), th.linalg.eigvalsh(S).min()))
