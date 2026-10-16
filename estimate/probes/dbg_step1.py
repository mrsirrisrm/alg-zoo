import sys; sys.path.insert(0,'.')
import torch as th, numpy as np
from gauss import *
from ground_truth import load_weights
import numpy.polynomial.legendre as L
np.set_printoptions(precision=2, suppress=True, linewidth=220)
dt=th.float64
Wi,Wh,Wo=(th.tensor(a,dtype=dt) for a in load_weights('weights.npz'))
vS=1.1; N=1000000
for lo,hi in [(0.0,0.5),(0.5,Phi(th.tensor(vS)).item())]:
    x0=Phi_inv(lo+(hi-lo)*th.rand(N,dtype=dt))
    h0=th.relu(x0[:,None]*Wi); h1=th.relu(vS*Wi+h0@Wh.T)
    g,gw=L.leggauss(8); v=(g+1)/2; gw=th.tensor(gw/2,dtype=dt)
    xs=Phi_inv(th.tensor(lo+(hi-lo)*v,dtype=dt))
    m1,m2=relu_moments(xs[:,None]*Wi, th.zeros(16,16,dtype=dt))
    mu=(gw[:,None]*m1).sum(0); S=(gw[:,None,None]*m2).sum(0)-mu[:,None]*mu[None]
    print('branch',lo,hi,' h0 mean err',(mu-h0.mean(0)).abs().max().item())
    a1,a2=relu_moments(vS*Wi+Wh@mu, Wh@S@Wh.T)
    print(' h1 mean MC ', h1.mean(0).numpy()); print(' h1 mean clo', a1.numpy())
    print(' h1 sd MC   ', h1.std(0).numpy()); print(' h1 sd clo  ', (a2.diagonal()-a1**2).clamp_min(0).sqrt().numpy())
    print(' pre1 mean', (vS*Wi+Wh@mu).numpy()); print(' pre1 sd', (Wh@S@Wh.T).diagonal().sqrt().numpy())
