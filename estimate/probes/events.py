import sys; sys.path.insert(0,'.')
import torch as th, numpy as np
from ground_truth import load_weights
np.set_printoptions(precision=3, suppress=True, linewidth=220)
Wi,Wh,Wo=(th.tensor(a) for a in load_weights('weights.npz'))
th.manual_seed(0); N=1<<20; x=th.randn(N,10)
h=th.zeros(N,16); A=[]
for t in range(10):
    pre=x[:,t:t+1]*Wi+h@Wh.T; A.append(pre>0); h=th.relu(pre)
A=th.stack(A,1)
cm=th.cummax(x,1).values
newmax=th.zeros(N,10,dtype=th.bool); new2=th.zeros(N,10,dtype=th.bool)
for t in range(1,10):
    top2=x[:,:t].topk(min(2,t),1).values
    newmax[:,t]=x[:,t]>top2[:,0]
    new2[:,t]=(x[:,t]<=top2[:,0])&((x[:,t]>top2[:,1]) if t>=2 else th.ones(N,dtype=th.bool))
pos=x>0
for name,ev in [('newmax',newmax),('new2nd',new2),('x>0',pos),('newmax|new2nd',newmax|new2)]:
    print(name)
    agree=(A==ev[:,:,None]).float().mean(0)[1:]  # t=1..9 x neurons
    print('  best neuron agreement per t:', [(int(agree[t].argmax()), round(agree[t].max().item(),3)) for t in range(9)])
