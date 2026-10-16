import sys; sys.path.insert(0,'.')
import torch as th, numpy as np
from ground_truth import load_weights
np.set_printoptions(precision=2, suppress=True, linewidth=220)
Wi,Wh,Wo=(th.tensor(a) for a in load_weights('weights.npz'))
th.manual_seed(0); N=1<<20; x=th.randn(N,10)
h=th.zeros(N,16); act=[]; pres=[]
for t in range(10):
    pre=x[:,t:t+1]*Wi+h@Wh.T; act.append(pre>0); pres.append(pre); h=th.relu(pre)
A=th.stack(act,1).float()  # N,10,16
print('P(active) [t, neuron]'); print(A.mean(0).numpy())
# pattern counts over subsets
def npat(bits):
    w=(1<<th.arange(bits.shape[1],dtype=th.int64))
    if bits.shape[1]>62:
        k=th.unique(bits.long(),dim=0); return len(k)
    return len(th.unique((bits.long()*w).sum(1)))
big=[1,4,6,7,8]
print('distinct patterns big-input neurons only:', npat(A[:,:,big].reshape(N,-1)))
print('distinct sign(x) patterns:', npat((x>0).reshape(N,-1)))
small=[i for i in range(16) if i not in big]
print('distinct, small neurons:', npat(A[:,:,small].reshape(N,-1)))
for n in range(16):
    print(n, 'distinct over time', npat(A[:,:,n]))
