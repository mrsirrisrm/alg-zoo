import sys; sys.path.insert(0,'.')
import torch as th, numpy as np
from ground_truth import load_weights
from pattern_census import patterns
Wi,Wh,Wo=(th.tensor(a) for a in load_weights('weights.npz'))
th.manual_seed(2); N=1<<22; x=th.randn(N,10)
p,h=patterns(x,Wi,Wh)
ok=(h@Wo.T).argmax(-1)==x.topk(2,1).indices[:,1]
cm=th.cummax(x,1).values; rec=th.zeros(N,10,dtype=th.bool); rec[:,1:]=x[:,1:]>cm[:,:-1]
w=1<<th.arange(10)
order=th.argsort(th.argsort(x,1),1)  # ranks
key_variants={
 'sign+record': (x>0).long()@w + 1024*(rec.long()@w),
 'sign+record+top2pos': (x>0).long()@w + 1024*(rec.long()@w) + (1<<20)*(x.topk(2,1).indices.long()@th.tensor([1,10])),
}
for name,key in key_variants.items():
    full=th.stack([key,p[:,0],p[:,1],p[:,2]],1)
    uk,ik,ck=th.unique(key,return_inverse=True,return_counts=True)
    uf,iff,cf=th.unique(full,dim=0,return_inverse=True,return_counts=True)
    # dominant pattern mass per key
    fkey=uf[:,0]; order_=th.argsort(cf,descending=True)
    best={}
    for kk,c in zip(fkey[order_].tolist(), cf[order_].tolist()):
        if kk not in best: best[kk]=c
    dom=sum(best.values())/N
    # correctness purity per key
    okk=th.zeros(len(uk)).index_add_(0,ik,ok.float())/ck
    purity=(th.minimum(okk,1-okk)*ck).sum().item()/N
    print(f'{name}: keys={len(uk)} full patterns={len(uf)} mass in dominant pattern per key={dom:.3f}; error-if-predict-majority-per-key={purity:.4f}')
