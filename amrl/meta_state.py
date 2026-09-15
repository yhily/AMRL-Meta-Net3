from collections import deque
import numpy as np
import torch

class MetaState:
    """Recent complete episode returns and optimizer diagnostics within an environment-step window."""
    def __init__(self,window=1000):
        self.window=window;self.returns=deque();self.updates=deque()
    def episode(self,step,value):self.returns.append((step,float(value)))
    def optimization(self,step,pl,vl,gn):self.updates.append((step,float(pl),float(vl),float(gn)))
    def get(self,step):
        for q in [self.returns,self.updates]:
            while q and q[0][0]<=step-self.window:q.popleft()
        r=[x[1] for x in self.returns];u=np.array([x[1:] for x in self.updates])
        return torch.tensor([np.mean(r) if r else 0,np.std(r) if r else 0,
            u[:,0].mean() if len(u) else 0,u[:,1].mean() if len(u) else 0,
            u[:,2].mean() if len(u) else 0,u[:,2].std() if len(u) else 0],dtype=torch.float32)

class Normalizer:
    def __init__(self,state=None):
        self.n=0;self.mean=np.zeros(6);self.m2=np.zeros(6)
        if state:self.n=state['n'];self.mean=np.array(state['mean']);self.m2=np.array(state['m2'])
    def transform(self,x,update=False):
        x=np.asarray(x,dtype=float)
        if update:
            self.n+=1;delta=x-self.mean;self.mean+=delta/self.n;self.m2+=delta*(x-self.mean)
        std=np.sqrt(self.m2/max(self.n-1,1));std=np.where(std<1e-6,1.,std)
        return torch.tensor(np.clip((x-self.mean)/std,-5,5),dtype=torch.float32)
    def state(self):return dict(n=self.n,mean=self.mean.tolist(),m2=self.m2.tolist())
