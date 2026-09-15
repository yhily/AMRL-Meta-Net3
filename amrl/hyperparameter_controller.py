"""Component-wise actuation matching the manuscript, with named safeguard ablations."""
import numpy as np

class HPController:
    def __init__(self, a=3e-4, g=.99, b=.01, safeguard='full'):
        self.a0 = float(a)
        self.a, self.g, self.b = map(float, (a,g,b))
        self.safeguard = safeguard

    def update(self, d):
        da,dg,db = map(float,d)
        low,high = 1e-5,1e-2
        if self.safeguard != 'no_drift':
            low,high = max(low,.1*self.a0),min(high,10*self.a0)
        self.a = float(np.clip(self.a*(1+da),low,high))
        self.g = float(np.clip(self.g+dg,.90,.999))
        self.b = float(np.clip(self.b+db,0,1))
        return self.get()

    def get(self):
        return self.a,self.g,self.b

    def apply(self, model):
        model.learning_rate = self.a
        model.lr_schedule = lambda progress: self.a
        for group in model.policy.optimizer.param_groups:
            group['lr'] = self.a
        model.gamma = self.g
        if hasattr(model,'rollout_buffer'):
            model.rollout_buffer.gamma = self.g
        if hasattr(model,'exploration_rate'):
            model.exploration_rate = self.b
            model.exploration_schedule = lambda progress: self.b
        else:
            model.ent_coef = self.b
