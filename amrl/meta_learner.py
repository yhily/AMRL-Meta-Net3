"""Gaussian recurrent controller; tanh Jacobian included in PPO likelihoods."""
import torch
from torch import nn
from torch.distributions import Normal

class MetaLearner(nn.Module):
    def __init__(self, architecture='lstm', hidden=64, bound_scale=1.):
        super().__init__()
        self.config = dict(architecture=architecture,hidden=hidden,bound_scale=bound_scale)
        self.architecture = architecture
        if architecture == 'lstm':
            self.encoder = nn.LSTM(6,hidden,batch_first=True)
        elif architecture == 'mlp':
            self.encoder = nn.Sequential(nn.Linear(6,hidden),nn.Tanh())
        else:
            raise ValueError(architecture)
        self.fc = nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.mean = nn.Linear(16,3)
        self.value = nn.Linear(16,1)
        self.log_std = nn.Parameter(torch.full((3,),-.5))
        self.register_buffer('bounds',torch.tensor([.30,.02,.05])*bound_scale)

    def forward(self,x,h=None):
        if self.architecture=='lstm': x,h=self.encoder(x,h)
        else: x=self.encoder(x)
        f=self.fc(x)
        return self.mean(f),self.value(f).squeeze(-1),h

    def log_prob(self,mean,z):
        # Stable log(1-tanh(z)^2), including the component-wise bounds.
        jac=2*(torch.log(torch.tensor(2.))-z-torch.nn.functional.softplus(-2*z))
        return (Normal(mean,self.log_std.exp()).log_prob(z)-jac-self.bounds.log()).sum(-1)

    def act(self,state,h=None,deterministic=False):
        with torch.no_grad():
            mean,value,h=self(state.reshape(1,1,6),h)
            z=mean if deterministic else Normal(mean,self.log_std.exp()).sample()
            action=(self.bounds*z.tanh()).reshape(3)
            return action,z.reshape(3),self.log_prob(mean,z).item(),value.item(),h


def matching_mlp_hidden():
    target=sum(p.numel() for p in MetaLearner().parameters())
    # Closed form parameter count for this topology; no dummy unused parameters.
    fixed=sum(p.numel() for p in MetaLearner('mlp',1).parameters())-39
    return max(1,round((target-fixed)/39))
