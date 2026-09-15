"""Recurrent PPO with per-episode GAE, complete sequence replay and frozen collection weights."""
import torch

def gae(rewards,values,gamma=.99,lam=.95):
    advantages=[]; last=0.;next_value=0.
    for r,v in zip(reversed(rewards),reversed(values)):
        last=r+gamma*next_value-v+gamma*lam*last
        advantages.append(last);next_value=v
    advantages=list(reversed(advantages))
    return torch.tensor(advantages),torch.tensor(advantages)+torch.tensor(values)

class MetaPPO:
    def __init__(self,policy,lr=3e-4):
        self.policy=policy
        self.opt=torch.optim.Adam(policy.parameters(),lr=lr)

    def update(self,episodes,epochs=4):
        episodes=[e for e in episodes if e]
        if not episodes: raise ValueError('Meta-training requires at least one completed intervention transition')
        batches=[]
        for e in episodes:
            adv,ret=gae([x['reward'] for x in e],[x['value'] for x in e])
            batches.append((e,adv,ret))
        all_adv=torch.cat([b[1] for b in batches]);scale=all_adv.std(unbiased=False).clamp_min(1e-8)
        logs=[]
        for _ in range(epochs):
            losses=[]
            for e,adv,ret in batches:
                states=torch.stack([x['state'] for x in e]).unsqueeze(0)
                z=torch.stack([x['z'] for x in e])
                old=torch.tensor([x['logp'] for x in e])
                mean,values,_=self.policy(states)
                ratio=(self.policy.log_prob(mean[0],z)-old).exp()
                a=(adv-all_adv.mean())/scale if len(all_adv)>1 else adv
                objective=torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()
                entropy=torch.distributions.Normal(mean,self.policy.log_std.exp()).entropy().sum(-1).mean()
                losses.append(-objective+.5*(values[0]-ret).square().mean()-.01*entropy)
            loss=torch.stack(losses).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite meta PPO objective')
            self.opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.policy.parameters(),.5);self.opt.step()
            logs.append(float(loss.detach()))
        return logs
