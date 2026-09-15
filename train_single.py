"""Train a real base learner, optionally with a saved frozen controller."""
import argparse,json
from pathlib import Path
from experiments.engine import run_base,load_controller

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--env',required=True)
    p.add_argument('--algorithm',choices=['ppo','a2c','dqn'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--hp',nargs=3,type=float,required=True,metavar=('ALPHA','GAMMA','BETA'))
    p.add_argument('--controller');p.add_argument('--output',required=True);args=p.parse_args()
    controller=norm=None
    if args.controller:controller,norm,_=load_controller(args.controller)
    r,_=run_base(args.env,args.algorithm,args.seed,json.loads(Path(args.config).read_text()),args.hp,controller,norm,output=args.output)
    print(json.dumps({k:r[k] for k in ['seed','mean_return','train_steps','online_seconds']}))
if __name__=='__main__':main()
