"""Evaluate a saved trained checkpoint; never instantiate an untrained agent."""
import argparse,json
from experiments.engine import ALGORITHMS,evaluate_model,save_json,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--algorithm',choices=list(ALGORITHMS),required=True)
    p.add_argument('--env',required=True);p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();model=ALGORITHMS[a.algorithm].load(a.checkpoint,device='cpu')
    scores,steps,info=evaluate_model(model,a.env,a.seeds)
    save_json(a.output,dict(checkpoint_sha256=digest(a.checkpoint),evaluation_seeds=a.seeds,scores=scores,evaluation_steps=steps,bsuite_info=info))
if __name__=='__main__':main()
