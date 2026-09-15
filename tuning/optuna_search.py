"""CLI for genuinely trained, separately seeded fixed-budget TPE trials."""
import argparse,json
from pathlib import Path
from experiments.engine import save_json
from experiments.run import tune

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--env',required=True)
    p.add_argument('--algorithm',choices=['ppo','a2c','dqn'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(tune(Path(a.output),a.env,a.algorithm,json.loads(Path(a.config).read_text()))))
if __name__=='__main__':main()
