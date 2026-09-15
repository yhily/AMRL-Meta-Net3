#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Standalone training example with explicitly supplied fixed hyperparameters, not a TPE result.
python train_single.py --config configs/review_pilot.json --env CartPole-v1 --algorithm ppo --seed 0 --hp 0.0003 0.99 0.01 --output results/cartpole_example
