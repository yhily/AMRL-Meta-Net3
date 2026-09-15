#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python train_single.py --config configs/review_pilot.json --env bsuite/memory_len/0 --algorithm ppo --seed 0 --hp 0.0003 0.99 0.01 --output results/bsuite_example
