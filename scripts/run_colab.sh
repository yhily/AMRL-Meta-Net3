#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m experiments.run --config configs/review_pilot.json --output results/reviewer_pilot
