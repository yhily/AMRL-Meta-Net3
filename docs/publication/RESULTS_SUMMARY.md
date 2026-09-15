# Verified publication result summary

This folder contains the compact, review-facing evidence package for the revised AMRL-Meta-Net manuscript.

- 265 formal runs across 53 experiment groups
- 5 seeds per group
- 5,300 evaluation episodes
- Paired seed-level contrasts with Holm adjustment
- 0 primary-family rejections at alpha = 0.05 in the final numerical validation
- Resource accounting separated into meta-training, TPE tuning, deployment, and query latency

The results support an environment-dependent interpretation. CartPole-PPO favors the static baseline in the paired summary, while Catch-PPO favors AMRL. MountainCar conditions are close to the reward floor, and Memory/0 is ceiling-limited. These observations motivate restrained claims about conditional adaptation rather than universal superiority.

The cost summary reports approximately 252M meta-training interactions, 180M TPE-training interactions, 26.5M deployment interactions, and median query latency around 0.173 ms in the audited run. These quantities are accounting boundaries for the reported experiments, not a claim of general end-to-end compute savings.

See `docs/publication/qa/final_numerical_validation.json` and `docs/publication/qa/data_validation.json` for the machine-readable checks.
