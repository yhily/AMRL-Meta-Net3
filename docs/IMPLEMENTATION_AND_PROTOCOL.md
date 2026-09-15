# Implementation audit and declared experimental protocol

## Original snapshot findings

The supplied 28-file tree contained no CSV results, saved weights, seed manifests or training timing logs. The old `train_meta.py` performed forward passes but never called a meta-optimizer; changing an HPController object did not modify the actual base optimizer, gamma or exploration settings. `evaluate.py` evaluated a newly initialized PPOAgent without loading learned weights. The old PPOAgent used a single immediate reward rather than a clipped PPO update with rollouts. Both MetaPPO stubs referenced missing imports and undefined functions. bsuite's dm_env interface was treated as old Gym. `train_single.py` referenced undefined variables; the `.ipynb` file was plain text rather than notebook JSON. These invalidate the old README's claim of a complete reproducible pipeline.

## Repaired execution

1. Use maintained SB3 base algorithms (PPO clipping/rollouts, A2C, DQN replay and target network). Legacy algorithm modules export these implementations; the old per-transition constructor/update API is intentionally retired.
2. Create a fresh base learner for every meta-episode and evaluation training seed. Base networks use two 64-unit hidden layers. On-policy rollout size, exact step budget and sparse adaptation interval are explicit.
3. Learn a separate 6D LSTM Gaussian policy with transformed actions, the tanh change-of-variables correction, value estimates, episodic GAE, and clipped recurrent PPO. Entire sequences are replayed from their initial hidden state. Collection weights stay unchanged within each meta-batch. Record actual parameter change; zero change aborts training.
4. Reward an intervention from the **next** interval's change in recent mean completed-episode return. No action is taken at the final checkpoint. A sparse-reward interval with no completed episodes uses zero for the current return feature; this is an explicit modeling convention, not a historical fact.
5. Normalize states only during controller meta-training, store the statistics, and freeze them for deployment. Feature ablation masks are applied after normalization and used consistently in both meta-training and deployment.
6. Learning-rate actuation is `alpha * (1 + delta_alpha)`, projected into `[1e-5,1e-2]` and, by default, `[0.1 alpha0,10 alpha0]`. Gamma is projected to `[0.90,0.999]`, beta to `[0,1]`. Updates are written into the actual optimizer/schedules, rollout-buffer gamma, and entropy/epsilon fields.
7. Evaluate the saved trained base model on separate deterministic-action episodes with reproducible reset seeds. Persist the individual episode returns and model checksum.

## Base learner and diagnostic boundaries

Each chunk uses the same learner and environment state; it does not restart training at every adaptation checkpoint. On-policy chunks divide exactly into rollouts and DQN chunks divide into its train frequency. Environment termination/truncation and TimeLimit bootstrapping are handled by the SB3/Gymnasium backend.

Completed training-episode returns feed the reward features. The recorded policy/value losses are the last SB3 update aggregates; gradient norms are observed at optimizer step time, after clipping. Gradient observations collected in an interval are assigned its end checkpoint. Thus the window is exact for episodic returns but interval-granular for optimizer diagnostics. These differences from ideal step-resolved diagnostics must be stated in any method description using these results.

## Matrix and controls

Main comparisons: four explicit environments × PPO/A2C/DQN × Static(TPE)/AMRL × five training seeds. The pooled controller trains on CartPole, MountainCar and memory_len using PPO. catch is excluded from pooled meta-training.

Cross-algorithm: on each environment with A2C and DQN, compare the frozen PPO-trained controller, separately A2C/DQN-trained controllers and a jointly PPO/A2C/DQN-trained controller. All controller roles receive the same number of full base-learner meta-episodes; the joint schedule balances all nine task–algorithm combinations. This fixes total meta-episode budget, not wall time or per-algorithm exposure. All source-task initializations are included in first-use cost accounting.

Ablations: CartPole–PPO and memory_len–PPO; Frozen/no actuation; MLP; reward-only; no loss; no gradient; no cumulative learning-rate drift guard; half/double per-action bounds; half/double adaptation interval. Each non-frozen variant is meta-trained under its own declared setting, with the same total meta-episode and inner-step budgets. The MLP width is selected to match the number of trainable LSTM-controller parameters within 0.1%, with actual counts reported. It changes architecture, not just state memory. Range projection is retained in all safeguard variants. `no_drift` must not be described as removing all bounds.

Cross-task: CartPole→MountainCar, CartPole→memory_len, MountainCar→catch using PPO; each source-specific frozen controller is compared with a separately target-trained controller and the same target Static(TPE). Pooled controllers and source-only controllers are separate saved artifacts.

## Seeds and selection

Default deployment/controller seeds: 0,1,2,3,4. Meta-episode base seeds: `seed * 10000 + 2000 + episode_index`. Tuning base seeds: 700/701/702 (full) or 700 (pilot). Evaluation reset seeds: `base_seed * 1000 + 1000000 + episode_index`. TPE sampler seed: 900. TPE selection uses only tuning seeds; it does not select configurations using the five evaluation training seeds.

These paired comparisons capture variation in both learner and controller training seed, conditional on one shared TPE study per task/algorithm. They do not measure independent repeated TPE-study variability. The wrapper replaces only the bsuite sweep instance's random seed and preserves its task parameters; it does not claim that `/0` covers an entire family.

## Statistical interpretation

The independent replicate is a training seed, with its final score averaged over separately seeded evaluation episodes. A paired contrast requires **all planned seeds** in both conditions. Export paired mean differences, t-based 95% intervals and p-values, exact sign-flip p-values, and paired standardized differences when defined. With five nonzero pairs, the minimum possible two-sided exact p-value is 0.0625. Degenerate/undefined t tests are null, not invented as tiny p-values.

The twelve primary AMRL-versus-static tests form one prespecified family; Holm correction is computed separately for parametric and exact p-values when all twelve are available. Other contrasts are exploratory and unadjusted. No cross-task pooling of incomparable raw reward scales is performed.

## Complete measured cost boundaries

- TPE: training and evaluation of all candidate trials, measured over the entire study call.
- Meta-training: controller initialization, all inner-learner rollouts/updates and diagnostics, outer PPO optimization, and controller checkpoint writing. Record total seconds and environment interactions, not just outer optimizer time.
- Online: fixed-budget base training with diagnostics and controller intervention path. Evaluation and checkpoint saving are separate from the online-stage timer.
- First use: unique initialization studies needed by the chosen controller's source tasks/algorithms **and** the target pair + complete meta-training + target online stage. Shared task/algorithm searches are counted once.
- Reuse: only the meta-training portion may be amortized as `meta_seconds / K`; do not charge a reused checkpoint as newly trained each time.
- Whole experiment: sum unique stage records, not the sum of per-run first-use charges. The latter double-counts reusable assets.

All timings are measurements on the recorded machine; no T4 timing, energy consumption, dollar cost, or historical 0.8–1.2% claim is inferred from this CPU reference run. Inline instrumentation is not itself a paired end-to-end overhead experiment.

## Upstream sources checked

- [SB3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)
- [Official bsuite environment wrapper](https://github.com/google-deepmind/bsuite/blob/main/bsuite/utils/gym_wrapper.py)

Code is pinned to SB3 2.7.1 and Gymnasium 1.2.3 in the declared environment; published master documentation may refer to a newer release. Installed source behavior was exercised by the tests.
