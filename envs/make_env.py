"""Gymnasium interface; bsuite uses explicit IDs, never implicit full-suite claims."""
import gymnasium as gym
import numpy as np

class BSuiteEnv(gym.Env):
    def __init__(self, name, seed):
        import bsuite
        self.name = name.removeprefix('bsuite/')
        if '/' not in self.name:
            raise ValueError('Use an explicit bsuite ID, e.g. bsuite/memory_len/0')
        self._env = bsuite.load_from_id(self.name)
        self._seed = seed
        self._rebuild(seed)
        spec = self._env.observation_spec()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=spec.shape, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self._env.action_spec().num_values)

    def _rebuild(self, seed):
        # Resolve the official sweep's task arguments, overriding only its random seed.
        from bsuite import sweep
        import bsuite
        kwargs = dict(sweep.SETTINGS[self.name])
        kwargs['seed'] = int(seed)
        self._env = bsuite.load(self.name.split('/')[0], kwargs)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rebuild(seed)
        ts = self._env.reset()
        return np.asarray(ts.observation, dtype=np.float32), {}

    def step(self, action):
        ts = self._env.step(int(action))
        return np.asarray(ts.observation, dtype=np.float32), float(ts.reward or 0), bool(ts.last()), False, {}

    def bsuite_info(self):
        return dict(self._env.bsuite_info())


def make_env(name, seed=0):
    if name.startswith('bsuite/'):
        env = BSuiteEnv(name, seed)
    else:
        env = gym.make(name, **({'max_episode_steps': 1000} if name == 'CartPole-v1' else {}))
    env.action_space.seed(seed)
    return env
