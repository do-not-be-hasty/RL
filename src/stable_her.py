import sys
from collections import OrderedDict
from functools import partial

from gym import GoalEnv, spaces
from gym_rubik.envs import RubikEnv
from stable_baselines import HER, DQN, SAC, DDPG, TD3
# from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.bit_flipping_env import BitFlippingEnv
from stable_baselines.deepq import MlpPolicy

import networks
from environment_builders import make_env_Rubik
import numpy as np

from networks import CustomPolicy
from utility import her_callback


class HerRubikEnv(RubikEnv, GoalEnv):
    def __init__(self, step_limit=100, shuffles=50):
        super(HerRubikEnv, self).__init__(step_limit, shuffles)
        self.goal_obs = self._get_state().flatten()

    def create_observation_space(self):
        # self.observation_space = spaces.Box(low=0, high=1, shape=(6, 3, 3, 12), dtype=np.float32)
        # self.observation_space = spaces.Dict({
        #         'observation': spaces.Box(low=0, high=1, shape=(6, 3, 3, 12), dtype=np.float32),
        #         'achieved_goal': spaces.Box(low=0, high=1, shape=(6, 3, 3, 12), dtype=np.float32),
        #         'desired_goal': spaces.Box(low=0, high=1, shape=(6, 3, 3, 12), dtype=np.float32),
        #     })
        self.observation_space = spaces.Dict({
            'observation': spaces.MultiBinary(6 * 3 * 3 * 6),
            'achieved_goal': spaces.MultiBinary(6 * 3 * 3 * 6),
            'desired_goal': spaces.MultiBinary(6 * 3 * 3 * 6),
        })

    def step(self, action):
        obs, reward, done, info = super(HerRubikEnv, self).step(action)

        obs = self._get_goal_observation(obs)
        reward = self._calculate_reward(obs['observation'], obs['achieved_goal'], obs['desired_goal'])

        return obs, reward, done, info

    def reset(self):
        obs = super(HerRubikEnv, self).reset()
        # print(obs.flatten())
        # print(self._get_goal_observation(obs))
        return self._get_goal_observation(obs)

    def _get_goal_observation(self, obs):
        return self._convert_observation(obs.flatten(), obs.flatten(), self.goal_obs)

    def _convert_observation(self, obs, state, goal):
        # print(obs.shape, state.shape, goal.shape)
        return OrderedDict([
            ('observation', obs),
            ('achieved_goal', state),
            ('desired_goal', goal)
        ])

    def _calculate_reward(self, obs, state, goal):
        return 0 if np.array_equal(state, goal) else -1

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._calculate_reward(None, achieved_goal, desired_goal)


model_class = DQN  # works also with SAC, DDPG and TD3

# env = BitFlippingEnv(10, continuous=model_class in [DDPG, SAC, TD3], max_steps=10)
# env = HERGoalEnvWrapper(make_env_Rubik(step_limit=100, shuffles=5))
env = HerRubikEnv(step_limit=40, shuffles=50)

# Available strategies (cf paper): future, final, episode, random
goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE

# Wrap the model
policy = partial(MlpPolicy, layers=[1024, 1024])
# policy = partial(CustomPolicy, arch_fun=networks.arch_color_embedding),
model = HER(policy, env, model_class, n_sampled_goal=12, goal_selection_strategy=goal_selection_strategy,
            verbose=1)

# Train the model
model.learn(1000000, callback=her_callback)

sys.exit(0)

model.save("./her_bit_env")

# WARNING: you must pass an env
# or wrap your environment with HERGoalEnvWrapper to use the predict method
model = HER.load('./her_bit_env', env=env)

ends = []
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    # print(obs)

    if done:
        ends.append(reward)
        obs = env.reset()

print(np.mean(ends))