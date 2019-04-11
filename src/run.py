import os
from functools import partial

import gym
import numpy as np
import sys
# from gym_maze.envs import MazeEnv
from gym_maze import RandomMazeGenerator

from utility import make_env_BitFlipper, make_env_GoalBitFlipper, make_env_GoalMaze
from stable_baselines.bench import Monitor

from stable_baselines.common.policies import MlpPolicy as PPO2_Policy, FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, DDPG
from stable_baselines.deepq import MlpPolicy as DQN_Policy
from stable_baselines.ddpg import MlpPolicy as DDPG_Policy
from DQN_HER import DQN_HER as HER
from DQN_MTR import DQN_MTR as MTR
from stable_baselines.results_plotter import ts2xy, load_results

from utility import resources_dir, get_cur_time_str

from neptune_utils.neptune_utils import get_configuration
from neptune_utils.neptune_utils import neptune_logger
from neptune import ChannelType


def callback(_locals, _globals):
    if len(_locals['episode_rewards']) % 100 == 0:
        neptune_logger('success rate', np.mean(_locals['episode_success']))
        neptune_logger('distance', np.mean(_locals['episode_finals']))

    return False


def DQN_model(env):
    env = DummyVecEnv([lambda: env])
    return DQN(
        policy=partial(DQN_Policy, layers=[256]),
        env=env,
        learning_rate=1e-3,
        buffer_size=1000000,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        verbose=1,
    )


def HER_model(env):
    return HER(
        policy=partial(DQN_Policy, layers=[256]),
        env=env,
        hindsight=1,
        learning_rate=1e-4,
        buffer_size=1000000,
        exploration_fraction=0.02,
        exploration_final_eps=0.0005,
        gamma=0.98,
        verbose=1,
    )


def PPO_model(env):
    env = DummyVecEnv([lambda: env])
    return PPO2(
        policy=PPO2_Policy,
        env=env,
        learning_rate=1e-3,
        verbose=1,
    )


def evaluate(model, env, steps=1000, verbose=True):
    print('--- Evaluation\n')

    obs = env.reset()
    if verbose:
        print(obs)
    succ = 0
    n_ep = 0

    for i in range(steps):
        action, _states = model.predict(np.concatenate((obs['observation'], obs['desired_goal'])))
        # action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if verbose:
            env.render()

        if rewards >= 0:
            succ += 1

        if dones:
            n_ep += 1
            obs = env.reset()
            if verbose:
                # env.print_rewards_info()
                print()

    print('Success rate: ', succ / n_ep)


def learn_BitFlipper_HER():
    n = 15
    print("BitFlipper({0}), DQN+HER".format(n))

    env = make_env_GoalBitFlipper(n=n, space_seed=15)
    model = HER_model(env)

    try:
        model = model.learn(total_timesteps=10000*16*n,
                            # callback=callback,
                            )
    except KeyboardInterrupt:
        pass

    evaluate(model, env)


def learn_Maze_HER():
    n = 40
    print("Maze({0}), DQN+HER".format(n))

    env = make_env_GoalMaze(
        maze_generator=RandomMazeGenerator,
        width=n,
        height=n,
        complexity=.05,
        density=.1,
        seed=17,
        obs_type='discrete',
        reward_type='sparse',
        step_limit=100)
    model = HER_model(env)

    print("Initial distance: {0}".format(env._distance_diameter()))

    try:
        model = model.learn(total_timesteps=3500000,
                            callback=callback,
                            )
    except KeyboardInterrupt:
        pass

    env._set_live_display(True)
    evaluate(model, env)


def learn_Maze_MTR():
    n = 10
    print("Maze({0}), DQN+MTR".format(n))

    env = make_env_GoalMaze(
        maze_generator=RandomMazeGenerator,
        width=n,
        height=n,
        complexity=.05,
        density=.1,
        seed=15,
        obs_type='discrete',
        reward_type='sparse',
        step_limit=100)
    model = HER_model(env)

    try:
        model = model.learn(total_timesteps=5000,
                            callback=callback,
                            )
    except KeyboardInterrupt:
        pass

    env._set_live_display(True)
    evaluate(model, env)


def main():
    ctx, exp_dir_path = get_configuration()
    debug_info = ctx.create_channel('debug info', channel_type=ChannelType.TEXT)
    os.environ['MRUNNER_UNDER_NEPTUNE'] = '1'

    # learn_BitFlipper_HER()
    learn_Maze_HER()
    # learn_Maze_MTR()

if __name__ == '__main__':
    main()
