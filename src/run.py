from functools import partial

import gym
import numpy as np
import sys
# from gym_maze.envs import MazeEnv
from utility import make_env_BitFlipper, make_env_GoalBitFlipper
from stable_baselines.bench import Monitor

from stable_baselines.common.policies import MlpPolicy as PPO2_Policy, FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, DDPG
from stable_baselines.deepq import MlpPolicy as DQN_Policy
from stable_baselines.ddpg import MlpPolicy as DDPG_Policy
from DQN_HER import DQN_HER as HER
from stable_baselines.results_plotter import ts2xy, load_results

from utility import resources_dir, get_cur_time_str


n_steps = 0


def callback(_locals, _globals):
    global n_steps

    n_steps += 1

    if n_steps >= 100:
        n_steps = 0
    else:
        return False

    is_printed = False

    try:
        # For gym.Env, eg HER
        _locals['self'].get_env().print_rewards_info()
        is_printed = True
    except AttributeError:
        pass

    try:
        # For DummyVecEnv, eg PPO2
        _locals['self'].get_env().env_method('print_rewards_info')
        is_printed = True
    except AttributeError:
        pass

    try:
        # For _UnvecWrapper, eg DQN
        _locals['self'].get_env().venv.env_method('print_rewards_info')
        is_printed = True
    except AttributeError:
        pass

    if not is_printed:
        print('Cannot print logs')

    print('')

    return False

def DQN_model(env):
    env = DummyVecEnv([lambda: env])
    return DQN(
        policy=partial(DQN_Policy, layers=[256]),
        env=env,
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.9,
        exploration_final_eps=0.05,
        verbose=1,
    )

def HER_model(env):
    return HER(
        policy=partial(DQN_Policy, layers=[256]),
        env=env,
        learning_rate=1e-3,
        buffer_size=1000000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
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

def main():
    env = make_env_GoalBitFlipper(n=40, space_seed=15)
    # env = make_env_BitFlipper(n=10, space_seed=10)

    # model = DQN_model(env)
    model = HER_model(env)
    # model = PPO_model(env)

    try:
        model = model.learn(total_timesteps=10000*16*40,
                            # callback=callback,
                            )
    except KeyboardInterrupt:
        pass
    model.save(str(resources_dir().joinpath('model.pkl')))

    # env._set_live_display(True)
    # env._set_step_limit(100)
    # env.reset_rewards_info()

    print('--- Evaluation\n')

    obs = env.reset()
    print(obs)
    succ = 0
    n_ep = 0

    for i in range(1000):
        action, _states = model.predict(np.concatenate((obs['observation'], obs['desired_goal'])))
        # action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

        if rewards >= 0:
            succ += 1

        if dones:
            n_ep += 1
            obs = env.reset()
            # env.print_rewards_info()
            print()

    print('Success rate: ', succ/n_ep)

if __name__ == '__main__':
    main()
