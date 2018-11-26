import gym
from gym_maze.envs import MazeEnv

from stable_baselines.common.policies import MlpPolicy as PPO2_Policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, DDPG
from stable_baselines.deepq import MlpPolicy as DQN_Policy
from stable_baselines.ddpg import MlpPolicy as DDPG_Policy

from utility import resources_dir, get_cur_time_str

from DQN_HER import DQN_HER as HER


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
        # For pure gym env, eg DQN_HER
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


def main():
    env = gym.make('GoalMazeEnv-v0')
    # env = DummyVecEnv([lambda: env])

    model = HER(
        policy=DQN_Policy,
        env=env,
        learning_rate=1e-3,
        # buffer_size=50000,
        # exploration_fraction=0.1,
        # exploration_final_eps=0.02,
        # verbose=1,
    )

    model = model.learn(total_timesteps=50000, callback=callback)
    model.save(str(resources_dir().joinpath('model.pkl')))

    env._set_live_display(True)
    env._set_step_limit(100)
    env.reset_rewards_info()

    print('--- Evaluation\n')

    obs_dict = env.reset()
    obs = obs_dict['observation']
    for i in range(1000):
        action, _states = model.predict(obs)
        obs_dict, rewards, dones, info = env.step(action)
        obs = obs_dict['observation']
        env.render()

        if dones:
            obs_dict = env.reset()
            obs = obs_dict['observation']
            env.print_rewards_info()
            print()

if __name__ == '__main__':
    main()
