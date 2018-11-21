import gym
from gym_maze.envs import MazeEnv

from stable_baselines.common.policies import MlpPolicy as PPO2_Policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, DDPG
from stable_baselines.deepq import MlpPolicy as DQN_Policy
from stable_baselines.ddpg import MlpPolicy as DDPG_Policy
from stable_baselines.her import HER

from utility import resources_dir, get_cur_time_str


def main():
    env = gym.make('MazeEnv-v0')
    env = DummyVecEnv([lambda: env])

    model = PPO2(
        policy=PPO2_Policy,
        env=env,
        learning_rate=1e-3,
        # buffer_size=50000,
        # exploration_fraction=0.1,
        # exploration_final_eps=0.02,
        verbose=1,
    )

    model = model.learn(total_timesteps=40000)
    model.save(str(resources_dir().joinpath('model.pkl')))

    env.env_method('set_live_display', True)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main()
