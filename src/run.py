import gym
import time
import numpy as np

from baselines import deepq
from baselines.deepq.deepq import load_act
from utility import resources_dir, get_cur_time_str

from gym_maze.envs import MazeEnv
from gym_maze.envs.generators import SimpleMazeGenerator, RandomMazeGenerator, RandomBlockMazeGenerator, \
                                     UMazeGenerator, TMazeGenerator, WaterMazeGenerator



def train(env, save=True, name='model'+get_cur_time_str()+'.pkl', **model_kwargs):
    act = deepq.learn(
        env,
        network='mlp',
        **model_kwargs,
        lr=1e-3,
        total_timesteps=50000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
    )
    
    if save:
        print("Saving model to " + name)
        act.save(str(resources_dir().joinpath(name)))


def show(name, env, **model_kwargs):
    model = deepq.learn(
        env,
        network='mlp',
        load_path=str(resources_dir().joinpath(name)),
        total_timesteps=0,
        **model_kwargs
    )
    
    obs = env.reset()
    
    while True:
        actions, _, state, _ = model.step(obs)
        obs, _, done, _ = env.step(actions[0])
        env.render()
        print(obs)
        
        if done:
            obs = env.reset()
    
    env.close()


def main():
    np.random.seed(100)
    env = MazeEnv(RandomMazeGenerator(width=20, height=20, complexity=.5, density=.5), live_display=False, obs_type='discrete')
    
    train(env, hiddens=[64], name='lab.pkl')
    #show('lab.pkl', env, hiddens=[64])


if __name__ == '__main__':
    main()
