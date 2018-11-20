from argparse import Namespace

import gym

from baselines import deepq, ppo2, run
from baselines.run import get_env_type, get_learn_function, get_learn_function_defaults, build_env, get_default_network, _game_envs
from utility import resources_dir, get_cur_time_str

from gym_maze.envs import MazeEnv
from gym_maze.envs.generators import SimpleMazeGenerator, RandomMazeGenerator, RandomBlockMazeGenerator, \
                                     UMazeGenerator, TMazeGenerator, WaterMazeGenerator


_game_envs['custom'].add('MazeEnv-v0')


def train(env, learn_func, save=True, name='model'+get_cur_time_str()+'.pkl', **model_kwargs):
    act = learn_func(
        env=env,
        network='mlp',
        **model_kwargs,
        # lr=1e-3,
        total_timesteps=7000,
        # buffer_size=50000,
        # exploration_fraction=0.1,
        # exploration_final_eps=0.02,
        # print_freq=10,
    )
    
    if save:
        print("Saving model to " + name)
        act.save(str(resources_dir().joinpath(name)))


def show(name, env, learn_func, **model_kwargs):
    env.set_live_display(True)

    model = learn_func(
        env=env,
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


def main():
    # env = MazeEnv(RandomMazeGenerator(width=7, height=7, complexity=.5, density=.5, seed=5), obs_type='discrete')
    # env = gym.make('CartPole-v0')
    # env = gym.make('MazeEnv-v0')
    # print(env.actions)
    # train(env, run.get_learn_function('deepq'), hiddens=[64, 64], name='lab.pkl')
    # train(env, run.get_learn_function('ddpg'), name='lab.pkl')
    # show('lab.pkl', env, deepq.learn, hiddens=[256,256])

    args = Namespace(alg='deepq', env='MazeEnv-v0', gamestate=None, network=None,
                     num_env=None, num_timesteps=100000.0, play=False, reward_scale=1.0,
                     save_path=None, save_video_interval=0, save_video_length=200, seed=None)

    model, env = run.train(args, {})

if __name__ == '__main__':
    main()
