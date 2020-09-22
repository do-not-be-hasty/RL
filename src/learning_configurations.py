from gym_maze import RandomMazeGenerator

import networks
from environment_builders import make_env_BitFlipper, make_env_GoalBitFlipper, make_env_GoalMaze, \
    make_env_Rubik, make_env_GoalRubik
from networks import CustomPolicy
from utility import callback, evaluate, hard_eval, neptune_logger, bitflipper_callback, maze_callback
from DQN_HER import DQN_HER as HER
from functools import partial
from stable_baselines.deepq import MlpPolicy as DQN_Policy


def restore_HER_model(path, env, **kwargs):
    return HER.load(path, env, **kwargs)


def learn_BitFlipper_HER(n):
    print("BitFlipper({0}), DQN+HER".format(n))

    env = make_env_GoalBitFlipper(n=n, space_seed=None)
    model = HER(
        policy=partial(DQN_Policy, layers=[1024]),
        env=env,
        hindsight=7,
        learning_rate=1e-4,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
    )

    try:
        model.learn(total_timesteps=250 * 100 * n,
                    callback=bitflipper_callback,
                    )
    except KeyboardInterrupt:
        pass

    evaluate(model, env)


def learn_Maze_HER(n):
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

    model = HER(
        policy=partial(DQN_Policy, layers=[1024]),
        env=env,
        hindsight=3,
        learning_rate=1e-4,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
    )

    print("Initial distance: {0}".format(env._distance_diameter()))

    try:
        model.learn(total_timesteps=200 * 100 * n,
                    callback=maze_callback,
                    log_interval=20,
                    )
    except KeyboardInterrupt:
        pass

    # env._set_live_display(True)
    evaluate(model, env)


def learn_Rubik_HER():
    print("Rubik, DQN+HER")

    env = make_env_GoalRubik(
        step_limit=100,
        shuffles=100,
    )
    model = HER(
        policy=partial(CustomPolicy, arch_fun=networks.arch_color_embedding),
        env=env,
        hindsight=1000,
        learning_rate=3e-5,
        exploration_fraction=0.01,
        exploration_final_eps=0.05,
    )

    try:
        model.learn(total_timesteps=120000000,
                    callback=callback,
                    log_interval=200,
                    )
    except KeyboardInterrupt:
        pass
