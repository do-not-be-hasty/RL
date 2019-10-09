from gym_maze import RandomMazeGenerator

from environment_builders import make_env_BitFlipper, make_env_GoalBitFlipper, make_env_GoalMaze, make_env_Sokoban, \
    make_env_GoalSokoban
from utility import callback, evaluate
from models import HER_model, MTR_model, DQN_model, HER_model_conv


def learn_BitFlipper_HER():
    n = 15
    print("BitFlipper({0}), DQN+HER".format(n))

    env = make_env_GoalBitFlipper(n=n, space_seed=15)
    model = HER_model(env)

    try:
        model.learn(total_timesteps=10000*16*n)
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
        model.learn(total_timesteps=2000000,
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
        resample_state=True,
        step_limit=100)
    model = MTR_model(env)

    try:
        model = model.learn(total_timesteps=200000,
                            callback=callback,
                            )
    except KeyboardInterrupt:
        pass

    data_x, data_y = env.get_dist_data()
    print(model.model.evaluate(data_x, data_y))

def learn_Sokoban_DQN():
    print("Sokoban, DQN")

    env = make_env_Sokoban(
        dim_room=(8, 8),
        max_steps=100,
        num_boxes=2,
        mode='one_hot',
        seed=None,
        curriculum=300,  # depth of DFS in reverse_play
    )
    model = DQN_model(env)

    try:
        model.learn(total_timesteps=8000000,
                    callback=callback,
                    )
    except KeyboardInterrupt:
        pass

def learn_Sokoban_HER():
    print("Sokoban, DQN+HER")

    env = make_env_GoalSokoban(
        dim_room=(8,8),
        max_steps=100,
        num_boxes=2,
        mode='one_hot',
        seed=None,
        curriculum=300,  # depth of DFS in reverse_play
        )
    # model = HER_model(env)
    model = HER_model_conv(env)

    try:
        model.learn(total_timesteps=8000000,
                    callback=callback,
                    )
    except KeyboardInterrupt:
        pass