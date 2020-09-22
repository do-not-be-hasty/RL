from gym_maze import RandomMazeGenerator

from environment_builders import make_env_BitFlipper, make_env_GoalBitFlipper, make_env_GoalMaze, make_env_Sokoban, \
    make_env_GoalSokoban, make_env_Rubik, make_env_GoalRubik
from utility import callback, evaluate, hard_eval, neptune_logger, bitflipper_callback
from models import HER_model, MTR_model, DQN_model, HER_model_conv, restore_HER_model


def learn_BitFlipper_HER():
    n = 150
    print("BitFlipper({0}), DQN+HER".format(n))

    env = make_env_GoalBitFlipper(n=n, space_seed=None)
    model = HER_model(env)

    try:
        model.learn(total_timesteps=100000 * 16 * n,
                    callback=bitflipper_callback,
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
        model.learn(total_timesteps=20000000,
                    callback=callback,
                    log_interval=20,
                    )
    except KeyboardInterrupt:
        pass

    # env._set_live_display(True)
    # evaluate(model, env)


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
        dim_room=(8, 8),
        max_steps=100,
        num_boxes=2,
        mode='one_hot',
        seed=None,
        curriculum=300,  # depth of DFS in reverse_play
    )
    model = HER_model(env)

    try:
        model.learn(total_timesteps=8000000,
                    callback=callback,
                    log_interval=10,
                    )
    except KeyboardInterrupt:
        pass


def learn_Sokoban_HER_conv():
    print("Sokoban, DQN+HER, CNN")

    env = make_env_GoalSokoban(
        dim_room=(8, 8),
        max_steps=100,
        num_boxes=2,
        mode='one_hot',
        seed=None,
        curriculum=300,  # depth of DFS in reverse_play
    )
    model = HER_model_conv(env)

    try:
        model.learn(total_timesteps=8000000,
                    callback=callback,
                    )
    except KeyboardInterrupt:
        pass


def learn_Rubik_DQN():
    print("Rubik, DQN")

    env = make_env_Rubik(
        step_limit=100,
        shuffles=5,
    )
    model = DQN_model(env)

    try:
        model.learn(total_timesteps=8000000,
                    # callback=callback,
                    )
    except KeyboardInterrupt:
        pass


def learn_Rubik_HER():
    print("Rubik, DQN+HER")

    env = make_env_GoalRubik(
        step_limit=100,
        shuffles=100,
    )
    model = HER_model(env)
    # model = restore_HER_model('/home/plgrid/plgmizaw/checkpoints/checkpoints_test_2020-07-04-05:03:24_100000', env,
    #                           learning_rate=3e-7, learning_starts=100 * 200 * 4000000, exploration_fraction=0.0001,
    #                           exploration_final_eps=0.01, model_save_episode_freq=-1)  # solved collector
    # model = restore_HER_model('/home/plgrid/plgmizaw/checkpoints/checkpoints_test_2020-07-14-11:46:46_140000', env, learning_rate=1e-6, learning_starts=100*200*40)

    try:
        model.learn(total_timesteps=120000000,
                    callback=callback,
                    log_interval=200,
                    )
    except KeyboardInterrupt:
        pass


def hard_eval_Rubik_HER():
    print("Rubik eval, DQN+HER")

    env = make_env_GoalRubik(
        step_limit=1e3,
        shuffles=5,
    )
    # model = HER_model(env)
    # model = restore_HER_model('/home/michal/Projekty/RL/RL/resources/baseline_2020-04-07-05:32:42_80000.pkl', env)
    model = restore_HER_model('/home/michal/Projekty/RL/RL/resources/network_2k_2020-05-21-09:02:01_40000.pkl', env)

    count = 0
    solved = 0
    while True:
        solved += hard_eval(model, env)
        count += 1
        neptune_logger('solved', solved/count)
