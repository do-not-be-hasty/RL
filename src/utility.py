import os
from pathlib import Path
import datetime

import gym
import sys
from gym.envs import register
from gym_BitFlipper.envs import BitFlipperEnv
import gym_sokoban


def resources_dir():
    env_var_str = 'RESOURCES_DIR'
    assert env_var_str in os.environ
    return Path(os.environ[env_var_str])


def get_cur_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def make_env_BitFlipper(n=10, space_seed=0):
    id = "BitFlipper"+str(n)+":"+str(space_seed)+"-v0"
    try:
        register(id=id,entry_point='gym_BitFlipper.envs:BitFlipperEnv',kwargs = {"space_seed":space_seed,"n":n})
    except:
        print("Environment with id = "+id+" already registered. Continuing with that environment.")

    env=gym.make(id)
    env.seed(0)

    return env


def make_env_GoalBitFlipper(n=10, space_seed=0):
    id = "GoalBitFlipper" + str(n) + ":" + str(space_seed) + "-v0"
    try:
        register(id=id, entry_point='gym_BitFlipper.envs:GoalBitFlipperEnv', kwargs={"space_seed": space_seed, "n": n})
    except:
        print("Environment with id = " + id + " already registered. Continuing with that environment.")

    env = gym.make(id)
    env.seed(0)

    return env


def make_env_GoalMaze(**kwargs):
    id = ("GoalMaze-" + str(kwargs) + "-v0").translate(str.maketrans('',''," {}'<>()_"))
    id = id.replace(',', '-')

    try:
        register(id=id, entry_point='gym_maze.envs:GoalMazeEnv', kwargs=kwargs)
        print("Registered environment with id = " + id)
    except:
        print("Environment with id = " + id + " already registered. Continuing with that environment.")

    env = gym.make(id)
    env.seed(0)

    return env


def make_env_Sokoban(**kwargs):
    id = ("Sokoban-" + str(kwargs) + "-v0").translate(str.maketrans('', '', " {}'<>()_"))
    id = id.replace(',', '-')

    try:
        register(id=id, entry_point='gym_sokoban.envs:SokobanEnv', kwargs=kwargs)
        print("Registered environment with id = " + id)
    except:
        print("Environment with id = " + id + " already registered. Continuing with that environment.")

    env = gym.make(id)

    return env


def make_env_GoalSokoban(**kwargs):
    id = ("GoalSokoban-" + str(kwargs) + "-v0").translate(str.maketrans('', '', " {}'<>()_"))
    id = id.replace(',', '-')

    try:
        register(id=id, entry_point='gym_sokoban.envs:GoalSokobanEnv', kwargs=kwargs)
        print("Registered environment with id = " + id)
    except:
        print("Environment with id = " + id + " already registered. Continuing with that environment.")

    env = gym.make(id)

    return env