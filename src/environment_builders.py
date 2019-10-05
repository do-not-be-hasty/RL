import gym
from gym.envs import register


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