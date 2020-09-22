from mrunner.helpers.client_helper import get_configuration

from learning_configurations import learn_Rubik_HER, learn_BitFlipper_HER, learn_Maze_HER
import numpy as np


def main():
    params = get_configuration(print_diagnostics=True, with_neptune=True)
    np.random.seed(None)

    # learn_BitFlipper_HER(50)
    learn_Maze_HER(50)
    # learn_Rubik_HER()


if __name__ == '__main__':
    main()
