from mrunner.helpers.client_helper import get_configuration

from learning_configurations import learn_Rubik_HER, learn_BitFlipper_HER, learn_Maze_HER
import numpy as np
import os
import sys


def main():
    with_neptune = True
    if len(os.environ.get('NEPTUNE_API_TOKEN')) == 0:
        print('empty token, run without neptune', file=sys.stderr)
        with_neptune = False

    params = get_configuration(print_diagnostics=True, with_neptune=with_neptune)
    np.random.seed(None)

    learn_BitFlipper_HER(10)
    # learn_Maze_HER(50)
    # learn_Rubik_HER()


if __name__ == '__main__':
    main()
