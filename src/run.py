from mrunner.helpers.client_helper import get_configuration

from learn import learn_Sokoban_HER, learn_Rubik_DQN, learn_Sokoban_HER_conv, learn_Rubik_HER, learn_BitFlipper_HER, \
    learn_Maze_HER, hard_eval_Rubik_HER
from supervised import supervised_Rubik, prepare_solutions


def main():
    params = get_configuration(print_diagnostics=True, with_neptune=True)

    # learn_Sokoban_HER_conv()
    learn_Rubik_HER()
    # hard_eval_Rubik_HER()
    # learn_BitFlipper_HER()
    # learn_Maze_HER()
    # supervised_Rubik()
    # prepare_solutions(shuffles=100, count=10000)


if __name__ == '__main__':
    main()
