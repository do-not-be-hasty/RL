from mrunner.helpers.client_helper import get_configuration

from learn import learn_Sokoban_HER, learn_Rubik_DQN, learn_Sokoban_HER_conv


def main():
    params = get_configuration(print_diagnostics=True, with_neptune=True)

    learn_Sokoban_HER_conv()


if __name__ == '__main__':
    main()