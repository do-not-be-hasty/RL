import curses
import sys

from pycolab import ascii_art
from pycolab import human_ui
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

from maze_generator import generate

LEVELS = [
    ['XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
     'XX                  XX      XX          XX',
     'XX  XX  XXXXXX  XXXXXXXXXX  XX  XXXXXX  XX',
     'XX  XX      XX                  XX  XX  XX',
     'XXXXXXXXXX  XX  XXXXXX  XXXXXX  XX  XX  XX',
     'XX      XX  XX  XX          XX      XX  XX',
     'XXXXXX  XXXXXXXXXXXXXX  XXXXXX  XXXXXX  XX',
     'XX              XX  XX  XX  XX  XX      XX',
     'XX  XXXXXXXXXXXXXX  XX  XX  XXXXXXXXXX  XX',
     'XX      XX                  XX          XX',
     'XXXXXX  XXXXXXXXXX  XXXXXXXXXXXXXX  XXXXXX',
     'XX              XX          XX          XX',
     'XX  XX  XXXXXXXXXXXXXX  XX  XXXXXX  XXXXXX',
     'XX  XX  XX              XX  XX          XX',
     'XXXXXX  XX  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
     'XX  XX      XX  XX              XX      XX',
     'XX  XX  XX  XX  XXXXXX  XXXXXX  XXXXXX^ XX',
     'XX      XX      XX      XX      XX  XX  XX',
     'XXXXXXXXXX  XX  XX  XXXXXX  XXXXXX  XX  XX',
     'XX          XX          XX     P        XX',
     'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'],
]


COLOURS = {'P': (0, 999, 0),      # Player
           ' ': (500, 500, 500),  # Empty space
           '^': (700, 700, 700),  # Target
           'X': (999, 600, 200)}  # Wall


class PlayerSprite(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character):
        super(PlayerSprite, self).__init__(
              corner, position, character, impassable='X', confined_to_board=True)

    def update(self, actions, board, layers, backdrop, things, the_plot):

        if actions == 0:
            self._north(board, the_plot)
        elif actions == 1:
            self._west(board, the_plot)
        elif actions == 2:
            self._east(board, the_plot)
        elif actions == 3:
            self._south(board, the_plot)


class GoalDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        player_position = things['P'].position

        if layers['^'][player_position]:
            the_plot.add_reward(1)
            the_plot.terminate_episode()


def make_game(level):
    if level == -1:
        level_art = generate()
    else:
        level_art = LEVELS[level]

    return ascii_art.ascii_art_to_game(
        level_art,
        what_lies_beneath=' ',
        sprites={'P': PlayerSprite},
        drapes={'^': GoalDrape},
        update_schedule=['P', '^'],
        z_order=['^', 'P'],
    )


def main(argv=()):
    game = make_game(int(argv[1]) if len(argv) > 1 else 0)

    keys_to_actions = {
        curses.KEY_UP: 0,
        curses.KEY_LEFT: 1,
        curses.KEY_RIGHT: 2,
        curses.KEY_DOWN: 3,
    }

    ui = human_ui.CursesUi(
            keys_to_actions=keys_to_actions,
            delay=500, colour_fg=COLOURS)

    ui.play(game)


if __name__ == '__main__':
    main(sys.argv)
