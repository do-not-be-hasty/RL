import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten
from rubik_solver import utils as rubik_solver
from gym_rubik.envs.cube import Actions, Cube

from environment_builders import make_env_Rubik

env = make_env_Rubik(step_limit=100, shuffles=5)
print(env.observation_space.sample().shape)

model = Sequential()
model.add(Flatten(input_shape=env.observation_space.sample().shape))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(env.action_space.n, activation='softmax'))

model.summary()


def cube_labels():
    return 'ywrogb'


def cube_bin_to_str(binary_obs):
    ordered_faces = [binary_obs[i] for i in [0, 5, 2, 4, 3, 1]]
    aligned_faces = np.array([np.rot90(face, axes=(0, 1)) for face in ordered_faces])
    sticker_list = aligned_faces.reshape((-1, 6))
    string_obs = ''.join([cube_labels()[label] for label in np.where(sticker_list)[1]])
    return string_obs


def quarterize(moves):
    quarter_moves = []

    for move in moves:
        if move.double:
            move.double = False
            quarter_moves.append(move)
            quarter_moves.append(move)
        else:
            quarter_moves.append(move)

    return quarter_moves


action_to_move_lookup = {
    0: "U", 1: "U'", 2: "D", 3: "D'", 4: "F", 5: "F'",
    6: "B", 7: "B'", 8: "R", 9: "R'", 10: "L", 11: "L'"
}

move_to_action_lookup = {v: k for k, v in action_to_move_lookup.items()}


def move_to_action(move):
    return move_to_action_lookup[str(move)]


def run_policy(model, env):
    obs = env.reset()

    while (2):
        action = model.predict(np.array([obs]))[0].argmax()
        print(action)
        obs, rewards, dones, info = env.step(action)
        print(cube_bin_to_str(obs))

        if dones:
            return rewards


obs = env.reset()
for action in [move_to_action(move) for move in quarterize(rubik_solver.solve(cube_bin_to_str(obs), 'Kociemba'))]:
    obs, rewards, dones, info = env.step(action)
    print(cube_bin_to_str(obs))
    print(dones)


# run_policy(model, env)
