import copy
import random

import keras
import numpy as np
from keras import Sequential
from keras import backend as K
from keras.layers import Dense, Flatten, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from rubik_solver import utils as rubik_solver
from mrunner.helpers.client_helper import logger as neptune_logger

from environment_builders import make_env_Rubik


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


def run_policy_single(model, env):
    obs = env.reset()

    while True:
        action = model.predict(np.array([obs]))[0].argmax()
        # action = random.randint(0, 11)
        obs, rewards, dones, info = env.step(action)

        if dones:
            return rewards == 0


def evaluate(model, env, num_episodes, num_shuffles):
    env = copy.deepcopy(env)
    # env.config(scramble_size=num_shuffles)
    env.scrambleSize = num_shuffles
    return np.mean([run_policy_single(model, env) for _ in range(num_episodes)])


action_encoder = np.eye(len(action_to_move_lookup.keys()))


def action_to_one_hot(action):
    return action_encoder[action]


class Buffer:
    def __init__(self, size):
        self.buffer = []
        self.place = 0
        self.size = size

    def add(self, entry):
        if len(self.buffer) < self.size:
            self.buffer.append(entry)
            return

        if self.place >= len(self.buffer):
            self.place = 0

        self.buffer[self.place] = entry
        self.place += 1

    def one_hot_actions(self, actions):
        return [action_to_one_hot(action) for action in actions]

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
        observations = [self.buffer[i][0] for i in idxes]
        actions = [self.buffer[i][1] for i in idxes]
        return np.array(observations), np.array(self.one_hot_actions(actions), dtype=np.float)


def reverse_move(move):
    if len(move) == 1:
        return move + "'"
    else:
        return move[:-1]


def push_drift(env, expert_action, buffer):
    for move in action_to_move_lookup.values():
        if move == expert_action:
            continue
        if random.randint(0, 3) != 0:
            continue

        tmp_env = copy.deepcopy(env)
        obs, _, _, _ = tmp_env.step(move_to_action(move))
        buffer.add((obs, move_to_action(reverse_move(move))))


def push_simple_solver(obs, env, buffer):
    env = copy.deepcopy(env)
    solution = [move_to_action(move) for move in quarterize(rubik_solver.solve(cube_bin_to_str(obs), 'Kociemba'))]

    for action in solution:
        buffer.add((obs, action))
        obs, rewards, dones, info = env.step(action)


def policy_rollout(model, env, buffer, shuffles, solver_freq):
    env = copy.deepcopy(env)
    env.scrambleSize = shuffles
    obs = env.reset()

    while True:
        action = model.predict(np.array([obs]))[0].argmax()
        obs, rewards, dones, info = env.step(action)

        if dones:
            return

        if random.random() < solver_freq:
            push_simple_solver(obs, env, buffer)


def supervised_Rubik():
    buffer = Buffer(1e6)
    step = 0
    batch = 32
    decay = 0.

    env = make_env_Rubik(step_limit=50, shuffles=10)

    model = Sequential()
    model.add(Flatten(input_shape=env.observation_space.sample().shape))
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(decay)))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(decay)))
    model.add(BatchNormalization())
    model.add(Dense(env.action_space.n, activation='softmax'))

    model.summary()

    lr = 1e-3
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    print(model.metrics_names)

    # simple imitation
    while True:
        # env.scrambleSize = random.randint(7, 20)
        obs = env.reset()
        print(cube_bin_to_str(obs))
        solution = [move_to_action(move) for move in quarterize(rubik_solver.solve(cube_bin_to_str(obs), 'Kociemba'))]

        for action in solution:
            # print(model.optimizer.get_config())
            push_drift(env, action, buffer)

            buffer.add((obs, action))
            obs, rewards, dones, info = env.step(action)
            # print(cube_bin_to_str(obs))
            # print(dones)

            if step > 100:
                inputs, targets = buffer.sample(batch)
                model.train_on_batch(inputs, targets)

                lr *= 0.999999
                K.set_value(model.optimizer.lr, lr)

            step += 1

            if step % 1000 == 0:
                print('\n{0} steps'.format(step))
                for i in [2, 4, 7, 10, 13, 19, 25]:
                    score = evaluate(model, env, 10, i)
                    neptune_logger('shuffles {0} success rate'.format(i), score)

                inputs, targets = buffer.sample(256)
                metrics = model.evaluate(inputs, targets)
                neptune_logger('loss', metrics[0])
                neptune_logger('accuracy', metrics[1])
                neptune_logger('learning rate', lr)

    # dagger
    # while True:
    #     policy_rollout(model, env, buffer, 10, 0.1)
    #
    #     if step > 100:
    #         for _ in range(100):
    #             inputs, targets = buffer.sample(batch)
    #             model.train_on_batch(inputs, targets)
    #
    #         if step % 10 == 0:
    #             print('\n{0} steps'.format(step))
    #             for i in [2, 4, 7, 10, 13, 19, 25]:
    #                 score = evaluate(model, env, 10, i)
    #                 neptune_logger('shuffles {0} success rate'.format(i), score)
    #
    #             inputs, targets = buffer.sample(256)
    #             metrics = model.evaluate(inputs, targets)
    #             neptune_logger('loss', metrics[0])
    #             neptune_logger('accuracy', metrics[1])
    #
    #     step += 1


def prepare_solutions(shuffles, count):
    initial_pos = []
    moves = []

    env = make_env_Rubik(step_limit=100, shuffles=shuffles)

    for i in range(count):
        obs = env.reset()
        print(cube_bin_to_str(obs))
        solution = [move_to_action(move) for move in quarterize(rubik_solver.solve(cube_bin_to_str(obs), 'Kociemba'))]
        print(solution)

        initial_pos.append(obs)
        moves.append(solution)

    result = np.array([initial_pos, moves])
    print(result)
    np.save("solutions.npy", result)



if __name__ == '__main__':
    # supervised_Rubik()
    prepare_solutions(100, 5)
