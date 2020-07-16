import copy
from time import sleep

import gym
import numpy as np
from keras import Model, Input
from keras.layers import Dense, Flatten, BatchNormalization, Activation, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras_layer_normalization import LayerNormalization
from mrunner.helpers.client_helper import get_configuration

from environment_builders import make_env_GoalBitFlipper, make_env_BitFlipper, make_env_GoalRubik, make_env_Rubik
from mrunner.helpers.client_helper import logger as raw_neptune_logger
import tensorflow as tf


def add_goal(observation, goal_observation):
    return np.concatenate([observation['observation'], goal_observation['achieved_goal']], axis=-1)


def reached_goal_reward(observation, goal_observation):
    return 0 if np.array_equal(observation['achieved_goal'], goal_observation['achieved_goal']) else -1


def neptune_logger(message, value, use_stdout=True):
    raw_neptune_logger(message, value)
    if use_stdout:
        print(message.replace(' ', ''), value)


def huber_loss_fn(**huber_loss_kwargs):
    def keras_huber_loss(y_true, y_pred):
        return tf.losses.huber_loss(y_true, y_pred, **huber_loss_kwargs)

    return keras_huber_loss


def metric_loss_fn(huber_weight, huber_delta):
    def keras_metric_loss(y_true, y_pred):
        x = y_pred - y_true
        return K.mean(
            huber_weight * K.minimum(K.maximum(2 * huber_delta * K.abs(x) - huber_delta ** 2, huber_delta ** 2),
                                     x ** 2) + K.relu(x) ** 2)

    return keras_metric_loss


def evaluate_model(model, env, nevals=10):
    def single_eval(model, env):
        plain_observation = env.reset()
        observation = model.convert_observation(plain_observation)

        while True:
            action = model.act(observation, exploration=0)
            next_observation, reward, done, _ = env.step(action)

            plain_observation = next_observation
            observation = model.convert_observation(plain_observation)

            if done:
                return int(reward == 0)

    return np.mean([single_eval(model, copy.deepcopy(env)) for _ in range(nevals)])


def log_rubik_curriculum_eval(shuffles_list, model, env, nevals=10):
    for shuffle in shuffles_list:
        eval_env = copy.deepcopy(env)
        eval_env.config(scramble_size=shuffle, step_limit=shuffle + 4)
        neptune_logger('shuffles {0} success rate'.format(shuffle), evaluate_model(model, eval_env, nevals))


def log_mean_value(model, env, scrambles, nevals=10):
    eval_env = copy.deepcopy(env)
    eval_env.config(scramble_size=scrambles)
    values = []

    for _ in range(nevals):
        plain_observation = eval_env.reset()
        observation = model.convert_observation(plain_observation)
        values.append(model.predict_value(np.array([observation])))

    neptune_logger('infty on {0}'.format(scrambles), np.mean(values))


class MetricReplayBuffer:
    def __init__(self, size):
        self.storage = []
        self.max_size = size
        self.index = 0

    def add(self, episode):
        episode_range = self.index + len(episode)

        for transition in episode:
            if len(self.storage) < self.max_size:
                self.storage.append(transition + (episode_range,))
            else:
                self.storage[self.index] = (transition + (episode_range,))

            self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        observations, steps = [], []

        for _ in range(batch_size):
            transition_idx = np.random.randint(0, len(self.storage))
            observation, _, episode_range = self.storage[transition_idx]

            goal_idx = np.random.randint(transition_idx, episode_range)
            step = goal_idx - transition_idx + 1
            _, goal, _ = self.storage[goal_idx % self.max_size]

            if np.array_equal(observation, goal):
                step = 0
            else:
                for idx in range(transition_idx, goal_idx):
                    _, middle, _ = self.storage[idx % self.max_size]
                    if np.array_equal(middle, goal):
                        step = idx - transition_idx + 1
                        break

            observations.append(np.concatenate([observation, goal], axis=-1))
            # TODO debug mode on
            # steps.append(np.sum(observation != goal))
            steps.append(step)

        return np.array(observations), np.array(steps)


class MetricDQN:
    def __init__(self, env):
        self.network = None
        self.gamma = 0.98
        self.learning_rate = 1e-3
        self.batch_size = 32
        self.env = env
        self.replay_buffer = MetricReplayBuffer(2000000)
        self.update_target_freq = 10
        self.learning_start = 1000
        self.exploration = 0.01
        self.exploration_final_eps = 0.1

        self.step = 0
        self.maxloss = (0, None)

        self.optimizer = None
        self.network = self.setup_network()

    def setup_network(self, dummy=False):
        input_shape = self.env.observation_space.shape[:-1] + (self.env.observation_space.shape[-1] * 2,)
        input = Input(shape=input_shape)

        if len(self.env.observation_space.shape) > 1:
            layer = Flatten()(input)
        else:
            layer = input
        layer_width = 1024
        layer = Dense(layer_width)(layer)
        layer = LayerNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dense(layer_width)(layer)
        layer = LayerNormalization()(layer)
        features = Activation('relu')(layer)

        output = Dense(1)(features)

        network = Model(inputs=input, outputs=output)
        optimizer = Adam(lr=self.learning_rate, clipnorm=1.0)
        if not dummy:
            self.optimizer = optimizer
        # network.compile(loss='mean_squared_error', optimizer=optimizer)
        network.compile(loss=metric_loss_fn(huber_weight=0.1, huber_delta=1.0), optimizer=optimizer)

        if not dummy:
            network.summary()

        return network

    def learn(self, steps):
        observation = self.env.reset()

        num_episode = 0
        episode_transitions = []
        visited = set()
        episode_diversities = []
        losses = []

        for step in range(steps):
            self.step = step
            visited.add(tuple(observation.flatten()))
            action = self.env.action_space.sample()

            next_observation, reward, done, _ = self.env.step(action)
            episode_transitions.append((observation, next_observation))
            observation = next_observation

            if done:
                self.replay_buffer.add(episode_transitions)
                episode_transitions = []
                episode_diversities.append(len(visited))
                visited = set()

                if num_episode % 200 == 0:
                    neptune_logger('episodes', num_episode)
                    neptune_logger('diversity', np.mean(episode_diversities))
                    # neptune_logger('success rate', evaluate(self, nevals=20))
                    log_evaluate_rubik(self, [3, 5, 7, 10], nevals=30)
                    log_rubik_infty(self, [1, 3, 5, 7, 50], nevals=20)
                    neptune_logger('loss', np.mean(losses))
                    # log_distances(self, nevals=10)
                    episode_diversities = []
                    losses = []

                observation = self.env.reset()
                num_episode += 1

            if step >= self.learning_start:
                # K.set_value(self.network.optimizer.lr, 1e-8)
                step_loss = self.train_step()
                losses.append(step_loss)

    def predict(self, data):
        return self.network.predict(data)

    def act(self, env, goal, exploration=0.):
        if np.random.rand() < exploration:
            return env.action_space.sample()
        else:
            values = []
            for a in range(env.action_space.n):
                action_env = copy.deepcopy(env)
                next_observation, reward, done, _ = action_env.step(a)
                values.append(self.network.predict(np.array([np.concatenate([next_observation, goal], axis=-1)])))

            return np.argmin(values)

    def train_step(self):
        observations, steps = self.replay_buffer.sample(self.batch_size)
        distances = (1 - self.gamma ** steps) / (1 - self.gamma)
        distances = np.expand_dims(distances, axis=-1)

        return self.update_weights(observations, distances)

    def update_weights(self, data, target):
        loss = self.network.train_on_batch(x=data, y=target) * self.env.action_space.n * len(data)
        return loss


def evaluate(model, nevals=100):
    env = copy.deepcopy(model.env)

    def single_eval():
        observation = env.reset()
        goal = env.goal
        episode_reward = 0.
        # print('goal:', goal)

        while True:
            # print(observation)
            action = model.act(env, goal)
            next_observation, reward, done, _ = env.step(action)

            episode_reward += reward
            observation = next_observation

            if done:
                # print(observation)
                return 0 if reward < 0 else 1

    return np.mean([single_eval() for _ in range(nevals)])


def log_evaluate_rubik(model, shuffles_list, nevals=100):
    def evaluate_shuffle(model, shuffles, nevals):
        env = copy.deepcopy(model.env)
        env.config(scramble_size=shuffles, step_limit=(shuffles + 2) * 2)

        def single_eval():
            observation = env.reset()
            goal = env.goal
            episode_reward = 0.
            # print('goal:', goal)

            while True:
                # print(observation)
                action = model.act(env, goal)
                next_observation, reward, done, _ = env.step(action)

                episode_reward += reward
                observation = next_observation

                if done:
                    # print(observation)
                    return 0 if reward < 0 else 1

        return np.mean([single_eval() for _ in range(nevals)])

    for shuffles in shuffles_list:
        neptune_logger('shuffles {0} success rate'.format(shuffles), evaluate_shuffle(model, shuffles, nevals))


def log_rubik_infty(model, distance_list, nevals=100):
    def infty_distance(model, distance, nevals):
        env = copy.deepcopy(model.env)
        env.config(scramble_size=distance)

        def single_infty():
            observation = env.reset()
            goal = env.goal
            return model.predict(np.array([np.concatenate([observation, goal], axis=-1)]))[0][0]

        return np.mean([single_infty() for _ in range(nevals)])

    for distance in distance_list:
        neptune_logger('infty on {0}'.format(distance), infty_distance(model, distance, nevals))


def log_distances(model, nevals=10):
    data = np.array(
        [np.concatenate([model.env.observation_space.sample(), model.env.observation_space.sample()], axis=-1) for _
         in range(nevals)])
    print(data)
    print(model.predict(data).flatten(),
          np.sum(data[:, :model.env.observation_space.shape[-1]] != data[:, model.env.observation_space.shape[-1]:],
                 axis=-1))


def main():
    try:
        params = get_configuration(print_diagnostics=True, with_neptune=True)
    except TypeError:
        print(' ************************************************\n',
              '                NEPTUNE DISABLED                \n',
              '************************************************')
        sleep(2)
    # env = gym.make("CartPole-v1")
    # env = gym.make("MountainCar-v0")
    # env = make_env_BitFlipper(n=50, space_seed=None)
    # env = make_env_GoalBitFlipper(n=5, space_seed=None)
    # env = make_env_GoalRubik(step_limit=100, shuffles=100)
    env = make_env_Rubik(step_limit=50, shuffles=100)
    model = MetricDQN(env)
    # model.learn(100000 * 16 * 50)
    model.learn(120000000)


if __name__ == '__main__':
    main()
