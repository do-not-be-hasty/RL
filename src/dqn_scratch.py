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

from environment_builders import make_env_GoalBitFlipper, make_env_BitFlipper, make_env_GoalRubik
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


class ReplayBuffer:
    def __init__(self, size):
        self.storage = []
        self.max_size = size
        self.index = 0

    def add(self, episode):
        for transition in episode:
            if len(self.storage) < self.max_size:
                self.storage.append(transition)
            else:
                self.storage[self.index] = transition
                self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        observations, actions, nexts, rewards, dones, steps = [], [], [], [], [], []

        for _ in range(batch_size):
            observation, action, next, reward, done = self.storage[np.random.randint(0, len(self.storage))]
            observations.append(observation)
            actions.append(action)
            nexts.append(next)
            rewards.append(reward)
            dones.append(done)
            steps.append(1)

        return np.array(observations), np.array(actions), np.array(nexts), np.array(rewards), np.array(dones), \
               np.array(steps)


class HindsightReplayBuffer:
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
        observations, actions, nexts, rewards, dones, ranges, steps = [], [], [], [], [], [], []

        for _ in range(batch_size):
            transition_idx = np.random.randint(0, len(self.storage))
            observation, action, next, _, _, episode_range = self.storage[transition_idx]

            goal_idx = np.random.randint(transition_idx, episode_range)
            goal_idx = goal_idx % self.max_size
            _, _, goal_observation, _, _, _ = self.storage[goal_idx]

            reward = reached_goal_reward(next, goal_observation)

            observations.append(add_goal(observation, goal_observation))
            actions.append(action)
            nexts.append(add_goal(next, goal_observation))
            rewards.append(reward)
            dones.append(np.array_equal(next, goal_observation))
            steps.append(1)

        return np.array(observations), np.array(actions), np.array(nexts), np.array(rewards), np.array(dones), \
               np.array(steps)


class MultistepHindsightReplayBuffer:
    def __init__(self, size, gamma, multistep=1):
        self.storage = []
        self.max_size = size
        self.index = 0
        self.multistep = multistep
        self.rewards_list = [(1 - gamma ** i) / (gamma - 1) for i in range(self.multistep + 1)]

    def add(self, episode):
        episode_range = self.index + len(episode)

        for transition in episode:
            if len(self.storage) < self.max_size:
                self.storage.append(transition + (episode_range,))
            else:
                self.storage[self.index] = (transition + (episode_range,))

            self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        observations, actions, nexts, rewards, dones, ranges, steps = [], [], [], [], [], [], []

        for _ in range(batch_size):
            while True:
                step = np.random.randint(0, self.multistep)
                transition_idx = np.random.randint(0, len(self.storage))
                observation, action, next, _, _, episode_range = self.storage[transition_idx]

                goal_idx = np.random.randint(transition_idx, episode_range)
                next_idx = transition_idx + step
                if next_idx > goal_idx:
                    next_idx = goal_idx
                goal_idx = goal_idx % self.max_size
                next_idx = next_idx % self.max_size
                _, _, goal_observation, _, _, _ = self.storage[goal_idx]
                _, _, next, _, _, _ = self.storage[next_idx]

                for i in range(step):
                    _, _, middle, _, _, _ = self.storage[(transition_idx + i) % self.max_size]
                    if np.array_equal(middle, next) or np.array_equal(middle, goal_observation):
                        next = middle
                        step = i
                        break

                reward = self.rewards_list[step] if np.array_equal(next['achieved_goal'], goal_observation[
                    'achieved_goal']) else self.rewards_list[step + 1]

                if np.array_equal(observation['achieved_goal'], next['achieved_goal']):
                    continue

                observations.append(add_goal(observation, goal_observation))
                actions.append(action)
                nexts.append(add_goal(next, goal_observation))
                rewards.append(reward)
                dones.append(np.array_equal(next, goal_observation))
                steps.append(step + 1)
                break

        return np.array(observations), np.array(actions), np.array(nexts), np.array(rewards), np.array(dones), \
               np.array(steps)


class DQN:
    def __init__(self, env):
        self.network = None
        self.gamma = 0.98
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.env = env
        self.replay_buffer = ReplayBuffer(2000000)
        self.update_target_freq = 10
        self.learning_start = 1000
        self.exploration = 0.01
        self.exploration_final_eps = 0.1

        self.step = 0
        self.maxloss = (0, None)

        self.optimizer = None
        self.network = self.setup_network()
        self.target_network = self.setup_network(dummy=True)

    def setup_network(self, dummy=False):
        input = Input(shape=self.env.observation_space.shape)
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

        value = Dense(layer_width)(features)
        value = LayerNormalization()(value)
        value = Activation('relu')(value)
        value = Dense(1)(value)

        advantage = Dense(layer_width)(features)
        advantage = LayerNormalization()(advantage)
        advantage = Activation('relu')(advantage)
        advantage = Dense(self.env.action_space.n, activation='linear')(advantage)

        output = Lambda(lambda x: x[0] + (x[1] - K.expand_dims(K.mean(x[1], axis=1), axis=1)))([value, advantage])

        network = Model(inputs=input, outputs=output)
        optimizer = Adam(lr=self.learning_rate)
        if not dummy:
            self.optimizer = optimizer
        # network.compile(loss='mean_squared_error', optimizer=optimizer)
        network.compile(loss=huber_loss_fn(delta=1.0), optimizer=optimizer)

        if not dummy:
            network.summary()

        return network

    def convert_observation(self, observation):
        return observation

    def learn(self, steps):
        plain_observation = self.env.reset()
        observation = self.convert_observation(plain_observation)

        episode_reward = 0.
        num_episode = 0
        episode_rewards = []
        episode_transitions = []
        episode_success = []
        visited = set()
        episode_diversities = []
        losses = []

        for step in range(steps):
            self.step = step
            exploration_plain_eps = max(1. - step / (steps * self.exploration), self.exploration_final_eps)
            if tuple(observation.flatten()) in visited:
                exploration_eps = 1.
            else:
                exploration_eps = exploration_plain_eps
                visited.add(tuple(observation.flatten()))
            action = self.act(observation, exploration_eps)

            next_observation, reward, done, _ = self.env.step(action)
            episode_transitions.append((plain_observation, action, next_observation, reward, done))

            plain_observation = next_observation
            observation = self.convert_observation(plain_observation)
            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward)
                self.replay_buffer.add(episode_transitions)
                episode_success.append(int(reward > -1e-4))
                episode_transitions = []
                episode_diversities.append(len(visited))
                visited = set()

                if num_episode % 200 == 0:
                    neptune_logger('episodes', num_episode)
                    neptune_logger('reward', np.mean(episode_rewards))
                    neptune_logger('success rate', np.mean(episode_success))
                    # neptune_logger('no exploration success rate', evaluate_model(self, self.env, nevals=10))
                    neptune_logger('diversity', np.mean(episode_diversities))
                    neptune_logger('exploration', exploration_plain_eps)
                    neptune_logger('loss', np.mean(losses))
                    log_rubik_curriculum_eval([1, 2, 3, 4, 5, 7], self, self.env, nevals=30)
                    log_mean_value(self, self.env, scrambles=1, nevals=30)
                    log_mean_value(self, self.env, scrambles=2, nevals=30)
                    log_mean_value(self, self.env, scrambles=3, nevals=30)
                    log_mean_value(self, self.env, scrambles=5, nevals=30)
                    log_mean_value(self, self.env, scrambles=7, nevals=30)
                    log_mean_value(self, self.env, scrambles=50, nevals=30)
                    episode_rewards = []
                    episode_success = []
                    episode_diversities = []
                    losses = []

                    # print(self.maxloss)
                    # self.maxloss = (0, None)

                plain_observation = self.env.reset()
                observation = self.convert_observation(plain_observation)
                episode_reward = 0.
                num_episode += 1

            if step % self.update_target_freq == 0:
                self.update_target()

            if step >= self.learning_start:
                # K.set_value(self.network.optimizer.lr, 1e-8)
                step_loss = self.train_step()
                losses.append(step_loss)

    def predict(self, data):
        return self.network.predict(data)

    def predict_value(self, data):
        return np.max(self.network.predict(data), axis=-1)

    def act(self, observation, exploration=0.):
        if np.random.rand() < exploration:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.network.predict(np.array([observation]))[0], axis=-1)

    def train_step(self):
        observations, actions, nexts, rewards, dones, steps = self.replay_buffer.sample(self.batch_size)
        gammas = self.gamma ** steps

        network_next_q = self.network.predict(nexts)
        target_next_q = self.target_network.predict(nexts)

        values = np.transpose(
            [target_next_q[np.arange(len(nexts)), np.argmax(network_next_q, axis=-1)] * gammas * (
                    1. - dones) + rewards])

        actions_ind = np.eye(self.env.action_space.n)[actions]
        noactions_ind = np.ones(shape=actions_ind.shape) - actions_ind

        target_q = self.target_network.predict(observations)
        targets = target_q * noactions_ind + actions_ind * values

        # print('debug')
        # print(observations)
        # print(targets)

        return self.update_weights(observations, targets)

    def update_weights(self, data, target):
        loss = self.network.train_on_batch(x=data, y=target) * self.env.action_space.n * len(data)
        # if loss > self.maxloss[0]:
        #     self.maxloss = (loss, data, target)
        return loss
        # self.network.fit(x=data, y=target, epochs=10, verbose=0)
        # return 0

    def update_target(self):
        for (model_layer, target_layer) in zip(self.network.layers, self.target_network.layers):
            weights = model_layer.get_weights()
            target_layer.set_weights(weights)


class HER(DQN):
    def __init__(self, env):
        super().__init__(env)
        # self.replay_buffer = MultistepHindsightReplayBuffer(2000000, self.gamma, multistep=2)
        self.replay_buffer = HindsightReplayBuffer(2000000)

    def convert_observation(self, observation):
        return np.concatenate([observation['observation'], observation['desired_goal']], axis=-1)


def evaluate(model):
    print('evaluation')

    while True:
        observation = model.env.reset()
        episode_reward = 0.

        while True:
            action = model.act(observation)
            next_observation, reward, done, _ = model.env.step(action)

            episode_reward += reward
            observation = next_observation

            if done:
                print('reward:', episode_reward)
                break


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
    # env = make_env_BitFlipper(n=5, space_seed=None)
    # env = make_env_GoalBitFlipper(n=5, space_seed=None)
    env = make_env_GoalRubik(step_limit=100, shuffles=100)
    model = HER(env)
    # model.learn(100000 * 16 * 50)
    model.learn(120000000)


if __name__ == '__main__':
    main()
