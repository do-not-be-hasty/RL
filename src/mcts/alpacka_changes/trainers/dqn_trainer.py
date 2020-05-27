"""Supervised trainer."""

import gin
import numpy as np
import copy

from alpacka import data
from alpacka.trainers import base
from alpacka.trainers import replay_buffers

from alpacka.trainers.episode_replay_buffer import ReplayBuffer as EpisodeReplayBuffer
from gym_BitFlipper.envs import GoalBitFlipperEnv
from mcts.utility import evaluation_infos


class DqnTrainer(base.Trainer):
    """DQN-style trainer.
    """

    def __init__(
            self,
            network_signature,
            batch_size=64,
            n_steps_per_epoch=1000,
            replay_buffer_capacity=1000000,
            hindsight=4,
    ):
        """Initializes SupervisedTrainer.

        Args:
            network_signature (pytree): Input signature for the network.
            target (pytree): Pytree of functions episode -> target for
                determining the targets for network training. The structure of
                the tree should reflect the structure of a target.
            batch_size (int): Batch size.
            n_steps_per_epoch (int): Number of optimizer steps to do per
                epoch.
            replay_buffer_capacity (int): Maximum size of the replay buffer.
            replay_buffer_sampling_hierarchy (tuple): Sequence of Episode
                attribute names, defining the sampling hierarchy.
        """
        super().__init__(network_signature)
        self._batch_size = batch_size
        self._n_steps_per_epoch = n_steps_per_epoch
        self._replay_buffer = EpisodeReplayBuffer(replay_buffer_capacity, hindsight=hindsight)
        self._gamma = 0.98
        self._step = 0
        self._callback_step = 0
        self._ref_obs = None

    def add_episode(self, episode):
        episode = [(trans.observation, trans.action, trans.reward, trans.next_observation, trans.done) for trans in
                   data.nested_unstack(episode.transition_batch)]
        self._replay_buffer.add(episode)

    def train_epoch(self, network):
        loss = []

        def single_eval():
            env = GoalBitFlipperEnv()

            full_obs = env.reset()
            obs = np.concatenate([full_obs['observation'], full_obs['desired_goal']], axis=-1)
            done = False
            # print(obs)

            while True:
                action = network.predict_action(obs)[0]
                assert(action == np.argmax(network.predict_q_values(obs)))
                # print('q_values', network.predict_q_values(obs))
                # print('value', network.predict(obs))

                (full_next, reward, done, info) = env.step(action)
                next = np.concatenate([full_next['observation'], full_next['desired_goal']], axis=-1)
                # print(next)
                obs = next

                if done:
                    print('SOLVED' if reward >= 0 else 'nope')
                    return reward+1

        # print('play success rate', np.mean([single_eval() for _ in range(10)]))
        # if self._ref_obs is None:
        if True:
            sampled_obs, sampled_action, sampled_reward, sampled_next, sampled_done = self._replay_buffer.sample(self._batch_size)
            self._ref_obs = sampled_next
        # print(self._ref_obs)
        # print('values_beg', list(network.predict([self._ref_obs]).flatten()))

        for _ in range(self._n_steps_per_epoch):
            self._step += 1
            network.learning_step(self._step, self._replay_buffer, None, loss)
        # print('values_fin', list(network.predict([self._ref_obs]).flatten()))

        self._callback_step += 1
        info = evaluation_infos(network, self._callback_step)
        info['training_loss'] = 0. if len(loss)==0 else np.mean(loss)

        return info
