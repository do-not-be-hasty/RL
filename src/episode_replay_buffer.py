import random

import numpy as np
import sys

from stable_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size, hindsight=1):
        """
        Create Replay buffer.

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._hindsight = hindsight

    def __len__(self):
        return len(self._storage)

    # def add(self, obs_t, action, reward, obs_tp1, done):
    def add(self, episode):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        ep_range = self._next_idx + len(episode)
        for d in episode:
            data = d + (ep_range,)
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes, batch_size):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        rep = np.random.multinomial(batch_size-len(idxes), np.ones(len(idxes))/len(idxes))

        for it in range(len(idxes)):
            i = idxes[it]
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, ep_range = data
            # print(obs_t, action, reward, obs_tp1, done, ep_range)

            def push_trans(goal, true_replay=False):
                obses_t.append(np.array(np.concatenate([obs_t['observation'], goal], axis=-1), copy=False))
                actions.append(np.array(action, copy=False))
                if np.array_equal(obs_tp1['achieved_goal'], goal):
                    rewards.append(0)
                    dones.append(1)
                else:
                    rewards.append(-1)
                    if true_replay:
                        dones.append(done)
                    else:
                        dones.append(0)
                obses_tp1.append(np.array(np.concatenate([obs_tp1['observation'], goal], axis=-1), copy=False))

                # print(obses_t[-1], actions[-1], rewards[-1], obses_tp1[-1], dones[-1])

            push_trans(obs_t['desired_goal'], true_replay=True)

            if rep[it] == 0:
                continue

            if ep_range <= i:
                ep_range += self._maxsize

            offsets = np.random.choice(ep_range - i, rep[it])

            for j in offsets:
                _, _, _, new_obs, _, _ = self._storage[(i+j) % self._maxsize]
                add_goal = new_obs['achieved_goal']
                push_trans(add_goal)

            # if done == 1:
            #     sys.exit(0)

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _encode_mtr_sample(self, idxes):
        obses_beg, obses_step, obses_fin, dist = [], [], [], []

        for i in idxes:
            obs_1, _, _, obs_s, _, ep_range = self._storage[i]

            if ep_range <= i:
                ep_range += self._maxsize

            d = np.random.randint(0, ep_range-i)
            _, _, _, obs_2, _, _ = self._storage[(i+d) % self._maxsize]

            obses_beg.append(obs_1['observation'])
            obses_step.append(obs_s['observation'])
            obses_fin.append(obs_2['observation'])

            # if np.array_equal(obs_1['observation'], obs_2['observation']):
            #     dist.append(0.)
            # else:
            dist.append(d+1.)

        return np.array(obses_beg), np.array(obses_step), np.array(obses_fin), np.array(dist)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        if len(self._storage) == 0:
            print("Sampling from empty buffer")
            return np.array([])

        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(int(batch_size//(self._hindsight+1)))]
        return self._encode_sample(idxes, batch_size)

    def mtr_sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences for mtr.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_mtr_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta=0):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
