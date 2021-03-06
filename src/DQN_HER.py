import pickle
import random
import sys
from functools import partial

import tensorflow as tf
import numpy as np
import gym
from scipy.special import huber

from stable_baselines import logger, deepq
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from episode_replay_buffer import ReplayBuffer as EpisodeReplayBuffer
from stable_baselines.deepq import ReplayBuffer as SimpleReplayBuffer, \
    PrioritizedReplayBuffer as SimplePrioritizedReplayBuffer
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.a2c.utils import find_trainable_variables, total_episode_reward_logger

from utility import get_cur_time_str, timeit, MazeEnv_printable_obs


class DQN_HER(OffPolicyRLModel):
    """
    The DQN model class. DQN paper: https://arxiv.org/pdf/1312.5602.pdf
    :param policy: (DQNPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) size of the replay buffer
    :param exploration_fraction: (float) fraction of entire training period over which the exploration rate is
            annealed
    :param exploration_final_eps: (float) final value of random action probability
    :param train_freq: (int) update the model every `train_freq` steps. set to None to disable printing
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param checkpoint_freq: (int) how often to save the model. This is so that the best version is restored at the
            end of the training. If you do not wish to restore the best version
            at the end of the training set this variable to None.
    :param checkpoint_path: (str) replacement path used if you need to log to somewhere else than a temporary
            directory.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param prioritized_replay: (bool) if True prioritized replay buffer will be used.
    :param prioritized_replay_alpha: (float) alpha parameter for prioritized replay buffer
    :param prioritized_replay_beta0: (float) initial value of beta for prioritized replay buffer
    :param prioritized_replay_beta_iters: (int) number of iterations over which beta will be annealed from initial
            value to 1.0. If set to None equals to max_timesteps.
    :param prioritized_replay_eps: (float) epsilon to add to the TD errors when updating priorities.
    :param param_noise: (bool) Whether or not to apply noise to the parameters of the policy.
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy, env, hindsight, gamma=0.98, learning_rate=5e-4, buffer_size=2000000,
                 exploration_fraction=0.01,
                 exploration_final_eps=0.05, train_freq=1, batch_size=32, checkpoint_freq=10000, checkpoint_path=None,
                 learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                 beta_fraction=1.0,
                 prioritized_replay_eps=1e-6, param_noise=False, verbose=1, tensorboard_log=None,
                 _init_setup_model=True, model_save_path="saved_model", model_save_episode_freq=-1, loop_breaking=True,
                 multistep=1, boltzmann=False):

        # TODO: replay_buffer refactoring
        super(DQN_HER, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                      policy_base=DQNPolicy,
                                      requires_vec_env=False)

        self.checkpoint_path = checkpoint_path
        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.checkpoint_freq = checkpoint_freq
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.beta_fraction = beta_fraction
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.hindsight = hindsight
        self.tensorboard_log = tensorboard_log
        self.model_save_path = model_save_path
        self.model_save_freq = model_save_episode_freq
        self.loop_breaking = loop_breaking
        self.multistep = multistep
        self.boltzmann = boltzmann

        self.graph = None
        self.sess = None
        self._train_step = None
        self.step_model = None
        self.update_target = None
        self.act = None
        self.proba_step = None
        self.replay_buffer = None
        self.solved_replay_buffer = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None
        self.summary = None
        self.episode_reward = None
        self.steps_made = 0
        self.episodes_completed = 0
        self.solved_episodes = []

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        with SetVerbosity(self.verbose):
            assert not isinstance(self.action_space, gym.spaces.Box), \
                "Error: DQN cannot output a gym.spaces.Box action space."

            # If the policy is wrap in functool.partial (e.g. to disable dueling)
            # unwrap it to check the class type
            if isinstance(self.policy, partial):
                test_policy = self.policy.func
            else:
                test_policy = self.policy
            assert issubclass(test_policy, DQNPolicy), "Error: the input policy for the DQN model must be " \
                                                       "an instance of DQNPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(graph=self.graph)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                # optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate)
                # optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True)

                self.act, self._train_step, self.update_target, self.step_model = deepq.build_train(
                    q_func=self.policy,
                    ob_space=self.observation_space,
                    ac_space=self.action_space,
                    optimizer=optimizer,
                    gamma=self.gamma,
                    grad_norm_clipping=10,
                    param_noise=self.param_noise,
                    sess=self.sess
                )
                self.proba_step = self.step_model.proba_step
                self.params = find_trainable_variables("deepq")

                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess)
                self.update_target(sess=self.sess)

                self.summary = tf.summary.merge_all()

    def save_model_checkpoint(self):
        save_path = self.model_save_path + "_" + get_cur_time_str() + "_" + str(self.episodes_completed)
        print("Saving checkpoint to {0}".format(save_path), file=sys.stderr)
        self.save(save_path)

    def dump_solved_episodes(self):
        save_path = self.model_save_path + "_solvedEpisodes_" + get_cur_time_str() + "_" + str(self.episodes_completed)
        print('SOLVED episodes saved to {0}'.format(save_path))
        with open(save_path, 'wb') as outfile:
            pickle.dump(self.solved_episodes, outfile)

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="DQN"):
        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:
            self._setup_learn(seed)

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = SimplePrioritizedReplayBuffer(self.buffer_size,
                                                                   alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps * self.beta_fraction
                    self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                        initial_p=self.prioritized_replay_beta0,
                                                        final_p=1.0)
            else:
                # self.replay_buffer = ReplayBuffer(self.buffer_size, gamma=self.gamma, hindsight=self.hindsight, multistep=self.multistep)
                self.replay_buffer = EpisodeReplayBuffer(self.buffer_size, hindsight=self.hindsight)
                self.solved_replay_buffer = EpisodeReplayBuffer(self.buffer_size, hindsight=self.hindsight)
                # self.replay_buffer = SimpleReplayBuffer(self.buffer_size)
                self.beta_schedule = None
            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=1.0,
                                              final_p=self.exploration_final_eps)

            episode_rewards = [0.0]
            episode_trans = []
            episode_replays = []
            episode_success = [0] * log_interval
            episode_finals = [0] * log_interval
            episode_losses = []
            is_in_loop = False
            loss_accumulator = [0.] * 50

            episode_places = set()
            episode_div = [0] * log_interval

            full_obs = self.env.reset()
            part_obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=-1)
            begin_obs = [full_obs] * log_interval

            reset = True
            self.episode_reward = np.zeros((1,))

            for step in range(total_timesteps):
                # self.steps_made += 1
                # if step >= 7 * 100 * 150:
                #     raise Exception("trigger")
                # curriculum
                # curriculum_scrambles = 1 + int(self.steps_made ** (0.50)) // 500
                # curriculum_step_limit = min((curriculum_scrambles + 2) * 2, 100)
                # self.replay_buffer.set_sampling_cut(curriculum_step_limit)
                # self.env.scrambleSize = curriculum_scrambles
                # self.env.step_limit = curriculum_step_limit

                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(step)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = \
                        -np.log(1. - self.exploration.value(step) +
                                self.exploration.value(step) / float(self.env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                with self.sess.as_default():
                    # Loop breaking
                    if self.loop_breaking and is_in_loop:
                        # update_eps_value = (update_eps + 1.) / 2.
                        update_eps_value = 1.
                    else:
                        update_eps_value = update_eps
                    if self.boltzmann:
                        values = self.predict_q_values(np.array(part_obs))[0]
                        exp = 1. / update_eps_value
                        action = np.random.choice(np.arange(0, values.shape[0]), p=(exp ** values) / sum(exp ** values))
                    else:
                        action = self.act(np.array(part_obs)[None], update_eps=update_eps_value, **kwargs)[0]
                # action = self.env.action_space.sample()
                env_action = action
                reset = False
                new_obs, rew, done, _ = self.env.step(env_action)

                current_place = None
                is_in_loop = False
                try:
                    current_place = tuple(self.env.room_state.flatten())
                except AttributeError:
                    current_place = tuple(new_obs['observation'].flatten())
                if current_place in episode_places:
                    is_in_loop = True
                episode_places.add(current_place)

                # Store transition in the replay buffer.
                # self.replay_buffer.add(part_obs, action, rew, np.concatenate((new_obs['observation'], new_obs['desired_goal'])), float(done))
                episode_replays.append((full_obs, action, rew, new_obs, float(done)))
                episode_trans.append((full_obs, action, rew, new_obs))
                full_obs = new_obs
                part_obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=-1)

                if writer is not None:
                    ep_rew = np.array([rew]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                                      step)

                episode_rewards[-1] += rew
                if done:
                    if np.array_equal(full_obs['achieved_goal'], full_obs['desired_goal']):
                        episode_success.append(1.)
                        self.solved_episodes.append(episode_replays)
                    else:
                        episode_success.append(0.)
                    episode_success = episode_success[1:]
                    episode_div.append(len(episode_places))
                    episode_div = episode_div[1:]
                    self.episodes_completed += 1
                    if self.model_save_freq > 0 and self.episodes_completed % self.model_save_freq == 0:
                        self.save_model_checkpoint()
                    if self.episodes_completed % (200 * 100) == 0:
                        self.dump_solved_episodes()

                    if not isinstance(self.env, VecEnv):
                        full_obs = self.env.reset()
                        # print(full_obs)
                        part_obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=-1)

                    def postprocess_replays(raw_replays, buffer, prioritized_replay):
                        if not prioritized_replay:
                            buffer.add(raw_replays)
                            return

                        for _ in range(10):
                            for id, (full_obs, action, rew, new_obs, done) in enumerate(raw_replays):
                                offset = np.random.randint(id, len(raw_replays))
                                target = raw_replays[offset][3]['achieved_goal']
                                obs = np.concatenate([full_obs['observation'], target], axis=-1)
                                step = np.concatenate([new_obs['observation'], target], axis=-1)
                                if np.array_equal(new_obs['achieved_goal'], target):
                                    rew = 0.
                                    done = 1.
                                else:
                                    rew = -1.
                                    done = 0.

                                buffer.add(obs, action, rew, step, done)

                    postprocess_replays(episode_replays, self.replay_buffer, self.prioritized_replay)

                    begin_obs.append(full_obs)
                    begin_obs = begin_obs[1:]

                    if callback is not None:
                        callback(locals(), globals())

                    episode_rewards.append(0.0)
                    episode_trans = []
                    episode_replays = []
                    episode_places = set()
                    episode_losses = []
                    reset = True
                    is_in_loop = False

                if step > self.learning_starts and step % self.train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if self.prioritized_replay:
                        experience = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(step))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                        weights /= np.mean(weights)
                    else:
                        if np.random.randint(0, 100) < 100:  # always
                            obses_t, actions, rewards, obses_tp1, dones, info = self.replay_buffer.sample(self.batch_size)
                        else:
                            obses_t, actions, rewards, obses_tp1, dones, info = self.solved_replay_buffer.sample(
                                self.batch_size)
                        weights, batch_idxes = np.ones_like(rewards), None

                    if writer is not None:
                        # run loss backprop with summary, but once every 100 steps save the metadata
                        # (memory, compute time, ...)
                        if (1 + step) % 100 == 0:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                  dones, weights, sess=self.sess, options=run_options,
                                                                  run_metadata=run_metadata)
                            writer.add_run_metadata(run_metadata, 'step%d' % step)
                        else:
                            summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                  dones, weights, sess=self.sess)
                        writer.add_summary(summary, step)
                    else:
                        _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                        sess=self.sess)

                    if not self.prioritized_replay:
                        for (dist, error) in zip(info, td_errors):
                            if len(loss_accumulator) < dist + 1:
                                loss_accumulator += [0.] * (dist + 1 - len(loss_accumulator))
                            loss_accumulator[dist] = loss_accumulator[dist] * 0.99 + huber(1., error)

                        # if step % 1000 == 0:
                        #     print('accumulator', [int(x) for x in loss_accumulator])
                        #     weights_sum = sum(loss_accumulator)
                        #     print('normalized ', ['%.2f' % (x / weights_sum) for x in loss_accumulator])
                        #     print('distance   ', info)

                    loss = np.mean(np.dot(weights, [huber(1., error) for error in td_errors]))
                    episode_losses.append(loss)

                    if self.prioritized_replay:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                if step > self.learning_starts and step % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if len(episode_rewards[-(log_interval + 1):-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-(log_interval + 1):-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.record_tabular("steps", step)
                    logger.record_tabular("episodes", num_episodes)
                    logger.record_tabular("mean {0} episode reward".format(log_interval), mean_100ep_reward)
                    logger.record_tabular("{0} episode success".format(log_interval), np.mean(episode_success))
                    logger.record_tabular("% time spent exploring", int(100 * self.exploration.value(step)))
                    logger.dump_tabular()

        return self

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, _, _ = self.step_model.step(observation, deterministic=deterministic)

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def predict_q_values(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        observation = observation.reshape((-1,) + self.observation_space.shape)

        with self.sess.as_default():
            _, q_values, _ = self.step_model.step(observation, deterministic=deterministic)

        return q_values

    def action_probability(self, observation, state=None, mask=None):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions_proba = self.proba_step(observation, state, mask)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions_proba = actions_proba[0]

        return actions_proba

    def save(self, save_path):
        # params
        data = {
            "checkpoint_path": self.checkpoint_path,
            "param_noise": self.param_noise,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "prioritized_replay": self.prioritized_replay,
            "prioritized_replay_eps": self.prioritized_replay_eps,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "checkpoint_freq": self.checkpoint_freq,
            "prioritized_replay_alpha": self.prioritized_replay_alpha,
            "prioritized_replay_beta0": self.prioritized_replay_beta0,
            "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "hindsight": self.hindsight,
            "model_save_path": self.model_save_path,
            "model_save_freq": self.model_save_freq,
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        print('loaded data:', data, 'kwargs:', kwargs)

        model = cls(policy=data["policy"], env=env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model
