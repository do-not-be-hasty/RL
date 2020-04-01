"""Environments."""

import collections
import copy

import gin
import gym
import numpy as np
from gym import wrappers
from gym.envs import classic_control
from gym_sokoban.envs import sokoban_env_fast

from alpacka.utils import attribute

try:
    import gfootball.env as football_env
except ImportError:
    football_env = None


class ModelEnv(gym.Env):
    """Environment interface used by model-based agents.

    This class defines an additional interface over gym.Env that is assumed by
    model-based agents. It's just for documentation purposes, doesn't have to be
    subclassed by envs used as models (but it can be).
    """

    def clone_state(self):
        """Returns the current environment state."""
        raise NotImplementedError

    def restore_state(self, state):
        """Restores environment state, returns the observation."""
        raise NotImplementedError


class ModelWrapper(gym.Wrapper):
    """Base class for wrappers intended for use with model-based environments.

    This class defines an additional interface over gym.Wrapper that is assumed
    by model-based agents. It's just for documentation purposes, doesn't have to
    be subclassed by wrappers used with models (but it can be).
    """

    def clone_state(self):
        """Returns the current environment state."""
        return self.env.clone_state()

    def restore_state(self, state):
        """Restores environment state, returns the observation."""
        return self.env.restore_state(state)


@gin.configurable
class CartPole(classic_control.CartPoleEnv, ModelEnv):
    """CartPole with state clone/restore and returning a "solved" flag."""

    def __init__(self, solved_at=500, reward_scale=1., **kwargs):
        super().__init__(**kwargs)

        self.solved_at = solved_at
        self.reward_scale = reward_scale

        self._step = None

    def reset(self):
        self._step = 0
        return super().reset()

    def step(self, action):
        (observation, reward, done, info) = super().step(action)
        if done:
            info['solved'] = self._step >= self.solved_at
        self._step += 1
        return (observation, reward * self.reward_scale, done, info)

    def clone_state(self):
        return (tuple(self.state), self.steps_beyond_done, self._step)

    def restore_state(self, state):
        (state, self.steps_beyond_done, self._step) = state
        self.state = np.array(state)
        return self.state


@gin.configurable
class Sokoban(sokoban_env_fast.SokobanEnvFast, ModelEnv):
    """Sokoban with state clone/restore and returning a "solved" flag.

    Returns observations in one-hot encoding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Return observations as float32, so we don't have to cast them in the
        # network training pipeline.
        self.observation_space.dtype = np.float32

    def reset(self):
        return super().reset().astype(np.float32)

    def step(self, action):
        (observation, reward, done, info) = super().step(action)
        return (observation.astype(np.float32), reward, done, info)

    def clone_state(self):
        return self.clone_full_state()

    def restore_state(self, state):
        self.restore_full_state(state)
        return self.render(mode=self.mode)


@gin.configurable
class GoogleFootball(ModelEnv):
    """Google Research Football with state clone/restore and
    returning a 'solved' flag."""
    state_size = 480000

    def __init__(self,
                 env_name='academy_empty_goal_close',
                 representation='simple115',
                 rewards='scoring,checkpoints',
                 stacked=False,
                 dump_path=None,
                 solved_at=1,
                 **kwargs):
        if football_env is None:
            raise ImportError('Could not import gfootball! '
                              'HINT: Perform the setup instructions here: '
                              'https://github.com/google-research/football')

        self._solved_at = solved_at
        self._env = football_env.create_environment(
            env_name=env_name,
            representation=representation,
            rewards=rewards,
            stacked=stacked,
            write_full_episode_dumps=dump_path is not None,
            write_goal_dumps=False,
            logdir=dump_path or '',
            **kwargs
        )

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self):
        # pylint: disable=protected-access
        obs = self._env.reset()
        env = self._env.unwrapped
        env._env._trace._trace = collections.deque([], 4)

        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        if done:
            info['solved'] = info['score_reward'] >= self._solved_at
        return obs, reward, done, info

    def render(self, mode='human'):
        return self._env.render(mode)

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        raise NotImplementedError

    def clone_state(self):
        # pylint: disable=protected-access
        raw_state = self._env.get_state()
        size_encoded = len(raw_state).to_bytes(3, byteorder='big')
        # Byte suffix to enforce self.state_size of state.
        suffix = bytes(self.state_size - len(size_encoded) - len(raw_state))
        resized_state = size_encoded + raw_state + suffix
        state = np.frombuffer(resized_state, dtype=np.uint8)

        env = self._env.unwrapped

        # Temporary fix for a reward bug in Google Football: put everything in
        # state. Long-term fix on the way:
        # https://github.com/google-research/football/pull/115
        trace = env._env._trace
        trace_copy = attribute.deep_copy_without_fields(trace, ['_config',
                                                      '_dump_config'])
        # Placeholder to prevent exceptions when gc this object
        trace_copy._dump_config = []
        return (
            state,
            copy.deepcopy(env._env._steps_time),
            copy.deepcopy(env._env._step),
            copy.deepcopy(env._env._cumulative_reward),
            copy.deepcopy(env._env._observation),
            trace_copy,
        )

    def restore_state(self, state):
        # pylint: disable=protected-access
        env = self._env.unwrapped
        (
            state,
            env._env._steps_time,
            env._env._step,
            env._env._cumulative_reward,
            env._env._observation,
            trace,
        ) = state

        env = self._env.unwrapped
        trace_old = env._env._trace
        trace_copy = attribute.deep_copy_merge(trace, trace_old,
                                               ['_config', '_dump_config'])
        env._env._trace = trace_copy

        assert state.size == self.state_size, (
            f'State size does not match: {state.size} != {self.state_size}')

        # First 3 bytes encodes size of state.
        size_decoded = int.from_bytes(list(state[:3]), byteorder='big')
        raw_state = state[3:(size_decoded + 3)]
        assert (state[(size_decoded + 3):] == 0).all()

        self._env.set_state(bytes(raw_state))
        return self._observation

    @property
    def _observation(self):
        # TODO(kc): Hacky, clean it when implementation of football allow it
        # pylint: disable=protected-access
        observation = self._env.unwrapped._env.observation()
        observation = self._env.unwrapped._convert_observations(
            observation, self._env.unwrapped._agent,
            self._env.unwrapped._agent_left_position,
            self._env.unwrapped._agent_right_position
        )
        # pylint: enable=protected-access

        # Lets apply observation transformations from wrappers.
        # WARNING: This assumes that only ObservationWrapper(s) in the wrappers
        # stack transform observation.
        env = self._env
        observation_wrappers = []
        while True:
            if isinstance(env, gym.ObservationWrapper):
                observation_wrappers.append(env)
            if isinstance(env, football_env.wrappers.FrameStack):
                # TODO(pj): Black magic! We know that FrameStack keeps already
                # processed observations and we can return it here. Loose this
                # assumption.
                return env._get_observation()  # pylint: disable=protected-access
            if isinstance(env, gym.Wrapper):
                env = env.env
            else:
                break

        for wrapper in reversed(observation_wrappers):
            observation = wrapper.observation(observation)

        return observation


TimeLimitWrapperState = collections.namedtuple(
    'TimeLimitWrapperState',
    ['super_state', 'elapsed_steps']
)


class TimeLimitWrapper(wrappers.TimeLimit, ModelWrapper):
    """Model-based TimeLimit gym.Env wrapper."""

    def clone_state(self):
        """Returns the current environment state."""
        assert self._elapsed_steps is not None, (
            'Environment must be reset before the first clone_state().'
        )

        return TimeLimitWrapperState(
            super().clone_state(), self._elapsed_steps)

    def restore_state(self, state):
        """Restores environment state, returns the observation."""
        try:
            self._elapsed_steps = state.elapsed_steps
            state = state.super_state
        except AttributeError:
            self._elapsed_steps = 0

        return super().restore_state(state)
