import gin
from gym_rubik.envs import GoalRubikEnv, RubikEnv
from alpacka.envs import ModelEnv
import copy
import numpy as np
from gym_BitFlipper.envs import GoalBitFlipperEnv as GoalBitFlipper


@gin.configurable
class TestGoalRubikEnv(GoalRubikEnv, ModelEnv):
    """Rubik Cube with state clone/restore and returning a "solved" flag."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        return super().reset()

    def step(self, action):
        (observation, reward, done, info) = super().step(action)
        if done:
            info['solved'] = reward >= 0
        return observation, reward, done, info

    def clone_state(self):
        # print("clone\n\n")
        return (
            copy.deepcopy(self.cube),
            # self.goal_obs,
            # self.action_space,
            self.fig,
            # self.solved_state,
            # self.observation_space,
            # self.scramble,
            self.debugLevel,
            self.renderViews,
            self.renderFlat,
            self.renderCube,
            self.scrambleSize,
            self.num_steps,
            self.step_limit,
            tuple(self.goal_obs.flatten()),
            self.goal_obs.shape,
        )

    def restore_state(self, state):
        (
            cube,
            # self.goal_obs,
            # self.action_space,
            self.fig,
            # self.solved_state,
            # self.observation_space,
            # self.scramble,
            self.debugLevel,
            self.renderViews,
            self.renderFlat,
            self.renderCube,
            self.scrambleSize,
            self.num_steps,
            self.step_limit,
            goal_obs,
            goal_obs_shape,
        ) = state
        self.cube = copy.deepcopy(cube)
        self.goal_obs = np.array(goal_obs).reshape(goal_obs_shape)

        # return self._get_goal_observation(self._get_state())
        return self._get_state()


@gin.configurable
class TestSimpleRubikEnv(RubikEnv, ModelEnv):
    """Rubik Cube with state clone/restore and returning a "solved" flag."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        return super().reset()

    def step(self, action):
        (observation, reward, done, info) = super().step(action)
        if done:
            info['solved'] = reward >= 0
        return observation, reward, done, info

    def clone_state(self):
        # print("clone\n\n")
        return (
            copy.deepcopy(self.cube),
            # self.goal_obs,
            # self.action_space,
            self.fig,
            # self.solved_state,
            # self.observation_space,
            # self.scramble,
            self.debugLevel,
            self.renderViews,
            self.renderFlat,
            self.renderCube,
            self.scrambleSize,
            self.num_steps,
            self.step_limit,
        )

    def restore_state(self, state):
        (
            cube,
            # self.goal_obs,
            # self.action_space,
            self.fig,
            # self.solved_state,
            # self.observation_space,
            # self.scramble,
            self.debugLevel,
            self.renderViews,
            self.renderFlat,
            self.renderCube,
            self.scrambleSize,
            self.num_steps,
            self.step_limit,
        ) = state
        self.cube = copy.deepcopy(cube)

        return self._get_state()


@gin.configurable
class TestGoalBitFlipper(GoalBitFlipper, ModelEnv):
    """Rubik Cube with state clone/restore and returning a "solved" flag."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, seed=None):
        return super().reset()

    def step(self, action):
        (observation, reward, done, info) = super().step(action)
        if done:
            info['solved'] = reward >= 0
        return observation, reward, done, info

    def clone_state(self):
        # print("BitFlipper clone\n\n")
        return (
            self.n,
            self.reward_range,
            self.space_seed,
            tuple(self.initial_state),
            tuple(self.goal),
            tuple(self.state),
            self.envstepcount,
            self.reward_max,
        )

    def restore_state(self, state):
        (
            self.n,
            self.reward_range,
            self.space_seed,
            self.initial_state,
            self.goal,
            self.state,
            self.envstepcount,
            self.reward_max,
        ) = state
        self.initial_state = np.array(self.initial_state)
        self.goal = np.array(self.goal)
        self.state = np.array(self.state)

        return self._convert_observation(self.state, self.state, self.goal)
