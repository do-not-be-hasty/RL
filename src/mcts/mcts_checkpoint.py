import gin
import numpy as np

from alpacka.networks import core

from DQN_HER import DQN_HER as HER
from environment_builders import make_env_GoalRubik


def restore_HER_model(path, env, **kwargs):
    return HER.load(path, env, **kwargs)

@gin.configurable
class MctsCheckpoint(core.TrainableNetwork):
    def __init__(self, network_signature):
        super().__init__(network_signature)
        self.env = make_env_GoalRubik(
            step_limit=100,
            shuffles=100,
        )
        self.network_signature = network_signature

        # NOTE 2k models
        # self.model = restore_HER_model('/home/michal/Projekty/RL/RL/resources/network_2k_2020-05-21-09:02:01_40000.pkl', self.env)
        # self.model = restore_HER_model('/home/plgrid/plgmizaw/checkpoints/network_2k_2020-05-21-09:02:01_40000.pkl', self.env)

        # NOTE 1k models
        # self.model = restore_HER_model('/home/michal/Projekty/RL/RL/resources/checkpoints_test_2020-05-30-06:27:16_120000.pkl', self.env)
        self.model = restore_HER_model('/home/plgrid/plgmizaw/checkpoints/checkpoints_test_2020-06-22-07:38:16_80000', self.env)

    def setup_model(self):
        self.model.setup_model()

    @property
    def params(self):
        return (
            self.network_signature,
            self.model.verbose,
            self.model.observation_space,
            self.model.action_space,
            self.model.policy,
            self.model.learning_rate,
            self.model.gamma,
            self.model.param_noise,
            self.model.loop_breaking,
            self.model.prioritized_replay,
            self.model.learning_starts,
            self.model.train_freq,
            self.model.batch_size,
            self.model.prioritized_replay_eps,
            self.model.target_network_update_freq,

            self.model.beta_schedule,

            self.model.sess,  # disable for ray
            self.model.graph,  # disable for ray
            self.model.act,  # disable for ray
            self.model._train_step,  # disable for ray
            self.model.update_target,  # disable for ray
            self.model.step_model,  # disable for ray
            self.model.summary,  # disable for ray
            self.model.proba_step,  # disable for ray
            self.model.params,  # disable for ray

            self.model.env,
        )

    @params.setter
    def params(self, new_params):
        pass

    def make_action(self, is_in_loop, update_eps, part_obs, kwargs):
        with self.sess.as_default():
            # Loop breaking
            if self.model.loop_breaking and is_in_loop:
                update_eps_value = 1.
            else:
                update_eps_value = update_eps

            action = self.model.act(np.array(part_obs)[None], update_eps=update_eps_value, **kwargs)[0]

        return action

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return np.max(self.predict_q_values(observation, state, mask, deterministic), axis=-1)

    def predict_action(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        # vectorized_env = self._is_vectorized_observation(observation, self.observation_space)
        vectorized_env = False

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.model.sess.as_default():
            actions, _, _ = self.model.step_model.step(observation, deterministic=deterministic)

        if not vectorized_env:
            actions = np.array([actions[0]])

        return actions

    def predict_q_values(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        observation = observation.reshape((-1,) + self.model.observation_space.shape)

        with self.model.sess.as_default():
            _, q_values, _ = self.model.step_model.step(observation, deterministic=deterministic)

        return q_values

    def action_probability(self, observation, state=None, mask=None):
        observation = np.array(observation)
        # vectorized_env = self._is_vectorized_observation(observation, self.observation_space)
        vectorized_env = False

        observation = observation.reshape((-1,) + self.model.observation_space.shape)
        actions_proba = self.model.proba_step(observation, state, mask)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions_proba = actions_proba[0]

        return actions_proba

    def learning_step(self, step, replay_buffer, writer, episode_losses):
        # print(sys.stderr, "Not learning in Checkpoint mode.")
        pass

    def train(self, data_stream):
        raise NotImplementedError('Do not use train() in MctsCheckpoint network')
