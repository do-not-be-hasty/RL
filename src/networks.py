import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

from stable_baselines.deepq.policies import DQNPolicy

from utility import model_summary


class CustomPolicy(DQNPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, arch_fun, reuse=False,
                 feature_extraction="mlp",
                 obs_phs=None, dueling=True, act_fun=tf.nn.relu, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                           n_batch, dueling=dueling, reuse=reuse,
                                           scale=(feature_extraction == "cnn"), obs_phs=obs_phs)

        self._kwargs_check(feature_extraction, kwargs)

        with tf.variable_scope("model", reuse=reuse):
            q_out = arch_fun(self.processed_obs, act_fun, self.n_actions, self.dueling)

        model_summary()

        self.q_values = q_out
        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


def arch_simpleFf(processed_obs, act_fun, n_actions, dueling):
    with tf.variable_scope("action_value"):
        extracted_features = tf.layers.flatten(processed_obs)
        action_out = extracted_features

        action_out = tf_layers.fully_connected(action_out, num_outputs=1024, activation_fn=None)
        action_out = act_fun(action_out)
        action_out = tf_layers.fully_connected(action_out, num_outputs=1024, activation_fn=None)
        action_out = act_fun(action_out)

        action_scores = tf_layers.fully_connected(action_out, num_outputs=n_actions, activation_fn=None)

    if dueling:
        with tf.variable_scope("state_value"):
            state_out = extracted_features

            state_out = tf_layers.fully_connected(state_out, num_outputs=1024, activation_fn=None)
            state_out = act_fun(state_out)
            state_out = tf_layers.fully_connected(state_out, num_outputs=1024, activation_fn=None)
            state_out = act_fun(state_out)

            state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)

        action_scores_mean = tf.reduce_mean(action_scores, axis=1)
        action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
        q_out = state_score + action_scores_centered
    else:
        q_out = action_scores

    return q_out


def arch_batchnorm(processed_obs, act_fun, n_actions, dueling):
    with tf.variable_scope("action_value"):
        extracted_features = tf.layers.flatten(processed_obs)
        action_out = extracted_features

        action_out = tf_layers.fully_connected(action_out, num_outputs=1024, activation_fn=None)
        action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
        action_out = act_fun(action_out)
        action_out = tf_layers.fully_connected(action_out, num_outputs=1024, activation_fn=None)
        action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
        action_out = act_fun(action_out)

        action_scores = tf_layers.fully_connected(action_out, num_outputs=n_actions, activation_fn=None)

    if dueling:
        with tf.variable_scope("state_value"):
            state_out = extracted_features

            state_out = tf_layers.fully_connected(state_out, num_outputs=1024, activation_fn=None)
            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
            state_out = act_fun(state_out)
            state_out = tf_layers.fully_connected(state_out, num_outputs=1024, activation_fn=None)
            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
            state_out = act_fun(state_out)

            state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)

        action_scores_mean = tf.reduce_mean(action_scores, axis=1)
        action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
        q_out = state_score + action_scores_centered
    else:
        q_out = action_scores

    return q_out


def arch_dropout(processed_obs, act_fun, n_actions, dueling):
    with tf.variable_scope("action_value"):
        extracted_features = tf.layers.flatten(processed_obs)
        action_out = extracted_features

        action_out = tf_layers.fully_connected(action_out, num_outputs=1024, activation_fn=None)
        action_out = tf_layers.dropout(action_out, keep_prob=0.8)
        action_out = act_fun(action_out)
        action_out = tf_layers.fully_connected(action_out, num_outputs=1024, activation_fn=None)
        action_out = tf_layers.dropout(action_out, keep_prob=0.8)
        action_out = act_fun(action_out)

        action_scores = tf_layers.fully_connected(action_out, num_outputs=n_actions, activation_fn=None)

    if dueling:
        with tf.variable_scope("state_value"):
            state_out = extracted_features

            state_out = tf_layers.fully_connected(state_out, num_outputs=1024, activation_fn=None)
            state_out = tf_layers.dropout(state_out, keep_prob=0.8)
            state_out = act_fun(state_out)
            state_out = tf_layers.fully_connected(state_out, num_outputs=1024, activation_fn=None)
            state_out = tf_layers.dropout(state_out, keep_prob=0.8)
            state_out = act_fun(state_out)

            state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)

        action_scores_mean = tf.reduce_mean(action_scores, axis=1)
        action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
        q_out = state_score + action_scores_centered
    else:
        q_out = action_scores

    return q_out


def arch_shared_features(processed_obs, act_fun, n_actions, dueling):
    with tf.variable_scope("action_value"):
        extracted_features = tf.layers.flatten(processed_obs)
        extracted_features = tf_layers.fully_connected(extracted_features, num_outputs=1024, activation_fn=None)
        extracted_features = tf_layers.layer_norm(extracted_features, center=True, scale=True)
        extracted_features = act_fun(extracted_features)
        extracted_features = tf_layers.fully_connected(extracted_features, num_outputs=1024, activation_fn=None)
        extracted_features = tf_layers.layer_norm(extracted_features, center=True, scale=True)
        extracted_features = act_fun(extracted_features)

        action_out = extracted_features
        action_scores = tf_layers.fully_connected(action_out, num_outputs=n_actions, activation_fn=None)

    if dueling:
        with tf.variable_scope("state_value"):
            state_out = extracted_features
            state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)

        action_scores_mean = tf.reduce_mean(action_scores, axis=1)
        action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
        q_out = state_score + action_scores_centered
    else:
        q_out = action_scores

    return q_out


def arch_batchnorm_regularization(processed_obs, act_fun, n_actions, dueling):
    regularizer = tf.keras.regularizers.l2(l=0.01)

    with tf.variable_scope("action_value"):
        extracted_features = tf.layers.flatten(processed_obs)
        action_out = extracted_features

        action_out = tf_layers.fully_connected(action_out, num_outputs=1024, activation_fn=None, weights_regularizer=regularizer)
        action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
        action_out = act_fun(action_out)
        action_out = tf_layers.fully_connected(action_out, num_outputs=1024, activation_fn=None, weights_regularizer=regularizer)
        action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
        action_out = act_fun(action_out)

        action_scores = tf_layers.fully_connected(action_out, num_outputs=n_actions, activation_fn=None, weights_regularizer=regularizer)

    if dueling:
        with tf.variable_scope("state_value"):
            state_out = extracted_features

            state_out = tf_layers.fully_connected(state_out, num_outputs=1024, activation_fn=None, weights_regularizer=regularizer)
            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
            state_out = act_fun(state_out)
            state_out = tf_layers.fully_connected(state_out, num_outputs=1024, activation_fn=None, weights_regularizer=regularizer)
            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
            state_out = act_fun(state_out)

            state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)

        action_scores_mean = tf.reduce_mean(action_scores, axis=1)
        action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
        q_out = state_score + action_scores_centered
    else:
        q_out = action_scores

    return q_out
