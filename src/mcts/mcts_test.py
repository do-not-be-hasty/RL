"""Monte Carlo Tree Search for deterministic environments."""

# TODO(koz4k): Clean up more, add comments and tests.
import copy
import time
import traceback
from functools import partial

import gin
import gym
import numpy as np

from alpacka import data, metric_logging
from alpacka.agents import base
from alpacka.utils import space as space_utils
from alpacka.networks import core

import tensorflow as tf
from mrunner.helpers.client_helper import get_configuration
from stable_baselines import logger, deepq
from stable_baselines.common import tf_util
from stable_baselines.a2c.utils import find_trainable_variables
from stable_baselines.deepq import MlpPolicy as DQN_Policy
from scipy.special import huber

import networks
from mcts.utility import neptune_logger
from networks import CustomPolicy


class ValueTraits:
    """Value traits base class.

    Defines constants for abstract value types.
    """

    zero = None
    dead_end = None


@gin.configurable
class ScalarValueTraits(ValueTraits):
    """Scalar value traits.

    Defines constants for the most basic case of scalar values.
    """

    zero = 0.0

    def __init__(self, dead_end_value=-2.0):
        self.dead_end = dead_end_value


class ValueAccumulator:
    """Value accumulator base class.

    Accumulates abstract values for a given node across multiple MCTS passes.
    """

    def __init__(self, value):
        # Creates and initializes with typical add
        self.add(value)

    def add(self, value):
        """Adds an abstract value to the accumulator.

        Args:
            value: Abstract value to add.
        """
        raise NotImplementedError

    def get(self):
        """Returns the accumulated abstract value for backpropagation.

        May be non-deterministic.  # TODO(pm): What does it mean?
        """
        raise NotImplementedError

    def index(self):
        """Returns an index for selecting the best node."""
        raise NotImplementedError

    def target(self):
        """Returns a target for value function training."""
        raise NotImplementedError

    def count(self):
        """Returns the number of accumulated values."""
        raise NotImplementedError


@gin.configurable
class ScalarValueAccumulator(ValueAccumulator):
    """Scalar value accumulator.

    Calculates a mean over accumulated values and returns it as the
    backpropagated value, node index and target for value network training.
    """

    def __init__(self, value):
        self._sum = 0.0
        self._count = 0
        super().__init__(value)

    def add(self, value):
        self._sum += value
        self._count += 1

    def get(self):
        return self._sum / self._count

    def index(self):
        return self.get()

    def target(self):
        return self.get()

    def count(self):
        return self._count


@gin.configurable
class MaxValueAccumulator(ValueAccumulator):
    """Scalar value accumulator.

    Calculates a mean over accumulated values and returns it as the
    backpropagated value, node index and target for value network training.
    """

    def __init__(self, value):
        self._value = 0.0
        self._count = 0
        super().__init__(value)

    def add(self, value):
        self._value = max(self._value, value)
        self._count += 1

    def get(self):
        return self._value

    def index(self):
        return self.get()

    def target(self):
        return self.get()

    def count(self):
        return self._count


class GraphNode:
    """Graph node, corresponding 1-1 to an environment state.

    Accumulates value across multiple passes through the same environment state.
    """

    def __init__(
            self,
            value_acc,
            state=None,
            terminal=False,
            solved=False,
    ):
        self.value_acc = value_acc
        self.rewards = {}
        self.state = state
        self.terminal = terminal
        self.solved = solved
        self.visits = 1


class TreeNode:
    """Node in the search tree, corresponding many-1 to GraphNode.

    Stores children, and so defines the structure of the search tree. Many
    TreeNodes can point to the same GraphNode, because multiple paths from the
    root of the search tree can lead to the same environment state.
    """

    def __init__(self, node):
        self.node = node
        self.children = {}  # {valid_action: Node}

    @property
    def rewards(self):
        return self.node.rewards

    @property
    def value_acc(self):
        return self.node.value_acc

    @property
    def state(self):
        return self.node.state

    @state.setter
    def state(self, state):
        self.node.state = state

    def expanded(self):
        return bool(self.children)

    @property
    def terminal(self):
        return self.node.terminal

    @property
    def solved(self):
        return self.node.solved

    @terminal.setter
    def terminal(self, terminal):
        self.node.terminal = terminal

    @property
    def visits(self):
        return self.node.visits

    def visit(self):
        self.node.visits += 1


@gin.configurable
class TestDeterministicMCTSAgent(base.OnlineAgent):
    """Monte Carlo Tree Search for deterministic environments.

    Implements transposition tables (sharing value estimates between multiple
    tree nodes corresponding to the same environment state) and loop avoidance.
    """

    def __init__(
            self,
            gamma=0.99,
            n_passes=10,
            avoid_loops=True,
            value_traits_class=ScalarValueTraits,
            value_accumulator_class=ScalarValueAccumulator,
            exploration_length=1e5,
            exploration_final_eps=0.1,
            metrics_frequency=40,
            cumulate=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        # print("MCTS init")
        self._gamma = gamma
        self._n_passes = n_passes
        self._avoid_loops = avoid_loops
        self._value_traits = value_traits_class()
        self._value_acc_class = value_accumulator_class
        self._state2node = {}
        self._model = None
        self._root = None
        self._step = 0
        self._exploration_length = exploration_length
        self._exploration_final_eps = exploration_final_eps
        self._metrics_frequency = metrics_frequency
        self._metrics_table = dict()
        self._cumulate = cumulate

    def _children_of_state(self, parent_state):
        # print("MCTS ch")
        old_state = self._model.clone_state()

        self._model.restore_state(parent_state)

        def step_and_rewind(action):
            (full_observation, reward, done, info) = self._model.step(action)
            observation = np.concatenate([full_observation['observation'], full_observation['desired_goal']], axis=-1)
            state = self._model.clone_state()
            solved = 'solved' in info and info['solved']
            self._model.restore_state(parent_state)
            return (observation, reward, done, solved, state)

        results = zip(*[
            step_and_rewind(action)
            for action in space_utils.element_iter(
                self._model.action_space
            )
        ])
        self._model.restore_state(old_state)
        return results

    def run_mcts_pass(self):
        # print("MCTS run")
        # search_path = list of tuples (node, action)
        # leaf does not belong to search_path (important for not double counting
        # its value)
        leaf, search_path = self._traverse()
        value = yield from self._expand_leaf(leaf)
        # print("mcts_pass, leaf expanded")
        self._backpropagate(search_path, value)

    def _traverse(self):
        # print("MCTS trav")
        node = self._root
        seen_states = set()
        search_path = []
        # new_node is None iff node has no unseen children, i.e. it is Dead
        # End
        while node is not None and node.expanded():
            seen_states.add(node.state)
            # INFO: if node Dead End, (new_node, action) = (None, None)
            # INFO: _select_child can SAMPLE an action (to break tie)
            states_to_avoid = seen_states if self._avoid_loops else set()
            new_node, action = self._select_child(node, states_to_avoid)  #
            search_path.append((node, action))
            node.visit()
            node = new_node
        # at this point node represents a leaf in the tree (and is None for Dead
        # End). node does not belong to search_path.
        return node, search_path

    def _backpropagate(self, search_path, value):
        # print("MCTS bp")
        # Note that a pair
        # (node, action) can have the following form:
        # (Terminal node, None),
        # (Dead End node, None),
        # (TreeNode, action)
        for node, action in reversed(search_path):
            # returns value if action is None
            value = td_backup(node, action, value, self._gamma)
            node.value_acc.add(value)

    def _initialize_graph_node(self, initial_value, state, done, solved):
        # print("MCTS node")
        value_acc = self._value_acc_class(initial_value)
        new_node = GraphNode(
            value_acc,
            state=state,
            terminal=done,
            solved=solved,
        )
        # store newly initialized node in _state2node
        self._state2node[state] = new_node
        return new_node

    def _expand_leaf(self, leaf):
        # print("MCTS leaf")
        if leaf is None:  # Dead End
            return self._value_traits.dead_end

        if leaf.terminal:  # Terminal state
            return self._value_traits.zero

        # neighbours are ordered in the order of actions:
        # 0, 1, ..., _model.num_actions
        obs, rewards, dones, solved, states = self._children_of_state(
            leaf.state
        )
        # print(full_obs)
        # obs = np.concatenate([full_obs['observation'], full_obs['desired_goal']], axis=-1)
        # print("obss\n\n\n", obs)
        # print("obss\n\n\n")

        # print('expand_leaf obs', np.array(obs).astype(np.float32))

        value_batch = yield np.array(obs).astype(np.float32)

        # print('value batch', value_batch)

        # print("expand_fin")

        for idx, action in enumerate(
                space_utils.element_iter(self._action_space)
        ):
            leaf.rewards[action] = rewards[idx]
            new_node = self._state2node.get(states[idx], None)
            if new_node is None:
                if dones[idx]:
                    child_value = self._value_traits.zero
                else:
                    child_value = value_batch[idx]
                new_node = self._initialize_graph_node(
                    child_value, states[idx], dones[idx], solved=solved[idx]
                )
            leaf.children[action] = TreeNode(new_node)

        return leaf.value_acc.get()

    def _child_index(self, parent, action, selection=False):
        # print("MCTS idx")
        accumulator = parent.children[action].value_acc
        value = accumulator.index()
        # TODO sprawdzić działanie przy różnych konfiguracjach selekcji (0: max value, może być też np. children.visits)
        ucb = 0 if selection else np.sqrt(np.log(parent.visits) / parent.children[action].visits)
        return td_backup(parent, action, value, self._gamma) + ucb
        # return td_backup(parent, action, value, self._gamma)
        # return td_backup(parent, action, value, self._gamma) if selection else 10.  # simple BFS

    def _rate_children(self, node, states_to_avoid, selection=False):
        # print("MCTS rate")
        assert self._avoid_loops or len(states_to_avoid) == 0
        return [
            (self._child_index(node, action, selection), action)
            for action, child in node.children.items()
            if child.state not in states_to_avoid
        ]

    # Select the child with the highest score
    def _select_child(self, node, states_to_avoid, explore=False, selection=False):
        # print("MCTS select")
        values_and_actions = self._rate_children(node, states_to_avoid, selection)
        if not values_and_actions:
            return None, None
        (max_value, _) = max(values_and_actions)
        argmax = [
            action for value, action in values_and_actions if (explore or (value == max_value))
        ]
        # INFO: here can be sampling
        if len(argmax) > 1:  # PM: This works faster
            action = np.random.choice(argmax)
        else:
            action = argmax[0]
        return node.children[action], action

    def reset(self, env, observation):
        # print("MCTS reset")
        yield from super().reset(env, observation)
        self._model = env
        # 'reset' mcts internal variables: _state2node and _model
        self._state2node = {}
        state = self._model.clone_state()
        # print("reset 2")
        (value,) = yield np.array([observation])
        # print("reset 4")
        # Initialize root.
        graph_node = self._initialize_graph_node(
            initial_value=value, state=state, done=False, solved=False
        )
        self._root = TreeNode(graph_node)
        # print("reset 3")

    def act(self, observation, force_explore=False):
        self._step += 1
        # print("MCTS act")
        # perform MCTS passes.
        # each pass = tree traversal + leaf evaluation + backprop
        for _ in range(self._n_passes):
            yield from self.run_mcts_pass()
            if force_explore:
                break
        # print("act, mcts_pass-ed")
        info = {'node': self._root}
        # INFO: below line guarantees that we do not perform one-step loop (may
        # be considered slight hack)
        # states_to_avoid = {self._root.state} if self._avoid_loops else set()
        states_to_avoid = set()
        # INFO: possible sampling for exploration
        # explore = (np.random.random() < max((self._exploration_length - self._step) / 1e5, self._exploration_final_eps))
        explore = (np.random.random() < 0.01)
        if force_explore:
            explore = True
        # print('explore', explore, self._step)
        self._root, action = self._select_child(self._root, states_to_avoid, explore, selection=True)

        return (action, info)

    @staticmethod
    def postprocess_transitions(transitions):
        # print("MCTS postp")
        # print(transitions)
        postprocessed_transitions = transitions
        # for (i, transition) in enumerate(transitions):
        #     node = transition.agent_info['node']
        #     value = node.value_acc.target().item()
        #
        #     offset = np.random.randint(i, len(transitions))
        #     goal = transitions[offset].next_observation['achieved_goal']
        #     # goal = transitions[offset].next_observation['desired_goal']
        #     done = np.array_equal(transition.next_observation['achieved_goal'], goal)
        #     reward = 0 if done else transition.reward
        #     observation = np.concatenate([transition.observation['observation'], goal], axis=-1)
        #     next_observation = np.concatenate([transition.next_observation['observation'], goal], axis=-1)
        #
        #     postprocessed_transitions.append(
        #         transition._replace(observation=observation, reward=reward, done=done, next_observation=next_observation,
        #                             agent_info={'value': value}))
        return postprocessed_transitions

    @staticmethod
    def network_signature(observation_space, action_space):
        # print("MCTS sign")
        return data.NetworkSignature(
            input=space_utils.signature(observation_space),
            output=data.TensorSignature(shape=(action_space.n,)),
        )

    @staticmethod
    def compute_metrics(episodes):
        info = dict()
        count = dict()

        for episode in episodes:
            for (key, value) in episode.info.items():
                if key not in info:
                    info[key] = 0.
                    count[key] = 0
                info[key] += value
                count[key] += 1

        for (key, val) in count.items():
            info[key] /= val

        return info

    def add_metrics(self, info, model_env, epoch):
        yield from self.hard_eval_run(model_env)

        if epoch % self._metrics_frequency == 0:
            model_passes = self._n_passes

            eval_rate = 10

            for steps in [13, 16, 19, 22, 25, 30]:
                for passes in [10, 200, 2000]:
                    self._n_passes = passes
                    env = copy.deepcopy(model_env)
                    env.env.scrambleSize = steps
                    env.env.step_limit = steps + 5

                    solved_acc = 0.
                    for _ in range(eval_rate):
                        episode = yield from self.solve(env, epoch, dummy=True)
                        solved_acc += int(episode.solved)

                    metric_name = 'mcts({0}) shuffles({1})'.format(passes, steps)

                    if self._cumulate:
                        (solved_total, count) = self._metrics_table[
                            metric_name] if metric_name in self._metrics_table.keys() else (0, 0)
                        solved_total += solved_acc
                        count += eval_rate
                        info[metric_name] = solved_total / count
                        self._metrics_table[metric_name] = (solved_total, count)
                    else:
                        info[metric_name] = solved_acc / eval_rate

            self._n_passes = model_passes

    def hard_eval_run(self, env):
        print('Hard eval started')

        env.env.scrambleSize = 100
        env.env.step_limit = 150
        self._n_passes = 8000

        count = 0
        solved = 0
        while True:
            beg_time = time.time()
            episode = yield from self.solve(env, count, dummy=True, hard=True)
            run_time = time.time() - beg_time
            solved += int(episode.solved)
            count += 1

            metric_logging.log_scalar_metrics(
                'hard_eval',
                count,
                {'solved': solved / count, 'solving time': run_time, 'steps': 1 - episode.return_}
            )


def td_backup(node, action, value, gamma):
    if action is None:
        return value
    return node.rewards[action] + gamma * value


@gin.configurable
def DqnRubikPolicy():
    return partial(CustomPolicy, arch_fun=networks.arch_color_embedding)


@gin.configurable
def DqnBitFlipperPolicy():
    return partial(DQN_Policy, layers=[512, 512])


@gin.configurable
class DqnInternalNetwork(core.TrainableNetwork):
    def __init__(self, network_signature, verbose, policy, learning_rate, gamma, param_noise, loop_breaking,
                 prioritized_replay,
                 learning_starts, train_freq, batch_size, prioritized_replay_eps, target_network_update_freq,
                 env_class):
        super().__init__(network_signature)
        self.env = env_class()

        self.verbose = verbose
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.policy = policy()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.param_noise = param_noise
        self.loop_breaking = loop_breaking
        self.prioritized_replay = prioritized_replay
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.prioritized_replay_eps = prioritized_replay_eps
        self.target_network_update_freq = target_network_update_freq

        self.beta_schedule = None

        self.sess = None
        self.graph = None
        self.act = None
        self._train_step = None
        self.update_target = None
        self.step_model = None
        self.summary = None
        self.proba_step = None
        self.net_params = None

        self.setup_model()

    def setup_model(self):
        assert not isinstance(self.action_space, gym.spaces.Box), \
            "Error: DQN cannot output a gym.spaces.Box action space."

        # If the policy is wrap in functool.partial (e.g. to disable dueling)
        # unwrap it to check the class type
        if isinstance(self.policy, partial):
            test_policy = self.policy.func
        else:
            test_policy = self.policy

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
            self.net_params = find_trainable_variables("deepq")

            # Initialize the parameters and copy them to the target network.
            tf_util.initialize(self.sess)
            self.update_target(sess=self.sess)

            self.summary = tf.summary.merge_all()

    @property
    def params(self):
        """Returns network parameters."""

        return (
            self._network_signature,
            self.verbose,
            self.observation_space,
            self.action_space,
            self.policy,
            self.learning_rate,
            self.gamma,
            self.param_noise,
            self.loop_breaking,
            self.prioritized_replay,
            self.learning_starts,
            self.train_freq,
            self.batch_size,
            self.prioritized_replay_eps,
            self.target_network_update_freq,

            self.beta_schedule,

            self.sess,  # disable for ray
            self.graph,  # disable for ray
            self.act,  # disable for ray
            self._train_step,  # disable for ray
            self.update_target,  # disable for ray
            self.step_model,  # disable for ray
            self.summary,  # disable for ray
            self.proba_step,  # disable for ray
            self.net_params,  # disable for ray

            self.env,
        )

    @params.setter
    def params(self, new_params):
        """Sets network parameters."""

        (
            self._network_signature,
            self.verbose,
            self.observation_space,
            self.action_space,
            self.policy,
            self.learning_rate,
            self.gamma,
            self.param_noise,
            self.loop_breaking,
            self.prioritized_replay,
            self.learning_starts,
            self.train_freq,
            self.batch_size,
            self.prioritized_replay_eps,
            self.target_network_update_freq,

            self.beta_schedule,

            self.sess,  # disable for ray
            self.graph,  # disable for ray
            self.act,  # disable for ray
            self._train_step,  # disable for ray
            self.update_target,  # disable for ray
            self.step_model,  # disable for ray
            self.summary,  # disable for ray
            self.proba_step,  # disable for ray
            self.net_params,  # disable for ray

            self.env,
        ) = new_params

    def make_action(self, is_in_loop, update_eps, part_obs, kwargs):
        with self.sess.as_default():
            # Loop breaking
            if self.loop_breaking and is_in_loop:
                # update_eps_value = (update_eps + 1.) / 2.
                update_eps_value = 1.
            else:
                update_eps_value = update_eps
            # if self.boltzmann:
            #     values = self.predict_q_values(np.array(part_obs))[0]
            #     exp = 1. / update_eps_value
            #     action = np.random.choice(np.arange(0, values.shape[0]), p=(exp ** values) / sum(exp ** values))
            # else:
            #     action = self.act(np.array(part_obs)[None], update_eps=update_eps_value, **kwargs)[0]
            action = self.act(np.array(part_obs)[None], update_eps=update_eps_value, **kwargs)[0]

        return action

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return np.max(self.predict_q_values(observation, state, mask, deterministic), axis=-1)

    def predict_action(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        # vectorized_env = self._is_vectorized_observation(observation, self.observation_space)
        vectorized_env = False

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, _, _ = self.step_model.step(observation, deterministic=deterministic)

        if not vectorized_env:
            actions = np.array([actions[0]])

        return actions

    def predict_q_values(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        observation = observation.reshape((-1,) + self.observation_space.shape)

        with self.sess.as_default():
            _, q_values, _ = self.step_model.step(observation, deterministic=deterministic)

        return q_values

    def action_probability(self, observation, state=None, mask=None):
        observation = np.array(observation)
        # vectorized_env = self._is_vectorized_observation(observation, self.observation_space)
        vectorized_env = False

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions_proba = self.proba_step(observation, state, mask)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions_proba = actions_proba[0]

        return actions_proba

    def learning_step(self, step, replay_buffer, writer, episode_losses):
        if step > self.learning_starts and step % self.train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if self.prioritized_replay:
                experience = replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(step))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(self.batch_size)
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

            loss = np.mean(np.dot(weights, [huber(1., error) for error in td_errors]))
            episode_losses.append(loss)

            if self.prioritized_replay:
                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

        if step > self.learning_starts and step % self.target_network_update_freq == 0:
            # Update target network periodically.
            self.update_target(sess=self.sess)

    def train(self, data_stream):
        raise NotImplementedError('Do not use train() in DqnInternalNetwork')
