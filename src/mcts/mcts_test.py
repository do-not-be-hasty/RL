"""Monte Carlo Tree Search for deterministic environments."""

# TODO(koz4k): Clean up more, add comments and tests.
import traceback

import gin
import numpy as np

from alpacka import data
from alpacka.agents import base
from alpacka.utils import space as space_utils


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

        # print('value batch preparation')
        value_batch = yield [np.array(obs), np.ones([12, 12])]
        # print('value batch pre', value_batch)
        value_batch = np.max(value_batch, axis=-1)

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

    def _child_index(self, parent, action):
        # print("MCTS idx")
        accumulator = parent.children[action].value_acc
        value = accumulator.index()
        return td_backup(parent, action, value, self._gamma)

    def _rate_children(self, node, states_to_avoid):
        # print("MCTS rate")
        assert self._avoid_loops or len(states_to_avoid) == 0
        return [
            (self._child_index(node, action), action)
            for action, child in node.children.items()
            if child.state not in states_to_avoid
        ]

    # Select the child with the highest score
    def _select_child(self, node, states_to_avoid):
        # print("MCTS select")
        values_and_actions = self._rate_children(node, states_to_avoid)
        if not values_and_actions:
            return None, None
        (max_value, _) = max(values_and_actions)
        argmax = [
            action for value, action in values_and_actions if value == max_value
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
        (value,) = yield [np.array([observation]), np.array([np.ones(12)])]
        # print("reset 4")
        # Initialize root.
        graph_node = self._initialize_graph_node(
            initial_value=value, state=state, done=False, solved=False
        )
        self._root = TreeNode(graph_node)
        # print("reset 3")

    def act(self, observation):
        # print("MCTS act")
        # perform MCTS passes.
        # each pass = tree traversal + leaf evaluation + backprop
        for _ in range(self._n_passes):
            yield from self.run_mcts_pass()
        # print("act, mcts_pass-ed")
        info = {'node': self._root}
        # INFO: below line guarantees that we do not perform one-step loop (may
        # be considered slight hack)
        #states_to_avoid = {self._root.state} if self._avoid_loops else set()
        states_to_avoid = set()
        # INFO: possible sampling for exploration
        self._root, action = self._select_child(self._root, states_to_avoid)

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


def td_backup(node, action, value, gamma):
    if action is None:
        return value
    return node.rewards[action] + gamma * value
