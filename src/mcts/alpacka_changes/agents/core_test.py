"""Tests for alpacka.agents.core."""

import collections

import gym
import numpy as np
import pytest

from alpacka import agents
from alpacka import testing
from alpacka import utils

mock_env = testing.mock_env_fixture


@pytest.mark.parametrize('with_critic', [True, False])
@pytest.mark.parametrize('agent_class',
                         [agents.SoftmaxAgent,
                          agents.EpsilonGreedyAgent])
def test_agents_network_signature(agent_class, with_critic):
    # Set up
    obs_space = gym.spaces.Box(low=0, high=255, shape=(7, 7), dtype=np.uint8)
    act_space = gym.spaces.Discrete(n=7)

    # Run
    agent = agent_class(with_critic=with_critic)
    signature = agent.network_signature(obs_space, act_space)

    # Test
    assert signature.input.shape == obs_space.shape
    assert signature.input.dtype == obs_space.dtype
    if with_critic:
        assert signature.output[0].shape == (1, )
        assert signature.output[0].dtype == np.float32
        assert signature.output[1].shape == (act_space.n, )
        assert signature.output[1].dtype == np.float32
    else:
        assert signature.output.shape == (act_space.n, )
        assert signature.output.dtype == np.float32


@pytest.mark.parametrize('with_critic', [True, False])
@pytest.mark.parametrize('logits',
                         [np.array([[3, 2, 1]]),
                          np.array([[1, 3, 2]]),
                          np.array([[2, 1, 3]])])
@pytest.mark.parametrize('agent_class,rtol',
                         [(agents.SoftmaxAgent, 0.25),
                          (agents.EpsilonGreedyAgent, 0.35)])
def test_agents_the_most_common_action_and_agent_info_is_correct(agent_class,
                                                                 logits,
                                                                 rtol,
                                                                 with_critic):
    # Set up
    agent = agent_class(with_critic=with_critic)
    expected = np.argmax(logits)
    value = np.array([[7]])  # Batch of size 1.
    actions = []
    infos = []

    # Run
    for _ in range(6000):
        action, info = testing.run_with_constant_network_prediction(
            agent.act(np.zeros((7, 7))),
            (value, logits) if with_critic else logits
        )
        actions.append(action)
        infos.append(info)
    (_, counts) = np.unique(actions, return_counts=True)

    most_common = np.argmax(counts)
    sample_prob = counts / np.sum(counts)
    sample_logp = np.log(sample_prob)
    sample_entropy = -np.sum(sample_prob * sample_logp)

    # Test
    info = infos[0]
    for other in infos[1:]:
        for info_value, other_value in zip(info.values(), other.values()):
            np.testing.assert_array_equal(info_value, other_value)

    if with_critic:
        assert info['value'] == value
    else:
        assert 'value' not in info

    assert most_common == expected
    np.testing.assert_allclose(sample_prob, info['prob'], rtol=rtol)
    np.testing.assert_allclose(sample_logp, info['logp'], rtol=rtol)
    np.testing.assert_allclose(sample_entropy, info['entropy'], rtol=rtol)


@pytest.mark.parametrize('agent_class,attr_name',
                         [(agents.SoftmaxAgent, 'distribution.temperature'),
                          (agents.EpsilonGreedyAgent, 'distribution.epsilon')])
def test_agents_linear_annealing_exploration_parameter(
        agent_class, attr_name, mock_env):
    # Set up
    param_values = list(range(10, 0, -1))
    max_value = max(param_values)
    min_value = min(param_values)
    n_epochs = len(param_values)

    agent = agent_class(linear_annealing_kwargs={
        'max_value': max_value,
        'min_value': min_value,
        'n_epochs': n_epochs
    })

    # Run & Test
    for epoch, x_value in enumerate(param_values):
        testing.run_with_constant_network_prediction(
            agent.solve(mock_env, epoch=epoch),
            logits=np.array([[3, 2, 1]])
        )
        assert utils.recursive_getattr(agent, attr_name) == x_value


def test_softmax_agent_action_counts_for_different_temperature():
    # Set up
    low_temp_agent = agents.SoftmaxAgent(temperature=.5)
    high_temp_agent = agents.SoftmaxAgent(temperature=2.)
    low_temp_action_count = collections.defaultdict(int)
    high_temp_action_count = collections.defaultdict(int)
    logits = ((2, 1, 1, 1, 2), )  # Batch of size 1.

    # Run
    for agent, action_count in [
        (low_temp_agent, low_temp_action_count),
        (high_temp_agent, high_temp_action_count),
    ]:
        for _ in range(200):
            action, _ = testing.run_with_constant_network_prediction(
                agent.act(np.zeros((7, 7))),
                logits
            )
            action_count[action] += 1

    # Test
    assert low_temp_action_count[0] > high_temp_action_count[0]
    assert low_temp_action_count[1] < high_temp_action_count[1]
    assert low_temp_action_count[2] < high_temp_action_count[2]
    assert low_temp_action_count[3] < high_temp_action_count[3]
    assert low_temp_action_count[4] > high_temp_action_count[4]


def test_egreedy_agent_action_counts_for_different_epsilon():
    # Set up
    low_eps_agent = agents.EpsilonGreedyAgent(epsilon=.05)
    high_eps_agent = agents.EpsilonGreedyAgent(epsilon=.5)
    low_eps_action_count = collections.defaultdict(int)
    high_eps_action_count = collections.defaultdict(int)
    logits = ((5, 4, 3, 2, 1), )  # Batch of size 1.

    # Run
    for agent, action_count in [
        (low_eps_agent, low_eps_action_count),
        (high_eps_agent, high_eps_action_count),
    ]:
        for _ in range(200):
            action, _ = testing.run_with_constant_network_prediction(
                agent.act(np.zeros((7, 7))),
                logits
            )
            action_count[action] += 1

    # Test
    assert low_eps_action_count[0] > high_eps_action_count[0]
    assert low_eps_action_count[1] < high_eps_action_count[1]
    assert low_eps_action_count[2] < high_eps_action_count[2]
    assert low_eps_action_count[3] < high_eps_action_count[3]
    assert low_eps_action_count[4] < high_eps_action_count[4]
