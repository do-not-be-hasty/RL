import copy
import os
import sys
import time
from pathlib import Path
import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from mrunner.helpers.client_helper import logger as raw_neptune_logger


def resources_dir():
    env_var_str = 'RESOURCES_DIR'
    assert env_var_str in os.environ
    return Path(os.environ[env_var_str])


def get_cur_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def neptune_logger(message, value, use_stdout=True):
    raw_neptune_logger(message, value)
    if use_stdout:
        print(message.replace(' ', ''), value)


def timeit(method):
    def timed(*args, **kwargs):
        begin_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()

        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = int((end_time - begin_time))
        else:
            print('%r  %2.2f ms' % (method.__name__, (end_time - begin_time)), file=sys.stderr)
        return result

    return timed


def ordering(preds, data_y):
    res = 0.
    cnt = 0.

    for i in range(preds.shape[0]):
        res += np.sum((preds < preds[i]) * (data_y < data_y[i]) + (preds > preds[i]) * (data_y > data_y[i]))
        cnt += np.sum(data_y != data_y[i])

    return res / cnt


def clear_eval(model, env, neval=100, loop_break=False, with_diversity=False):
    def single_eval():
        visited = set()
        steps = 0
        obs = env.reset()

        while True:
            if loop_break and tuple(obs['observation'].flatten()) in visited:
                action = env.action_space.sample()
            else:
                action, _states = model.predict(np.concatenate((obs['observation'], obs['desired_goal']), axis=-1))
                visited.add(tuple(obs['observation'].flatten()))
            steps += 1

            obs, rewards, dones, info = env.step(action)

            if rewards >= -1e-5:
                return 1, steps

            if dones:
                return 0, steps

    runs = [single_eval() for i in range(neval)]
    scores, raw_divers = zip(*runs)
    divers = 0 if np.sum(scores) == 0 else np.dot(raw_divers, scores) / np.sum(scores)

    if with_diversity:
        return np.mean(scores), divers
    else:
        return np.mean(scores)


def hard_eval(model, env):
    beg_time = time.time()
    visited = set()
    steps = 0
    obs = env.reset()

    while True:
        end = False
        if tuple(obs['observation'].flatten()) in visited:
            for i in range(10):
                action = env.action_space.sample()
                obs, rewards, dones, info = env.step(action)
                steps += 1
                if steps % 100 == 0:
                    visited = set()
                if dones or rewards >= -1e-5:
                    end = True
                    break

        if end:
            break

        action, _states = model.predict(np.concatenate((obs['observation'], obs['desired_goal']), axis=-1))
        visited.add(tuple(obs['observation'].flatten()))
        steps += 1
        if steps % 100 == 0:
            visited = set()

        obs, rewards, dones, info = env.step(action)

        if dones or rewards >= -1e-5:
            break

    if rewards >= -1e-5:
        neptune_logger('time', time.time() - beg_time)
        neptune_logger('steps', steps)
        return 1

    if dones:
        print('No solution found')
        neptune_logger('time', time.time() - beg_time)
        return 0


def rubik_ultimate_eval(model, env, neval=100):
    env = copy.deepcopy(env)
    env.scrambling = True
    scrambles = env.scrambleSize
    env.scrambleSize = 0

    def single_eval():
        env.reset()
        env.randomize(100)
        env.set_goal(env._get_state())
        env.randomize(scrambles)
        obs = env._get_goal_observation(env._get_state())

        while True:
            action, _states = model.predict(np.concatenate((obs['observation'], obs['desired_goal']), axis=-1))
            obs, rewards, dones, info = env.step(action)

            if rewards >= -1e-5:
                return 1

            if dones:
                return 0

    vals = [single_eval() for i in range(neval)]
    return np.mean(vals)


def log_rubik_curriculum_eval(shuffles_list, model, env, neval=10, loop_break=False, with_diversity=False):
    env = copy.deepcopy(env)
    env.scrambling = True

    for shuffle in shuffles_list:
        env.scrambleSize = shuffle
        env.step_limit = 2 * (shuffle + 2)
        success, diversity = clear_eval(model, env, neval, loop_break, with_diversity=True)

        neptune_logger('shuffles {0} success rate'.format(shuffle), success)
        if with_diversity:
            neptune_logger('shuffles {0} move diversity'.format(shuffle), diversity)


def log_rubik_ultimate_eval(shuffles_list, model, env, neval=10):
    env = copy.deepcopy(env)
    env.scrambling = True

    for shuffle in shuffles_list:
        env.scrambleSize = shuffle
        env.step_limit = 2 * (shuffle + 2)
        neptune_logger('ultimate {0} success rate'.format(shuffle), rubik_ultimate_eval(model, env, neval))


def log_rubik_infty(model, distances):
    env = copy.deepcopy(model.env)

    def single_infty(model, env, distance):
        env.scrambleSize = distance
        env.reset()
        obs = env._get_state()
        q_values = model.predict_q_values(
            np.concatenate((obs, env.goal_obs), axis=-1)).flatten()
        return np.max(q_values)

    for distance in distances:
        distance_infty = np.mean([single_infty(model, env, distance) for _ in range(100)])
        neptune_logger('infty on {0}'.format(distance), distance_infty)


def log_bitflipper_infty(model, distances):
    env = copy.deepcopy(model.env)

    def single_infty(model, env, distance):
        env.reset()
        env.goal = env.state
        for i in range(distance):
            env.step(env.action_space.sample())
        obs = env.state
        q_values = model.predict_q_values(
            np.concatenate((obs, env.goal), axis=-1)).flatten()
        return np.max(q_values)

    for distance in distances:
        distance_infty = np.mean([single_infty(model, env, distance) for _ in range(100)])
        neptune_logger('infty on {0}'.format(distance), distance_infty)


def log_rubik_ultimate_infty(model, distances):
    env = copy.deepcopy(model.env)

    def single_infty(model, env, distance):
        env.randomize(100)
        goal = env._get_state()
        env.randomize(distance)
        obs = env._get_state()
        q_values = model.predict_q_values(
            np.concatenate((obs, goal), axis=-1)).flatten()
        return np.mean(q_values)

    for distance in distances:
        distance_infty = np.mean([single_infty(model, env, distance) for _ in range(100)])
        neptune_logger('ultimate infty on {0}'.format(distance), distance_infty)


def log_distance_weights(idxes, weights, weight_sum):
    for i in idxes:
        neptune_logger('weight {0}'.format(i), weights[i] / (weight_sum + 1e-3))


def callback(_locals, _globals):
    interval = 100 if _locals['log_interval'] is None else _locals['log_interval']

    if len(_locals['episode_rewards']) % interval == 0:
        neptune_logger('success rate', np.mean(_locals['episode_success']))
        # neptune_logger('no exploration success rate',
        #                clear_eval(_locals['self'], copy.deepcopy(_locals['self'].env), neval=25))
        neptune_logger('exploration', _locals['update_eps'])
        ep_div = np.array(_locals['episode_div'])
        ep_succ = np.array(_locals['episode_success'])
        neptune_logger('success move diversity',
                       np.sum(ep_div * ep_succ) / np.sum(ep_succ) if np.sum(ep_succ) != 0 else 0)
        neptune_logger('failure move diversity',
                       np.sum(ep_div * (1 - ep_succ)) / np.sum(1 - ep_succ) if np.sum(1 - ep_succ) != 0 else 0)
        neptune_logger('loss', np.mean(_locals['episode_losses']))
        # neptune_logger('loss_min', np.min(_locals['episode_losses']))
        # neptune_logger('loss_max', np.max(_locals['episode_losses']))

        # neptune_logger('shuffles', _locals['self'].env.scrambleSize)
        # neptune_logger('sampling beta', _locals['self'].replay_buffer._beta)
        # neptune_logger('sampling cut', _locals['self'].replay_buffer._sampling_cut)

        log_rubik_curriculum_eval([5, 7, 10, 13, 16, 20, 25, 30], _locals['self'], _locals['self'].env, neval=30, with_diversity=True, loop_break=True)
        # log_rubik_curriculum_eval([13, 16], _locals['self'], _locals['self'].env, neval=30)
        # log_rubik_curriculum_eval([7], _locals['self'], _locals['self'].env, loop_break=True)
        log_rubik_ultimate_eval([7, 10, 13, 16], _locals['self'], _locals['self'].env, neval=20)

        log_rubik_infty(_locals['self'], [1, 2, 3, 5, 7, 10, 13, 18, 50])
        # log_rubik_infty(_locals['self'], [1, 5, 7, 10, 50])
        log_rubik_ultimate_infty(_locals['self'], [1, 2, 3, 5, 7, 10, 13, 18, 50])

        # neptune_logger('weight sum', sum(_locals['loss_accumulator']))
        # log_distance_weights([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20], _locals['loss_accumulator'], sum(_locals['loss_accumulator']))

    return False


def bitflipper_callback(_locals, _globals):
    interval = 100 if _locals['log_interval'] is None else _locals['log_interval']

    if len(_locals['episode_rewards']) % interval == 0:
        neptune_logger('success rate', np.mean(_locals['episode_success']))
        neptune_logger('no exploration success rate',
                       clear_eval(_locals['self'], copy.deepcopy(_locals['self'].env), neval=30))
        neptune_logger('exploration', _locals['update_eps'])
        ep_div = np.array(_locals['episode_div'])
        ep_succ = np.array(_locals['episode_success'])
        neptune_logger('success move diversity',
                       np.sum(ep_div * ep_succ) / np.sum(ep_succ) if np.sum(ep_succ) != 0 else 0)
        neptune_logger('failure move diversity',
                       np.sum(ep_div * (1 - ep_succ)) / np.sum(1 - ep_succ) if np.sum(1 - ep_succ) != 0 else 0)
        neptune_logger('loss', np.mean(_locals['episode_losses']))
        log_bitflipper_infty(_locals['self'], [1, 2, 3, 5, 7, 10, 13, 18, 50])
        # neptune_logger('loss_min', np.min(_locals['episode_losses']))
        # neptune_logger('loss_max', np.max(_locals['episode_losses']))

        # neptune_logger('shuffles', _locals['self'].env.scrambleSize)
        # neptune_logger('sampling beta', _locals['self'].replay_buffer._beta)
        # neptune_logger('sampling cut', _locals['self'].replay_buffer._sampling_cut)

        # log_rubik_curriculum_eval([5, 7, 10], _locals['self'], _locals['self'].env, neval=30, with_diversity=True)
        # log_rubik_curriculum_eval([3, 8, 9, 11, 13, 16], _locals['self'], _locals['self'].env, neval=30)
        # log_rubik_curriculum_eval([7], _locals['self'], _locals['self'].env, loop_break=True)
        # log_rubik_ultimate_eval([7], _locals['self'], _locals['self'].env, neval=30)

        # log_rubik_infty(_locals['self'], [1, 2, 3, 5, 7, 10, 13, 18, 50])
        # log_rubik_ultimate_infty(_locals['self'], [7, 50])

        # neptune_logger('weight sum', sum(_locals['loss_accumulator']))
        # log_distance_weights([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20], _locals['loss_accumulator'], sum(_locals['loss_accumulator']))

    return False


def evaluate(model, env, steps=1000, verbose=True):
    print('--- Evaluation\n')

    obs = env.reset()
    if verbose:
        print(obs)
    succ = 0
    n_ep = 0

    for i in range(steps):
        action, _states = model.predict(np.concatenate((obs['observation'], obs['desired_goal'])))
        # action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if verbose:
            env.render()

        if rewards >= 0:
            succ += 1

        if dones:
            n_ep += 1
            obs = env.reset()
            if verbose:
                # env.print_rewards_info()
                print()

    print('Success rate: ', succ / n_ep)


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def MazeEnv_printable_goal_obs(obs):
    n = obs.shape[0] // 4
    return '[({0},{1})->({2},{3})]'.format(np.where(obs[:n] == 1)[0][0],
                                           np.where(obs[n:2 * n] == 1)[0][0],
                                           np.where(obs[2 * n:3 * n] == 1)[0][0],
                                           np.where(obs[3 * n:4 * n] == 1)[0][0])


def MazeEnv_printable_obs(obs):
    n = obs.shape[0] // 2
    return '({0},{1})'.format(np.where(obs[:n] == 1)[0][0],
                              np.where(obs[n:2 * n] == 1)[0][0])
