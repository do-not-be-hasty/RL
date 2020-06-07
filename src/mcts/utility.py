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


def clear_eval(model, env, neval=100, loop_break=False):
    def single_eval():
        visited = set()
        obs = env.reset()

        while True:
            if loop_break and tuple(obs['observation'].flatten()) in visited:
                action = env.action_space.sample()
            else:
                action = model.predict_action(np.concatenate([obs['observation'], obs['desired_goal']], axis=-1))[0]
                visited.add(tuple(obs['observation'].flatten()))

            obs, rewards, dones, info = env.step(action)

            if rewards >= -1e-5:
                return 1

            if dones:
                return 0

    vals = [single_eval() for i in range(neval)]
    return np.mean(vals)


def rubik_ultimate_eval(model, env, neval=100):
    env = copy.deepcopy(env)
    scrambles = env.scrambleSize
    env.scrambleSize = 0

    def single_eval():
        env.reset()
        env.randomize(100)
        env.goal_obs = env._get_state()
        env.randomize(scrambles)
        obs = env._get_goal_observation(env._get_state())

        while True:
            action = model.predict_action(np.concatenate([obs['observation'], obs['desired_goal']], axis=-1))[0]
            obs, rewards, dones, info = env.step(action)

            if rewards >= -1e-5:
                return 1

            if dones:
                return 0

    vals = [single_eval() for i in range(neval)]
    return np.mean(vals)


def log_rubik_curriculum_eval(shuffles_list, model, env, info, neval=10, loop_break=False):
    env = copy.deepcopy(env)

    for shuffle in shuffles_list:
        env.scrambleSize = shuffle
        env.step_limit = 2 * (shuffle + 2)
        info['{0}shuffles {1} success rate'.format('[loop break] ' if loop_break else '', shuffle)] = clear_eval(model, env, neval, loop_break)


def log_rubik_ultimate_eval(shuffles_list, model, env, info, neval=10):
    env = copy.deepcopy(env)

    for shuffle in shuffles_list:
        env.scrambleSize = shuffle
        env.step_limit = 2 * (shuffle + 2)
        info['ultimate {0} success rate'.format(shuffle)] = rubik_ultimate_eval(model, env, neval)


def log_rubik_infty(model):
    env = copy.deepcopy(model.env)

    def single_infty(model, env):
        env.randomize(100)
        obs = env._get_state()
        q_values = model.predict_q_values(
            np.concatenate((obs, env.goal_obs), axis=-1)).flatten()
        return np.mean(q_values)

    return np.mean([single_infty(model, env) for _ in range(100)])


def log_rubik_ultimate_infty(model):
    env = copy.deepcopy(model.env)

    def single_infty(model, env):
        env.randomize(100)
        goal = env._get_state()
        env.randomize(100)
        obs = env._get_state()
        q_values = model.predict_q_values(
            np.concatenate((obs, goal), axis=-1)).flatten()
        return np.mean(q_values)

    return np.mean([single_infty(model, env) for _ in range(100)])


def evaluation_infos(network, step):
    if not step % 100 == 0:
        return {}

    info = dict()

    log_rubik_curriculum_eval([2, 4, 7, 10, 13, 16, 19, 24], network, network.env, info)
    # log_rubik_curriculum_eval([7], network, network.env, loop_break=True, info=info)
    log_rubik_ultimate_eval([2, 4, 7], network, network.env, info)
    info['infty'] = log_rubik_infty(network)
    info['ultimate infty'] = log_rubik_ultimate_infty(network)

    return info


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
