import copy
import os
import sys
import time
from pathlib import Path
import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from mrunner.helpers.client_helper import logger as neptune_logger


def resources_dir():
    env_var_str = 'RESOURCES_DIR'
    assert env_var_str in os.environ
    return Path(os.environ[env_var_str])


def get_cur_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


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


def clear_eval(model, env, neval=100):
    def single_eval():
        obs = env.reset()
        while True:
            action, _states = model.predict(np.concatenate((obs['observation'], obs['desired_goal']), axis=-1))
            obs, rewards, dones, info = env.step(action)

            if rewards >= -1e-5:
                return 1

            if dones:
                return 0

    vals = [single_eval() for i in range(neval)]
    return np.mean(vals)


def log_rubik_curriculum_eval(shuffles_list, model, env, neval=10):
    for shuffle in shuffles_list:
        env.scrambleSize = shuffle
        env.step_limit = 2 * (shuffle + 2)
        neptune_logger('shuffles {0} success rate'.format(shuffle),
                       clear_eval(model, env, neval))


def callback(_locals, _globals):
    interval = 100 if _locals['log_interval'] is None else _locals['log_interval']

    if len(_locals['episode_rewards']) % interval == 0:
        neptune_logger('success rate', np.mean(_locals['episode_success']))
        neptune_logger('no exploration success rate',
                       clear_eval(_locals['self'], copy.deepcopy(_locals['self'].env), neval=25))
        neptune_logger('exploration', _locals['update_eps'])
        ep_div = np.array(_locals['episode_div'])
        ep_succ = np.array(_locals['episode_success'])
        neptune_logger('success move diversity',
                       np.sum(ep_div * ep_succ) / np.sum(ep_succ) if np.sum(ep_succ) != 0 else 0)
        neptune_logger('failure move diversity',
                       np.sum(ep_div * (1 - ep_succ)) / np.sum(1 - ep_succ) if np.sum(1 - ep_succ) != 0 else 0)
        neptune_logger('loss', np.mean(_locals['episode_losses']))
        # neptune_logger('shuffles', _locals['self'].env.scrambleSize)
        # neptune_logger('sampling beta', _locals['self'].replay_buffer._beta)
        # neptune_logger('sampling cut', _locals['self'].replay_buffer._sampling_cut)

        log_rubik_curriculum_eval([2, 4, 7, 10, 13, 16, 19, 24, 50], _locals['self'],
                                  copy.deepcopy(_locals['self'].env))

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
