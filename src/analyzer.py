import sys
from time import sleep, time
import tensorflow as tf

from DQN_HER import DQN_HER as HER
from environment_builders import make_env_GoalRubik, make_env_GoalBitFlipper
from models import HER_model
import numpy as np
from utility import clear_eval, model_summary, rubik_ultimate_eval

env = make_env_GoalRubik(step_limit=15)
# env = make_env_GoalBitFlipper(n=15, space_seed=None)

# model = HER.load("/home/michal/Projekty/RL/RL/resources/hincur_checkpoint_2019-11-11-04:59:42_30000.pkl", env) # RUB-10

# model = HER.load("/home/michal/Projekty/RL/RL/resources/hincur_checkpoint_2019-11-11-05:30:01_30000.pkl", env) # RUB-11
# model = HER.load("/home/michal/Projekty/RL/RL/resources/hincur_checkpoint_2019-11-11-18:02:54_60000.pkl", env) # RUB-11

# model = HER.load("/home/michal/Projekty/RL/RL/resources/R7_hind1_checkpoint_2019-11-11-08:00:53_20000.pkl", env) # RUB-17
# model = HER.load("/home/michal/Projekty/RL/RL/resources/R7_hind1_checkpoint_2019-11-11-14:24:29_40000.pkl", env) # RUB-17
# model = HER.load("/home/michal/Projekty/RL/RL/resources/R7_hind1_checkpoint_2019-11-11-20:04:16_60000.pkl", env) # RUB-17
# model = HER.load("/home/michal/Projekty/RL/RL/resources/R7_hind1_checkpoint_2019-11-12-00:48:03_80000.pkl", env) # RUB-17

# model = HER.load("/home/michal/Projekty/RL/RL/resources/fakeonly_2019-11-13-12:01:24_40000.pkl", env) # RUB-21

# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_normal_2019-12-02-13:41:05_1000.pkl", env) # 0.002
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_normal_2019-12-02-13:41:38_2000.pkl", env) # 0.002
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_normal_2019-12-02-13:42:10_3000.pkl", env) # 0.002
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_normal_2019-12-02-13:42:42_4000.pkl", env) # 0.006
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_normal_2019-12-02-13:43:14_5000.pkl", env) # 0.087
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_normal_2019-12-02-13:43:44_6000.pkl", env) # 0.571
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_normal_2019-12-02-13:44:11_7000.pkl", env) # 0.944
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_normal_2019-12-02-13:44:34_8000.pkl", env)
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_normal_2019-12-02-13:44:55_9000.pkl", env)
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_normal_2019-12-02-13:45:16_10000.pkl", env)

# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_fakeonly_2019-12-02-15:49:27_1000.pkl", env)
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_fakeonly_2019-12-02-15:50:00_2000.pkl", env)
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_fakeonly_2019-12-02-15:50:33_3000.pkl", env)
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_fakeonly_2019-12-02-15:51:06_4000.pkl", env) # 0.001
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_fakeonly_2019-12-02-15:51:39_5000.pkl", env)
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_fakeonly_2019-12-02-15:52:13_6000.pkl", env)
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_fakeonly_2019-12-02-15:52:45_7000.pkl", env)
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_fakeonly_2019-12-02-15:53:14_8000.pkl", env)  # 0.541
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_fakeonly_2019-12-02-15:53:37_9000.pkl", env)
# model = HER.load("/home/michal/Projekty/RL/RL/resources/bitflipper_fakeonly_2019-12-02-15:53:57_10000.pkl", env)

# model = HER.load("/home/michal/Projekty/RL/RL/resources/colour_emb__2020-02-09-09:47:20_100000.pkl", env) # RUB-134
# model = HER.load("/home/michal/Projekty/RL/RL/resources/colour_emb_longer_ep__2020-02-21-15:20:27_30000.pkl", env) # RUB-156

model = HER.load("/home/michal/Projekty/RL/RL/resources/network_2k_2020-05-21-09:02:01_40000.pkl", env) # RUB-309

def simple_eval_Rubik(env):
    for i in range(1, 10):
        env.config(render_cube=True, scramble_size=i)
        env.step_limit = 2 * (i + 2)
        print(i, "scrambles", clear_eval(model, env, 100), rubik_ultimate_eval(model, env, 100))

    env.config(render_cube=True, scramble_size=7)
    env.step_limit = 15


def simple_eval_BitFlipper(env):
    print(clear_eval(model, env, 1000))


def watch_play(hold=False, sleep_scale=1.):
    with model.sess.as_default(), model.graph.as_default():
        while True:
            obs = env.reset()
            env.render()

            print("RESET")
            if not hold:
                sleep(2 * sleep_scale)

            while True:
                if hold:
                    input()

                action, _states = model.predict(np.concatenate((obs['observation'], obs['desired_goal']), axis=-1))
                q_values = model.predict_q_values(
                    np.concatenate((obs['observation'], obs['desired_goal']), axis=-1)).flatten()

                assert (action == np.argmax(q_values))
                print(action, list(q_values))

                obs, rewards, dones, info = env.step(action)
                env.render()

                if dones:
                    break
                    if hold:
                        print("DONE")

                if not hold:
                    sleep(1 * sleep_scale)

            if not hold:
                sleep(3 * sleep_scale)


simple_eval_Rubik(env)
watch_play(hold=False, sleep_scale=0.2)
