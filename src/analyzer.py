from time import sleep, time
import tensorflow as tf

from DQN_HER import DQN_HER as HER
from environment_builders import make_env_GoalRubik
from models import HER_model
import numpy as np
from utility import clear_eval, model_summary, rubik_ultimate_eval

env = make_env_GoalRubik(
    step_limit=10,
)

# model = HER.load("/home/michal/Projekty/RL/RL/resources/hincur_checkpoint_2019-11-11-04:59:42_30000.pkl", env) # RUB-10

# model = HER.load("/home/michal/Projekty/RL/RL/resources/hincur_checkpoint_2019-11-11-05:30:01_30000.pkl", env) # RUB-11
# model = HER.load("/home/michal/Projekty/RL/RL/resources/hincur_checkpoint_2019-11-11-18:02:54_60000.pkl", env) # RUB-11

# model = HER.load("/home/michal/Projekty/RL/RL/resources/R7_hind1_checkpoint_2019-11-11-08:00:53_20000.pkl", env) # RUB-17
# model = HER.load("/home/michal/Projekty/RL/RL/resources/R7_hind1_checkpoint_2019-11-11-14:24:29_40000.pkl", env) # RUB-17
# model = HER.load("/home/michal/Projekty/RL/RL/resources/R7_hind1_checkpoint_2019-11-11-20:04:16_60000.pkl", env) # RUB-17
# model = HER.load("/home/michal/Projekty/RL/RL/resources/R7_hind1_checkpoint_2019-11-12-00:48:03_80000.pkl", env) # RUB-17

model = HER.load("/home/michal/Projekty/RL/RL/resources/fakeonly_2019-11-13-12:01:24_40000.pkl", env) # RUB-21

for i in range(1, 10):
    env.config(render_cube=True, scramble_size=i)
    env.step_limit = 2*(i+2)
    print(i, "scrambles", clear_eval(model, env, 100), rubik_ultimate_eval(model, env, 100))

env.config(render_cube=True, scramble_size=7)
env.step_limit = 15


def watch_play(hold=False, sleep_scale=1.):
    with model.sess.as_default(), model.graph.as_default():
        while True:
            obs = env.reset()
            env.render()

            if hold:
                print("RESET")
            else:
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


watch_play(hold=True, sleep_scale=0.2)
