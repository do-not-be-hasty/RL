from time import sleep, time
import tensorflow as tf

from DQN_HER import DQN_HER as HER
from environment_builders import make_env_GoalRubik
from models import HER_model
import numpy as np
from utility import clear_eval, model_summary

env = make_env_GoalRubik(
    step_limit=10,
)

model = HER.load("/home/michal/Projekty/RL/tmp/checkpoint_2019-11-09-17:47:42_10.pkl", env)

env.config(render_cube=True, scramble_size=2)


# print(clear_eval(model, env, 10))

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


watch_play(hold=True, sleep_scale=0.01)
