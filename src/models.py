from functools import partial

from stable_baselines.common.policies import MlpPolicy as PPO2_Policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN
from stable_baselines.deepq import MlpPolicy as DQN_Policy
from DQN_HER import DQN_HER as HER
from DQN_metric import DQN_MTR as MTR


def DQN_model(env):
    env = DummyVecEnv([lambda: env])
    return DQN(
        policy=partial(DQN_Policy, layers=[256]),
        env=env,
        learning_rate=1e-3,
        buffer_size=1000000,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        verbose=1,
    )


def HER_model(env):
    return HER(
        policy=partial(DQN_Policy, layers=[1024, 1024]),
        env=env,
        hindsight=1,
        learning_rate=1e-4,
        buffer_size=2000000,
        exploration_fraction=0.9,
        exploration_final_eps=0.2,
        gamma=0.98,
        verbose=1,
    )


def HER_model_conv(env):
    return HER(
        policy="convnet_mnist",
        env=env,
        hindsight=1,
        learning_rate=1e-4,
        buffer_size=2000000,
        exploration_fraction=0.9,
        exploration_final_eps=0.2,
        gamma=0.98,
        verbose=1,
    )


def MTR_model(env):
    return MTR(
        policy=partial(DQN_Policy, layers=[256]),
        env=env,
        hindsight=0.5,
        learning_rate=1e-4,
        buffer_size=1000000,
        exploration_fraction=0.05,
        exploration_final_eps=0.0005,
        gamma=0.98,
        verbose=1,
    )


def PPO_model(env):
    env = DummyVecEnv([lambda: env])
    return PPO2(
        policy=PPO2_Policy,
        env=env,
        learning_rate=1e-3,
        verbose=1,
    )