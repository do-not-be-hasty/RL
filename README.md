# Model-free approach to solving the Rubik’s Cube
This is an experimental repository for the project of solving the Rubik's Cube with sparse rewards, without built-in knowledge and without environment model.
For classic RL algorithms such as DQN, this is a virtually impossible task, since regardless of the exploration quality the agent cannot reach the solved state from a randomly scrambled cube and thus observes no positive reward at all.
However, a successful policy can be trained by using a technique of [Hindsight Experience Replay](https://arxiv.org/pdf/1707.01495.pdf).
The agent trained this way is capable of solving all the cubes scrambled with several moves despite observing no successful trajectory during the whole training procedure.

## Project structure
This repository allows to run the HER algorithm on the Rubik's Cube environment, with sparse rewards.
Additionally, it provides two benchmark goal-oriented environments - simple grid maze in which the agent has to reach the target field in a maze and BitFlipper environment in which the agent can change the values of *n* bits and has to reach the target configuration.
The latter was proposed by the authors of HER and used as a motivating example for their work.

All the sources are located in `src` directory.
Most important files contain:
* `run.py` - main function which selects the experiment to run and specifies its complexity.
* `learning_configurations.py` - specifies all the parameters of training, including its length, learning rate, exploration etc.
* `DQN_HER.py` - implementation for main loop of the training algorithm.
* `episode_replay_buffer.py` - replay buffer, which apart from storing the experience manages goal assignment.

Having the desired experiment set in `run.py`, run it with the command `./scripts/run_local.sh ANONYMOUS "test"`.
This script prepares a virtual envirorenment and runs the selected experiment locally.
The results can be send to [neptune](https://ui.neptune.ai) for visualization.

## Experimental results
The agent trained for solving the Rubik's Cube starts learning quickly and after about 140000 episodes it successfully solves over 80% of cubes scrambled with 8 random moves.
Charts presenting detailed performance of this agent during this stage of training can be found [here](https://ui.neptune.ai/do-not-be-hasty/rubik/e/RUB-709/charts).
Though the progress decreases with time, after 1500000 training episodes its performance is the following:
![Success rate](https://github.com/do-not-be-hasty/RL/blob/master/chart_ncubes.png)
The agent successfully solves moderately scrambled cubes, but still solves only few completely random instances.
Detailed results can be found [here](https://ui.neptune.ai/do-not-be-hasty/rubik/e/RUB-775/charts).

In case of the benchmarks, the proposed implementation of HER easily reaches almost perfect success rate:
* maze 10x10 [https://ui.neptune.ai/do-not-be-hasty/rubik/e/RUB-812/charts]
* maze 30x30 [https://ui.neptune.ai/do-not-be-hasty/rubik/e/RUB-811/charts]
* maze 60x60 [https://ui.neptune.ai/do-not-be-hasty/rubik/e/RUB-815/charts]
* BitFlipper, 10 bits [https://ui.neptune.ai/do-not-be-hasty/rubik/e/RUB-807/charts]
* BitFlipper, 50 bits [https://ui.neptune.ai/do-not-be-hasty/rubik/e/RUB-808/charts]
* BitFlipper, 200 bits [https://ui.neptune.ai/do-not-be-hasty/rubik/e/RUB-820/charts]

Note that the last example is far more difficult than showed in [HER](https://arxiv.org/pdf/1707.01495.pdf), though the authors did not aim to optimize this particular task.

## Code ownership
The main loop of training is based on DQN implementation from [stable baselines](https://github.com/hill-a/stable-baselines).
The Rubik's Cube environment is taken from [do-not-be-hasty/gym-rubik](https://github.com/do-not-be-hasty/gym-rubik), which was forked from [yoavain/gym-rubik](https://github.com/yoavain/gym-rubik).
The BitFlipper environment is taken from [do-not-be-hasty/BitFlipper](https://github.com/do-not-be-hasty/BitFlipper), which is a fork of [JoyChopra1298/BitFlipper](https://github.com/JoyChopra1298/BitFlipper).
The maze environment is taken from [do-not-be-hasty/mazelab](https://github.com/do-not-be-hasty/mazelab), a fork of [zuoxingdong/mazelab](https://github.com/zuoxingdong/mazelab).
The remaining code was developed by the author.
