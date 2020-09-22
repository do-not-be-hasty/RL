# Minimal mrunner experiment configuration
from munch import Munch

from mrunner.experiment import Experiment
import os

if "NEPTUNE_API_TOKEN" not in os.environ or "PROJECT_QUALIFIED_NAME" not in os.environ:
    print("Please set NEPTUNE_API_TOKEN and PROJECT_QUALIFIED_NAME env variables")
    print("Their values can be from up.neptune.ml. Click help and then quickstart.")
    exit(1)

tags = os.environ["PROJECT_TAG"].split(' ') if "PROJECT_TAG" in os.environ.keys() else []

exp = Experiment(name='HER experiment',
                 script='python3 run.py',
                 # script='python3 dqn_scratch.py',
                 # script='python3 dqn_episodic.py',
                 # script='python3 metric_scratch.py',
                 project=os.environ["PROJECT_QUALIFIED_NAME"],
                 tags=tags,
                 env={"NEPTUNE_API_TOKEN": os.environ["NEPTUNE_API_TOKEN"]},
                 parameters=Munch(param1=10))

# A specification file must contain list of experiments in experiments_list variable. Here just one.
experiments_list = [exp]
