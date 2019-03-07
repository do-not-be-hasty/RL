from mrunner.experiment import Experiment
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'neptune_utils'))
from spec_utils import get_git_head_info, get_combinations
# It might be a good practice to not change specification files if run
# successfully, to keep convenient history of experiments. When you want to run
# the same experiment with different hyper-parameters, just copy it.
# Starting name with (approximate) date of run is also helpful.

def create_experiment_for_spec(parameters):
    script = 'run.py'
    # this will be also displayed in jobs on prometheus
    name = 'MZ, rl exp'
    project_name = "my-project"
    python_path = '.:some_utils:some/other/utils/path'
    paths_to_dump = ''  # e.g. 'plgrid tensor2tensor', do we need it?
    tags = ''
    parameters['git_head'] = get_git_head_info()
    return Experiment(project=project_name, name=name, script=script,
                      parameters=parameters, python_path=python_path,
                      paths_to_dump=paths_to_dump, tags=tags,
                      time='0-8'  # days-hours
                      )

# Set params_configurations, eg. as combinations of grid.
# params are also good place for e.g. output path, or git hash
params_grid = dict(
  #delta=[0.5, 1],
  alpha=['a',],
)
params_configurations = get_combinations(params_grid)


def spec():
	experiments = [create_experiment_for_spec(params)
			for params in params_configurations]
	return experiments
