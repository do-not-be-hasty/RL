from mrunner.helpers.specification_helper import create_experiments_helper
import os

tags = [os.environ["PROJECT_TAG"]] if "PROJECT_TAG" in os.environ.keys() else []

experiments_list = create_experiments_helper(
    experiment_name='exp0',
    base_config={},
    params_grid={'TestDeterministicMCTSAgent.n_passes': [50]},
    script='python3 -m alpacka.runner --mrunner --output_dir=./out --config_file=mcts/mcts.gin',
    exclude=['.git', 'resources'],
    python_path='',
    tags=tags,
    with_neptune=True
)
