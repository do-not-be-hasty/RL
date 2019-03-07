import argparse
import os
from deepsense import neptune
import socket

_ctx = None

def is_neptune_online():
  # I wouldn't be suprised if this would depend on neptune version
  return 'NEPTUNE_ONLINE_CONTEXT' in os.environ


def get_configuration():
  global _ctx
  if is_neptune_online():
    # running under neptune
    ctx = neptune.Context()
    exp_dir_path = os.getcwd()
  else:
    parser = argparse.ArgumentParser(description='Debug run.')
    parser.add_argument('--ex', type=str)
    parser.add_argument("--exp_dir_path", default='/tmp')
    commandline_args = parser.parse_args()
    if commandline_args.ex != None:
      vars = {}
      exec(open(commandline_args.ex).read(), vars)
      spec_func = vars['spec']
      # take first experiment (params configuration)
      experiment = spec_func()[0]
      params = experiment.parameters
    else:
      params = {}
    # create offline context
    ctx = neptune.Context(offline_parameters=params)
    exp_dir_path = commandline_args.exp_dir_path
  _ctx = ctx
  ctx.properties['pwd'] = os.getcwd()
  ctx.properties['host'] = socket.gethostname()
  return ctx, exp_dir_path


def neptune_logger(m, v):
  global _ctx
  assert _ctx is not None, "Run first get_configuration"
  if _ctx.experiment_id is None:
    print(rf"{m}:{v}")
  else:
    _ctx.channel_send(name=m, x=None, y=v)