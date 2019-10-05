#!/usr/bin/env bash

cd scripts/

source ../py35/bin/activate

./basic_setup.sh plgmizaw sim2real

export PROJECT_QUALIFIED_NAME="do-not-be-hasty/sokoban"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiIyY2RlMTgwMi02ZjY1LTQ5NjItOTgxOC1lY2I4ZTAwNDI2OTcifQ=="

ssh-add

cd ../src/

mrunner --config /tmp/mrunner_config.yaml --context prometheus_cpu run sokoban_conf.py
# mrunner --config /tmp/mrunner_config.yaml --context prometheus_cpu --cmd_type srun run sokoban_conf.py

deactivate
