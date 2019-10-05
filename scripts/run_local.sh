#!/usr/bin/env bash

cd scripts/

source ../py35/bin/activate

./basic_setup.sh plgmizaw sim2real

export PROJECT_QUALIFIED_NAME="do-not-be-hasty/sokoban"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiIyY2RlMTgwMi02ZjY1LTQ5NjItOTgxOC1lY2I4ZTAwNDI2OTcifQ=="

echo "Run experiments locally"
set -o xtrace
python ../src/run.py --ex ../src/sokoban_conf.py
