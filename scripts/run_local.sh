#!/usr/bin/env bash

set -e # stop on any error

if [ "$#" -ne 2 ]; then
    echo "Usage: ./scripts/run_mrunner [project_name] [project_description]"
    exit 1
fi


PYTHON=python3.6
PROJECT_DIR=$(pwd)
SCRIPT_DIR=${PROJECT_DIR}/scripts
SRC_DIR=${PROJECT_DIR}/src
PIP_ENV=${PROJECT_DIR}/py36
REQ_DIR=${PROJECT_DIR}/requirements
RES_DIR=${PROJECT_DIR}/resources

PROJECT_NAME=$1


# Currently, pending projects are rubik and sokoban
# Add here if new appear

if [[ "$PROJECT_NAME" =~ ^(sokoban|rubik)$ ]]; then
    echo "Running $PROJECT_NAME project"
else
    echo "\"$PROJECT_NAME\" is not a valid project name"
    exit 1
fi


# Prepare virtualenv

if [ ! -d "$PIP_ENV" ]; then
	virtualenv -p $PYTHON "${PIP_ENV}"
fi

source ${PIP_ENV}/bin/activate
echo "Installing requirements..."
for f in "${REQ_DIR}"/*
do
        # Add -q to make the installation quiet
	$PYTHON -m pip install -r "$f" -q
done
echo "Requirements installed."

echo "Preparing resources..."
bash "$SCRIPT_DIR"/get_resources.sh
echo "Resources prepared"


# Prepare mrunner config

cd ${SCRIPT_DIR}
./basic_setup.sh plgmizaw sim2real

export PROJECT_QUALIFIED_NAME="do-not-be-hasty/local"
. export_api_token.sh
if [ ! -z "$2" ]; then
        export PROJECT_TAG="$2"
fi

ssh-add


# Run the project

echo "Run experiment locally"
set -o xtrace
python ../src/run.py --ex ../src/project_conf.py
