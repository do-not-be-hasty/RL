#!/bin/bash

set -e # stop on any error

cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)
PROJECT_DIR=${SCRIPT_DIR}/..

PIP_ENV=${PROJECT_DIR}/py35
REQ_DIR=${PROJECT_DIR}/requirements
RES_DIR=${PROJECT_DIR}/resources
SCRIPT_TO_RUN=${PROJECT_DIR}/$1

# Prepare virtualenv

if [ ! -d "$PIP_ENV" ]; then
	virtualenv -p python3.5 "${PIP_ENV}"
fi

source ${PIP_ENV}/bin/activate

echo "Installing requirements..."

for f in "${REQ_DIR}"/*
do
	python3.5 -m pip install -r "$f" -q
done

bash "$SCRIPT_DIR"/get_resources.sh

echo "Requirements installed."

# Run script with environment variables
PYTHONPATH=${PROJECT_DIR}/src \
RESOURCES_DIR=${PROJECT_DIR}/resources \
python3.5 ${SCRIPT_TO_RUN} "${@:2}"

deactivate
