#!/bin/bash

set -e # stop on any error

cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)
PROJECT_DIR=${SCRIPT_DIR}/..

PIP_ENV=${PROJECT_DIR}/py35
REQ_DIR=${PROJECT_DIR}/requirements
SCRIPT_TO_RUN=${PROJECT_DIR}/$1

# Prepare virtualenv

if [ ! -d "$PIP_ENV" ]; then
	virtualenv -p python3.5 "${PIP_ENV}"
fi

source ${PIP_ENV}/bin/activate

for f in "${REQ_DIR}"/*
do
	pip3 install -r "$f"
done

# Run script with environment variables
PYTHONPATH=${PROJECT_DIR}/src \
RESOURCES_DIR=${PROJECT_DIR}/resources \
python3.5 ${SCRIPT_TO_RUN} "${@:2}"

deactivate
