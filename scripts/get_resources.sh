#!/bin/bash

set -e # stop on any error

cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)
PROJECT_DIR=${SCRIPT_DIR}/..
RES_DIR=${PROJECT_DIR}/resources

if [ ! -d "$RES_DIR/environment" ]; then
	mkdir "$RES_DIR"/environment
	cd "$RES_DIR"/environment
	git clone https://github.com/do-not-be-hasty/mazelab.git
	pip3 install -e "$RES_DIR/environment/mazelab"
fi
