#!/bin/bash

set -e # stop on any error

cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)
PROJECT_DIR=${SCRIPT_DIR}/..
RES_DIR=${PROJECT_DIR}/resources

if [ ! -d "$RES_DIR" ]; then
	mkdir "$RES_DIR"
fi

if [ ! -d "$RES_DIR/environment" ]; then
	mkdir "$RES_DIR"/environment
fi

cd "$RES_DIR"/environment

if [ ! -d "$RES_DIR/environment/mazelab" ]; then
	git clone https://github.com/do-not-be-hasty/mazelab.git
	python3.5 -m pip install -e "$RES_DIR/environment/mazelab"
fi

if [ ! -d "$RES_DIR/environment/BitFlipper" ]; then
	git clone https://github.com/do-not-be-hasty/BitFlipper.git
	python3.5 -m pip install -e "$RES_DIR/environment/BitFlipper"
fi

if [ ! -d "$RES_DIR/environment/gym-rubik" ]; then
	git clone https://github.com/do-not-be-hasty/gym-rubik.git
	python3.5 -m pip install -e "$RES_DIR/environment/gym-rubik"
fi
