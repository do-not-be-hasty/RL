#!/bin/bash

set -e # stop on any error

ssh-add

source py35/bin/activate

cd src
cp -a ~/.neptune .
mkdir -p ~/.neptune_tokens
cp ~/.neptune/tokens/* ~/.neptune_tokens/  

mrunner --context rl_exp run ${1#*/}

deactivate
