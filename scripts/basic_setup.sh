#!/usr/bin/env bash

#run as basic_setup.sh PROMETHEUS_LOGIN

set -e
export PROMETHEUS_LOGIN=$1
export GRANT_NAME=$2

function prepare_local_venv {
    ENV_DIR=../py35
    source $ENV_DIR/bin/activate
    pip install wheel
    pip install psutil
    pip install -r ../src/neptune_resources/requirements_local.txt --quiet
}

function prepare_mrunner_config {
    sed "s/<username>/$PROMETHEUS_LOGIN/g" ../src/neptune_resources/prometheus_config_template.yaml > /tmp/mrunner_config_1.yaml
    sed "s/<grantname>/$GRANT_NAME/g" /tmp/mrunner_config_1.yaml > /tmp/mrunner_config.yaml

    rm /tmp/mrunner_config_1.yaml
}


function prepare_envs_and_mrunner_config {
    if [ -z "$PROMETHEUS_LOGIN" ];
    then
        echo "PROMETHEUS_LOGIN must be set. exiting";
        exit;
    fi
    
    prepare_local_venv
    prepare_mrunner_config
}

prepare_envs_and_mrunner_config
