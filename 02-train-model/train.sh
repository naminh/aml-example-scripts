#!/bin/bash

sh ../scripts/az_configure.sh
az ml environment create --name "NAME" --build-context "env/" --dockerfile-path "Dockerfile"

# az compute check if exist
az ml compute show -n "COMPUTE_NAME"

az ml job create -f "pipeline.yml" --set inputs.train_epoch_param=5