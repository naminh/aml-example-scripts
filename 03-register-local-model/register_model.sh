#!/bin/bash


sh ../scripts/az_configure.sh
az ml model create --name "NAME" --version 1 --path /model
