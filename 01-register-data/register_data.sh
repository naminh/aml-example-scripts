#!/bin/bash


sh ../scripts/az_configure.sh
az ml data create --file "data.yml"
