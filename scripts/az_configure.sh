#!/bin/bash

source ../config/aml_config.sh
az account set --subscription $subscription_id
az configure --defaults group=$resource_group workspace=$workspace_name

echo "completed az cli login with subsription_id=$subscription_id group=$resource_group workspace=$workspace_name"
