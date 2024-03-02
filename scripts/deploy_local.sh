#!/bin/bash

azmlinfsrv --entry_script src/score.py --model_dir models/

curl --request POST "0.0.0.0:5001/score" --header "Content-Type:application/json" --data @data/endpoint/sample_request.json