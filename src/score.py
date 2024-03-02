import json
import logging
import os

import joblib
import pandas as pd


def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    model = joblib.load(model_path)
    logging.info("Init complete")


def run(raw_data):
    """
    Runs the prediction on the given raw data.

    Args:
        raw_data (str): The raw data to be processed.

    Returns:
        str: The predictions in JSON format.
    """
    logging.info("Invoke request received...")
    json_data = json.loads(raw_data)
    print(json_data)

    predictions = model.predict_proba(json_data["inputs"])

    return pd.DataFrame(predictions).to_json()
