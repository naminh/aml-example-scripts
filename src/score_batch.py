import logging
import os
from typing import List

import joblib
import pandas as pd


def init():
    global model
    global output_path
    output_path = os.environ["AZUREML_BI_OUTPUT_PATH"]
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)
    logging.info("Init complete")

def run(mini_batch: List[str]):
    logging.info("Invoke request received...")

    results = pd.DataFrame()
    for batch in mini_batch:
        data = pd.read_csv(batch)

        data["prediction"] = model.predict(data)
        results = pd.concat([results, data], ignore_index=True)

    return results