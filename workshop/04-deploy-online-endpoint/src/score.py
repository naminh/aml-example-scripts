import os
import json
import mlflow
import joblib

from io import StringIO
from mlflow.pyfunc.scoring_server import predictions_to_json

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model/model.pkl")
    model = joblib.load(model_path)

def run(raw_data):
    json_data = json.loads(raw_data)

    predictions = model.predict(json_data["inputs"])

    result = StringIO()
    predictions_to_json(predictions, result)

    return result.getvalue()
