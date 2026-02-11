# score.py
import json
import numpy as np
import joblib
from azureml.core.model import Model

# Called when the service is started
def init():
    global model
    # Load the registered model
    model_path = Model.get_model_path('my-sklearn-model')
    model = joblib.load(model_path)

# Called for each request
def run(raw_data):
    try:
        # Convert input JSON to NumPy array
        data = np.array(json.loads(raw_data)['data'])
        predictions = model.predict(data)
        return predictions.tolist()
    except Exception as e:
        return str(e)