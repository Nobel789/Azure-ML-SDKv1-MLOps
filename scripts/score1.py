import json
import joblib
import numpy as np
import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    global model
    
    try:
        # 1. Get the root directory
        model_root = os.getenv('AZUREML_MODEL_DIR')
        target_filename = 'sklearn_reg_model.pkl'
        model_path = None

        logger.info(f"Searching for {target_filename} in {model_root}...")

        # --- SMART SEARCH ---
        # Walk through the directory tree to find the file.
        # This works if the file is at the root OR inside 'model_assets'
        for root, dirs, files in os.walk(model_root):
            if target_filename in files:
                model_path = os.path.join(root, target_filename)
                logger.info(f"Found model at: {model_path}")
                break
        
        # If still not found, crash with a helpful error
        if model_path is None:
            logger.error("Could not find file. Directory dump:")
            for root, dirs, files in os.walk(model_root):
                logger.error(f"{root}: {files}")
            raise FileNotFoundError(f"Could not find {target_filename} anywhere in {model_root}")

        # 2. Load the model
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
        
    except Exception as e:
        logger.error(f"CRITICAL INIT ERROR: {str(e)}")
        raise e

def run(raw_data):
    # Standard run function
    start_time = time.perf_counter()
    result = None
    success = False
    
    try:
        # 1. Parse Input
        json_data = json.loads(raw_data)
        
        if 'data' not in json_data:
            return json.dumps({"error": "Invalid input format. Key 'data' missing."})
        
        input_data = np.array(json_data['data'])
        
        # 2. Predict
        predictions = model.predict(input_data)
        
        # Convert to list for JSON serialization
        result = predictions.tolist()
        success = True
        
        return json.dumps({"result": result})

    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})

    finally:
        # 3. Log Metrics
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        
        log_payload = {
            "metric": "predict_latency_ms",
            "value": latency_ms,
            "success": success,
            "timestamp": time.time()
        }
        print(json.dumps(log_payload))
