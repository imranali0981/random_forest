from fastapi import FastAPI
import subprocess
import sys
import os

# Add randomforest folder to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "randomforest"))

# Import the new function from predict_model.py
from randomforest.predict_plantation import predict_plantation_need


app = FastAPI()

@app.get("/predict_plantation")
def run_training_and_predict():
    print("Starting the prediction process...")
    try:
        # training already completed
        print("Training already completed.")
        print("Running the prediction function...")
        # Step 2: Predict and get results
        result = predict_plantation_need()

        return {
            "status": "success",
            "result": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
