from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="CICIDS Intrusion Detection API")

# Load model and feature names
model = joblib.load("models/xgb_model.pkl")
feature_names = joblib.load("data/processed/feature_names.pkl")

# Path for logging live data
LOG_FILE = "monitoring/live_data.csv"


# Input schema
class InputData(BaseModel):
    data: dict



def log_input(data):
    try:
        df = pd.DataFrame([data])

        # Ensure folder exists
        os.makedirs("monitoring", exist_ok=True)

        if not os.path.exists(LOG_FILE):
            df.to_csv(LOG_FILE, index=False)
        else:
            df.to_csv(LOG_FILE, mode='a', header=False, index=False)

    except Exception as e:
        print(f"Logging failed: {e}")  # Don't break API if logging fails


@app.get("/")
def home():
    return {"message": "CICIDS Intrusion Detection API is running"}


@app.post("/predict")
def predict(input_data: InputData):
    try:
        data = input_data.data

        # Check for missing features
        missing_features = [col for col in feature_names if col not in data]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing_features}"
            )

        
        log_input(data)

        # Arrange features in correct order
        features = np.array([data[col] for col in feature_names]).reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return {
            "prediction": int(prediction),
            "label": "ATTACK" if prediction == 1 else "BENIGN",
            "probability": float(probability)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))