from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="CICIDS Intrusion Detection API")

# Load model and feature names
model = joblib.load("models/xgb_model.pkl")
feature_names = joblib.load("data/processed/feature_names.pkl")


# Input schema
class InputData(BaseModel):
    data: dict


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