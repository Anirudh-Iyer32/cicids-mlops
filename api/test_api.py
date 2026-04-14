import requests
import pandas as pd
import joblib

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv")
feature_names = joblib.load("data/processed/feature_names.pkl")

# Take one sample
sample = dict(zip(feature_names, X_test.iloc[0]))

# Send request
response = requests.post(url, json={"data": sample})

# Print result
print("Response:", response.json())