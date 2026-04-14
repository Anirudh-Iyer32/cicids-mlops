import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def preprocess(input_path, output_dir):
    # Load data
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Binary classification (simplify first)
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    # Split features and target
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Save
    pd.DataFrame(X_train).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    joblib.dump(scaler, f"{output_dir}/scaler.pkl")

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess(
        input_path="data/raw/wednesday.csv",
        output_dir="data/processed"
    )