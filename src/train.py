import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def train(data_dir, model_dir):
    print("Loading data...")

    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()

    with mlflow.start_run():

        print("Training XGBoost model...")
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train)

        print("Evaluating model...")
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log params
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 6)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy: {acc}")

        # Save model
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, f"{model_dir}/xgb_model.pkl")

        mlflow.sklearn.log_model(model, "model")

        print("Training completed!")

if __name__ == "__main__":
    train("data/processed", "models")