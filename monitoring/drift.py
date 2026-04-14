import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def run_drift_report(reference_path, current_path, output_path):
    print("Checking files...")

    # Check if live data exists
    if not os.path.exists(current_path):
        print(f"No live data found at {current_path}")
        print("Run API predictions first to generate live data.")
        return

    print("Loading data...")

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    # Handle small datasets safely
    reference = reference.sample(min(3000, len(reference)), random_state=42)
    current = current.sample(min(3000, len(current)), random_state=42)

    print(f"Reference shape: {reference.shape}")
    print(f"Current shape: {current.shape}")

    print("Creating report...")
    report = Report(metrics=[
        DataDriftPreset()
    ])

    print("Running report...")
    report.run(reference_data=reference, current_data=current)

    print("Saving report...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)

    print("Done! Report saved at:", output_path)


if __name__ == "__main__":
    run_drift_report(
        reference_path="data/processed/X_train.csv",
        current_path="monitoring/live_data.csv",   # 🔥 LIVE DATA
        output_path="monitoring/drift_report.html"
    )