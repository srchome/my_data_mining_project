# Entry point for the project

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import load_from_sql
from src.model import train_model, save_model
from src.evaluate import evaluate_model, plot_model
from src.db import upload_predictions_to_sql

# File paths
MODEL_PATH = "outputs/model.joblib"
LABEL_ENCODER_PATH = "outputs/label_encoders.joblib"
PLOT_PATH = "outputs/decision_tree.png"
WEBAPP_PLOT_PATH = "webapp/static/outputs/decision_tree.png"

if __name__ == "__main__":
    # Load data from SQL
    df, label_encoders = load_from_sql("LAPTOP-CDU52NMS\\MSSQLSERVER_SR", "DataMiningDB", "CustomerData")

    X = df.drop("Buys_Product", axis=1)
    y = df["Buys_Product"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate and save predictions
    y_pred = evaluate_model(model, X_test, y_test, label_encoders)

    # Upload to SQL
    results_df = pd.read_csv("outputs/predictions.csv")
    upload_predictions_to_sql(results_df)

    # Plot model
    plot_model(model, X.columns.tolist(), output_path=PLOT_PATH)

    # Copy plot to webapp static folder
    os.makedirs(os.path.dirname(WEBAPP_PLOT_PATH), exist_ok=True)
    shutil.copy(PLOT_PATH, WEBAPP_PLOT_PATH)

    # Save model and encoders
    save_model(model, MODEL_PATH)
    import joblib
    joblib.dump(label_encoders, LABEL_ENCODER_PATH)

    print(f"\n✅ Model saved to: {MODEL_PATH}")
    print(f"🖼️ Plot saved to: {PLOT_PATH}")
    print(f"🌐 Plot copied to: {WEBAPP_PLOT_PATH}")
