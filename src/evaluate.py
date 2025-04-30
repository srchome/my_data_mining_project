# Functions for model evaluation and visualization
import os
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd

def evaluate_model(model, X_test, y_test, label_encoders, output_csv=True):
    import numpy as np

    # Step 1: Get expected features only (drop unseen columns like 'Buys_Product')
    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
        X_test = X_test[[col for col in expected_features if col in X_test.columns]]
    else:
        # In case model doesn't store feature names (older scikit-learn)
        print("⚠️ Warning: Model does not store feature names. Using X_test as-is.")

    # Step 2: Predict
    y_pred = model.predict(X_test)
    
    # Step 3: Print accuracy and classification report
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    if output_csv:
        # Combine actual & predicted
        results_df = X_test.copy()
        results_df["Actual"] = y_test
        results_df["Predicted"] = y_pred

        # Decode labels if label encoder exists
        if "Buys_Product" in label_encoders:
            le = label_encoders["Buys_Product"]

            # Ensure y values are arrays for inverse_transform
            actual_decoded = le.inverse_transform(np.array(y_test))
            predicted_decoded = le.inverse_transform(np.array(y_pred))

            results_df["Actual_Label"] = actual_decoded
            results_df["Predicted_Label"] = predicted_decoded

        # Save results to CSV
        output_path = os.path.join("outputs", "predictions.csv")
        #added new to see human friendly value in table
        for col, le in label_encoders.items():
            if col in results_df.columns:
                results_df[col] = le.inverse_transform(results_df[col])
        #end new

        results_df.to_csv(output_path, index=False)
        print(f" Predictions saved to: {output_path}")

    return y_pred



def plot_model(model, feature_names, output_path=None):
    plt.figure(figsize=(10, 6))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=["No", "Yes"])
    if output_path:
        # Make sure the folder exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    else:
        plt.show()


def predict_with_model(model, X, label_encoders=None):
    """
    Predict using the trained model and return labels (decoded if encoders are provided).
    
    Parameters:
    - model: Trained model (e.g., DecisionTreeClassifier)
    - X: Preprocessed input features (DataFrame)
    - label_encoders: Optional dictionary of encoders to decode predictions
    
    Returns:
    - List or Series of predicted labels
    """
    predictions = model.predict(X)

    # Decode predictions if encoders are provided
    if label_encoders and "Buys_Product" in label_encoders:
        le = label_encoders["Buys_Product"]
        predictions = le.inverse_transform(predictions)

    return predictions
