import sys
import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from flask import flash

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluate import predict_with_model

app = Flask(__name__)
app.secret_key = "some-secret-value"  # Required for flashing message
UPLOAD_FOLDER = "webapp/static/uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = os.path.join(OUTPUT_FOLDER, "model.joblib")
ENCODER_PATH = os.path.join(OUTPUT_FOLDER, "label_encoders.joblib")
PLOT_PATH = os.path.join(OUTPUT_FOLDER, "decision_tree.png")
WEBAPP_PLOT_PATH = "webapp/static/outputs/decision_tree.png"
TARGET_COLUMN_PATH = os.path.join(OUTPUT_FOLDER, "target_column.txt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    table = None
    image_path = PLOT_PATH if os.path.exists(PLOT_PATH) else None
    download_file = None
    message = None
    filename = None
    prediction_counts = None
    income_counts = None
    # Read model info if available
    model_type = None
    training_file = None

    model_file = os.path.join(OUTPUT_FOLDER, "model_type.txt")
    train_file = os.path.join(OUTPUT_FOLDER, "training_filename.txt")

    if os.path.exists(model_file):
        with open(model_file) as f:
            model_type = f.read().strip()

    if os.path.exists(train_file):
        with open(train_file) as f:
            training_file = f.read().strip()

    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename:
            filename = uploaded_file.filename
            filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(filepath)

            try:
                input_df = pd.read_csv(filepath)

                # Load saved target column
                saved_target_column = None
                if os.path.exists(TARGET_COLUMN_PATH):
                    with open(TARGET_COLUMN_PATH, "r") as f:
                        saved_target_column = f.read().strip()

                # Drop target column if present
                if saved_target_column and saved_target_column in input_df.columns:
                    input_df = input_df.drop(saved_target_column, axis=1)

                model = joblib.load(MODEL_PATH)
                label_encoders = joblib.load(ENCODER_PATH)

                expected_features = list(model.feature_names_in_)
                missing = [col for col in expected_features if col not in input_df.columns]

                if missing:
                    return f"❗ Missing required columns: {missing} Please select the correct file to predict.<br> The columns in the CSV file for training (except target column) should match with the CSV file uploaded to predict!", 400

                input_df = input_df[expected_features]

                for col in input_df.columns:
                    if col in label_encoders and input_df[col].dtype == object:
                        encoder = label_encoders[col]
                        input_df[col] = encoder.transform(input_df[col])

                preds = predict_with_model(model, input_df, label_encoders)
                input_df["Prediction"] = preds

                #prediction_counts = input_df["Prediction"].value_counts().to_dict()

                # Handle income-like column dynamically
                income_column = None
                for col in input_df.columns:
                    if "income" in col.lower():
                        income_column = col
                        break

                if income_column:
                    income_counts = input_df[income_column].value_counts().sort_index().to_dict()

                # Decode categorical columns
                for col, le in label_encoders.items():
                    if col in input_df.columns:
                        input_df[col] = le.inverse_transform(input_df[col])

                # Decode prediction if target available
                if saved_target_column and saved_target_column in label_encoders:
                    if pd.api.types.is_numeric_dtype(input_df["Prediction"]):
                        input_df["Prediction"] = label_encoders[saved_target_column].inverse_transform(input_df["Prediction"])

                #For charts
                prediction_counts = input_df["Prediction"].value_counts().to_dict()

                # Save output
                predictions_filename = filename.replace(".csv", "_predictions.csv")
                predictions_path = os.path.join("webapp/static/uploads", predictions_filename)
                input_df.to_csv(predictions_path, index=False)
                download_file = os.path.join("uploads", predictions_filename).replace("\\", "/")

                table = input_df.to_html(classes="table table-striped table-bordered text-start", index=False)
                #message = "✅ Predictions created successfully!"
                # After successful prediction
                flash("✅ Predictions created successfully!", "success")

            except Exception as e:
                return f"Error: {e}", 500

    return render_template(
        "index.html",
        table=table,
        image_path=image_path if os.path.exists(PLOT_PATH) else None,
        download_file=download_file,
        prediction_counts=prediction_counts,
        income_counts=income_counts,
        message=message,
        model_type=model_type, 
        training_file=training_file
    )


@app.route("/train", methods=["GET", "POST"])
def train():
    message = None
    columns = []
    model_trained = False
    filename = None
    decision_tree_trained = False

    if request.method == "POST":
        if 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file and uploaded_file.filename.endswith(".csv"):
                filename = uploaded_file.filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                uploaded_file.save(filepath)

                df = pd.read_csv(filepath)
                columns = df.columns.tolist()

                message = "✅ File uploaded successfully! Now select the target column and model."
                return render_template(
                    "train.html",
                    message=message,
                    columns=columns,
                    filename=filename,
                    model_trained=False,
                    decision_tree_trained=False
                )

        elif 'target' in request.form and 'filename' in request.form and 'model_choice' in request.form:
            filename = request.form['filename']
            target_column = request.form['target']
            model_choice = request.form['model_choice']
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            if not os.path.exists(filepath):
                return "Uploaded file not found.", 400

            df = pd.read_csv(filepath)

            if target_column not in df.columns:
                return f"Target column {target_column} not found!", 400

            # Separate features and target
            X = df.drop(target_column, axis=1)
            y = df[target_column]

            label_encoders = {}
            for col in X.columns:
                if X[col].dtype == object:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    label_encoders[col] = le

            if y.dtype == object:
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y)
                label_encoders[target_column] = target_encoder

            # Create model based on choice
            if model_choice == "decision_tree":
                model = DecisionTreeClassifier(random_state=42)
                decision_tree_trained = True
            elif model_choice == "random_forest":
                model = RandomForestClassifier(random_state=42)
            elif model_choice == "logistic_regression":
                model = LogisticRegression(max_iter=1000)
            elif model_choice == "knn":
                model = KNeighborsClassifier()
            else:
                return "❌ Invalid model selected.", 400

            model.fit(X, y)

            joblib.dump(model, MODEL_PATH)
            joblib.dump(label_encoders, ENCODER_PATH)

            model.feature_names_in_ = X.columns

            # Save Decision Tree plot only if applicable
            if decision_tree_trained:
                plt.figure(figsize=(20, 10))
                plot_tree(
                    model,
                    feature_names=X.columns,
                    class_names=label_encoders[target_column].classes_,
                    filled=True
                )
                plt.savefig(PLOT_PATH)
                plt.savefig(WEBAPP_PLOT_PATH)
                plt.close()

            # Save target column
            with open(TARGET_COLUMN_PATH, "w") as f:
                f.write(target_column)
            
            # Save model type
            with open(os.path.join(OUTPUT_FOLDER, "model_type.txt"), "w") as f:
                f.write(model_choice)

            # Save training filename
            with open(os.path.join(OUTPUT_FOLDER, "training_filename.txt"), "w") as f:
                f.write(filename)


            message = f"✅ Model ({model_choice.replace('_', ' ').title()}) trained successfully!"

            return render_template(
                "train.html",
                message=message,
                columns=[],
                filename=None,
                model_trained=True,
                decision_tree_trained=decision_tree_trained
            )

    return render_template(
        "train.html",
        message=None,
        columns=[],
        filename=None,
        model_trained=False,
        decision_tree_trained=False
    )


@app.route("/clear_uploads", methods=["POST"])
def clear_uploads():
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        # Also clear the decision tree plot
        if os.path.exists(PLOT_PATH):
            os.remove(PLOT_PATH)
        if os.path.exists(WEBAPP_PLOT_PATH):
            os.remove(WEBAPP_PLOT_PATH)

        for file in ["model_type.txt", "training_filename.txt"]:
            path = os.path.join(OUTPUT_FOLDER, file)
            if os.path.exists(path):
                os.remove(path)

        flash("🗑️ Uploads cleared successfully!", "success")
        return redirect(url_for('index', cleared="true"))

    except Exception as e:
        return f"❌ Error clearing uploads: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
