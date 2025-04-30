# Functions to train and save the model
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def load_model(model_path="outputs/model.joblib"):
    model = joblib.load(model_path)
    label_encoders = joblib.load("outputs/label_encoders.joblib")
    return model, label_encoders


def train_model(X, y):
    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(X, y)
    return model

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
