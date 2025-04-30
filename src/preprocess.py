import pandas as pd
import pyodbc
from sklearn.preprocessing import LabelEncoder

import joblib

def preprocess_data(df, training=True):
    label_encoders = {}

    categorical_columns = ["Student", "Credit_Rating"]  # update based on your data

    for col in categorical_columns:
        le = LabelEncoder()
        if training:
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            # Load saved encoders
            label_encoders = joblib.load("outputs/label_encoders.joblib")
            le = label_encoders[col]
            df[col] = le.transform(df[col])

    X = df.drop(columns=["Buys_Product"], errors="ignore")
    y = df["Buys_Product"] if "Buys_Product" in df.columns else None

    return X, y, label_encoders


def load_from_sql(server, database, table):
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};DATABASE={database};Trusted_Connection=yes;"
    )
    conn = pyodbc.connect(conn_str)
    query = f"SELECT * FROM {table}"
    df = pd.read_sql(query, conn)
    conn.close()

    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    return df, label_encoders
