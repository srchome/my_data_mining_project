import pyodbc

def upload_predictions_to_sql(df, table_name="PredictionResults"):
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=LAPTOP-CDU52NMS\MSSQLSERVER_SR;'
        'DATABASE=DataMiningDB;'
        'Trusted_Connection=yes;'
    )
    cursor = conn.cursor()

    # Optional: clear existing data
    cursor.execute(f"DELETE FROM {table_name}")
    conn.commit()

    # Insert rows
    for _, row in df.iterrows():
        cursor.execute(f"""
            INSERT INTO {table_name} (Age, Income, Student, Credit_Rating, Actual, Predicted)
            VALUES (?, ?, ?, ?, ?, ?)
        """, row["Age"], row["Income"], row["Student"], row["Credit_Rating"],
             row["Actual_Label"], row["Predicted_Label"])

    conn.commit()
    conn.close()
    print("✅ Predictions uploaded to SQL Server.")
