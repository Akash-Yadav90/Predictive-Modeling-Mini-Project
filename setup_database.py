import sqlite3
import pandas as pd

from sklearn.datasets import load_diabetes, load_breast_cancer

# Create the database
conn = sqlite3.connect("data/predictive_models.db")

# Load regression dataset
diabetes = load_diabetes(as_frame=True)
regression_df = diabetes.frame

# Load classification dataset
cancer = load_breast_cancer(as_frame=True)
classification_df = cancer.frame

# Save datasets into SQLite tables
regression_df.to_sql(
    "diabetes_data",
    conn,
    if_exists="replace",
    index=False
)

classification_df.to_sql(
    "cancer_data",
    conn,
    if_exists="replace",
    index=False
)

# Verify SQL queries
print("Regression dataset:")
print(pd.read_sql_query(
    "SELECT * FROM diabetes_data LIMIT 5",
    conn
))

print("\nClassification dataset:")
print(pd.read_sql_query(
    "SELECT * FROM cancer_data LIMIT 5",
    conn
))

conn.close()

print("\nDatabase created successfully!")