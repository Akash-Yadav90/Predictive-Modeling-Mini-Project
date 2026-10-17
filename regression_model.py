import sqlite3
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Connect to SQLite database
conn = sqlite3.connect("data/predictive_models.db")

# Retrieve data using SQL
df = pd.read_sql_query(
    "SELECT * FROM diabetes_data",
    conn
)

conn.close()

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Split data before training
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("REGRESSION MODEL RESULTS")
print("------------------------")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Save trained model
joblib.dump(model, "models/regression_model.pkl")

# Save evaluation results
results = pd.DataFrame({
    "Metric": ["RMSE", "R2 Score"],
    "Value": [rmse, r2]
})

results.to_csv("reports/regression_results.csv", index=False)

print("\nModel saved successfully!")
print("Results saved to reports/regression_results.csv")