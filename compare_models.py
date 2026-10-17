import pandas as pd

# Load regression evaluation results
regression = pd.read_csv("reports/regression_results.csv")

# Load classification evaluation results
classification = pd.read_csv("reports/classification_results.csv")

# Display individual model results
print("=" * 50)
print("REGRESSION MODEL RESULTS")
print("=" * 50)
print(regression.to_string(index=False))

print("\n" + "=" * 50)
print("CLASSIFICATION MODEL RESULTS")
print("=" * 50)
print(classification.to_string(index=False))

# Save combined results
regression["Model"] = "Linear Regression"
classification["Model"] = "Logistic Regression"

comparison = pd.concat(
    [regression, classification],
    ignore_index=True
)

comparison = comparison[["Model", "Metric", "Value"]]

comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nComparison report saved successfully!")
print("Location: reports/model_comparison.csv")