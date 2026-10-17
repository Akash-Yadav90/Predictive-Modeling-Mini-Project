# Predictive Modeling Mini-Project

## Overview

This project demonstrates an end-to-end predictive modeling
pipeline using SQL, Python, and scikit-learn.

It builds and evaluates one regression model and one
classification model using separate datasets.

## Technologies

- Python
- SQLite
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Datasets

### Regression
Diabetes dataset from scikit-learn.

Target: Disease progression.

Model: Linear Regression.

Metrics: RMSE and R².

### Classification
Breast Cancer Wisconsin dataset from scikit-learn.

Target: Benign or malignant tumor classification.

Model: Logistic Regression with feature standardization.

Metrics: Accuracy, precision, recall, and F1-score.

## SQL Pipeline

The datasets are stored in SQLite tables:

- diabetes_data
- cancer_data

SQL queries retrieve the records used for model training.

## Regression Results

- RMSE: 53.85
- R²: 0.4526

## Classification Results

- Accuracy: 98.25%
- Precision: 98.61%
- Recall: 98.61%
- F1-score: 98.61%

These metrics are based on the current test split.
They should be reproduced before final submission.

## Project Structure

- src/: Python source code
- data/: SQLite database
- models/: Saved models
- reports/: Evaluation results

## How to Run

Create and activate a Python virtual environment.

Install dependencies:

    python -m pip install -r requirements.txt

Create the SQLite database:

    python src/setup_database.py

Train the regression model:

    python src/regression_model.py

Train the classification model:

    python src/classification_model.py

Generate the comparison report:

    python src/compare_models.py

Run these commands from the project root.

## Model Comparison

Regression and classification solve different prediction tasks,
so their metrics cannot be compared directly.

RMSE and R² measure regression performance. Accuracy, precision,
recall, and F1-score measure classification performance.

## Limitations

The models use different datasets and target variables.
Their metric values therefore do not establish that one model
is universally better than the other.

The classification results should be interpreted in the context
of the test split and the limitations of the dataset.