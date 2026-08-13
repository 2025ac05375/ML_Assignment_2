# ML Classification Models + Streamlit Demo

## Problem statement
Build, compare, and deploy multiple machine learning classification models on one public classification dataset. The Streamlit application allows a user to upload test data, choose a trained model, and view predictions, evaluation metrics, a confusion matrix, and a classification report.

## Dataset description
**Dataset:** Mushroom Classification  
**Original source:** UCI Machine Learning Repository  
**Task:** Binary classification to predict whether a mushroom is edible (`e`) or poisonous (`p`) from categorical mushroom attributes.

The checked-in dataset file is `dataset_full.csv`. It contains 8,124 records, 22 feature columns, and one target column named `poisonous`, satisfying the assignment requirement of at least 12 features and 500 instances. The test upload file used for app verification is `test_data.csv`.

## GitHub Repository Link
https://github.com/2025ac05375/ML_Assignment_2

## Models used
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbor Classifier
- Naive Bayes Classifier - GaussianNB
- Random Forest (Ensemble)

### Comparison table
Metrics below are calculated on `test_data.csv`, the test data file used for Streamlit app verification.

ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC
---|---:|---:|---:|---:|---:|---:
Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000
Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000
kNN | 0.9975 | 1.0000 | 1.0000 | 0.9949 | 0.9974 | 0.9951
Naive Bayes | 0.9509 | 0.9960 | 0.9112 | 0.9949 | 0.9512 | 0.9054
Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000

### Observations
ML Model Name | Observation about model performance
---|---
Logistic Regression | Achieved perfect scores on the test data after one-hot encoding the categorical features, showing that the classes are close to linearly separable in the transformed feature space.
Decision Tree | Achieved perfect scores on the test data and captured the rule-like structure of the categorical mushroom attributes.
kNN | Performed almost perfectly, with only a small recall loss compared with the top models.
Naive Bayes | Had the lowest accuracy and MCC among the five models, likely because its feature-independence assumption is less suitable for related categorical mushroom attributes.
Random Forest (Ensemble) | Tied with Logistic Regression and Decision Tree on the test-data metrics. It is preferred as the final model because combining many trees usually gives better robustness than a single tree.
Overall Winner for your dataset? | Logistic Regression, Decision Tree, and Random Forest tie on the observed test-data metrics; Random Forest is selected as the preferred final model for robustness.

## Repository structure
```text
ML_Assignment_2/
|-- app.py
|-- model_training.py
|-- training_utils.py
|-- requirements.txt
|-- README.md
|-- dataset_full.csv
|-- test_data.csv
|-- model/
|   |-- logistic_regression.pkl
|   |-- decision_tree.pkl
|   |-- knn.pkl
|   |-- naive_bayes.pkl
|   |-- random_forest_ensemble.pkl
|   |-- scaler.pkl
|   |-- label_encoder.pkl
|   |-- metrics.csv
```

## How to run
1. Install dependencies:
```bash
cd ML_Assignment_2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train models and regenerate saved artifacts:
```bash
python model_training.py
```

3. Start the Streamlit app:
```bash
streamlit run app.py
```

4. Quick demo upload:
- Use `test_data.csv`.
- Keep "CSV includes target column" checked.
- Select `poisonous` as the target column if prompted.

## Streamlit app features
- Dataset upload option for CSV test data
- Model selection dropdown with the five required models
- Display of Accuracy, AUC, Precision, Recall, F1, and MCC
- All-model comparison table after uploading test data with target labels
- Confusion matrix and classification report for uploaded test data with target labels
