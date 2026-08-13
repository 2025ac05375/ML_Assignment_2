from __future__ import annotations

import pickle
from sklearn.preprocessing import LabelEncoder

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split as split_sample
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


DATASET_NAME_MUSHROOM = "Mushroom Classification"

DATASET_NAME = DATASET_NAME_MUSHROOM


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: pd.Series
    target_name: str
    feature_names: list[str]


def load_dataset_from_csv(csv_path: Path) -> DatasetBundle:
    """Load dataset from a CSV file.

    Args:
        csv_path: Path to the CSV file.
    Returns:
        DatasetBundle containing the loaded dataset.
    """
    print(f"Loading dataset from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV file is empty.")

    print(f"\nDataset Preview:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return DatasetBundle(
        name=DATASET_NAME,
        X=X,
        y=y,
        target_name=y.name if y.name is not None else "target",
        feature_names=list(X.columns),
    )


def load_dataset(sample_size: float = 0.00) -> DatasetBundle:
    """Load the checked-in assignment dataset.

    Args:
        sample_size: Percentage of data to save as sample test set.

    Returns:
        DatasetBundle containing the loaded dataset.
    """
    dataset_path = Path(__file__).parent / "dataset_full.csv"
    bundle = load_dataset_from_csv(dataset_path)
    x = bundle.X.copy()
    y = bundle.y.to_frame()

    # Handle missing values
    fill_missing_values(x, y)

    # Save sample for testing and remove from original dataset only if sample_size > 0
    if sample_size > 0:
        print(f"\n[Step 1.5/6] Saving {sample_size*100}% sample as test set...")

        # Split to get sample
        X_remaining, X_sample, y_remaining, y_sample = train_test_split(
            x, y, test_size=sample_size, random_state=42, stratify=y
            )

        # Create DataFrame with sample data
        sample_df = pd.DataFrame(X_sample, columns=list(x.columns))
        sample_df[y.columns[0]] = y_sample.iloc[:, 0]

        # Save sample to CSV
        sample_path = Path(__file__).parent / "test_data.csv"
        sample_df.to_csv(sample_path, index=False)
        print(f"Sample saved to: {sample_path}")
        print(f"Sample size: {X_sample.shape[0]} samples ({sample_size*100}%)")
        print(f"Remaining dataset: {X_remaining.shape[0]} samples ({(1-sample_size)*100}%)")
    else:
        X_remaining = x
        y_remaining = y
        print("\nNo sample extraction (sample_size = 0)")

    return DatasetBundle(
        name=DATASET_NAME,
        X=X_remaining,
        y=y_remaining.iloc[:,0],
        target_name=y_remaining.columns[0],
        feature_names=list(x.columns),
    )

def fill_missing_values(x, y):
    print("\nChecking for missing values...")
    print(f"Missing values in features:\n{x.isnull().sum()}")
    print(f"Missing values in target:\n{y.isnull().sum()}")

    if x.isnull().any().any() or y.isnull().any().any():
        print("\nFilling missing values...")
        
        # Fill missing values in features based on column type
        for col in x.columns:
            if x[col].isnull().any():
                if x[col].dtype == 'object':
                    # Categorical: fill with mode
                    mode_value = x[col].mode()[0] if not x[col].mode().empty else 'Unknown'
                    x[col].fillna(mode_value, inplace=True)
                    print(f"  Filled {col} (categorical) with mode: {mode_value}")
                else:
                    # Numerical: fill with median
                    median_value = x[col].median()
                    x[col].fillna(median_value, inplace=True)
                    print(f"  Filled {col} (numerical) with median: {median_value}")
        
        # Fill missing values in target with mode
        if y.isnull().any().any():
            mode_value = y.iloc[:, 0].mode()[0]
            y.fillna(mode_value, inplace=True)
            print(f"  Filled target with mode: {mode_value}")
        
        print("Finished filling missing values")
        print(f"Missing values in features:\n{x.isnull().sum()}")
        print(f"Missing values in target:\n{y.isnull().sum()}")
    else:
        print("No missing values found")


def scale_features(X: pd.DataFrame) -> tuple[pd.DataFrame, BaseEstimator]:
    """Scale features using StandardScaler.
    
    Args:
    X: Features to fit and transform.
    
    Returns:
    Tuple of (scaled X, fitted scaler).
    """
    print("Scaling features...")
    # Identify categorical and numerical columns
    categorical_cols = [
        col
        for col in X.columns
        if (
            is_object_dtype(X[col])
            or is_string_dtype(X[col])
            or isinstance(X[col].dtype, pd.CategoricalDtype)
        )
    ]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    
    # Create transformers for each type
    transformers = []
    
    # Categorical columns - one-hot encode
    if categorical_cols:
        transformers.append(('categorical', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols))
    
    # Numerical columns - standard scale
    if numerical_cols:
        transformers.append(('numerical', StandardScaler(), numerical_cols))
    
    # Apply transformations
    scaler = ColumnTransformer(transformers=transformers, remainder='passthrough')
    X_scaled = scaler.fit_transform(X)

    # Save the scaler for later use during prediction
    print("Saving scaler to disk...")
    scaler_path = Path(__file__).parent / "model" / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    
    # Get feature names after transformation
    feature_names = []
    if categorical_cols:
        feature_names.extend(scaler.named_transformers_['categorical'].get_feature_names_out(categorical_cols))
    if numerical_cols:
        feature_names.extend(numerical_cols)

    # Convert back to DataFrame to preserve column names
    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=feature_names,
        index=X.index if len(X_scaled) == len(X) else None,
    )
    
    print("Finished scaling features")
    print("\nScaled Features Preview:")
    print(X_scaled_df.head())
    print(X_scaled_df.describe(include='all'))
    print(f"Scaled feature columns ({len(X_scaled_df.columns)}): {list(X_scaled_df.columns)}")
    # Write scaled data to CSV
    # scaled_data_path = Path(__file__).parent / "scaled_features.csv"
    # X_scaled_df.to_csv(scaled_data_path, index=False)
    # print(f"Scaled features saved to: {scaled_data_path}")
    return X_scaled_df, scaler


def scale_test_features(X: pd.DataFrame, scaler: BaseEstimator) -> pd.DataFrame:
    """Scale test features using a fitted scaler.
    
    Args:
    X: Features to transform.
    scaler: Fitted scaler.
    
    Returns:
    Scaled X.
    """
    print("Scaling test features...")
    X_scaled = scaler.transform(X)

    # Get feature names after transformation
    feature_names = []
    for transformer_name, transformer, columns in scaler.transformers_:
        if transformer_name == "categorical":
            feature_names.extend(transformer.get_feature_names_out(columns))
        elif transformer_name == "numerical":
            feature_names.extend(columns)

    # Convert back to DataFrame to preserve column names
    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=feature_names,
        index=X.index if len(X_scaled) == len(X) else None,
    )

    print("Finished scaling test features")
    return X_scaled_df


def encode_target(y: pd.Series, encoder: LabelEncoder = None, save_encoder: bool = False) -> tuple[pd.Series, LabelEncoder]:
    """Encode target labels using LabelEncoder.
    
    Args:
    y: Target labels to encode.
    
    Returns:
    Tuple of (encoded y, fitted label encoder).
    """
    print("Encoding target labels...")
    if encoder is None:
        label_encoder = LabelEncoder()
        y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index, name=y.name)
    else:
        label_encoder = encoder
        y_encoded = pd.Series(label_encoder.transform(y), index=y.index, name=y.name)
    print(f"Target classes: {label_encoder.classes_}")
    print(f"Target mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    print("Finished encoding target labels")

    # Save the encoder for later use during prediction if requested
    if save_encoder:
        print("Saving label encoder to disk...")
        encoder_path = Path(__file__).parent / "model" / "label_encoder.pkl"
        encoder_path.parent.mkdir(parents=True, exist_ok=True)
        with open(encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)
        print(f"Label encoder saved to: {encoder_path}")
    return y_encoded, label_encoder


def split_train_test(
        X: pd.DataFrame,
        y: pd.Series,
        *,
        test_size: float = 0.2,
        random_state: int = 42,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    result = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print("Finished splitting train/test data")
    return result


def _auc_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    unique = np.unique(y_true)
    if unique.size == 2:
        positive_class_index = 1
        result = float(roc_auc_score(y_true, y_proba[:, positive_class_index]))
    else:
        result = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
    print("Finished calculating AUC score")
    return result


def build_models(*, random_state: int = 42) -> dict[str, BaseEstimator]:
    print("Building classification models...")
    models: dict[str, BaseEstimator] = {}

    models["Logistic Regression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )

    models["Decision Tree"] = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=None,
    )

    models["kNN"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]
    )

    models["Naive Bayes"] = GaussianNB()

    models["Random Forest (Ensemble)"] = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )

    print("Finished building classification models")
    return models


def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    print(f"Evaluating model: {model}")
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        if not hasattr(model, "decision_function"):
            raise ValueError("Model has neither predict_proba nor decision_function for AUC.")
        scores = model.decision_function(X_test)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        y_proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    y_true = y_test.to_numpy()
    y_pred_np = np.asarray(y_pred)

    result = {
        "Accuracy": float(accuracy_score(y_true, y_pred_np)),
        "AUC": _auc_score(y_true, np.asarray(y_proba)),
        "Precision": float(precision_score(y_true, y_pred_np, average="binary")),
        "Recall": float(recall_score(y_true, y_pred_np, average="binary")),
        "F1": float(f1_score(y_true, y_pred_np, average="binary")),
        "MCC": float(matthews_corrcoef(y_true, y_pred_np)),
        "ConfusionMatrix": confusion_matrix(y_true, y_pred_np).tolist(),
        "ClassificationReport": classification_report(y_true, y_pred_np, output_dict=True),
    }
    print("Finished evaluating model")
    return result


def train_models(models: dict[str, BaseEstimator], X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, BaseEstimator]:
    print("Training models...")
    trained: dict[str, BaseEstimator] = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        trained[name] = model.fit(X_train, y_train)
    print("Finished training models")
    return trained


def metrics_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    print("Creating metrics metrics_table...")
    rows: list[dict[str, Any]] = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
            row[col] = metrics[col]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    print("Finished creating metrics metrics_table")
    return df


def save_artifacts_pkl(
        *,
        artifacts_dir: Path,
        trained_models: dict[str, BaseEstimator],
        metrics: dict[str, dict[str, Any]],
    ) -> None:
    print("Saving artifacts (pickle)...")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, model in trained_models.items():
        safe = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )
        with open(artifacts_dir / f"{safe}.pkl", "wb") as f:
            pickle.dump(model, f)

    df = metrics_table(metrics)
    print("\nEvaluation metrics (test split):\n")
    print(df.round(4).to_string())

    df.to_csv(artifacts_dir / "metrics.csv")
    print("Finished saving artifacts (pickle)")


def load_artifacts(artifacts_dir: Path) -> tuple[dict[str, BaseEstimator], pd.DataFrame] | None:
    metrics_path = artifacts_dir / "metrics.csv"
    if not metrics_path.exists():
        print("No artifacts found to load")
        return None

    metrics_df = pd.read_csv(metrics_path, index_col=0)

    models: dict[str, BaseEstimator] = {}
    for model_name in metrics_df.index.tolist():
        safe = (
            model_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )
        path = artifacts_dir / f"{safe}.pkl"
        if not path.exists():
            print(f"Artifact file {path} not found")
            return None
        with open(path, "rb") as f:
            models[model_name] = pickle.load(f)
    print("Finished loading artifacts")
    return models, metrics_df

def run_training_pipeline(test_size=0.2, random_state=42, save_models=False) -> None:
    """Execute the complete model training and evaluation pipeline."""
    bundle = load_dataset()
    print(f"\nDataset loaded: {bundle.name}")
    X_scaled, _ = scale_features(bundle.X)
    y_encoded, _ = encode_target(bundle.y, save_encoder=save_models)
    X_train, X_test, y_train, y_test = split_train_test(
        X_scaled,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
    )

    models = build_models(random_state=random_state)
    trained = train_models(models, X_train, y_train)

    results = {name: evaluate_model(model, X_test, y_test) for name, model in trained.items()}
    metrics_tbl = metrics_table(results)
    print("\nEvaluation metrics (test split):\n")
    print(metrics_tbl.round(4).to_string())

    if save_models:
        artifacts_dir = Path(__file__).parent / "model"
        # save_artifacts(artifacts_dir=artifacts_dir, trained_models=trained, metrics=results)
        save_artifacts_pkl(artifacts_dir=artifacts_dir, trained_models=trained, metrics=results)
        print(f"\nSaved artifacts to: {artifacts_dir}\n")

    return trained, metrics_tbl
