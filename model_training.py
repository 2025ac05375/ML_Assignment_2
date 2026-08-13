"""
Model Training Script

This module trains all classification models and saves them to the models directory.
Run this script to generate trained model artifacts that will be used by the Streamlit app.

Usage:
    python model_training.py
    python model_training.py --no-save  # Train without saving
"""

import argparse
from pathlib import Path

from training_utils import (
    build_models,
    evaluate_model,
    load_dataset_from_csv,
    scale_features,
    encode_target,
    save_artifacts_pkl,
    split_train_test,
    train_models,
)


def main(save_models: bool = True, models_dir: Path | None = None):
    """Main training pipeline execution.
    
    Args:
        save_models: Whether to save trained models to disk. Default is True.
        models_dir: Directory to save models. If None, uses 'models' subfolder.
    """
    print("=" * 80)
    print("Starting Model Training Pipeline")
    print("=" * 80)
    
    # Define models directory
    if models_dir is None:
        models_dir = Path(__file__).parent / "model"
    
    if save_models:
        print(f"\nModels will be saved to: {models_dir}")
    else:
        print("\nSave models: DISABLED (training only)")
    
    # Load dataset
    print("\n[Step 1/6] Loading dataset...")
    dataset_path = Path(__file__).parent / "dataset_full.csv"
    bundle = load_dataset_from_csv(dataset_path)
    print(f"Dataset loaded: {bundle.name}")
    print(f"Features shape: {bundle.X.shape}")
    print(f"Target shape: {bundle.y.shape}")

    # Scale features
    print("\n[Step 2/6] Scaling features...")
    X_scaled, scaler = scale_features(bundle.X)
    print("Features scaled successfully")

    # Convert target to numerical values for mushroom dataset
    print("\n[Step 2.5/6] Converting target to numerical values...")
    y_scaled, label_encoder = encode_target(bundle.y, save_encoder=True)
    print(f"Target encoding completed: {label_encoder.classes_}")

    # Split data
    print("\n[Step 3/6] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_train_test(
        X_scaled,
        y_scaled,
        test_size=0.2,
        random_state=42,
    )
    print(f"Train size: {X_train.shape[0]} samples")
    print(f"Test size: {X_test.shape[0]} samples")

    # Build models
    print("\n[Step 4/6] Building classification models...")
    models = build_models(random_state=42)
    print(f"Models created: {list(models.keys())}")
    
    # Train models
    print("\n[Step 5/6] Training models...")
    trained_models = train_models(models, X_train, y_train)
    print("All models trained successfully")
    
    # Evaluate models
    print("\n[Step 6/6] Evaluating models on test set...")
    results = {}
    for name, model in trained_models.items():
        print(f"  Evaluating {name}...")
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(f"    Accuracy: {metrics['Accuracy']:.4f}")
        print(f"    AUC: {metrics['AUC']:.4f}")
        print(f"    F1: {metrics['F1']:.4f}")
    
    # Save artifacts
    if save_models:
        print("\nSaving model artifacts...")
        save_artifacts_pkl(
            artifacts_dir=models_dir,
            trained_models=trained_models,
            metrics=results,
        )
        print(f"Artifacts saved to: {models_dir}")
    else:
        print("\nSkipping model save (save_models=False)")
    
    print("\n" + "=" * 80)
    print("Training Pipeline Completed Successfully!")
    print("=" * 80)
    
    if save_models:
        print("You can now run the Streamlit app with: streamlit run app.py")
    
    print("Finished main training pipeline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train classification models and optionally save them"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving models (train and evaluate only)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory to save models (default: ./model)",
    )
    
    args = parser.parse_args()
    
    main(save_models=not args.no_save, models_dir=args.models_dir)
