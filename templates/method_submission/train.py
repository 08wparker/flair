"""
FLAIR Training Script Template

This is a template for the train.py file required in FLAIR submissions.
Modify this file to implement your training logic.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(wide_dataset_path: str, labels_path: str):
    """Load and prepare data for training."""
    # Load datasets
    wide_df = pd.read_parquet(wide_dataset_path)
    labels_df = pd.read_parquet(labels_path)

    # Merge on hospitalization_id
    df = wide_df.merge(labels_df, on="hospitalization_id", how="inner")

    # Separate features and labels
    feature_cols = [c for c in wide_df.columns if c != "hospitalization_id"]
    X = df[feature_cols].values
    y = df["label"].values  # Adjust column name as needed

    return X, y, feature_cols


def train(
    wide_dataset_path: str,
    labels_path: str,
    output_dir: str,
    **kwargs,
) -> None:
    """
    Train model and save to output directory.

    Args:
        wide_dataset_path: Path to wide_dataset.parquet
        labels_path: Path to labels.parquet
        output_dir: Directory to save trained model
        **kwargs: Additional configuration
    """
    logger.info("Loading data...")
    X, y, feature_cols = load_data(wide_dataset_path, labels_path)

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Label distribution: {np.bincount(y.astype(int))}")

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train XGBoost model
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_model(output_path / "model.json")

    # Save feature names
    with open(output_path / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    # Save validation metrics
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]

    from sklearn.metrics import accuracy_score, roc_auc_score

    metrics = {
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_auroc": float(roc_auc_score(y_val, val_prob)),
    }

    with open(output_path / "val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model saved to {output_path}")
    logger.info(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FLAIR model")
    parser.add_argument("--wide-dataset", required=True, help="Path to wide_dataset.parquet")
    parser.add_argument("--labels", required=True, help="Path to labels.parquet")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()

    train(
        wide_dataset_path=args.wide_dataset,
        labels_path=args.labels,
        output_dir=args.output_dir,
    )
