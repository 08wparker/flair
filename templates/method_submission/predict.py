"""
FLAIR Prediction Script Template

This is a template for the predict.py file required in FLAIR submissions.
Modify this file to implement your prediction logic.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict(
    wide_dataset_path: str,
    model_path: str,
    output_path: str,
    **kwargs,
) -> None:
    """
    Load model and generate predictions.

    Args:
        wide_dataset_path: Path to wide_dataset.parquet (test set)
        model_path: Path to trained model directory
        output_path: Path to save predictions
        **kwargs: Additional configuration
    """
    model_dir = Path(model_path)

    # Load model
    logger.info(f"Loading model from {model_dir}")
    model = xgb.XGBClassifier()
    model.load_model(model_dir / "model.json")

    # Load feature columns
    with open(model_dir / "feature_cols.json", "r") as f:
        feature_cols = json.load(f)

    # Load test data
    logger.info(f"Loading test data from {wide_dataset_path}")
    wide_df = pd.read_parquet(wide_dataset_path)

    # Extract features
    X_test = wide_df[feature_cols].values
    hosp_ids = wide_df["hospitalization_id"].values

    logger.info(f"Test data shape: {X_test.shape}")

    # Generate predictions
    logger.info("Generating predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        "hospitalization_id": hosp_ids,
        "prediction": y_pred,
        "probability": y_prob,
    })

    # Save predictions
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(output_file, index=False)

    logger.info(f"Predictions saved to {output_file}")
    logger.info(f"Total predictions: {len(predictions_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FLAIR predictions")
    parser.add_argument("--wide-dataset", required=True, help="Path to wide_dataset.parquet")
    parser.add_argument("--model-path", required=True, help="Path to model directory")
    parser.add_argument("--output", required=True, help="Path to save predictions")

    args = parser.parse_args()

    predict(
        wide_dataset_path=args.wide_dataset,
        model_path=args.model_path,
        output_path=args.output,
    )
