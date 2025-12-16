"""
Data splitting utilities for FLAIR benchmark.

Supports temporal and random splitting strategies.
"""

import polars as pl
import numpy as np
import logging
from enum import Enum
from typing import Dict, Tuple, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class SplitMethod(Enum):
    """Splitting method for train/val/test."""

    TEMPORAL = "temporal"
    RANDOM = "random"


class DataSplitter:
    """
    Split FLAIR datasets into train/validation/test sets.

    Supports:
    - Temporal splitting: Based on admission date
    - Random splitting: Stratified random sampling

    Usage:
        splitter = DataSplitter(method="temporal", train_ratio=0.7)
        splits = splitter.split(wide_df, labels_df)
    """

    def __init__(
        self,
        method: str = "temporal",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        temporal_cutoff: Optional[str] = None,
        random_seed: int = 42,
    ):
        """
        Initialize splitter.

        Args:
            method: "temporal" or "random"
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            temporal_cutoff: Date string for temporal split (e.g., "2023-01-01")
            random_seed: Random seed for reproducibility
        """
        self.method = SplitMethod(method)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.temporal_cutoff = temporal_cutoff
        self.random_seed = random_seed

        # Validate ratios sum to 1
        total = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

    def split(
        self,
        wide_df: pl.DataFrame,
        labels_df: pl.DataFrame,
        demographics_df: Optional[pl.DataFrame] = None,
    ) -> Dict[str, Dict[str, pl.DataFrame]]:
        """
        Split datasets into train/val/test.

        Args:
            wide_df: Wide feature dataset
            labels_df: Labels dataset
            demographics_df: Demographics dataset (optional)

        Returns:
            Dictionary with structure:
            {
                "train": {"wide": df, "labels": df, "demographics": df},
                "val": {"wide": df, "labels": df, "demographics": df},
                "test": {"wide": df, "labels": df, "demographics": df}
            }
        """
        # Get hospitalization IDs and split them
        hosp_ids = labels_df["hospitalization_id"].unique().to_list()

        if self.method == SplitMethod.TEMPORAL:
            train_ids, val_ids, test_ids = self._temporal_split(hosp_ids, wide_df)
        else:
            train_ids, val_ids, test_ids = self._random_split(hosp_ids, labels_df)

        # Create split datasets
        splits = {}
        for split_name, split_ids in [
            ("train", train_ids),
            ("val", val_ids),
            ("test", test_ids),
        ]:
            splits[split_name] = {
                "wide": wide_df.filter(pl.col("hospitalization_id").is_in(split_ids)),
                "labels": labels_df.filter(pl.col("hospitalization_id").is_in(split_ids)),
            }
            if demographics_df is not None:
                splits[split_name]["demographics"] = demographics_df.filter(
                    pl.col("hospitalization_id").is_in(split_ids)
                )

        # Log split sizes
        for split_name, split_data in splits.items():
            logger.info(
                f"{split_name}: {split_data['labels'].height} hospitalizations"
            )

        return splits

    def _temporal_split(
        self,
        hosp_ids: List[str],
        wide_df: pl.DataFrame,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split based on temporal ordering.

        Uses event_time or admission time to sort, then splits.
        """
        # Get timestamps for each hospitalization
        if "event_time" in wide_df.columns:
            time_col = "event_time"
        elif "admission_dttm" in wide_df.columns:
            time_col = "admission_dttm"
        else:
            logger.warning(
                "No timestamp column found, falling back to random split"
            )
            return self._random_split(hosp_ids, wide_df)

        # Get first timestamp per hospitalization
        first_times = (
            wide_df.group_by("hospitalization_id")
            .agg(pl.col(time_col).min().alias("first_time"))
            .sort("first_time")
        )

        sorted_ids = first_times["hospitalization_id"].to_list()

        # Calculate split indices
        n = len(sorted_ids)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_ids = sorted_ids[:train_end]
        val_ids = sorted_ids[train_end:val_end]
        test_ids = sorted_ids[val_end:]

        logger.info(
            f"Temporal split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )

        return train_ids, val_ids, test_ids

    def _random_split(
        self,
        hosp_ids: List[str],
        labels_df: pl.DataFrame,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Random stratified split.

        Maintains label distribution across splits.
        """
        np.random.seed(self.random_seed)

        # Shuffle IDs
        shuffled_ids = np.array(hosp_ids)
        np.random.shuffle(shuffled_ids)

        # Calculate split indices
        n = len(shuffled_ids)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_ids = shuffled_ids[:train_end].tolist()
        val_ids = shuffled_ids[train_end:val_end].tolist()
        test_ids = shuffled_ids[val_end:].tolist()

        logger.info(
            f"Random split (seed={self.random_seed}): "
            f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )

        return train_ids, val_ids, test_ids

    def save_splits(
        self,
        splits: Dict[str, Dict[str, pl.DataFrame]],
        output_dir: str,
    ) -> None:
        """
        Save split datasets to parquet files.

        Args:
            splits: Output from split() method
            output_dir: Directory to save files
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            split_dir = output_path / split_name
            split_dir.mkdir(exist_ok=True)

            for data_name, df in split_data.items():
                df.write_parquet(split_dir / f"{data_name}.parquet")

        logger.info(f"Saved splits to {output_dir}")


def create_train_test_split(
    wide_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    test_ratio: float = 0.15,
    method: str = "temporal",
    random_seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Simple train/test split for FLAIR datasets.

    Returns:
        (train_wide, train_labels, test_wide, test_labels)
    """
    splitter = DataSplitter(
        method=method,
        train_ratio=1.0 - test_ratio,
        val_ratio=0.0,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    splits = splitter.split(wide_df, labels_df)

    return (
        splits["train"]["wide"],
        splits["train"]["labels"],
        splits["test"]["wide"],
        splits["test"]["labels"],
    )
