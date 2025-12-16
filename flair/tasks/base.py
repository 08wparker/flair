"""
Base task class for FLAIR benchmark.

All 4 tasks inherit from this base class and share the same cohort
from tokenETL (ICU hospitalizations).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import polars as pl
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Type of prediction task."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass"
    REGRESSION = "regression"


@dataclass
class TaskConfig:
    """Configuration for a FLAIR task."""

    name: str
    display_name: str
    description: str
    task_type: TaskType
    input_window_hours: int = 24
    prediction_window: Optional[Tuple[int, int]] = None
    cohort_filter: Optional[str] = None
    label_column: str = "label"
    positive_class: Optional[str] = None
    class_names: Optional[List[str]] = None
    evaluation_metrics: List[str] = None

    def __post_init__(self):
        if self.evaluation_metrics is None:
            if self.task_type == TaskType.BINARY_CLASSIFICATION:
                self.evaluation_metrics = [
                    "auroc",
                    "auprc",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "specificity",
                ]
            elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                self.evaluation_metrics = [
                    "accuracy",
                    "macro_f1",
                    "weighted_f1",
                    "per_class_f1",
                ]
            else:  # Regression
                self.evaluation_metrics = ["mse", "rmse", "mae", "r2"]


class BaseTask(ABC):
    """
    Abstract base class for FLAIR benchmark tasks.

    All tasks share the same cohort from tokenETL and use clifpy's
    create_wide_dataset() for feature generation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize task.

        Args:
            config: Task-specific configuration overrides
        """
        self._config_overrides = config or {}
        self._task_config = self._get_default_config()

        # Apply overrides
        task_settings = self._config_overrides.get("tasks", {}).get(self.name, {})
        if task_settings:
            if "input_window_hours" in task_settings:
                self._task_config.input_window_hours = task_settings["input_window_hours"]
            if "prediction_window" in task_settings:
                self._task_config.prediction_window = tuple(task_settings["prediction_window"])

    @property
    @abstractmethod
    def name(self) -> str:
        """Task identifier."""
        pass

    @abstractmethod
    def _get_default_config(self) -> TaskConfig:
        """Get default task configuration."""
        pass

    @property
    def config(self) -> TaskConfig:
        """Get task configuration."""
        return self._task_config

    @property
    def task_type(self) -> TaskType:
        """Get task type."""
        return self._task_config.task_type

    def filter_cohort(self, cohort_df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply task-specific cohort filters.

        Args:
            cohort_df: Full cohort DataFrame from tokenETL

        Returns:
            Filtered cohort for this task
        """
        # Base implementation: no filtering
        # Override in subclasses for task-specific filters
        return cohort_df

    @abstractmethod
    def build_labels(
        self,
        cohort_df: pl.DataFrame,
        narratives_dir: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Build labels DataFrame for this task.

        Args:
            cohort_df: Cohort DataFrame
            narratives_dir: Path to narratives directory (for extracting labels)

        Returns:
            DataFrame with columns: [hospitalization_id, label]
        """
        pass

    def build_demographics(self, cohort_df: pl.DataFrame) -> pl.DataFrame:
        """
        Extract demographics for the task cohort.

        Args:
            cohort_df: Cohort DataFrame

        Returns:
            DataFrame with demographic columns
        """
        # Select available demographic columns
        available_cols = ["hospitalization_id"]
        demographic_cols = [
            "patient_id",
            "age_at_admission",
            "sex_category",
            "race_category",
            "ethnicity_category",
        ]

        for col in demographic_cols:
            if col in cohort_df.columns:
                available_cols.append(col)

        return cohort_df.select(available_cols)

    def create_time_windowed_cohort(
        self,
        cohort_df: pl.DataFrame,
        input_window_hours: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Create cohort DataFrame with time windows for clifpy.

        Args:
            cohort_df: Cohort DataFrame with admission times
            input_window_hours: Hours of data to use (default: config value)

        Returns:
            pandas DataFrame with hospitalization_id, start_time, end_time
        """
        window_hours = input_window_hours or self._task_config.input_window_hours

        # Determine admission time column
        if "admission_dttm" in cohort_df.columns:
            time_col = "admission_dttm"
        elif "first_icu_time" in cohort_df.columns:
            time_col = "first_icu_time"
        else:
            raise ValueError("Cohort must have admission_dttm or first_icu_time column")

        # Create time windows
        result = cohort_df.select(
            [
                pl.col("hospitalization_id"),
                pl.col(time_col).alias("start_time"),
                (pl.col(time_col) + pl.duration(hours=window_hours)).alias("end_time"),
            ]
        )

        # Convert to pandas for clifpy compatibility
        return result.to_pandas()

    def get_evaluation_metrics(self) -> List[str]:
        """Get list of evaluation metrics for this task."""
        return self._task_config.evaluation_metrics

    def validate_labels(self, labels_df: pl.DataFrame) -> bool:
        """
        Validate labels DataFrame.

        Args:
            labels_df: Labels DataFrame to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        if "hospitalization_id" not in labels_df.columns:
            raise ValueError("Labels must have 'hospitalization_id' column")

        if self._task_config.label_column not in labels_df.columns:
            raise ValueError(
                f"Labels must have '{self._task_config.label_column}' column"
            )

        # Check for null labels
        null_count = labels_df.filter(
            pl.col(self._task_config.label_column).is_null()
        ).height

        if null_count > 0:
            logger.warning(f"Found {null_count} null labels")

        # Task-specific validation
        if self._task_config.task_type == TaskType.BINARY_CLASSIFICATION:
            unique_values = labels_df[self._task_config.label_column].unique().to_list()
            # Allow 0/1 or True/False
            valid_values = [{0, 1}, {True, False}, {0.0, 1.0}]
            if set(unique_values) not in valid_values and len(unique_values) != 2:
                logger.warning(
                    f"Binary classification labels have {len(unique_values)} unique values: {unique_values}"
                )

        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.task_type.value})"
