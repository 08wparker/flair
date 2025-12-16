"""Dataset generation for FLAIR benchmark.

Uses clifpy's create_wide_dataset() for feature generation.
"""

from flair.datasets.builder import FLAIRDatasetBuilder
from flair.datasets.splitter import DataSplitter, SplitMethod

__all__ = [
    "FLAIRDatasetBuilder",
    "DataSplitter",
    "SplitMethod",
]
