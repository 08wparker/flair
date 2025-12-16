"""
FLAIR: Federated Learning Assessment for ICU Research

A privacy-first benchmark framework for evaluating ML/AI methods on ICU prediction
tasks using the CLIF (Common Longitudinal ICU Format) data standard.

Key Features:
- Privacy-first: Network requests blocked, PHI detection, review process
- CLIF-native: Uses clifpy's create_wide_dataset() for feature generation
- Federated: Methods developed on MIMIC-CLIF, evaluated at 17+ sites
- 4 Tasks: Discharged home, LTACH, 72hr outcome, hypoxic proportion
"""

__version__ = "1.0.0"
__author__ = "CLIF Consortium"
__email__ = "clif_consortium@uchicago.edu"

from flair.privacy.network_guard import NetworkBlocker, network_blocked
from flair.helpers.table1 import get_table1
from flair.helpers.tripod_ai import generate_tripod_ai_report

__all__ = [
    "__version__",
    "NetworkBlocker",
    "network_blocked",
    "get_table1",
    "generate_tripod_ai_report",
]
