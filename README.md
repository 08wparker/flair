# FLAIR - Federated Learning Assessment for ICU Research

A privacy-first benchmark framework for evaluating ML/AI methods on ICU prediction tasks using the CLIF (Common Longitudinal ICU Format) data standard.

## Overview

FLAIR enables researchers to develop and evaluate prediction models across 17+ hospital sites using a federated approach:

1. **Develop** your method on MIMIC-CLIF (publicly available on PhysioNet)
2. **Submit** your code for evaluation
3. **PIs at each site** review and run your code on their private data
4. **Aggregated results** are shared back (no individual data leaves sites)

## Key Features

- **Privacy-First**: Network requests blocked, PHI detection, audit logging
- **CLIF-Native**: Uses `clifpy` for standardized wide dataset generation
- **4 Benchmark Tasks**: Discharged home, LTACH, 72hr outcome, hypoxic proportion
- **Shared Cohort**: All tasks use the same ICU cohort from tokenETL
- **TRIPOD-AI Compliant**: Standardized reporting following TRIPOD-AI guidelines

## Installation

```bash
# Install from source
pip install -e .

# Or with uv
uv pip install -e .
```

## Quick Start

### 1. Initialize Configuration

```bash
flair init --site my_hospital --output flair_config.yaml
```

Edit `flair_config.yaml` to configure paths for your site.

### 2. Build Datasets

```bash
# Build all tasks
flair build-datasets

# Or build specific task
flair build-datasets --task task1_discharged_home
```

### 3. Generate Table 1

```bash
flair table1 task1_discharged_home --format markdown
```

### 4. Validate Submission

```bash
flair validate /path/to/my_method
```

## Tasks

| Task | Type | Description | Cohort |
|------|------|-------------|--------|
| Task 1 | Binary | Discharged Home | All ICU (~62% positive) |
| Task 2 | Binary | Discharged to LTACH | All ICU (~3.6% positive) |
| Task 3 | Multi-class | 72hr Respiratory Outcome | IMV at 24hr |
| Task 4 | Regression | Hypoxic Failure Proportion | IMV at 24hr |

## Privacy Policy

```
╔════════════════════════════════════════════════════════════════╗
║                     FLAIR PRIVACY POLICY                        ║
╠════════════════════════════════════════════════════════════════╣
║ 1. NO NETWORK REQUESTS                                          ║
║    - All network access is blocked at Python socket level       ║
║    - Packages like requests, urllib3, httpx are banned          ║
║    - Violation = immediate submission rejection                 ║
║                                                                  ║
║ 2. PHI PROTECTION                                                ║
║    - All outputs scanned for PHI patterns                       ║
║    - Cell counts < 10 are suppressed                            ║
║    - Individual-level data never leaves the site                ║
║                                                                  ║
║ 3. REVIEW PROCESS                                                ║
║    - PIs at each site review code before execution              ║
║    - PIs have final say - they are not required to run          ║
║    - Code inspection for data exfiltration attempts             ║
║                                                                  ║
║ 4. CONSEQUENCES                                                  ║
║    - If data is found shared to internet during review:         ║
║      → Submitter is BANNED from FLAIR                           ║
║      → Incident reported to institution                         ║
╚════════════════════════════════════════════════════════════════╝
```

## Method Submission

### Required Files

```
my_method/
├── README.md           # Describe your method
├── requirements.txt    # Dependencies (see banned packages)
├── train.py           # Training script
└── predict.py         # Prediction script
```

### train.py Interface

```python
def train(
    wide_dataset_path: str,      # Path to wide_dataset.parquet
    labels_path: str,            # Path to labels.parquet
    output_dir: str,             # Directory to save trained model
    **kwargs
) -> None:
    pass
```

### predict.py Interface

```python
def predict(
    wide_dataset_path: str,      # Path to test wide_dataset.parquet
    model_path: str,             # Path to trained model
    output_path: str,            # Path to save predictions
    **kwargs
) -> None:
    pass
```

### Banned Packages

The following packages will cause your submission to be **rejected**:

- HTTP clients: `requests`, `urllib3`, `httpx`, `aiohttp`
- Network: `socket`, `websocket`, `paramiko`
- Protocols: `ftplib`, `smtplib`, `telnetlib`
- Cloud SDKs: `boto3`, `google.cloud`, `azure`

## Dataset Output Format

Each task produces three parquet files:

| File | Description |
|------|-------------|
| `wide_dataset.parquet` | Features from first 24 hours (via clifpy) |
| `labels.parquet` | Task-specific labels |
| `demographics.parquet` | Age, sex, race, ethnicity |

## CLI Commands

```bash
flair --help                          # Show all commands
flair build-datasets                  # Build all task datasets
flair validate /path/to/submission    # Validate submission
flair table1 task_name               # Generate Table 1
flair package-results                # Package results for submission
flair privacy-warning                # Show privacy policy
flair version                        # Show version
```

## Configuration

See `flair_config.yaml.template` for all options:

```yaml
site:
  name: "your_site_name"
  timezone: "US/Central"

data:
  clif_config_path: "clif_config.json"
  cohort_path: "OutputTokens/tokentables/cohort.parquet"

tasks:
  enabled:
    - task1_discharged_home
    - task2_discharged_ltach
    - task3_outcome_72hr
    - task4_hypoxic_proportion

privacy:
  enable_network_blocking: true
  enable_phi_detection: true
  min_cell_count: 10

# NOTE: No wandb_api_key - tracking must be offline only
```

## Helper Functions

### get_table1()

Generate HIPAA-compliant summary statistics:

```python
from flair.helpers import get_table1

table1 = get_table1(
    cohort_df=demographics,
    labels_df=labels,
    min_cell_count=10  # Suppress cells < 10
)
```

### generate_tripod_ai_report()

Generate TRIPOD-AI compliant reports:

```python
from flair.helpers import generate_tripod_ai_report

report = generate_tripod_ai_report(
    task_config=task_config,
    model_info={"model_type": "XGBoost"},
    results={"metrics": metrics},
    table1=table1
)

print(report.to_markdown())
```

## Architecture

```
FLAIR/
├── flair/                      # Main package
│   ├── config/                 # Configuration management
│   ├── privacy/                # Network blocking, PHI detection
│   ├── tasks/                  # Task definitions
│   ├── datasets/               # Dataset builder (clifpy integration)
│   ├── helpers/                # table1, tripod_ai, metrics
│   ├── submission/             # Validation, packaging
│   └── cli.py                  # Command-line interface
├── configs/tasks/              # Task configurations
├── templates/method_submission/ # Submission template
└── tests/                      # Test suite
```

## Integration

- **clifpy**: Uses `create_wide_dataset()` for feature generation
- **tokenETL**: Cohort from `cohort.parquet`
- **benchmark/**: Compatible with existing benchmark utilities

## License

MIT License

## Contact

- Website: [clif-icu.com](https://clif-icu.com)
- Email: clif_consortium@uchicago.edu
