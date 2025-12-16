# FLAIR Method Submission Template

This template provides the structure for submitting a method to the FLAIR benchmark.

## Directory Structure

```
my_method/
├── README.md           # This file - describe your method
├── requirements.txt    # Python dependencies (see restrictions below)
├── train.py           # Training script
├── predict.py         # Prediction script
└── config.yaml        # Optional: method configuration
```

## Required Files

### train.py

Your training script must accept the following arguments:

```python
def train(
    wide_dataset_path: str,      # Path to wide_dataset.parquet
    labels_path: str,            # Path to labels.parquet
    output_dir: str,             # Directory to save trained model
    **kwargs                     # Additional configuration
) -> None:
    """Train your model and save to output_dir."""
    pass
```

### predict.py

Your prediction script must accept the following arguments:

```python
def predict(
    wide_dataset_path: str,      # Path to wide_dataset.parquet (test set)
    model_path: str,             # Path to trained model
    output_path: str,            # Path to save predictions
    **kwargs
) -> None:
    """Load model and generate predictions."""
    pass
```

## CRITICAL: Package Restrictions

The following packages are **BANNED** and will cause your submission to be rejected:

- `requests`, `urllib3`, `httpx`, `aiohttp` - HTTP clients
- `socket`, `websocket`, `websockets` - Direct network access
- `paramiko`, `fabric` - SSH clients
- `ftplib`, `smtplib`, `telnetlib` - Protocol clients
- `boto3`, `google.cloud`, `azure` - Cloud SDKs

**All network access is blocked at the Python socket level.**

## Allowed Dependencies

Common ML/data science packages are allowed:
- `numpy`, `pandas`, `polars`
- `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- `torch`, `tensorflow` (local execution only)
- `scipy`, `statsmodels`

## Privacy Requirements

1. **No Network Requests**: Your code cannot make any network requests
2. **No Data Exfiltration**: Do not write patient data to unauthorized locations
3. **Aggregated Outputs Only**: Return metrics, not individual predictions
4. **No Hardcoded Data**: Do not embed or print patient identifiers

## Validation

Before submission, validate your code:

```bash
flair validate /path/to/my_method
```

## Example Submission

See the example files in this template for a working XGBoost classifier.
