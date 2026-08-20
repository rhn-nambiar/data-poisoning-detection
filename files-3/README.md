# License Plate Data-Poisoning / Anomaly Detection

A hybrid anomaly-detection pipeline for identifying invalid or poisoned
vehicle license plate records, combining domain-specific rule-based
validation with unsupervised machine learning.

## Overview

This project detects anomalous or "poisoned" license plate entries in a
dataset of Indian vehicle registration numbers using a two-stage approach:

1. **Rule-based filtering** — validates each plate against real Regional
   Transport Office (RTO) code ranges for every Indian state and union
   territory, flagging entries with invalid state codes, malformed series
   letters, digits appearing where letters should be, or RTO numbers
   outside a state's valid range.
2. **Statistical anomaly detection** — on the data that passes the
   rule-based filter, categorical fields are encoded and two unsupervised
   methods are applied:
   - **Isolation Forest** to flag statistical outliers
   - **DBSCAN** clustering to catch anomalies that fall outside
     high-density regions
   - Both scores are combined into a single weighted anomaly indicator

The pipeline is evaluated against a ground-truth dataset of known-valid
plates, reporting accuracy, precision, recall, and F1-score.

## How it works

- `data['state']`, `data['rto']`, `data['series']`, and `data['lnum']`
  are parsed directly from the plate string.
- `rule_based_detection()` applies six separate rule checks (invalid
  state code, invalid series, invalid characters in series, digits in
  series position, RTO number out of range, malformed plate number) and
  merges the results.
- Records that pass the rule-based filter are encoded with
  `LabelEncoder`, scaled with `StandardScaler`, and passed to both
  `IsolationForest` and `DBSCAN`.
- A combined anomaly score is computed as a weighted sum
  (0.2 × Isolation Forest flag + 0.8 × DBSCAN flag), and any record
  either caught by the rule-based filter or exceeding the combined
  score threshold is marked as an anomaly.
- Final predictions are compared against a ground-truth valid-plates
  dataset to compute accuracy, precision, recall, and F1-score.

## Data

This project uses two CSV inputs, both self-generated/simulated for
this project (not sourced from a real vehicle registry):

- `data/poisoned_combined4.csv` — a dataset of license plate records to
  be evaluated, containing a mix of valid and poisoned/anomalous
  entries
- `data/valid_license_plates.csv` — a ground-truth dataset of known-valid
  license plates, used to evaluate detection performance

The script reads both files directly from the `data/` folder, so it
runs as-is after cloning this repo.

## Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python anomaly_detection.py
```

The script prints all detected anomalies and final evaluation metrics
(accuracy, precision, recall, F1-score) to stdout.

## Results

- Accuracy: 0.9941
- Precision: 0.9858
- Recall: 1.0000
- F1-Score: 0.9929

(On a simulated dataset of 20,000 license plate records.)

## Notes

This was originally developed and run on Kaggle; paths have since been
updated to run locally against the `data/` folder in this repo.
