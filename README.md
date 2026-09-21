# BAN6440 Final Project

## End-to-End Machine Learning Architecture & Interoperability Proposal

**Student:** Emmanuel Fru  
**Course:** BAN6440 Machine Learning  
**Project:** Telecom Churn Retention Decision System  
**Repository:** https://github.com/fruStoic/BAN6440_Final_Project

---

## 1. Project Overview

This project develops an end-to-end machine learning decision system for telecommunications customer churn.

The objective is not only to predict whether a customer is likely to churn, but to determine when the estimated churn probability is high enough to justify a retention intervention.

The project extends the earlier Module 6 churn-classification work by adding:

- a fresh development and holdout protocol;
- repeated cross-validation;
- model comparison;
- probability calibration;
- protected-attribute ablation;
- fairness analysis;
- an economically derived intervention threshold;
- business-impact analysis;
- a production deployment architecture;
- model governance and monitoring;
- reproducibility controls.

The final selected model is **logistic regression** with **sigmoid calibration** and a fixed operational threshold of **0.40**.

---

## 2. Dataset

The project uses:

```text
data/teleconnect.csv
```

### Dataset summary

| Item | Value |
|---|---:|
| Total rows | 7,043 |
| Total columns | 21 |
| Churn customers | 1,869 |
| Churn prevalence | 26.54% |
| Development rows | 5,634 |
| Final holdout rows | 1,409 |

The final project uses a fixed stratified 80/20 split with:

```text
Random seed: 20260920
Development: 80%
Holdout:     20%
```

The split assignments are saved permanently and are not regenerated during the workflow.

---

## 3. Final Model

| Component | Final specification |
|---|---|
| Model | Logistic Regression |
| Calibration | Sigmoid |
| Operational threshold | 0.40 |
| Raw predictive inputs | 17 |
| Encoded features | 28 |
| Primary selection metric | PR-AUC / Average Precision |
| Deployment mode | Weekly batch scoring |

The following variables are **excluded from predictive inputs**:

```text
gender
SeniorCitizen
```

They are retained separately for fairness auditing.

---

## 4. Project Structure

```text
BAN6440_Final_Project/
│
├── data/
│   └── teleconnect.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── create_final_project_split.py
│   ├── data_audit.py
│   ├── preprocessing.py
│   ├── preprocessing_check.py
│   ├── model_comparison.py
│   ├── calibration.py
│   ├── demographic_ablation.py
│   ├── final_feature_calibration.py
│   ├── fairness_analysis.py
│   ├── freeze_pipeline.py
│   ├── holdout_preflight.py
│   ├── final_evaluation.py
│   └── final_business_postprocess.py
│
├── tests/
│   └── test_pipeline.py
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   ├── protocol_lock.json
│   ├── split_manifest.json
│   ├── split_assignments.csv
│   ├── pipeline_freeze.json
│   ├── final_holdout_results.json
│   └── final_business_impact_operational.json
│
├── docs/
│   └── model_card.md
│
├── report/
│
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

---

## 5. Environment Setup

The project was developed with **Python 3.11.9**.

### Create a virtual environment

```powershell
python -m venv .venv
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

For exact environment reproduction:

```powershell
pip install -r requirements-lock.txt
```

---

## 6. Main Dependencies

The project uses:

```text
pandas
numpy
scikit-learn
scipy
matplotlib
tensorflow
joblib
pytest
```

Exact installed package versions are stored in:

```text
requirements-lock.txt
```

---

## 7. Reproducibility Workflow

The project was executed in a controlled sequence.

### Step 1: Create the permanent project split

```powershell
python -m src.create_final_project_split
```

Creates:

```text
outputs/protocol_lock.json
outputs/split_manifest.json
outputs/split_assignments.csv
```

> Do not regenerate the split after creation.

---

### Step 2: Audit development data

```powershell
python -m src.data_audit
```

Produces development-only data-quality and IQR audit evidence.

---

### Step 3: Validate preprocessing

```powershell
python -m src.preprocessing_check
```

Checks:

- feature counts;
- encoded schema;
- numeric scaling;
- missing values;
- finite transformed values.

---

### Step 4: Compare candidate models

```powershell
python -m src.model_comparison
```

Models compared:

```text
Majority baseline
Logistic Regression
HistGradientBoosting
Module 6 ANN
```

Model selection uses mean PR-AUC with a pre-registered practical-tie rule.

---

### Step 5: Evaluate calibration

```powershell
python -m src.calibration
```

Sigmoid calibration was specified before results were observed.

---

### Step 6: Run protected-attribute ablation

```powershell
python -m src.demographic_ablation
```

Compares logistic regression with and without:

```text
gender
SeniorCitizen
```

---

### Step 7: Evaluate the final reduced feature set

```powershell
python -m src.final_feature_calibration
```

---

### Step 8: Run the development fairness audit

```powershell
python -m src.fairness_analysis
```

Fairness is assessed separately for:

```text
gender
SeniorCitizen
```

---

### Step 9: Freeze the pipeline

```powershell
python -m src.freeze_pipeline
```

This records the final:

- model;
- feature policy;
- calibration method;
- operational threshold.

---

### Step 10: Run automated tests

```powershell
pytest -q
```

Final pre-holdout result:

```text
11 passed
```

---

### Step 11: Run holdout preflight

```powershell
python -m src.holdout_preflight
```

The preflight confirms that:

- no previous final holdout evaluation exists;
- required frozen artifacts are present;
- artifact hashes are unchanged;
- the expected 5,634 / 1,409 split is intact;
- the final feature policy is unchanged.

---

### Step 12: Run final holdout evaluation

This command is intended to be run **once only**:

```powershell
python -m src.final_evaluation
```

The script creates a holdout-open marker before loading the final holdout.

> No model tuning, feature selection, calibration-method selection, or threshold optimization is permitted after this stage.

---

### Step 13: Recompute operational business output at 0.40

The original final-evaluation business summary used the exact economic threshold of 0.398860, while the pre-registered operational threshold was 0.40.

The holdout was **not rerun**.

The corrected operational result was calculated directly from the saved immutable holdout predictions:

```powershell
python -m src.final_business_postprocess
```

The corrected result is saved in:

```text
outputs/final_business_impact_operational.json
```

---

## 8. Final Holdout Results

At the fixed 0.40 threshold:

| Metric | Result |
|---|---:|
| Accuracy | 0.7899 |
| Precision | 0.5942 |
| Recall | 0.6578 |
| F1 | 0.6244 |
| PR-AUC | 0.6484 |
| ROC-AUC | 0.8470 |
| Brier score | 0.1356 |

### Confusion matrix

```text
TN = 867
FP = 168
FN = 128
TP = 246
```

### Selected customers

```text
414 / 1,409
29.38%
```

---

## 9. Business Decision Rule

A customer is selected for intervention when:

```text
calibrated churn probability >= 0.40
```

The exact economic break-even threshold is:

```text
0.398860
```

The pre-registered operational threshold is:

```text
0.40
```

### Base-case assumptions

| Assumption | Value |
|---|---:|
| Contact cost | $2 |
| Offer cost | $65 |
| Offer acceptance probability | 40% |
| Incremental retention uplift | 15% |
| Contribution margin rate | 30% |
| Monthly revenue proxy | $65 |
| Value horizon | 24 months |

These values are **planning assumptions**, not dataset facts.

### Operational holdout result

| Business measure | Result |
|---|---:|
| Selected customers | 414 |
| Selected share | 29.38% |
| Expected prevented churn | 36.48 |
| Campaign cost | $11,592.00 |
| Expected benefit | $17,073.27 |
| Expected net value | $5,481.27 |
| Expected ROI | 47.28% |

These are **model-based expected values**, not realized financial outcomes.

---

## 10. Key Output Artifacts

### Protocol and model decisions

```text
outputs/protocol_lock.json
outputs/split_manifest.json
outputs/pipeline_freeze.json
```

### Model evaluation

```text
outputs/final_holdout_results.json
outputs/tables/final_model_comparison.csv
outputs/tables/final_metric_confidence_intervals.csv
```

### Holdout predictions

```text
outputs/tables/final_holdout_predictions.csv
```

### Fairness analysis

```text
outputs/tables/final_holdout_fairness.csv
```

### Business analysis

```text
outputs/tables/business_sensitivity_grid.csv
outputs/final_business_impact.json
outputs/final_business_impact_operational.json
```

### Final model artifact

```text
outputs/models/final_model_bundle.joblib
```

### Figures

```text
outputs/figures/final_confusion_matrix.png
outputs/figures/final_roc_comparison.png
outputs/figures/final_pr_comparison.png
outputs/figures/final_holdout_reliability.png
```

---

## 11. Evaluation Restrictions

The final holdout must not be used for:

- hyperparameter tuning;
- feature selection;
- threshold optimization;
- calibration-method selection;
- model selection.

The holdout has already been consumed for final evaluation.

Any future model changes require either:

1. a new validation protocol, or
2. genuinely new data.

---

## 12. Known Limitations

The project has several important limitations:

- The final holdout is a fresh internal split, but the underlying dataset had already been studied during Module 6.
- The evaluation is therefore not equivalent to external validation.
- The dataset contains no timestamp field, so temporal generalization cannot be tested.
- The assumed 15% intervention uplift is not estimated from the dataset.
- The model estimates churn risk, not causal treatment effect.
- Expected ROI is sensitive to intervention uplift and customer-value assumptions.
- Group-level metrics, especially for `SeniorCitizen`, require continued monitoring.
- Production performance may change as customer behavior, pricing, campaigns, or market conditions change.

---

## 13. Responsible Use

The model is intended as a **retention decision-support tool**.

It should not independently:

- change customer prices;
- alter customer contracts;
- deny or restrict services;
- make credit decisions;
- generate unrestricted financial offers;
- make causal claims about intervention effectiveness.

Campaign outcomes and treatment exposure should be recorded so that future analysis can distinguish natural churn from intervention-driven outcomes.

---

## 14. Production Monitoring

Recommended monitoring includes:

```text
Input schema and missing values
Feature drift
Prediction drift
Churn prevalence
Overall Brier score
Group-level calibration
Precision and recall
False-positive and false-negative rates
Campaign selection rate
Batch completion
Score freshness
Model version
Treatment status
Campaign response
Observed churn outcome
```

A monitoring alert should trigger investigation rather than automatic retraining.

---

## 15. Supporting Documentation

The project submission includes:

- final report;
- architecture diagram;
- Model Card;
- AI Usage Disclosure;
- model evaluation tables;
- business analysis;
- fairness evidence;
- source code;
- environment requirements.

---

## 16. Repository

Public GitHub repository:

**https://github.com/fruStoic/BAN6440_Final_Project**
