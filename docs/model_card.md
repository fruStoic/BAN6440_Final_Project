# Model Card: Telecom Churn Retention Decision Model

## 1. Model Overview

| Field | Value |
|---|---|
| Model name | Telecom Churn Retention Decision Model |
| Model type | Logistic Regression |
| Task | Binary churn-risk estimation |
| Positive class | Customer churn |
| Final operating threshold | 0.40 calibrated churn probability |
| Probability calibration | Sigmoid calibration |
| Deployment mode | Weekly batch scoring |
| Development seed | 20260920 |
| Predictive inputs | 17 raw predictors, 28 encoded features |
| Excluded predictive variables | `gender`, `SeniorCitizen` |
| Primary development metric | PR-AUC / Average Precision |

The model estimates the probability that a telecommunications customer will churn. It is designed to support a retention-campaign decision rather than simply produce a binary churn classification.

A customer is selected for intervention when the calibrated churn probability is at least **0.40**, based on the pre-defined economic assumptions used in the project.

---

## 2. Intended Use

The intended use is to support a telecommunications retention team in identifying customers for whom a retention intervention is expected to be economically justified.

The model supports the decision:

> Should this customer receive a retention intervention under the current campaign assumptions?

The model is intended for scheduled batch campaigns rather than real-time transactional decisions.

It should be used as a **decision-support system**, with campaign policy, offer design, and customer communication remaining under human oversight.

---

## 3. Out-of-Scope Uses

The model should not be used to:

- automatically terminate, restrict, or change customer services;
- determine customer creditworthiness;
- autonomously change prices or contract terms;
- infer demographic characteristics;
- replace controlled experimentation for measuring retention uplift;
- claim causal effects of a retention intervention;
- make decisions using the churn score without considering the campaign economics under which the 0.40 threshold was derived.

---

## 4. Training and Evaluation Data

The supplied dataset contains:

| Dataset characteristic | Value |
|---|---:|
| Total customers | 7,043 |
| Churn customers | 1,869 |
| Churn prevalence | 26.54% |
| Development set | 5,634 |
| Final holdout | 1,409 |
| Development churn | 1,495 |
| Holdout churn | 374 |

The final project used a new stratified 80/20 split with random seed `20260920`.

The development set was used for model comparison, probability-calibration assessment, demographic-feature ablation, and preliminary fairness analysis.

The 1,409-customer final holdout was opened only after the pipeline was frozen.

The dataset contains no timestamp field. The evaluation is therefore an internal random-split evaluation rather than a temporal or external validation.

---

## 5. Data Preparation

`customerID` is excluded from prediction.

`TotalCharges` is converted from text to numeric. Blank values are accepted only where `tenure = 0`, in which case they are assigned a value of zero.

Numeric features are standardized using `StandardScaler`.

Categorical variables are transformed using:

```text
OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)
```

Preprocessing is fitted only on training data within each development fold.

The final predictive feature policy excludes:

```text
gender
SeniorCitizen
```

These variables are retained separately for fairness auditing.

---

## 6. Model Selection

Four candidates were evaluated using 5 × 3 repeated stratified cross-validation:

| Model | Mean PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|
| Majority baseline | 0.2654 | 0.5000 | 0.1949 |
| Logistic regression | **0.6549** | **0.8437** | **0.1361** |
| HistGradientBoosting | 0.6425 | 0.8339 | 0.1413 |
| ANN | 0.6532 | 0.8414 | 0.1370 |

The predictive candidates were treated as practical ties under the pre-registered paired-fold rule.

Logistic regression was selected because the additional complexity of HistGradientBoosting and the ANN was not supported by a meaningful improvement in development performance.

---

## 7. Final Holdout Performance

At the fixed 0.40 operating threshold:

| Metric | Estimate | 95% bootstrap CI |
|---|---:|---:|
| Accuracy | 0.7899 | 0.7693 to 0.8119 |
| Precision | 0.5942 | 0.5459 to 0.6437 |
| Recall | 0.6578 | 0.6081 to 0.7062 |
| F1 | 0.6244 | 0.5845 to 0.6650 |
| PR-AUC | 0.6484 | 0.5958 to 0.7019 |
| ROC-AUC | 0.8470 | 0.8255 to 0.8680 |
| Brier score | 0.1356 | 0.1260 to 0.1452 |

Final confusion matrix:

```text
TN = 867
FP = 168
FN = 128
TP = 246
```

The model selected **414 of 1,409 customers**, or **29.38%** of the holdout population.

---

## 8. Probability Calibration

Sigmoid calibration was specified before the final evaluation.

Development Brier score changed from approximately:

```text
Raw         0.13583
Calibrated  0.13587
```

Final holdout Brier score changed from:

```text
Raw         0.13554
Calibrated  0.13562
```

The hypothesis that logistic-regression probabilities would be materially over-confident was not supported.

The pre-specified sigmoid calibrator was retained rather than selecting another calibration method after inspecting results.

---

## 9. Business Decision Rule

The expected value of contact is:

\[
EV(\text{contact})
=
p_{\text{churn}}uM
-
C_{\text{contact}}
-
p_{\text{accept}}C_{\text{offer}}
\]

Base-case planning assumptions are:

| Assumption | Value |
|---|---:|
| Contact cost | \$2 |
| Offer cost | \$65 |
| Acceptance probability | 40% |
| Incremental retention uplift | 15% |
| Contribution-margin rate | 30% |
| Monthly revenue proxy | \$65 |
| Value horizon | 24 months |

These are **planning assumptions and are not observed properties of the dataset**.

The exact calculated break-even threshold is:

\[
0.398860
\]

The pre-registered operational threshold is:

\[
0.40
\]

On the holdout, this produced an estimated:

```text
Selected customers       414
Expected prevented churn 36.48
Campaign cost            $11,592.00
Expected benefit         $17,073.27
Expected net value       $5,481.27
Expected ROI             47.28%
```

These are model-based expected values, not realized financial results.

---

## 10. Fairness and Group Performance

`gender` and `SeniorCitizen` are excluded from prediction but retained for monitoring.

### Gender

| Metric | Female | Male |
|---|---:|---:|
| Selection rate | 30.06% | 28.68% |
| Recall | 67.72% | 63.78% |
| FPR | 16.70% | 15.74% |

Holdout differences by gender were relatively small.

### SeniorCitizen

| Metric | Non-senior | Senior |
|---|---:|---:|
| Churn prevalence | 22.77% | 44.81% |
| Selection rate | 25.60% | 47.72% |
| Precision | 55.18% | 70.43% |
| Recall | 62.03% | 75.00% |
| FPR | 14.86% | 25.56% |
| Brier | 0.1270 | 0.1773 |
| Calibration gap | +1.12 pp | -6.30 pp |

These differences require monitoring but do not by themselves establish discriminatory model behaviour. `SeniorCitizen` is not a predictive input, and the groups also differ substantially in observed churn prevalence.

---

## 11. Known Limitations

The model has several important limitations.

The final holdout is a fresh internal split, but the underlying dataset had already been used during Module 6. The evaluation therefore does not provide fully independent external validation.

The source dataset has no timestamp information, so temporal generalization cannot be tested.

The business uplift assumption of 15% is not estimated from the dataset. The data contain no randomized treatment information.

The model estimates churn risk rather than causal responsiveness to an intervention.

Expected campaign ROI is highly sensitive to the assumed intervention uplift and retained-customer contribution margin.

The SeniorCitizen group showed poorer calibration and a higher false-positive rate on the final holdout.

The model may also be affected by future changes in customer behaviour, pricing, service offerings, competitors, or campaign policy.

---

## 12. Production Monitoring

Production monitoring should include:

```text
Input schema and missing values
Feature distribution drift
Prediction distribution drift
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

## 13. Feedback and Treatment Bias

Retention intervention changes the outcome the model is attempting to predict.

A customer correctly identified as high risk may receive an intervention and subsequently remain. Without treatment information, later training data could incorrectly represent that customer as a natural non-churner.

Production data should therefore retain treatment assignment, contact status, offer acceptance, model version, original predicted probability, and subsequent churn outcome.

A randomized untreated control group should be maintained where operationally and ethically appropriate so that true campaign uplift can be estimated.

---

## 14. Versioning and Reproducibility

The production artifact should version the following components together:

```text
Feature policy
StandardScaler
OneHotEncoder
Logistic regression model
Sigmoid calibrator
0.40 operating threshold
Dependency versions
```

Scoring jobs should reference a pinned registry version rather than an implicit latest model.

The final-project pipeline was frozen before holdout evaluation, and **11 automated tests passed** before the holdout was opened.

---

## 15. Responsible Owner

The production model should have named owners for:

```text
Model performance
Data quality
Retention campaign policy
Privacy and access control
Fairness monitoring
Model promotion and rollback
```

Changes to the model, feature policy, calibration method, or decision threshold should pass through a documented approval process rather than being modified directly in production.

---

That gives us the **required separate Model Card** and keeps it consistent with the report evidence. 

